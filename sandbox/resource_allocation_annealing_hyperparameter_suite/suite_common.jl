import Pkg

const SUITE_DIR = @__DIR__
const REPO_ROOT = normpath(joinpath(SUITE_DIR, "..", ".."))
const TRAINING_PROJECT_DIR =
    joinpath(REPO_ROOT, "src", "ContextualDFL", "ContextualDFLTraining")

Pkg.activate(TRAINING_PROJECT_DIR; io=devnull)

using CSV
using ContextualDFL
using ContextualDFLExperiments
using Dates
using Flux
using Random
using Serialization
using SHA
using Statistics

const CONFIG_VERSION = "resource-allocation-annealing-suite-v1"
const DEFAULT_REMOTE_HOST = "gcp-big"
const DEFAULT_REMOTE_REPO = "/home/rwl/ProblemBasedScenarioGeneration"
const DEFAULT_REMOTE_JULIA = "/home/rwl/.juliaup/bin/julia"
const DEFAULT_REMOTE_PROJECT =
    joinpath(DEFAULT_REMOTE_REPO, "src", "ContextualDFL", "ContextualDFLTraining")

const BASE_TRAINING_CONTEXTS = 100
const BASE_TEST_CONTEXTS = 30
const BASE_TEST_SCENARIOS_PER_CONTEXT = 100
const SMOKE_TRAINING_CONTEXTS = 2
const SMOKE_TEST_CONTEXTS = 2
const SMOKE_TEST_SCENARIOS_PER_CONTEXT = 2

const DEMAND_SIGMA = 5.0
const DEMAND_POWER = 2.0
const CONTEXT_TERMS = 3
const NR_SCENARIOS = 1
const HIDDEN_SIZE = 128
const LEARNING_RATE = 1e-3
const BASE_BATCH_SIZE = 1
const BASE_TOTAL_EPOCHS = 130
const BASE_FINAL_EPOCHS = 10
const REPLICATES = 6
const SEED_BASE = 94_000

const DISCRETE_MU_VALUES =
    [1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.08, 0.06, 0.04, 0.02, 0.01]
const DISCRETE_MU_EPOCHS =
    [20, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]

const PHASES = (:depth, :activation, :mu_schedule, :rho, :batch_size)

suite_path(parts...) = joinpath(SUITE_DIR, parts...)
artifact_path(parts...) = suite_path("artifacts", parts...)
state_path(; smoke=false) = suite_path(smoke ? "suite_state_smoke.jls" : "suite_state.jls")

function unix_milliseconds()
    return round(Int64, time() * 1000)
end

function problem_objects()
    problem = ContextualDFLExperiments.ResourceAllocationProblem(
        ContextualDFLExperiments.default_resource_allocation_problem_data(),
    )
    solver = ContextualDFL.Solver(ContextualDFL.IpoptSolver(), ContextualDFL.HiGHSSolver())
    return (;
        problem=problem,
        solver=solver,
        program=ContextualDFLExperiments.stochastic_program(problem),
        scenario_decoder=ContextualDFLExperiments.ResourceAllocationDemandVectorDecoder(problem),
        reference_scenario_decoder=
            ContextualDFLExperiments.ResourceAllocationDemandParametricDecoder(problem),
    )
end

function demand_count(problem)
    return size(problem.problem_data.service_rate_parameters, 2)
end

function activation_function(name)
    symbol = Symbol(name)
    symbol == :relu && return Flux.relu
    symbol == :gelu && return Flux.gelu
    symbol in (:silu, :swish) && return Flux.swish
    throw(ArgumentError("unsupported activation: $(name)"))
end

function build_model(config, problem)
    depth = Int(config.depth)
    depth > 0 || throw(ArgumentError("depth must be positive."))
    activation = activation_function(config.activation)
    output_dimension = demand_count(problem) * Int(config.nr_scenarios)

    layers = Any[Flux.Dense(3 => Int(config.hidden_size), activation)]
    for _ in 2:depth
        push!(layers, Flux.Dense(Int(config.hidden_size) => Int(config.hidden_size), activation))
    end
    push!(layers, Flux.Dense(Int(config.hidden_size) => output_dimension, Flux.relu))
    return Flux.Chain(layers...) |> Flux.f64
end

function generate_resource_allocation_dataset(;
    seed,
    context_count,
    scenarios_per_context,
)
    objects = problem_objects()
    rng = Random.MersenneTwister(Int(seed))
    context_generator = ContextualDFLExperiments.ResourceAllocationContextDataGenerator(rng=rng)
    scenario_generator = ContextualDFLExperiments.ResourceAllocationScenarioDataGenerator(
        objects.problem;
        sigma=DEMAND_SIGMA,
        p=DEMAND_POWER,
        L=CONTEXT_TERMS,
        rng=rng,
    )

    contexts = [Vector{Float64}(context_generator()) for _ in 1:Int(context_count)]
    scenario_collections = [
        [scenario_generator(context) for _ in 1:Int(scenarios_per_context)] for
        context in contexts
    ]
    return ContextualDFLExperiments.generate_contextual_data_set(
        contexts,
        scenario_collections,
    )
end

function generate_training_dataset(config)
    return generate_resource_allocation_dataset(
        seed=Int(config.seed),
        context_count=Int(config.training_contexts),
        scenarios_per_context=1,
    )
end

function test_cache_dir(; smoke=false)
    return artifact_path(smoke ? "smoke_test_data" : "test_data")
end

function test_cache_paths(; smoke=false)
    dir = test_cache_dir(smoke=smoke)
    return (;
        dir=dir,
        dataset=joinpath(dir, "test_dataset.jls"),
        optimal_results=joinpath(dir, "test_optimal_results.jls"),
        metadata_jls=joinpath(dir, "metadata.jls"),
        metadata_csv=joinpath(dir, "metadata.csv"),
    )
end

function expected_test_shape(; smoke=false)
    if smoke
        return (;
            contexts=SMOKE_TEST_CONTEXTS,
            scenarios_per_context=SMOKE_TEST_SCENARIOS_PER_CONTEXT,
        )
    end
    return (;
        contexts=BASE_TEST_CONTEXTS,
        scenarios_per_context=BASE_TEST_SCENARIOS_PER_CONTEXT,
    )
end

function test_cache_exists(; smoke=false)
    paths = test_cache_paths(smoke=smoke)
    return isfile(paths.dataset) && isfile(paths.optimal_results) && isfile(paths.metadata_jls)
end

function serialized_digest(value)
    io = IOBuffer()
    Serialization.serialize(io, value)
    return "sha1:" * bytes2hex(sha1(take!(io)))
end

function ensure_test_cache!(; smoke=false, force=false)
    paths = test_cache_paths(smoke=smoke)
    if !force && test_cache_exists(smoke=smoke)
        return load_test_cache(smoke=smoke)
    end

    mkpath(paths.dir)
    shape = expected_test_shape(smoke=smoke)
    seed = smoke ? 11 : 1
    started_at = unix_milliseconds()
    dataset = generate_resource_allocation_dataset(
        seed=seed,
        context_count=shape.contexts,
        scenarios_per_context=shape.scenarios_per_context,
    )
    objects = problem_objects()

    optimal_results = nothing
    solve_seconds = @elapsed begin
        optimal_results = ContextualDFLExperiments.solve_dataset_to_optimality(
            dataset,
            objects.program,
            objects.reference_scenario_decoder,
            objects.solver;
            mu=0.0,
            rho=0.0,
        )
    end

    metadata = (;
        version=CONFIG_VERSION,
        smoke=Bool(smoke),
        seed=seed,
        contexts=shape.contexts,
        scenarios_per_context=shape.scenarios_per_context,
        solve_seconds=solve_seconds,
        dataset_digest=serialized_digest(dataset),
        optimal_results_digest=serialized_digest(optimal_results),
        started_at=started_at,
        finished_at=unix_milliseconds(),
        dataset_path=paths.dataset,
        optimal_results_path=paths.optimal_results,
    )

    Serialization.serialize(paths.dataset, dataset)
    Serialization.serialize(paths.optimal_results, optimal_results)
    Serialization.serialize(paths.metadata_jls, metadata)
    write_rows_csv(paths.metadata_csv, [metadata])
    return (; dataset=dataset, optimal_results=optimal_results, metadata=metadata)
end

function load_test_cache(; smoke=false)
    paths = test_cache_paths(smoke=smoke)
    test_cache_exists(smoke=smoke) ||
        error("Missing precomputed test cache in $(paths.dir). Run run_single.jl --precompute first.")
    return (;
        dataset=Serialization.deserialize(paths.dataset),
        optimal_results=Serialization.deserialize(paths.optimal_results),
        metadata=Serialization.deserialize(paths.metadata_jls),
    )
end

function discrete_mu_schedules(total_epochs, final_epochs)
    total_epochs == BASE_TOTAL_EPOCHS && final_epochs == BASE_FINAL_EPOCHS ||
        throw(ArgumentError("discrete schedule is defined for 130 epochs with 10 final epochs."))

    mu_main = Float64[]
    for (value, epochs) in zip(DISCRETE_MU_VALUES, DISCRETE_MU_EPOCHS)
        append!(mu_main, fill(Float64(value), Int(epochs)))
    end

    mu_in = vcat(mu_main, fill(last(DISCRETE_MU_VALUES), final_epochs))
    mu_ref = vcat(copy(mu_main), zeros(Float64, final_epochs))
    return mu_in, mu_ref
end

function continuous_mu_main(kind, main_epochs)
    main_epochs <= 0 && return Float64[]
    main_epochs == 1 && return [1.0]

    start = 1.0
    stop = 0.01
    t_values = collect(range(0.0, 1.0; length=main_epochs))
    schedule = Symbol(kind)

    if schedule == :linear
        return collect(range(start, stop; length=main_epochs))
    elseif schedule == :geometric
        return exp.(range(log(start), log(stop); length=main_epochs))
    elseif schedule == :delayed_log_1_5
        return [exp(log(start) + (log(stop) - log(start)) * t^1.5) for t in t_values]
    elseif schedule == :delayed_log_2_0
        return [exp(log(start) + (log(stop) - log(start)) * t^2.0) for t in t_values]
    elseif schedule == :warmup20_geometric
        warmup = min(20, main_epochs)
        tail = main_epochs - warmup
        tail_values = tail <= 0 ? Float64[] : exp.(range(log(start), log(stop); length=tail + 1))[2:end]
        return vcat(fill(start, warmup), tail_values)
    end

    throw(ArgumentError("unsupported mu schedule kind: $(kind)"))
end

function mu_schedules_for_config(config)
    total_epochs = Int(config.epochs)
    final_epochs = Int(config.final_epochs)
    total_epochs >= 0 || throw(ArgumentError("epochs must be non-negative."))
    final_epochs >= 0 || throw(ArgumentError("final_epochs must be non-negative."))
    final_epochs <= total_epochs ||
        throw(ArgumentError("final_epochs cannot exceed epochs."))

    kind = Symbol(config.mu_schedule_kind)
    if kind == :discrete
        return discrete_mu_schedules(total_epochs, final_epochs)
    end

    main_epochs = total_epochs - final_epochs
    mu_main = continuous_mu_main(kind, main_epochs)
    mu_in = vcat(mu_main, fill(isempty(mu_main) ? 0.01 : last(mu_main), final_epochs))
    mu_ref = vcat(copy(mu_main), zeros(Float64, final_epochs))
    return mu_in, mu_ref
end

function rho_schedules_for_config(config)
    total_epochs = Int(config.epochs)
    final_epochs = Int(config.final_epochs)
    main_epochs = total_epochs - final_epochs
    rho = Float64(config.rho)
    rho_in = fill(rho, total_epochs)
    rho_ref = vcat(fill(rho, main_epochs), zeros(Float64, final_epochs))
    return rho_in, rho_ref
end

function schedule_preview(values; n=5)
    isempty(values) && return ""
    keep = min(Int(n), length(values))
    head = join(round.(values[1:keep]; digits=6), "|")
    if length(values) <= keep
        return head
    end
    return head * "|...|" * string(round(last(values); digits=6))
end

function replicate_seed(replicate)
    return SEED_BASE + Int(replicate)
end

function final_epochs_for(total_epochs; smoke=false)
    smoke && return 0
    total_epochs == BASE_TOTAL_EPOCHS && return BASE_FINAL_EPOCHS
    return max(1, round(Int, total_epochs * BASE_FINAL_EPOCHS / BASE_TOTAL_EPOCHS))
end

function base_run_config(; phase, candidate_name, replicate, smoke=false, overrides...)
    total_epochs = smoke ? 1 : BASE_TOTAL_EPOCHS
    config = (;
        version=CONFIG_VERSION,
        smoke=Bool(smoke),
        phase=Symbol(phase),
        candidate_name=String(candidate_name),
        replicate=Int(replicate),
        run_id=run_id(Symbol(phase), candidate_name, replicate; smoke=smoke),
        seed=replicate_seed(replicate),
        training_contexts=smoke ? SMOKE_TRAINING_CONTEXTS : BASE_TRAINING_CONTEXTS,
        nr_scenarios=NR_SCENARIOS,
        hidden_size=HIDDEN_SIZE,
        depth=3,
        activation=:relu,
        mu_schedule_kind=smoke ? :geometric : :discrete,
        rho=0.0,
        batch_size=BASE_BATCH_SIZE,
        epochs=total_epochs,
        final_epochs=final_epochs_for(total_epochs; smoke=smoke),
        learning_rate=LEARNING_RATE,
        reset_optimizer_each_epoch=true,
        metric=:mean_test_relative_regret,
    )
    merged = merge(config, NamedTuple(overrides))
    return merge(merged, (; run_rel_dir=run_rel_dir(merged)))
end

function run_id(phase, candidate_name, replicate; smoke=false)
    prefix = smoke ? "smoke" : "full"
    return join(
        (prefix, string(phase), String(candidate_name), "rep" * lpad(string(Int(replicate)), 2, "0")),
        "_",
    )
end

function safe_path_part(text)
    return replace(String(text), r"[^A-Za-z0-9_.=-]" => "_")
end

function run_rel_dir(config)
    return joinpath(
        "runs",
        config.smoke ? "smoke" : "full",
        safe_path_part(config.phase),
        safe_path_part(config.candidate_name),
        "rep" * lpad(string(Int(config.replicate)), 2, "0"),
    )
end

run_dir(config) = suite_path(config.run_rel_dir)
config_path(config) = joinpath(run_dir(config), "config.jls")
checkpoint_path(config) = joinpath(run_dir(config), "checkpoint.jls")
epochs_csv_path(config) = joinpath(run_dir(config), "epochs.csv")
run_result_csv_path(config) = joinpath(run_dir(config), "run_result.csv")
run_result_jls_path(config) = joinpath(run_dir(config), "run_result.jls")
test_per_sample_csv_path(config) = joinpath(run_dir(config), "test_per_sample.csv")
error_path(config) = joinpath(run_dir(config), "error.txt")

function write_config!(config)
    mkpath(run_dir(config))
    Serialization.serialize(config_path(config), config)
    write_rows_csv(joinpath(run_dir(config), "config.csv"), [config_summary_row(config)])
    return config_path(config)
end

read_config(path) = Serialization.deserialize(path)

function config_summary_row(config)
    mu_in, mu_ref = mu_schedules_for_config(config)
    rho_in, rho_ref = rho_schedules_for_config(config)
    return (;
        run_id=config.run_id,
        phase=String(config.phase),
        candidate_name=config.candidate_name,
        replicate=config.replicate,
        seed=config.seed,
        training_contexts=config.training_contexts,
        depth=config.depth,
        activation=String(config.activation),
        mu_schedule_kind=String(config.mu_schedule_kind),
        rho=config.rho,
        batch_size=config.batch_size,
        epochs=config.epochs,
        final_epochs=config.final_epochs,
        learning_rate=config.learning_rate,
        mu_in_preview=schedule_preview(mu_in),
        mu_ref_preview=schedule_preview(mu_ref),
        rho_in_preview=schedule_preview(rho_in),
        rho_ref_preview=schedule_preview(rho_ref),
        run_rel_dir=config.run_rel_dir,
    )
end

function csv_value(value)
    value === nothing && return missing
    value === missing && return missing
    value isa Symbol && return String(value)
    value isa AbstractVector && return join(string.(value), "|")
    value isa Tuple && return join(string.(value), "|")
    return value
end

function row_pairs(row)
    row isa NamedTuple && return pairs(row)
    row isa AbstractDict && return pairs(row)
    return pairs(Dict{Symbol,Any}(:value => row))
end

function write_rows_csv(path, rows)
    mkpath(dirname(path))
    if isempty(rows)
        write(path, "")
        return path
    end

    headers = Symbol[]
    seen = Set{Symbol}()
    for row in rows
        for (key, _) in row_pairs(row)
            symbol = Symbol(key)
            symbol in seen && continue
            push!(seen, symbol)
            push!(headers, symbol)
        end
    end
    sort!(headers; by=String)

    columns = map(headers) do header
        header => [
            csv_value(get(Dict(Symbol(k) => v for (k, v) in row_pairs(row)), header, missing))
            for row in rows
        ]
    end
    CSV.write(path, (; columns...); missingstring="")
    return path
end

function run_complete(config)
    path = run_result_jls_path(config)
    isfile(path) || return false
    result = Serialization.deserialize(path)
    return hasproperty(result, :status) && result.status == "ok"
end

function phase_dir(phase; smoke=false)
    return suite_path("phases", smoke ? "smoke" : "full", String(phase))
end

function phase_paths(phase; smoke=false)
    dir = phase_dir(phase; smoke=smoke)
    return (;
        dir=dir,
        configs=joinpath(dir, "configs.csv"),
        runs=joinpath(dir, "runs.csv"),
        epochs=joinpath(dir, "epochs.csv"),
        summary=joinpath(dir, "summary.csv"),
        decision=joinpath(dir, "decision.csv"),
        decision_md=joinpath(dir, "decision.md"),
    )
end

function phase1_configs(; smoke=false)
    reps = smoke ? 1 : REPLICATES
    return [
        base_run_config(
            phase=:depth,
            candidate_name="depth$(depth)",
            replicate=replicate,
            smoke=smoke,
            depth=depth,
        ) for depth in (1, 2, 3) for replicate in 1:reps
    ]
end

function phase2_configs(selection; smoke=false)
    reps = smoke ? 1 : REPLICATES
    depth = Int(selection.depth)
    return [
        base_run_config(
            phase=:activation,
            candidate_name=String(activation),
            replicate=replicate,
            smoke=smoke,
            depth=depth,
            activation=activation,
        ) for activation in (:gelu, :relu, :silu) for replicate in 1:reps
    ]
end

function phase3_configs(selection; smoke=false)
    reps = smoke ? 1 : REPLICATES
    kinds = (:linear, :geometric, :delayed_log_1_5, :delayed_log_2_0, :warmup20_geometric)
    return [
        base_run_config(
            phase=:mu_schedule,
            candidate_name=String(kind),
            replicate=replicate,
            smoke=smoke,
            depth=Int(selection.depth),
            activation=Symbol(selection.activation),
            mu_schedule_kind=kind,
        ) for kind in kinds for replicate in 1:reps
    ]
end

function phase4_configs(selection; smoke=false)
    reps = smoke ? 1 : REPLICATES
    return [
        base_run_config(
            phase=:rho,
            candidate_name="rho_" * replace(string(rho), "." => "p", "-" => "m"),
            replicate=replicate,
            smoke=smoke,
            depth=Int(selection.depth),
            activation=Symbol(selection.activation),
            mu_schedule_kind=Symbol(selection.mu_schedule_kind),
            rho=rho,
        ) for rho in (0.0, 1e-3, 1e-2) for replicate in 1:reps
    ]
end

function batch_epochs(batch_size; smoke=false)
    smoke && return 1
    batch_size == 1 && return 130
    batch_size == 8 && return 1000
    batch_size == 16 && return 1858
    batch_size == 32 && return 3250
    throw(ArgumentError("unsupported batch size: $(batch_size)"))
end

function phase5_configs(selection; smoke=false)
    reps = smoke ? 1 : REPLICATES
    smoothing_rho = Float64(selection.best_nonzero_rho)
    rhos = unique([0.0, smoothing_rho])
    configs = NamedTuple[]
    for rho in rhos, batch_size in (1, 8, 16, 32), replicate in 1:reps
        epochs = batch_epochs(batch_size; smoke=smoke)
        push!(
            configs,
            base_run_config(
                phase=:batch_size,
                candidate_name="batch$(batch_size)_rho_" *
                    replace(string(rho), "." => "p", "-" => "m"),
                replicate=replicate,
                smoke=smoke,
                depth=Int(selection.depth),
                activation=Symbol(selection.activation),
                mu_schedule_kind=Symbol(selection.mu_schedule_kind),
                rho=rho,
                batch_size=batch_size,
                epochs=epochs,
                final_epochs=final_epochs_for(epochs; smoke=smoke),
            ),
        )
    end
    return configs
end

function phase_configs(phase, selection; smoke=false)
    phase == :depth && return phase1_configs(smoke=smoke)
    phase == :activation && return phase2_configs(selection; smoke=smoke)
    phase == :mu_schedule && return phase3_configs(selection; smoke=smoke)
    phase == :rho && return phase4_configs(selection; smoke=smoke)
    phase == :batch_size && return phase5_configs(selection; smoke=smoke)
    throw(ArgumentError("unsupported phase: $(phase)"))
end

function default_selection()
    return (;
        depth=3,
        activation=:relu,
        mu_schedule_kind=:discrete,
        rho=0.0,
        best_nonzero_rho=1e-3,
        batch_size=1,
    )
end

function result_metric(result)
    hasproperty(result, :average_test_loss) && return Float64(result.average_test_loss)
    hasproperty(result, :mean_test_relative_regret) &&
        return Float64(result.mean_test_relative_regret)
    return Inf
end

function mean_or_nan(values)
    clean = Float64[x for x in values if isfinite(x)]
    isempty(clean) && return NaN
    return Statistics.mean(clean)
end

function std_or_nan(values)
    clean = Float64[x for x in values if isfinite(x)]
    length(clean) <= 1 && return NaN
    return Statistics.std(clean)
end

function summarize_phase_results(configs)
    rows = NamedTuple[]
    epoch_rows = NamedTuple[]
    for config in configs
        if isfile(run_result_jls_path(config))
            push!(rows, Serialization.deserialize(run_result_jls_path(config)))
        else
            push!(
                rows,
                (;
                    status="missing",
                    run_id=config.run_id,
                    phase=config.phase,
                    candidate_name=config.candidate_name,
                    replicate=config.replicate,
                    average_test_loss=Inf,
                    mean_test_relative_regret=Inf,
                    error="missing run_result.jls",
                ),
            )
        end

        if isfile(checkpoint_path(config))
            checkpoint = Serialization.deserialize(checkpoint_path(config))
            if hasproperty(checkpoint, :epoch_rows)
                append!(epoch_rows, checkpoint.epoch_rows)
            end
        end
    end

    candidates = unique(String(row.candidate_name) for row in rows)
    summary_rows = NamedTuple[]
    for candidate in sort(candidates)
        candidate_rows = [row for row in rows if String(row.candidate_name) == candidate]
        ok_rows = [row for row in candidate_rows if row.status == "ok"]
        metrics = [result_metric(row) for row in ok_rows]
        push!(
            summary_rows,
            (;
                candidate_name=candidate,
                run_count=length(candidate_rows),
                ok_count=length(ok_rows),
                failed_count=length(candidate_rows) - length(ok_rows),
                mean_test_relative_regret=mean_or_nan(metrics),
                std_test_relative_regret=std_or_nan(metrics),
                min_test_relative_regret=isempty(metrics) ? NaN : minimum(metrics),
                max_test_relative_regret=isempty(metrics) ? NaN : maximum(metrics),
            ),
        )
    end

    return (; runs=rows, epochs=epoch_rows, summary=summary_rows)
end

function choose_candidate(summary_rows)
    complete_rows = [
        row for row in summary_rows if row.ok_count == row.run_count && row.run_count > 0
    ]
    isempty(complete_rows) &&
        error("No candidate has all replicates completed successfully; refusing to auto-select.")
    return first(sort(complete_rows; by=row -> row.mean_test_relative_regret))
end

function parse_rho_candidate(name)
    text = replace(String(name), "rho_" => "", "p" => ".")
    parsed = tryparse(Float64, text)
    parsed === nothing && return 0.0
    return parsed
end

function update_selection(selection, phase, decision_row, summary_rows)
    if phase == :depth
        depth = parse(Int, replace(String(decision_row.candidate_name), "depth" => ""))
        return merge(selection, (; depth=depth))
    elseif phase == :activation
        return merge(selection, (; activation=Symbol(decision_row.candidate_name)))
    elseif phase == :mu_schedule
        return merge(selection, (; mu_schedule_kind=Symbol(decision_row.candidate_name)))
    elseif phase == :rho
        best_rho = parse_rho_candidate(decision_row.candidate_name)
        nonzero = [row for row in summary_rows if parse_rho_candidate(row.candidate_name) > 0]
        best_nonzero = isempty(nonzero) ? 0.0 :
            parse_rho_candidate(first(sort(nonzero; by=row -> row.mean_test_relative_regret)).candidate_name)
        return merge(selection, (; rho=best_rho, best_nonzero_rho=best_nonzero))
    elseif phase == :batch_size
        parts = split(String(decision_row.candidate_name), "_")
        batch = parse(Int, replace(parts[1], "batch" => ""))
        rho = parse_rho_candidate(join(parts[2:end], "_"))
        return merge(selection, (; batch_size=batch, rho=rho))
    end
    return selection
end

function write_phase_outputs!(phase, configs; smoke=false)
    paths = phase_paths(phase; smoke=smoke)
    mkpath(paths.dir)
    results = summarize_phase_results(configs)
    write_rows_csv(paths.configs, [config_summary_row(config) for config in configs])
    write_rows_csv(paths.runs, results.runs)
    write_rows_csv(paths.epochs, results.epochs)
    write_rows_csv(paths.summary, results.summary)
    decision = choose_candidate(results.summary)
    write_rows_csv(paths.decision, [decision])
    write(
        paths.decision_md,
        "# Phase $(phase) Decision\n\n" *
        "- Selected candidate: `$(decision.candidate_name)`\n" *
        "- Mean test relative regret: `$(decision.mean_test_relative_regret)`\n" *
        "- Successful replicates: `$(decision.ok_count)/$(decision.run_count)`\n",
    )
    return (; decision=decision, results=results)
end

function save_suite_state!(state; smoke=false)
    Serialization.serialize(state_path(smoke=smoke), state)
    return state
end

function load_suite_state(; smoke=false)
    path = state_path(smoke=smoke)
    isfile(path) || return (;
        version=CONFIG_VERSION,
        smoke=Bool(smoke),
        completed_phases=Symbol[],
        selection=default_selection(),
        updated_at=unix_milliseconds(),
    )
    return Serialization.deserialize(path)
end

function shell_quote(text)
    return "'" * replace(String(text), "'" => "'\\''") * "'"
end
