import Pkg

const SUITE_DIR = @__DIR__
const TEMP_EXPERIMENTS_DIR = dirname(SUITE_DIR)
const REPO_ROOT = normpath(joinpath(SUITE_DIR, "..", ".."))
const TRAINING_PROJECT_DIR =
    joinpath(REPO_ROOT, "src", "ContextualDFL", "ContextualDFLTraining")

Pkg.activate(TRAINING_PROJECT_DIR; io=devnull)
if get(ENV, "DFL_SUITE_INSTANTIATE", "0") == "1"
    Pkg.instantiate()
end

using CSV
using ContextualDFL
using ContextualDFLExperiments
using Dates
using Flux
using LinearAlgebra
using Random
using Serialization
using SHA
using Statistics

const CONFIG_VERSION = "temp-dfl-suite-v1"
const BASE_TRAINING_CONTEXTS = 100
const BASE_TEST_CONTEXTS = 30
const BASE_TEST_SCENARIOS_PER_CONTEXT = 100
const SMOKE_TRAINING_CONTEXTS = 2
const SMOKE_TEST_CONTEXTS = 2
const SMOKE_TEST_SCENARIOS_PER_CONTEXT = 2
const DEMAND_SIGMA = 5.0
const DEMAND_POWER = 2.0
const CONTEXT_TERMS = 3
const BASE_HIDDEN_SIZE = 128
const BASE_DEPTH = 4
const BASE_ACTIVATION = :silu
const BASE_BATCH_SIZE = 1
const BASE_LEARNING_RATE = 1e-3
const BASE_SEED = 202_605_050
const TEST_SEED = 1
const SMOKE_TEST_SEED = 11
const BASE_MU_VALUES =
    [1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.08, 0.06, 0.04, 0.02, 0.01]
const BASE_STAGE_EPOCHS = [20, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]
const ANNEALED_RHOS = [1e-4, 3e-4, 1e-3, 3e-3]
const PURE_RHOS = [3e-4, 1e-3, 3e-3, 1e-2, 3e-2]
const KNN_VALUES = [2, 3, 5, 10]
const BATCH_VALUES = [4, 8, 16, 32]
const DEEP_DEPTHS = [5, 6, 7, 8]
const WIDE_WIDTHS = [64, 256, 512, 1024]

suite_path(parts...) = joinpath(SUITE_DIR, parts...)
artifact_path(parts...) = suite_path("artifacts", parts...)
run_root(; smoke=false) = suite_path("runs", smoke ? "smoke" : "full")
summary_root(; smoke=false) = suite_path("summaries", smoke ? "smoke" : "full")

unix_milliseconds() = round(Int64, time() * 1000)

function serialized_digest(value)
    io = IOBuffer()
    Serialization.serialize(io, value)
    return "sha1:" * bytes2hex(sha1(take!(io)))
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

demand_count(problem) = size(problem.problem_data.service_rate_parameters, 2)

function activation_function(name)
    symbol = Symbol(lowercase(String(string(name))))
    symbol == :relu && return Flux.relu
    symbol in (:silu, :swish) && return Flux.swish
    symbol in (:gelu, :geelu) && return Flux.gelu
    symbol == :tanh && return tanh
    symbol in (:identity, :linear, :none) && return identity
    throw(ArgumentError("unsupported activation: $(name)"))
end

function dense_initializer(seed)
    seed === nothing && return Flux.glorot_uniform
    return Flux.glorot_uniform(Random.MersenneTwister(Int(seed)))
end

softplus_output(x) = Flux.softplus.(x)

function build_model(config, problem)
    depth = Int(config.depth)
    depth > 0 || throw(ArgumentError("depth must be positive."))
    hidden_size = Int(config.hidden_size)
    hidden_size > 0 || throw(ArgumentError("hidden_size must be positive."))
    activation = activation_function(config.activation)
    output_dimension = demand_count(problem) * Int(config.nr_scenarios)
    init = dense_initializer(config.model_seed)

    layers = Any[Flux.Dense(3 => hidden_size, activation; init=init)]
    for _ in 2:depth
        push!(layers, Flux.Dense(hidden_size => hidden_size, activation; init=init))
    end
    push!(layers, Flux.Dense(hidden_size => output_dimension; init=init))
    push!(layers, softplus_output)
    return Flux.Chain(layers...) |> Flux.f64
end

function build_dfl_loss(objects, config)
    return ContextualDFL.DflScenLoss(
        objects.scenario_decoder,
        objects.reference_scenario_decoder,
        objects.solver,
        objects.program;
        nr_scenarios=Int(config.nr_scenarios),
    )
end

function generate_resource_allocation_dataset(; seed, context_count, scenarios_per_context)
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
    dataset = generate_resource_allocation_dataset(
        seed=Int(config.data_seed),
        context_count=Int(config.training_contexts),
        scenarios_per_context=1,
    )
    if Int(config.knn_k) > 0
        return homogenize_training_dataset(dataset, Int(config.knn_k))
    end
    return dataset
end

function squared_distance(a, b)
    length(a) == length(b) || throw(DimensionMismatch("context dimensions differ."))
    return sum(abs2, Float64.(a) .- Float64.(b))
end

function nearest_other_indices(contexts, index, k)
    distances = [
        (j, squared_distance(contexts[index], contexts[j])) for j in eachindex(contexts) if j != index
    ]
    sort!(distances; by=item -> item[2])
    return [item[1] for item in distances[1:min(k, length(distances))]]
end

function homogenize_training_dataset(dataset, k)
    k > 0 || return dataset
    length(dataset) > 1 || return dataset
    contexts = [point.context for point in dataset]
    homogenized = ContextualDFL.ContextualDataPoint[]
    for index in eachindex(dataset)
        scenarios = copy(dataset[index].scenario_parameters)
        for neighbor in nearest_other_indices(contexts, index, k)
            append!(scenarios, dataset[neighbor].scenario_parameters)
        end
        push!(
            homogenized,
            ContextualDFL.ContextualDataPoint(copy(dataset[index].context), scenarios),
        )
    end
    return homogenized
end

function test_cache_paths(; smoke=false)
    dir = artifact_path(smoke ? "smoke_test_cache" : "test_cache")
    return (;
        dir=dir,
        dataset=joinpath(dir, "test_dataset.jls"),
        optimal_results=joinpath(dir, "test_optimal_results.jls"),
        metadata_jls=joinpath(dir, "metadata.jls"),
        metadata_csv=joinpath(dir, "metadata.csv"),
    )
end

function test_cache_exists(; smoke=false)
    paths = test_cache_paths(smoke=smoke)
    return isfile(paths.dataset) && isfile(paths.optimal_results) && isfile(paths.metadata_jls)
end

function load_test_cache(; smoke=false)
    paths = test_cache_paths(smoke=smoke)
    test_cache_exists(smoke=smoke) ||
        error("missing test cache at $(paths.dir); run run_suite.jl --precompute first")
    return (;
        dataset=Serialization.deserialize(paths.dataset),
        optimal_results=Serialization.deserialize(paths.optimal_results),
        metadata=Serialization.deserialize(paths.metadata_jls),
    )
end

function ensure_test_cache!(; smoke=false, force=false)
    paths = test_cache_paths(smoke=smoke)
    if !force && test_cache_exists(smoke=smoke)
        return load_test_cache(smoke=smoke)
    end

    mkpath(paths.dir)
    context_count = smoke ? SMOKE_TEST_CONTEXTS : BASE_TEST_CONTEXTS
    scenarios_per_context = smoke ? SMOKE_TEST_SCENARIOS_PER_CONTEXT : BASE_TEST_SCENARIOS_PER_CONTEXT
    seed = smoke ? SMOKE_TEST_SEED : TEST_SEED
    started_at = unix_milliseconds()
    dataset = generate_resource_allocation_dataset(
        seed=seed,
        context_count=context_count,
        scenarios_per_context=scenarios_per_context,
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
        contexts=context_count,
        scenarios_per_context=scenarios_per_context,
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

function scaled_stage_epochs(batch_size; smoke=false)
    smoke && return [1, 1]
    factor = sqrt(Float64(batch_size))
    return [max(1, round(Int, epochs * factor)) for epochs in BASE_STAGE_EPOCHS]
end

function baseline_stage_epochs(; smoke=false)
    smoke && return [1, 1]
    return copy(BASE_STAGE_EPOCHS)
end

function schedule_values_for_config(config)
    epochs_by_stage = collect(Int.(config.stage_epochs))
    if Bool(config.smoke)
        total = sum(epochs_by_stage)
        if Symbol(config.schedule_kind) == :pure_rho
            return zeros(Float64, total), zeros(Float64, total)
        end
        mu_in = total == 1 ? [1.0] : [1.0, fill(0.01, total - 1)...]
        mu_ref = total == 1 ? [1.0] : [1.0, zeros(Float64, total - 1)...]
        Symbol(config.schedule_kind) == :no_finetune && (mu_ref[end] = mu_in[end])
        return mu_in, mu_ref
    end

    kind = Symbol(config.schedule_kind)
    kind == :pure_rho && return zeros(Float64, sum(epochs_by_stage)), zeros(Float64, sum(epochs_by_stage))
    length(epochs_by_stage) == length(BASE_MU_VALUES) + 1 ||
        throw(ArgumentError("stage_epochs must have $(length(BASE_MU_VALUES) + 1) entries."))

    if kind == :no_annealing
        mu_in = vcat(
            fill(BASE_MU_VALUES[1], epochs_by_stage[1]),
            fill(last(BASE_MU_VALUES), sum(epochs_by_stage[2:end])),
        )
        main_epochs = sum(epochs_by_stage[1:(end - 1)])
        mu_ref = vcat(
            fill(BASE_MU_VALUES[1], epochs_by_stage[1]),
            fill(last(BASE_MU_VALUES), main_epochs - epochs_by_stage[1]),
            zeros(Float64, epochs_by_stage[end]),
        )
        return mu_in, mu_ref
    end

    mu_main = Float64[]
    for (mu, epochs) in zip(BASE_MU_VALUES, epochs_by_stage[1:(end - 1)])
        append!(mu_main, fill(mu, epochs))
    end
    mu_in = vcat(mu_main, fill(last(BASE_MU_VALUES), epochs_by_stage[end]))
    mu_ref = if kind == :no_finetune
        copy(mu_in)
    else
        vcat(copy(mu_main), zeros(Float64, epochs_by_stage[end]))
    end
    return mu_in, mu_ref
end

function rho_values_for_config(config)
    total_epochs = sum(Int.(config.stage_epochs))
    rho = Float64(config.rho)
    kind = Symbol(config.schedule_kind)
    if kind == :pure_rho
        return fill(rho, total_epochs), fill(rho, total_epochs)
    end
    final_epochs = last(Int.(config.stage_epochs))
    main_epochs = total_epochs - final_epochs
    return fill(rho, total_epochs), vcat(fill(rho, main_epochs), zeros(Float64, final_epochs))
end

function schedule_preview(values; n=5)
    isempty(values) && return ""
    keep = min(Int(n), length(values))
    head = join(round.(values[1:keep]; digits=6), "|")
    length(values) <= keep && return head
    return head * "|...|" * string(round(last(values); digits=6))
end

function seed_bundle(replicate)
    base = BASE_SEED + 10_000 * Int(replicate)
    return (;
        data_seed=base + 1,
        model_seed=base + 2,
        training_seed=base + 3,
    )
end

function safe_path_part(text)
    return replace(String(text), r"[^A-Za-z0-9_.=-]" => "_")
end

function run_id(group, candidate, replicate; smoke=false)
    prefix = smoke ? "smoke" : "full"
    return join((prefix, String(group), String(candidate), "rep" * lpad(string(Int(replicate)), 2, "0")), "_")
end

function run_rel_dir(config)
    return joinpath(
        Bool(config.smoke) ? "smoke" : "full",
        safe_path_part(config.group),
        safe_path_part(config.candidate_name),
        "rep" * lpad(string(Int(config.replicate)), 2, "0"),
    )
end

function base_config(; group, candidate_name, replicate, smoke=false, overrides...)
    seeds = seed_bundle(replicate)
    stage_epochs = smoke ? [1, 1] : baseline_stage_epochs()
    config = (;
        version=CONFIG_VERSION,
        smoke=Bool(smoke),
        group=String(group),
        candidate_name=String(candidate_name),
        replicate=Int(replicate),
        run_id=run_id(group, candidate_name, replicate; smoke=smoke),
        data_seed=seeds.data_seed,
        model_seed=seeds.model_seed,
        training_seed=seeds.training_seed,
        test_seed=smoke ? SMOKE_TEST_SEED : TEST_SEED,
        training_contexts=smoke ? SMOKE_TRAINING_CONTEXTS : BASE_TRAINING_CONTEXTS,
        hidden_size=BASE_HIDDEN_SIZE,
        depth=BASE_DEPTH,
        activation=BASE_ACTIVATION,
        nr_scenarios=1,
        loss_kind=:dfl,
        schedule_kind=:baseline,
        rho=0.0,
        batch_size=BASE_BATCH_SIZE,
        stage_epochs=stage_epochs,
        learning_rate=BASE_LEARNING_RATE,
        reset_optimizer_each_epoch=true,
        knn_k=0,
        checkpoint_interval=smoke ? 1 : 10,
    )
    merged = merge(config, NamedTuple(overrides))
    return merge(merged, (; run_rel_dir=run_rel_dir(merged)))
end

function experiment_configs(; smoke=false)
    configs = NamedTuple[]
    push_config!(; kwargs...) = push!(configs, base_config(; smoke=smoke, kwargs...))
    reps10 = smoke ? 1 : 10
    reps5 = smoke ? 1 : 5
    reps4 = smoke ? 1 : 4

    for replicate in 1:reps10
        push_config!(group="00_baseline", candidate_name="standard", replicate=replicate)
        push_config!(
            group="00_baseline",
            candidate_name="persistent_adam",
            replicate=replicate,
            reset_optimizer_each_epoch=false,
        )
    end

    for replicate in 1:reps10
        push_config!(group="01_ablations", candidate_name="no_finetune", replicate=replicate, schedule_kind=:no_finetune)
        push_config!(group="01_ablations", candidate_name="no_annealing", replicate=replicate, schedule_kind=:no_annealing)
        push_config!(group="01_ablations", candidate_name="mse", replicate=replicate, loss_kind=:mse)
    end

    for scenarios in (smoke ? [2] : [2, 4]), replicate in 1:reps5
        push_config!(
            group="02_multi_scenario",
            candidate_name="scenarios$(scenarios)",
            replicate=replicate,
            nr_scenarios=scenarios,
        )
    end

    for rho in (smoke ? [first(ANNEALED_RHOS)] : ANNEALED_RHOS), replicate in 1:reps5
        push_config!(
            group="03_quadratic_smoothing",
            candidate_name="rho_" * replace(string(rho), "." => "p", "-" => "m"),
            replicate=replicate,
            rho=rho,
        )
    end

    for rho in (smoke ? [first(PURE_RHOS)] : PURE_RHOS), replicate in 1:reps5
        push_config!(
            group="04_pure_quadratic",
            candidate_name="pure_rho_" * replace(string(rho), "." => "p", "-" => "m"),
            replicate=replicate,
            rho=rho,
            schedule_kind=:pure_rho,
        )
    end

    for k in (smoke ? [first(KNN_VALUES)] : KNN_VALUES), replicate in 1:reps5
        push_config!(
            group="05_knn_equilibration",
            candidate_name="k$(k)",
            replicate=replicate,
            knn_k=k,
        )
    end

    for batch_size in (smoke ? [first(BATCH_VALUES)] : BATCH_VALUES), replicate in 1:reps4
        push_config!(
            group="06_batch_size",
            candidate_name="batch$(batch_size)",
            replicate=replicate,
            batch_size=batch_size,
            stage_epochs=scaled_stage_epochs(batch_size; smoke=smoke),
        )
    end

    for depth in (smoke ? [first(DEEP_DEPTHS)] : DEEP_DEPTHS), replicate in 1:reps5
        push_config!(
            group="07_deep_network",
            candidate_name="depth$(depth)",
            replicate=replicate,
            depth=depth,
        )
    end

    for width in (smoke ? [first(WIDE_WIDTHS)] : WIDE_WIDTHS), replicate in 1:reps5
        push_config!(
            group="08_wide_network",
            candidate_name="width$(width)",
            replicate=replicate,
            hidden_size=width,
        )
    end

    return configs
end

run_dir(config) = suite_path("runs", config.run_rel_dir)
config_path(config) = joinpath(run_dir(config), "config.jls")
config_csv_path(config) = joinpath(run_dir(config), "config.csv")
epochs_csv_path(config) = joinpath(run_dir(config), "epochs.csv")
run_result_csv_path(config) = joinpath(run_dir(config), "run_result.csv")
run_result_jls_path(config) = joinpath(run_dir(config), "run_result.jls")
test_per_sample_csv_path(config) = joinpath(run_dir(config), "test_per_sample.csv")
checkpoint_path(config) = joinpath(run_dir(config), "checkpoint_latest.jls")
checkpoint_dir(config) = joinpath(run_dir(config), "checkpoints")
stdout_path(config) = joinpath(run_dir(config), "stdout.log")
stderr_path(config) = joinpath(run_dir(config), "stderr.log")
error_path(config) = joinpath(run_dir(config), "error.txt")

function write_config!(config)
    mkpath(run_dir(config))
    Serialization.serialize(config_path(config), config)
    write_rows_csv(config_csv_path(config), [config_summary_row(config)])
    return config_path(config)
end

read_config(path) = Serialization.deserialize(path)

function config_summary_row(config)
    mu_in, mu_ref = schedule_values_for_config(config)
    rho_in, rho_ref = rho_values_for_config(config)
    return (;
        run_id=config.run_id,
        group=config.group,
        candidate_name=config.candidate_name,
        replicate=config.replicate,
        data_seed=config.data_seed,
        model_seed=config.model_seed,
        training_seed=config.training_seed,
        test_seed=config.test_seed,
        training_contexts=config.training_contexts,
        hidden_size=config.hidden_size,
        depth=config.depth,
        activation=String(config.activation),
        nr_scenarios=config.nr_scenarios,
        loss_kind=String(config.loss_kind),
        schedule_kind=String(config.schedule_kind),
        rho=config.rho,
        batch_size=config.batch_size,
        total_epochs=sum(Int.(config.stage_epochs)),
        stage_epochs=join(string.(config.stage_epochs), "|"),
        learning_rate=config.learning_rate,
        reset_optimizer_each_epoch=config.reset_optimizer_each_epoch,
        knn_k=config.knn_k,
        checkpoint_interval=config.checkpoint_interval,
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
    isfile(run_result_jls_path(config)) || return false
    result = Serialization.deserialize(run_result_jls_path(config))
    return hasproperty(result, :status) && result.status == "ok"
end

function final_training_loss_from_epochs(epoch_rows)
    isempty(epoch_rows) && return NaN
    return Float64(last(epoch_rows).training_loss)
end

function policy_mu_rho(config)
    mu_in, _ = schedule_values_for_config(config)
    rho_in, _ = rho_values_for_config(config)
    return (;
        mu=isempty(mu_in) ? 0.0 : Float64(last(mu_in)),
        rho=isempty(rho_in) ? 0.0 : Float64(last(rho_in)),
    )
end

function infer_decision(model, context, objects; nr_scenarios, mu, rho)
    output = model(context)
    W_eq, W_ineq, T_eq, T_ineq, h_eq, h_ineq, q =
        ContextualDFL.decode_scenario_collection(
            objects.scenario_decoder,
            output;
            nr_scenarios=Int(nr_scenarios),
        )
    z, _, _, _, _, _ = ContextualDFL.solve(
        objects.solver,
        objects.program,
        W_eq,
        W_ineq,
        T_eq,
        T_ineq,
        h_eq,
        h_ineq,
        q;
        μ=mu,
        ρ=rho,
    )
    return z
end

function decision_set_for_model(model, dataset, objects, config)
    try
        Flux.testmode!(model)
    catch
    end
    policy = policy_mu_rho(config)
    decisions = [
        infer_decision(
            model,
            point.context,
            objects;
            nr_scenarios=Int(config.nr_scenarios),
            mu=policy.mu,
            rho=policy.rho,
        ) for point in dataset
    ]
    return reduce(hcat, decisions)
end

function evaluation_vector_field(row, preferred_key, fallback_key)
    if hasproperty(row, preferred_key)
        return getproperty(row, preferred_key)
    elseif hasproperty(row, fallback_key)
        return getproperty(row, fallback_key)
    end
    return Float64[]
end

function evaluate_model(config, model)
    cache = load_test_cache(smoke=Bool(config.smoke))
    objects = problem_objects()
    decision_set = decision_set_for_model(model, cache.dataset, objects, config)
    evaluation = ContextualDFLExperiments.evaluate_policy_against_optimum(
        decision_set,
        cache.dataset,
        objects.program,
        objects.reference_scenario_decoder,
        objects.solver;
        optimal_results=cache.optimal_results,
        split_name=:test,
        mu=0.0,
        rho=0.0,
    )

    per_sample_rows = [
        (;
            run_id=config.run_id,
            group=config.group,
            candidate_name=config.candidate_name,
            replicate=Int(config.replicate),
            sample_index=row.sample_index,
            policy_value=row.policy_value,
            optimal_value=row.optimal_value,
            regret=row.regret,
            relative_regret=row.relative_regret,
            optimality_gap_percent=100 * row.relative_regret,
            policy_collection_values=evaluation_vector_field(row, :policy_collection_values, :policy_batch_values),
            optimal_collection_values=evaluation_vector_field(row, :optimal_collection_values, :optimal_batch_values),
            gap_values=evaluation_vector_field(row, :gap_values, :gap_values),
        ) for row in evaluation.per_sample
    ]
    write_rows_csv(test_per_sample_csv_path(config), per_sample_rows)

    metrics = evaluation.metrics
    value(name, default=NaN) = hasproperty(metrics, name) ? Float64(getproperty(metrics, name)) : default
    mean_relative_regret = value(:test_relative_regret_mean)
    return (;
        mean_test_relative_regret=mean_relative_regret,
        optimality_gap_percent=100 * mean_relative_regret,
        mean_test_regret=value(:test_regret_mean),
        mean_test_policy_value=value(:test_policy_value_mean),
        mean_test_optimal_value=value(:test_optimal_value_mean),
        test_sample_count=value(:test_sample_count),
        test_policy_eval_seconds=value(:test_policy_eval_seconds),
    )
end

function scenario_target(point)
    targets = [Vector{Float64}(scenario.h_eq_xi) for scenario in point.scenario_parameters]
    return vec(mean(reduce(hcat, targets); dims=2))
end

function mse_batch_loss(model, data_set, indices)
    return Statistics.mean(
        mean(abs2, model(data_set[index].context) .- scenario_target(data_set[index]))
        for index in indices
    )
end

function train_mse_one_epoch!(model, data_set, config, opt_state)
    optimizer = Flux.Adam(Float64(config.learning_rate))
    state = if opt_state === nothing || Bool(config.reset_optimizer_each_epoch)
        Flux.setup(optimizer, model)
    else
        opt_state
    end
    indices = collect(eachindex(data_set))
    losses = Float64[]
    for batch in Iterators.partition(indices, Int(config.batch_size))
        batch_indices = collect(batch)
        loss_value, gradients = Flux.withgradient(model) do trainable_model
            mse_batch_loss(trainable_model, data_set, batch_indices)
        end
        Flux.update!(state, model, gradients[1])
        push!(losses, Float64(loss_value))
    end
    return (; model=model, loss=mean(losses), display_loss=mean(losses), opt_state=state, iterations=length(losses))
end

function mean_or_nan(values)
    clean = Float64[x for x in values if isfinite(x)]
    isempty(clean) && return NaN
    return mean(clean)
end

function std_or_nan(values)
    clean = Float64[x for x in values if isfinite(x)]
    length(clean) <= 1 && return NaN
    return std(clean)
end

function read_run_result_or_missing(config)
    if isfile(run_result_jls_path(config))
        return Serialization.deserialize(run_result_jls_path(config))
    end
    return (;
        status="missing",
        run_id=config.run_id,
        group=config.group,
        candidate_name=config.candidate_name,
        replicate=config.replicate,
        mean_test_relative_regret=Inf,
        optimality_gap_percent=Inf,
        final_training_loss=Inf,
        error="missing run_result.jls",
    )
end

function summarize_group(configs)
    rows = [read_run_result_or_missing(config) for config in configs]
    baseline_rows = [
        row for row in rows if row.group == "00_baseline" &&
        row.candidate_name == "standard" &&
        row.status == "ok" &&
        isfinite(Float64(row.final_training_loss))
    ]
    baseline_loss = Dict(Int(row.replicate) => Float64(row.final_training_loss) for row in baseline_rows)
    candidates = sort(unique(String(row.candidate_name) for row in rows))
    summary_rows = NamedTuple[]
    for candidate in candidates
        candidate_rows = [row for row in rows if String(row.candidate_name) == candidate]
        ok_rows = [row for row in candidate_rows if row.status == "ok"]
        regrets = [Float64(row.mean_test_relative_regret) for row in ok_rows]
        gaps = [Float64(row.optimality_gap_percent) for row in ok_rows]
        final_losses = [Float64(row.final_training_loss) for row in ok_rows]
        loss_gaps = Float64[]
        for row in ok_rows
            key = Int(row.replicate)
            haskey(baseline_loss, key) || continue
            push!(loss_gaps, Float64(row.final_training_loss) - baseline_loss[key])
        end
        push!(
            summary_rows,
            (;
                candidate_name=candidate,
                run_count=length(candidate_rows),
                ok_count=length(ok_rows),
                failed_count=length(candidate_rows) - length(ok_rows),
                mean_test_relative_regret=mean_or_nan(regrets),
                std_test_relative_regret=std_or_nan(regrets),
                mean_optimality_gap_percent=mean_or_nan(gaps),
                std_optimality_gap_percent=std_or_nan(gaps),
                mean_final_training_loss=mean_or_nan(final_losses),
                std_final_training_loss=std_or_nan(final_losses),
                average_final_training_loss_gap=mean_or_nan(loss_gaps),
            ),
        )
    end
    return (; runs=rows, summary=summary_rows)
end

function write_all_summaries!(configs; smoke=false)
    root = summary_root(smoke=smoke)
    mkpath(root)
    write_rows_csv(joinpath(root, "all_runs.csv"), [read_run_result_or_missing(config) for config in configs])
    write_rows_csv(joinpath(root, "all_configs.csv"), [config_summary_row(config) for config in configs])
    for group in sort(unique(config.group for config in configs))
        group_configs = [config for config in configs if config.group == group]
        group_summary = summarize_group(group_configs)
        dir = joinpath(root, group)
        mkpath(dir)
        write_rows_csv(joinpath(dir, "runs.csv"), group_summary.runs)
        write_rows_csv(joinpath(dir, "summary.csv"), group_summary.summary)
    end
    return root
end

function shell_quote(text)
    return "'" * replace(String(text), "'" => "'\\''") * "'"
end
