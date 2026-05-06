using ContextualDFL
using ContextualDFLExperiments
using ContextualDFLTraining
using Base64
using Distributed
using Flux
using Random
using Serialization
using Sockets
using Statistics

function ContextualDFL.solve(
    solver::ContextualDFL.HiGHSSolver,
    lp::ContextualDFL.LP;
    constraint_tolerance=1e-6,
    kwargs...,
)
    bound_lp, bound_map = ContextualDFL._extract_variable_bounds_for_solver(solver, lp)

    model = ContextualDFL.JuMP.Model(ContextualDFL.HiGHS.Optimizer)
    ContextualDFL.JuMP.set_silent(model)
    ContextualDFL.JuMP.set_optimizer_attribute(model, "threads", 1)
    for (attribute, value) in kwargs
        ContextualDFL.JuMP.set_optimizer_attribute(model, String(attribute), value)
    end

    n_variables = length(bound_lp.c)
    ContextualDFL.JuMP.@variable(model, z[1:n_variables])
    ContextualDFL._set_variable_bounds!(z, bound_lp.lower_bounds, bound_lp.upper_bounds)

    eq_constraints = ContextualDFL.JuMP.@constraint(model, bound_lp.A_eq * z .== bound_lp.b_eq)
    ineq_constraints = ContextualDFL.JuMP.@constraint(model, bound_lp.A_ineq * z .<= bound_lp.b_ineq)

    ContextualDFL.JuMP.@objective(model, Min, sum(bound_lp.c[j] * z[j] for j in 1:n_variables))
    ContextualDFL.JuMP.optimize!(model)

    status = ContextualDFL._assert_successful_solve(model, solver; accepted_statuses=("OPTIMAL",))
    z_value = ContextualDFL.JuMP.value.(z)
    ContextualDFL._assert_lp_solution_feasible(lp, z_value; atol=constraint_tolerance)

    lower_bound_dual, upper_bound_dual =
        ContextualDFL._normalized_variable_bound_duals(z, bound_lp.lower_bounds, bound_lp.upper_bounds)
    raw_result = ContextualDFL.BoundFormSolveResult(
        z_value,
        bound_lp.b_ineq - bound_lp.A_ineq * z_value,
        -ContextualDFL.JuMP.dual.(ineq_constraints),
        ContextualDFL.JuMP.dual.(eq_constraints),
        lower_bound_dual,
        upper_bound_dual,
        ContextualDFL.JuMP.objective_value(model),
        status,
        (;
            primal_status=ContextualDFL.JuMP.primal_status(model),
            dual_status=ContextualDFL.JuMP.dual_status(model),
            raw_status=ContextualDFL.JuMP.raw_status(model),
            solver=solver,
        ),
    )

    return ContextualDFL._reconstruct_original_lp_result(lp, bound_map, raw_result)
end

const SUITE_DIR = @__DIR__
const REPO_ROOT = normpath(joinpath(SUITE_DIR, "..", "..", ".."))
const TRAINING_PROJECT_DIR =
    joinpath(REPO_ROOT, "src", "ContextualDFL", "ContextualDFLTraining")

const SUITE_VERSION = "data-compute-mu-schedule-2026-05-06-v1"
const EXPERIMENT_SELECTOR = "resource_allocation/experiment_1_tiny"
const TRAINING_SEED_BASE = 20_260_506_000
const BASE_TRAINING_CONTEXTS = 100
const BASE_TOTAL_EPOCHS = 130
const BASE_HIDDEN_SIZE = 128
const BASE_DEPTH = 4
const BASE_ACTIVATION = :gelu
const BASE_BATCH_SIZE = 1
const BASE_LEARNING_RATE = 1e-3
const BASE_DROPOUT = 0.0
const BASE_OUTPUT_ACTIVATION = :softplus
const NR_SCENARIOS = 1
const CHECKPOINT_EVERY_EPOCHS = 5
const SMOKE_EPOCHS = 2
const SMOKE_TRAINING_CONTEXTS = 8
const SMOKE_TEST_CONTEXTS = 2
const SMOKE_EVALUATION_BATCHES = 1

const DEMAND_SIGMA = 5.0
const DEMAND_POWER = 2.0
const CONTEXT_TERMS = 3

const WORKER_TEST_CACHE = Ref{Any}(nothing)

const CONFIG_HEADERS = (
    :version,
    :run_id,
    :smoke,
    :wave_index,
    :phase,
    :candidate_name,
    :schedule_name,
    :replicate,
    :seed,
    :training_data_seed,
    :training_contexts,
    :epochs,
    :epoch_fraction,
    :depth,
    :hidden_size,
    :activation,
    :output_activation,
    :batch_size,
    :learning_rate,
    :policy_inference_mu,
    :mu_in_preview,
    :mu_ref_preview,
)

const ATTEMPT_HEADERS = (
    :attempt_id,
    :run_id,
    :wave_index,
    :phase,
    :candidate_name,
    :schedule_name,
    :replicate,
    :worker_id,
    :hostname,
    :pid,
    :started_at,
)

const EPOCH_HEADERS = (
    :attempt_id,
    :run_id,
    :wave_index,
    :phase,
    :candidate_name,
    :schedule_name,
    :replicate,
    :epoch,
    :mu_in,
    :mu_ref,
    :rho_in,
    :rho_ref,
    :iterations,
    :epoch_seconds,
    :training_loss,
    :display_loss,
    :train_target_mse,
    :train_target_mae,
    :test_target_mse,
    :test_target_mae,
    :training_contexts,
    :epochs,
    :depth,
    :hidden_size,
    :activation,
    :batch_size,
    :seed,
    :worker_id,
    :hostname,
    :created_at,
)

const RUN_HEADERS = (
    :attempt_id,
    :run_id,
    :status,
    :wave_index,
    :phase,
    :candidate_name,
    :schedule_name,
    :replicate,
    :seed,
    :training_contexts,
    :epochs,
    :epoch_fraction,
    :depth,
    :hidden_size,
    :activation,
    :batch_size,
    :policy_inference_mu,
    :test_relative_regret_mean,
    :test_relative_regret_median,
    :test_relative_regret_std,
    :test_relative_regret_p95,
    :test_regret_mean,
    :test_policy_value_mean,
    :test_optimal_value_mean,
    :test_sample_count,
    :test_evaluation_batches,
    :test_gap_std_mean,
    :test_gap_stderr_mean,
    :final_train_target_mse,
    :final_test_target_mse,
    :completed_epochs,
    :training_seconds,
    :evaluation_seconds,
    :total_seconds,
    :worker_id,
    :hostname,
    :pid,
    :started_at,
    :finished_at,
    :error,
)

const TEST_SAMPLE_HEADERS = (
    :attempt_id,
    :run_id,
    :wave_index,
    :phase,
    :candidate_name,
    :schedule_name,
    :replicate,
    :sample_index,
    :policy_value,
    :optimal_value,
    :regret,
    :relative_regret,
    :gap_std,
    :gap_stderr,
    :policy_collection_values,
    :optimal_collection_values,
    :gap_values,
)

const CHECKPOINT_HEADERS = (
    :attempt_id,
    :run_id,
    :wave_index,
    :phase,
    :candidate_name,
    :schedule_name,
    :replicate,
    :epoch,
    :checkpoint_kind,
    :checkpoint_path,
    :checkpoint_bytes,
    :worker_id,
    :hostname,
    :created_at,
)

const SUMMARY_HEADERS = (
    :phase,
    :candidate_name,
    :schedule_name,
    :run_count,
    :ok_count,
    :failed_count,
    :mean_test_relative_regret_mean,
    :std_test_relative_regret_mean,
    :min_test_relative_regret_mean,
    :max_test_relative_regret_mean,
    :mean_training_seconds,
    :mean_evaluation_seconds,
)

unix_milliseconds() = round(Int64, time() * 1000)

function result_paths(; smoke=false)
    dir = joinpath(SUITE_DIR, smoke ? "smoke_results" : "results")
    return (;
        dir=dir,
        planned_configs=joinpath(dir, "planned_configs.csv"),
        attempts=joinpath(dir, "run_attempts.csv"),
        configs=joinpath(dir, "configs.csv"),
        epochs=joinpath(dir, "epochs.csv"),
        runs=joinpath(dir, "runs.csv"),
        test_samples=joinpath(dir, "test_per_sample.csv"),
        checkpoints=joinpath(dir, "checkpoints.csv"),
        checkpoints_dir=joinpath(dir, "checkpoints"),
        summary=joinpath(dir, "summary.csv"),
    )
end

phase_summary_path(phase; smoke=false) =
    joinpath(result_paths(smoke=smoke).dir, string(phase) * "_summary.csv")

safe_path_part(text) = strip(replace(string(text), r"[^A-Za-z0-9_.=-]+" => "_"), '_')

function csv_escape(value)
    if value === nothing || value === missing
        return ""
    elseif value isa Symbol
        value = string(value)
    elseif value isa AbstractVector || value isa Tuple
        value = join(string.(value), "|")
    end

    text = replace(string(value), "\r\n" => "\\n", "\n" => "\\n", "\r" => "\\r")
    if occursin(",", text) || occursin("\"", text) || occursin("\n", text) || occursin("\r", text)
        return "\"" * replace(text, "\"" => "\"\"") * "\""
    end
    return text
end

function row_value(row, key::Symbol, default=missing)
    if row isa NamedTuple
        key in keys(row) && return getproperty(row, key)
        return default
    elseif row isa AbstractDict
        haskey(row, key) && return row[key]
        haskey(row, string(key)) && return row[string(key)]
        return default
    end
    key in propertynames(row) && return getproperty(row, key)
    return default
end

function append_csv_row(path, headers, row)
    mkpath(dirname(path))
    needs_header = !isfile(path) || filesize(path) == 0
    open(path, "a") do io
        needs_header && println(io, join(string.(headers), ","))
        println(io, join((csv_escape(row_value(row, header)) for header in headers), ","))
        flush(io)
    end
    return path
end

function write_csv_file(path, headers, rows)
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, join(string.(headers), ","))
        for row in rows
            println(io, join((csv_escape(row_value(row, header)) for header in headers), ","))
        end
    end
    return path
end

function parse_csv_line(line)
    fields = String[]
    buffer = IOBuffer()
    in_quotes = false
    index = firstindex(line)
    while index <= lastindex(line)
        char = line[index]
        if in_quotes
            if char == '"'
                next_index = nextind(line, index)
                if next_index <= lastindex(line) && line[next_index] == '"'
                    print(buffer, '"')
                    index = next_index
                else
                    in_quotes = false
                end
            else
                print(buffer, char)
            end
        else
            if char == ','
                push!(fields, String(take!(buffer)))
            elseif char == '"'
                in_quotes = true
            else
                print(buffer, char)
            end
        end
        index = nextind(line, index)
    end
    push!(fields, String(take!(buffer)))
    return fields
end

function csv_record_complete(text)
    in_quotes = false
    index = firstindex(text)
    while index <= lastindex(text)
        char = text[index]
        if in_quotes
            if char == '"'
                next_index = nextind(text, index)
                if next_index <= lastindex(text) && text[next_index] == '"'
                    index = next_index
                else
                    in_quotes = false
                end
            end
        elseif char == '"'
            in_quotes = true
        end
        index = nextind(text, index)
    end
    return !in_quotes
end

function csv_records(lines)
    records = String[]
    current = ""
    for line in lines
        current = isempty(current) ? line : current * "\n" * line
        if csv_record_complete(current)
            push!(records, current)
            current = ""
        end
    end
    isempty(current) || error("CSV file ended inside a quoted record.")
    return records
end

function parse_csv_value(text)
    isempty(text) && return missing
    lower = lowercase(text)
    lower == "true" && return true
    lower == "false" && return false
    if !occursin(r"[.eE]", text)
        integer = tryparse(Int, text)
        integer === nothing || return integer
    end
    float = tryparse(Float64, text)
    float === nothing || return float
    return text
end

function read_csv_rows(path)
    isfile(path) || return NamedTuple[]
    records = csv_records(readlines(path))
    length(records) <= 1 && return NamedTuple[]
    headers = Symbol.(parse_csv_line(first(records)))
    rows = NamedTuple[]
    for record in Iterators.drop(records, 1)
        isempty(record) && continue
        values = parse_csv_line(record)
        length(values) == length(headers) ||
            error("CSV row in $(path) has $(length(values)) fields; expected $(length(headers)).")
        push!(rows, NamedTuple{Tuple(headers)}(Tuple(parse_csv_value.(values))))
    end
    return rows
end

function parse_vector_cell(value)
    value === missing && return Float64[]
    text = string(value)
    isempty(text) && return Float64[]
    return [parse(Float64, item) for item in split(text, "|") if !isempty(item)]
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

function build_model(config, problem)
    return ContextualDFLTraining.build_neural_net(
        3,
        demand_count(problem) * NR_SCENARIOS;
        hidden_size=Int(config.hidden_size),
        depth=Int(config.depth),
        dropout=Float64(config.dropout),
        activation=Symbol(config.activation),
        output_activation=Symbol(config.output_activation),
        seed=Int(config.seed),
    )
end

function dense_layers(model)
    :layers in propertynames(model) ||
        throw(ArgumentError("checkpoint model does not look like a Flux.Chain"))
    return [layer for layer in getproperty(model, :layers) if layer isa Flux.Dense]
end

function copy_dense_parameters!(target, source)
    target_layers = dense_layers(target)
    source_layers = dense_layers(source)
    length(target_layers) == length(source_layers) ||
        throw(DimensionMismatch(
            "checkpoint has $(length(source_layers)) Dense layers, expected $(length(target_layers))",
        ))

    for (target_layer, source_layer) in zip(target_layers, source_layers)
        size(target_layer.weight) == size(source_layer.weight) ||
            throw(DimensionMismatch("checkpoint Dense weight shape mismatch"))
        size(target_layer.bias) == size(source_layer.bias) ||
            throw(DimensionMismatch("checkpoint Dense bias shape mismatch"))
        copyto!(target_layer.weight, source_layer.weight)
        copyto!(target_layer.bias, source_layer.bias)
    end
    return target
end

function model_from_checkpoint(config, objects, checkpoint_model)
    model = build_model(config, objects.problem)
    copy_dense_parameters!(model, checkpoint_model)
    return model
end

function generate_training_dataset(config)
    objects = problem_objects()
    rng = Random.MersenneTwister(Int(config.training_data_seed))
    context_generator = ContextualDFLExperiments.ResourceAllocationContextDataGenerator(rng=rng)
    scenario_generator = ContextualDFLExperiments.ResourceAllocationScenarioDataGenerator(
        objects.problem;
        sigma=DEMAND_SIGMA,
        p=DEMAND_POWER,
        L=CONTEXT_TERMS,
        rng=rng,
    )

    contexts = [Vector{Float64}(context_generator()) for _ in 1:Int(config.training_contexts)]
    scenario_collections = [
        [scenario_generator(context) for _ in 1:NR_SCENARIOS] for context in contexts
    ]
    return ContextualDFLExperiments.generate_contextual_data_set(
        contexts,
        scenario_collections,
    )
end

function build_loss(objects)
    return ContextualDFL.DflScenLoss(
        objects.scenario_decoder,
        objects.reference_scenario_decoder,
        objects.solver,
        objects.program;
        nr_scenarios=NR_SCENARIOS,
    )
end

function target_mean(point)
    scenarios = point.scenario_parameters
    isempty(scenarios) && throw(ArgumentError("data point has no scenario parameters."))
    total = zeros(Float64, length(first(scenarios).h_eq_xi))
    for scenario in scenarios
        total .+= Float64.(scenario.h_eq_xi)
    end
    return total ./ length(scenarios)
end

function target_matrix(dataset)
    isempty(dataset) && return zeros(Float64, 0, 0)
    return reduce(hcat, (target_mean(point) for point in dataset))
end

function context_matrix(dataset)
    isempty(dataset) && return zeros(Float64, 0, 0)
    return reduce(hcat, (Float64.(point.context) for point in dataset))
end

function prediction_metrics(model, contexts, target)
    isempty(target) && return (; mse=NaN, mae=NaN)
    try
        Flux.testmode!(model)
    catch
    end
    prediction = Array(model(contexts))
    if ndims(prediction) == 1
        prediction = reshape(prediction, :, 1)
    end
    size(prediction) == size(target) ||
        throw(DimensionMismatch("prediction size $(size(prediction)) != target size $(size(target))"))
    errors = prediction .- target
    return (; mse=Statistics.mean(abs2, errors), mae=Statistics.mean(abs.(errors)))
end

function experiment_spec()
    return ContextualDFLTraining.load_experiment(EXPERIMENT_SELECTOR)
end

function optimality_objective_values(result)
    if hasproperty(result, :objective_values)
        values = Float64.(collect(result.objective_values))
        isempty(values) && throw(ArgumentError("optimal objective_values must not be empty."))
        return values
    elseif hasproperty(result, :objective_value)
        return [Float64(result.objective_value)]
    end
    throw(ArgumentError("optimal results must contain objective_values."))
end

function limit_test_cache(dataset, optimal_results; context_limit=0, evaluation_batch_limit=0)
    selected_count = Int(context_limit) <= 0 ? length(dataset) : min(Int(context_limit), length(dataset))
    limited_dataset = ContextualDFL.ContextualDataPoint[]
    limited_results = NamedTuple[]

    for index in 1:selected_count
        data_point = dataset[index]
        result = optimal_results[index]
        objective_values = optimality_objective_values(result)
        source_batch_count = length(objective_values)
        batch_limit = Int(evaluation_batch_limit) <= 0 ?
            source_batch_count :
            min(Int(evaluation_batch_limit), source_batch_count)

        scenario_count = length(data_point.scenario_parameters)
        scenario_count % source_batch_count == 0 || throw(
            ArgumentError(
                "scenario count $scenario_count is not divisible by stored optimality batches $source_batch_count.",
            ),
        )
        scenarios_per_batch = scenario_count ÷ source_batch_count
        scenario_limit = batch_limit * scenarios_per_batch
        selected_objective_values = objective_values[1:batch_limit]

        push!(
            limited_dataset,
            ContextualDFL.ContextualDataPoint(
                data_point.context,
                data_point.scenario_parameters[1:scenario_limit],
            ),
        )
        push!(
            limited_results,
            merge(
                result,
                (;
                    evaluation_batches=batch_limit,
                    objective_values=selected_objective_values,
                    objective_value=Statistics.mean(Float64.(selected_objective_values)),
                ),
            ),
        )
    end
    return limited_dataset, limited_results
end

function load_test_cache(; smoke=false, context_limit=0, evaluation_batch_limit=0)
    spec = experiment_spec()
    artifact = ContextualDFLTraining.load_test_data_artifact(spec)
    optimal_results =
        ContextualDFLTraining.load_optimal_results(spec, :test; dataset=artifact.dataset)

    effective_context_limit =
        smoke && Int(context_limit) <= 0 ? SMOKE_TEST_CONTEXTS : Int(context_limit)
    effective_batch_limit =
        smoke && Int(evaluation_batch_limit) <= 0 ? SMOKE_EVALUATION_BATCHES : Int(evaluation_batch_limit)

    dataset, optima = limit_test_cache(
        artifact.dataset,
        optimal_results;
        context_limit=effective_context_limit,
        evaluation_batch_limit=effective_batch_limit,
    )

    return (;
        dataset=dataset,
        optimal_results=optima,
        metadata=merge(
            artifact.metadata,
            (;
                experiment_id=spec.id,
                context_limit=effective_context_limit,
                evaluation_batch_limit=effective_batch_limit,
                loaded_contexts=length(dataset),
                loaded_evaluation_batches=isempty(optima) ? 0 : first(optima).evaluation_batches,
            ),
        ),
        context_matrix=context_matrix(dataset),
        target_matrix=target_matrix(dataset),
    )
end

function set_worker_test_cache!(; smoke=false, context_limit=0, evaluation_batch_limit=0)
    cache = load_test_cache(
        smoke=smoke,
        context_limit=context_limit,
        evaluation_batch_limit=evaluation_batch_limit,
    )
    WORKER_TEST_CACHE[] = cache
    return (;
        worker_id=Distributed.myid(),
        hostname=Sockets.gethostname(),
        pid=getpid(),
        test_contexts=length(cache.dataset),
        evaluation_batches=cache.metadata.loaded_evaluation_batches,
    )
end

function replicate_seed(replicate)
    return TRAINING_SEED_BASE + Int(replicate)
end

base_schedule_segments() = begin
    values = Float64[1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.08, 0.06, 0.04, 0.02, 0.01]
    segments = Tuple{Float64,Float64,Int}[]
    for (index, value) in enumerate(values)
        push!(segments, (value, value, index == 1 ? 20 : 10))
    end
    push!(segments, (last(values), 0.0, 10))
    return segments
end

alt_schedule_segments() = Tuple{Float64,Float64,Int}[
    (1.0, 1.0, 20),
    (0.1, 0.1, 10),
    (0.01, 0.0, 20),
]

schedule_segments(schedule_name) =
    Symbol(schedule_name) == :base ? base_schedule_segments() :
    Symbol(schedule_name) == :alt ? alt_schedule_segments() :
    throw(ArgumentError("unknown schedule $(schedule_name)"))

function scaled_segment_lengths(segments, total_epochs)
    total_epochs = Int(total_epochs)
    total_epochs > 0 || return Int[]
    raw_lengths = [segment[3] for segment in segments]
    raw_total = sum(raw_lengths)
    ideal = Float64.(raw_lengths) .* total_epochs ./ raw_total
    lengths = floor.(Int, ideal)
    difference = total_epochs - sum(lengths)

    if difference > 0
        order = sortperm(ideal .- lengths; rev=true)
        for index in order
            lengths[index] += 1
            difference -= 1
            difference == 0 && break
        end
    elseif difference < 0
        order = sortperm(ideal .- lengths)
        for index in order
            lengths[index] <= 0 && continue
            lengths[index] -= 1
            difference += 1
            difference == 0 && break
        end
    end

    sum(lengths) == total_epochs ||
        throw(ArgumentError("could not scale schedule to $total_epochs epochs."))
    return lengths
end

function schedules_for_config(config)
    segments = schedule_segments(config.schedule_name)
    lengths = scaled_segment_lengths(segments, Int(config.epochs))
    mu_in = Float64[]
    mu_ref = Float64[]
    for ((mu_value, ref_value, _), length) in zip(segments, lengths)
        append!(mu_in, fill(mu_value, length))
        append!(mu_ref, fill(ref_value, length))
    end
    return mu_in, mu_ref
end

rho_schedule_for_config(config) = zeros(Float64, Int(config.epochs))

function final_stage_start_epoch(config)
    segments = schedule_segments(config.schedule_name)
    lengths = scaled_segment_lengths(segments, Int(config.epochs))
    last_nonzero = findlast(>(0), lengths)
    last_nonzero === nothing && return nothing
    last_nonzero <= 1 && return nothing
    return sum(lengths[1:(last_nonzero - 1)]) + 1
end

function checkpoint_epochs(config)
    epochs = Int(config.epochs)
    special = Set{Int}()
    epochs <= 0 && return special
    push!(special, max(1, round(Int, epochs / 2)))
    push!(special, max(1, round(Int, 2 * epochs / 3)))
    start = final_stage_start_epoch(config)
    if start !== nothing && start > 1
        push!(special, start - 1)
    end
    push!(special, epochs)
    return special
end

function checkpoint_kinds_for_epoch(config, epoch)
    kinds = Symbol[]
    epochs = Int(config.epochs)
    midpoint = max(1, round(Int, epochs / 2))
    twothirds = max(1, round(Int, 2 * epochs / 3))
    start = final_stage_start_epoch(config)
    pre_final = start === nothing || start <= 1 ? nothing : start - 1

    epoch == midpoint && push!(kinds, :midpoint)
    epoch == twothirds && push!(kinds, :twothirds)
    pre_final !== nothing && epoch == pre_final && push!(kinds, :pre_final_mu)

    checkpoint_every = Int(config.checkpoint_every_epochs)
    if epoch < epochs && (epoch % checkpoint_every == 0 || !isempty(kinds))
        push!(kinds, :latest)
    end
    epoch == epochs && push!(kinds, :final)
    return unique(kinds)
end

function schedule_preview(values; n=6)
    isempty(values) && return ""
    keep = min(Int(n), length(values))
    head = join(string.(round.(values[1:keep]; digits=6)), "|")
    length(values) <= keep && return head
    return head * "|...|" * string(round(last(values); digits=6))
end

function run_id(wave_index, candidate_name, schedule_name, replicate; smoke=false)
    return join(
        (
            smoke ? "smoke" : "full",
            "w" * lpad(string(Int(wave_index)), 2, "0"),
            safe_path_part(candidate_name),
            safe_path_part(schedule_name),
            "rep" * lpad(string(Int(replicate)), 2, "0"),
        ),
        "_",
    )
end

function base_config(;
    wave_index,
    phase,
    candidate_name,
    schedule_name,
    replicate,
    smoke=false,
    training_contexts=BASE_TRAINING_CONTEXTS,
    epochs=BASE_TOTAL_EPOCHS,
    epoch_fraction="1",
    batch_size=BASE_BATCH_SIZE,
    overrides...,
)
    effective_epochs = smoke ? min(Int(epochs), SMOKE_EPOCHS) : Int(epochs)
    effective_training_contexts =
        smoke ? min(Int(training_contexts), SMOKE_TRAINING_CONTEXTS) : Int(training_contexts)
    config = merge(
        (;
            version=SUITE_VERSION,
            smoke=Bool(smoke),
            wave_index=Int(wave_index),
            phase=Symbol(phase),
            candidate_name=String(candidate_name),
            schedule_name=Symbol(schedule_name),
            replicate=Int(replicate),
            seed=replicate_seed(replicate),
            training_data_seed=replicate_seed(replicate),
            training_contexts=effective_training_contexts,
            epochs=effective_epochs,
            epoch_fraction=String(epoch_fraction),
            depth=BASE_DEPTH,
            hidden_size=BASE_HIDDEN_SIZE,
            activation=BASE_ACTIVATION,
            output_activation=BASE_OUTPUT_ACTIVATION,
            dropout=BASE_DROPOUT,
            batch_size=Int(batch_size),
            learning_rate=BASE_LEARNING_RATE,
            checkpoint_every_epochs=smoke ? 1 : CHECKPOINT_EVERY_EPOCHS,
        ),
        NamedTuple(overrides),
    )
    mu_in, _ = schedules_for_config(config)
    return merge(
        config,
        (;
            policy_inference_mu=isempty(mu_in) ? 0.0 : Float64(last(mu_in)),
            run_id=run_id(
                Int(config.wave_index),
                String(config.candidate_name),
                Symbol(config.schedule_name),
                Int(config.replicate);
                smoke=Bool(config.smoke),
            ),
        ),
    )
end

function n1000_candidate_specs()
    return [
        (; candidate_name="n1000_b1_epochs_1over5", batch_size=1, epochs=26, epoch_fraction="1/5"),
        (; candidate_name="n1000_b2_epochs_1over10", batch_size=2, epochs=13, epoch_fraction="1/10"),
        (; candidate_name="n1000_b1_epochs_1over3", batch_size=1, epochs=43, epoch_fraction="1/3"),
        (; candidate_name="n1000_b3_epochs_full", batch_size=3, epochs=130, epoch_fraction="1"),
        (; candidate_name="n1000_b2_epochs_1over3", batch_size=2, epochs=43, epoch_fraction="1/3"),
        (; candidate_name="n1000_b2_epochs_1over5", batch_size=2, epochs=26, epoch_fraction="1/5"),
        (; candidate_name="n1000_b4_epochs_1over10", batch_size=4, epochs=13, epoch_fraction="1/10"),
    ]
end

function planned_waves(; smoke=false)
    waves = NamedTuple[]
    wave_index = 1
    baseline_reps = 1:(smoke ? 1 : 10)
    n1000_reps = 1:(smoke ? 1 : 5)

    push!(
        waves,
        (;
            wave_index=wave_index,
            phase=:baseline,
            wave_name="standard_n100_base_schedule",
            configs=[
                base_config(
                    wave_index=wave_index,
                    phase=:baseline,
                    candidate_name="standard_n100",
                    schedule_name=:base,
                    replicate=replicate,
                    smoke=smoke,
                ) for replicate in baseline_reps
            ],
        ),
    )
    wave_index += 1

    push!(
        waves,
        (;
            wave_index=wave_index,
            phase=:n100_schedule,
            wave_name="standard_n100_alt_schedule",
            configs=[
                base_config(
                    wave_index=wave_index,
                    phase=:n100_schedule,
                    candidate_name="standard_n100",
                    schedule_name=:alt,
                    replicate=replicate,
                    smoke=smoke,
                ) for replicate in baseline_reps
            ],
        ),
    )
    wave_index += 1

    candidate_specs = smoke ? first(n1000_candidate_specs(), 1) : n1000_candidate_specs()
    for schedule_name in (:base, :alt)
        for spec in candidate_specs
            push!(
                waves,
                (;
                    wave_index=wave_index,
                    phase=Symbol(:n1000_, schedule_name),
                    wave_name=spec.candidate_name * "_" * string(schedule_name),
                    configs=[
                        base_config(
                            wave_index=wave_index,
                            phase=Symbol(:n1000_, schedule_name),
                            candidate_name=spec.candidate_name,
                            schedule_name=schedule_name,
                            replicate=replicate,
                            smoke=smoke,
                            training_contexts=1000,
                            epochs=spec.epochs,
                            epoch_fraction=spec.epoch_fraction,
                            batch_size=spec.batch_size,
                        ) for replicate in n1000_reps
                    ],
                ),
            )
            wave_index += 1
        end
    end
    return waves
end

all_configs(; smoke=false) = reduce(vcat, (wave.configs for wave in planned_waves(smoke=smoke)))

function config_row(config)
    mu_in, mu_ref = schedules_for_config(config)
    return (;
        version=config.version,
        run_id=config.run_id,
        smoke=config.smoke,
        wave_index=config.wave_index,
        phase=String(config.phase),
        candidate_name=config.candidate_name,
        schedule_name=String(config.schedule_name),
        replicate=config.replicate,
        seed=config.seed,
        training_data_seed=config.training_data_seed,
        training_contexts=config.training_contexts,
        epochs=config.epochs,
        epoch_fraction=config.epoch_fraction,
        depth=config.depth,
        hidden_size=config.hidden_size,
        activation=String(config.activation),
        output_activation=String(config.output_activation),
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        policy_inference_mu=config.policy_inference_mu,
        mu_in_preview=schedule_preview(mu_in),
        mu_ref_preview=schedule_preview(mu_ref),
    )
end

function put_log!(logger, kind::Symbol, row)
    logger === nothing && return nothing
    put!(logger, (; kind=kind, row=row))
    return nothing
end

function epoch_row(config, attempt_id, epoch, history_row, train_metrics, test_metrics)
    return (;
        attempt_id=attempt_id,
        run_id=config.run_id,
        wave_index=Int(config.wave_index),
        phase=String(config.phase),
        candidate_name=config.candidate_name,
        schedule_name=String(config.schedule_name),
        replicate=Int(config.replicate),
        epoch=Int(epoch),
        mu_in=Float64(row_value(history_row, :mu_in, NaN)),
        mu_ref=Float64(row_value(history_row, :mu_ref, NaN)),
        rho_in=Float64(row_value(history_row, :rho_in, 0.0)),
        rho_ref=Float64(row_value(history_row, :rho_ref, 0.0)),
        iterations=Int(row_value(history_row, :iterations, 0)),
        epoch_seconds=Float64(row_value(history_row, :epoch_seconds, NaN)),
        training_loss=Float64(row_value(history_row, :loss, NaN)),
        display_loss=Float64(row_value(history_row, :display_loss, NaN)),
        train_target_mse=Float64(train_metrics.mse),
        train_target_mae=Float64(train_metrics.mae),
        test_target_mse=Float64(test_metrics.mse),
        test_target_mae=Float64(test_metrics.mae),
        training_contexts=Int(config.training_contexts),
        epochs=Int(config.epochs),
        depth=Int(config.depth),
        hidden_size=Int(config.hidden_size),
        activation=String(config.activation),
        batch_size=Int(config.batch_size),
        seed=Int(config.seed),
        worker_id=Distributed.myid(),
        hostname=Sockets.gethostname(),
        created_at=unix_milliseconds(),
    )
end

function metric_value(metrics, key, default=NaN)
    key in propertynames(metrics) || return default
    value = getproperty(metrics, key)
    value === missing && return default
    return Float64(value)
end

function evaluation_vector_field(row, key)
    key in propertynames(row) || return Float64[]
    return Float64.(collect(getproperty(row, key)))
end

function evaluate_trained_model(config, model, attempt_id)
    cache = WORKER_TEST_CACHE[]
    cache === nothing && error("worker test cache has not been initialized")
    objects = problem_objects()
    try
        Flux.testmode!(model)
    catch
    end

    scenario_generator = ContextualDFL.ScenarioGenerator(;
        neural_net=model,
        scenario_decoder=objects.scenario_decoder,
    )
    policy = ContextualDFLExperiments.ScenarioGenerationPolicy(
        scenario_generator,
        objects.solver,
        objects.program;
        mu=Float64(config.policy_inference_mu),
        rho=0.0,
    )

    evaluation = ContextualDFLExperiments.evaluate_policy_against_optimum(
        policy,
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
            attempt_id=attempt_id,
            run_id=config.run_id,
            wave_index=Int(config.wave_index),
            phase=String(config.phase),
            candidate_name=config.candidate_name,
            schedule_name=String(config.schedule_name),
            replicate=Int(config.replicate),
            sample_index=row.sample_index,
            policy_value=row.policy_value,
            optimal_value=row.optimal_value,
            regret=row.regret,
            relative_regret=row.relative_regret,
            gap_std=row.gap_std,
            gap_stderr=row.gap_stderr,
            policy_collection_values=evaluation_vector_field(row, :policy_collection_values),
            optimal_collection_values=evaluation_vector_field(row, :optimal_collection_values),
            gap_values=evaluation_vector_field(row, :gap_values),
        ) for row in evaluation.per_sample
    ]

    metrics = evaluation.metrics
    return (;
        metrics=metrics,
        per_sample_rows=per_sample_rows,
        test_relative_regret_mean=metric_value(metrics, :test_relative_regret_mean),
        test_relative_regret_median=metric_value(metrics, :test_relative_regret_median),
        test_relative_regret_std=metric_value(metrics, :test_relative_regret_std),
        test_relative_regret_p95=metric_value(metrics, :test_relative_regret_p95),
        test_regret_mean=metric_value(metrics, :test_regret_mean),
        test_policy_value_mean=metric_value(metrics, :test_policy_value_mean),
        test_optimal_value_mean=metric_value(metrics, :test_optimal_value_mean),
        test_sample_count=metric_value(metrics, :test_sample_count),
        test_evaluation_batches=metric_value(metrics, :test_evaluation_batches),
        test_gap_std_mean=metric_value(metrics, :test_gap_std_mean),
        test_gap_stderr_mean=metric_value(metrics, :test_gap_stderr_mean),
    )
end

function checkpoint_payload(config, attempt_id, model, completed_epoch, training_seconds)
    return (;
        version=SUITE_VERSION,
        attempt_id=attempt_id,
        run_id=config.run_id,
        config=config_for_checkpoint(config),
        model=model,
        completed_epoch=Int(completed_epoch),
        training_seconds=Float64(training_seconds),
        worker=(;
            worker_id=Distributed.myid(),
            hostname=Sockets.gethostname(),
            pid=getpid(),
        ),
        saved_at=unix_milliseconds(),
    )
end

function config_for_checkpoint(config)
    pairs = Pair{Symbol,Any}[]
    for key in keys(config)
        key == :resume_checkpoint_bytes && continue
        push!(pairs, key => getproperty(config, key))
    end
    return (; pairs...)
end

function serialize_to_bytes(value)
    io = IOBuffer()
    Serialization.serialize(io, value)
    return take!(io)
end

function deserialize_from_bytes(bytes)
    io = IOBuffer(Vector{UInt8}(bytes))
    return Serialization.deserialize(io)
end

function has_resume_checkpoint(config)
    return :resume_checkpoint_bytes in keys(config) &&
           getproperty(config, :resume_checkpoint_bytes) isa AbstractVector &&
           !isempty(getproperty(config, :resume_checkpoint_bytes))
end

function checkpoint_relative_path(config, kind)
    filename = safe_path_part(config.run_id) * "_" * string(kind) * ".jls"
    return joinpath("checkpoints", filename)
end

function checkpoint_row(config, attempt_id, epoch, kind, bytes)
    return (;
        attempt_id=attempt_id,
        run_id=config.run_id,
        wave_index=Int(config.wave_index),
        phase=String(config.phase),
        candidate_name=config.candidate_name,
        schedule_name=String(config.schedule_name),
        replicate=Int(config.replicate),
        epoch=Int(epoch),
        checkpoint_kind=String(kind),
        checkpoint_path=checkpoint_relative_path(config, kind),
        checkpoint_bytes=bytes,
        worker_id=Distributed.myid(),
        hostname=Sockets.gethostname(),
        created_at=unix_milliseconds(),
    )
end

function run_attempt_row(config, attempt_id, started_at)
    return (;
        attempt_id=attempt_id,
        run_id=config.run_id,
        wave_index=Int(config.wave_index),
        phase=String(config.phase),
        candidate_name=config.candidate_name,
        schedule_name=String(config.schedule_name),
        replicate=Int(config.replicate),
        worker_id=Distributed.myid(),
        hostname=Sockets.gethostname(),
        pid=getpid(),
        started_at=started_at,
    )
end

function run_result_row(
    config,
    attempt_id,
    status,
    started_at,
    finished_at;
    metrics=nothing,
    timings=nothing,
    completed_epochs=0,
    final_train_metrics=(; mse=NaN),
    final_test_metrics=(; mse=NaN),
    error="",
)
    metrics = metrics === nothing ? (;) : metrics
    timings = timings === nothing ? (; training_seconds=NaN, evaluation_seconds=NaN) : timings
    total_seconds = (finished_at - started_at) / 1000
    return (;
        attempt_id=attempt_id,
        run_id=config.run_id,
        status=status,
        wave_index=Int(config.wave_index),
        phase=String(config.phase),
        candidate_name=config.candidate_name,
        schedule_name=String(config.schedule_name),
        replicate=Int(config.replicate),
        seed=Int(config.seed),
        training_contexts=Int(config.training_contexts),
        epochs=Int(config.epochs),
        epoch_fraction=String(config.epoch_fraction),
        depth=Int(config.depth),
        hidden_size=Int(config.hidden_size),
        activation=String(config.activation),
        batch_size=Int(config.batch_size),
        policy_inference_mu=Float64(config.policy_inference_mu),
        test_relative_regret_mean=row_value(metrics, :test_relative_regret_mean, Inf),
        test_relative_regret_median=row_value(metrics, :test_relative_regret_median, Inf),
        test_relative_regret_std=row_value(metrics, :test_relative_regret_std, Inf),
        test_relative_regret_p95=row_value(metrics, :test_relative_regret_p95, Inf),
        test_regret_mean=row_value(metrics, :test_regret_mean, Inf),
        test_policy_value_mean=row_value(metrics, :test_policy_value_mean, NaN),
        test_optimal_value_mean=row_value(metrics, :test_optimal_value_mean, NaN),
        test_sample_count=row_value(metrics, :test_sample_count, 0),
        test_evaluation_batches=row_value(metrics, :test_evaluation_batches, 0),
        test_gap_std_mean=row_value(metrics, :test_gap_std_mean, NaN),
        test_gap_stderr_mean=row_value(metrics, :test_gap_stderr_mean, NaN),
        final_train_target_mse=Float64(final_train_metrics.mse),
        final_test_target_mse=Float64(final_test_metrics.mse),
        completed_epochs=Int(completed_epochs),
        training_seconds=row_value(timings, :training_seconds, NaN),
        evaluation_seconds=row_value(timings, :evaluation_seconds, NaN),
        total_seconds=total_seconds,
        worker_id=Distributed.myid(),
        hostname=Sockets.gethostname(),
        pid=getpid(),
        started_at=started_at,
        finished_at=finished_at,
        error=error,
    )
end

function history_first_row(result)
    history = result.history
    history isa AbstractVector && return only(history)
    return history
end

function run_experiment_config(config, logger)
    started_at = unix_milliseconds()
    attempt_id = config.run_id * "_attempt" * string(started_at) * "_w" * string(Distributed.myid())
    put_log!(logger, :attempt, run_attempt_row(config, attempt_id, started_at))
    completed_epoch = 0
    training_seconds_total = 0.0
    final_train_metrics = (; mse=NaN, mae=NaN)
    final_test_metrics = (; mse=NaN, mae=NaN)

    try
        objects = problem_objects()
        model = nothing
        if has_resume_checkpoint(config)
            payload = deserialize_from_bytes(config.resume_checkpoint_bytes)
            model = model_from_checkpoint(config, objects, payload.model)
            completed_epoch = Int(payload.completed_epoch)
            training_seconds_total = Float64(payload.training_seconds)
        else
            model = build_model(config, objects.problem)
        end

        data_set_training = generate_training_dataset(config)
        train_contexts = context_matrix(data_set_training)
        train_target = target_matrix(data_set_training)
        cache = WORKER_TEST_CACHE[]
        cache === nothing && error("worker test cache has not been initialized")
        loss = build_loss(objects)
        mu_in, mu_ref = schedules_for_config(config)
        rho_in = rho_schedule_for_config(config)
        rho_ref = rho_schedule_for_config(config)

        for epoch in (completed_epoch + 1):Int(config.epochs)
            result = nothing
            elapsed = @elapsed begin
                result = ContextualDFL.train!(
                    model,
                    loss,
                    [mu_in[epoch]],
                    [mu_ref[epoch]],
                    data_set_training;
                    rho_in_schedule=[rho_in[epoch]],
                    rho_ref_schedule=[rho_ref[epoch]],
                    opt=Flux.Adam(Float64(config.learning_rate)),
                    epochs=1,
                    batchsize=Int(config.batch_size),
                    shuffle=false,
                    display_iterations=false,
                    verbose=false,
                    display_plot=false,
                    save_model=false,
                    reset_optimizer_each_epoch=true,
                    nr_scenarios=NR_SCENARIOS,
                    display_smooth=false,
                )
            end
            training_seconds_total += elapsed
            model = result.model
            completed_epoch = epoch

            final_train_metrics = prediction_metrics(model, train_contexts, train_target)
            final_test_metrics = prediction_metrics(model, cache.context_matrix, cache.target_matrix)
            put_log!(
                logger,
                :epoch,
                epoch_row(
                    config,
                    attempt_id,
                    epoch,
                    history_first_row(result),
                    final_train_metrics,
                    final_test_metrics,
                ),
            )

            kinds = checkpoint_kinds_for_epoch(config, epoch)
            if !isempty(kinds)
                bytes = serialize_to_bytes(
                    checkpoint_payload(
                        config,
                        attempt_id,
                        model,
                        completed_epoch,
                        training_seconds_total,
                    ),
                )
                for kind in kinds
                    put_log!(logger, :checkpoint, checkpoint_row(config, attempt_id, epoch, kind, bytes))
                end
            end
        end

        final_train_metrics = prediction_metrics(model, train_contexts, train_target)
        final_test_metrics = prediction_metrics(model, cache.context_matrix, cache.target_matrix)

        evaluation = nothing
        evaluation_seconds = @elapsed begin
            evaluation = evaluate_trained_model(config, model, attempt_id)
        end
        for row in evaluation.per_sample_rows
            put_log!(logger, :test_sample, row)
        end
        finished_at = unix_milliseconds()
        result = run_result_row(
            config,
            attempt_id,
            "ok",
            started_at,
            finished_at;
            metrics=evaluation,
            timings=(;
                training_seconds=training_seconds_total,
                evaluation_seconds=evaluation_seconds,
            ),
            completed_epochs=completed_epoch,
            final_train_metrics=final_train_metrics,
            final_test_metrics=final_test_metrics,
        )
        put_log!(logger, :run, result)
        return result
    catch error
        finished_at = unix_milliseconds()
        text = sprint(showerror, error, catch_backtrace())
        result = run_result_row(
            config,
            attempt_id,
            "failed",
            started_at,
            finished_at;
            completed_epochs=completed_epoch,
            final_train_metrics=final_train_metrics,
            final_test_metrics=final_test_metrics,
            timings=(; training_seconds=training_seconds_total, evaluation_seconds=NaN),
            error=text,
        )
        put_log!(logger, :run, result)
        return result
    end
end

function completed_run_ids(; smoke=false)
    rows = read_csv_rows(result_paths(smoke=smoke).runs)
    return Set(
        string(row.run_id) for row in rows if
        string(row_value(row, :status, "")) == "ok"
    )
end

function latest_checkpoint_path(run_id; smoke=false)
    return joinpath(
        result_paths(smoke=smoke).checkpoints_dir,
        safe_path_part(run_id) * "_latest.jls",
    )
end

function final_checkpoint_path(run_id; smoke=false)
    return joinpath(
        result_paths(smoke=smoke).checkpoints_dir,
        safe_path_part(run_id) * "_final.jls",
    )
end

checkpoint_csv_mirror_path(path) = path * ".csv"

function checkpoint_exists(path)
    return isfile(path) || isfile(checkpoint_csv_mirror_path(path))
end

function read_checkpoint_bytes(path)
    isfile(path) && return read(path)

    mirror_path = checkpoint_csv_mirror_path(path)
    isfile(mirror_path) || throw(ArgumentError("checkpoint does not exist: $path"))
    lines = readlines(mirror_path)
    length(lines) >= 2 || throw(ArgumentError("checkpoint CSV mirror is empty: $mirror_path"))
    fields = split(lines[2], ","; limit=2)
    length(fields) == 2 || throw(ArgumentError("checkpoint CSV mirror has invalid row: $mirror_path"))
    return base64decode(fields[2])
end

function attach_resume_checkpoint(config)
    final_path = final_checkpoint_path(config.run_id; smoke=Bool(config.smoke))
    latest_path = latest_checkpoint_path(config.run_id; smoke=Bool(config.smoke))
    path = checkpoint_exists(final_path) ? final_path : latest_path
    checkpoint_exists(path) || return config
    return merge(config, (; resume_checkpoint_bytes=read_checkpoint_bytes(path)))
end

function pending_configs(configs; smoke=false)
    completed = completed_run_ids(smoke=smoke)
    return [attach_resume_checkpoint(config) for config in configs if !(config.run_id in completed)]
end

function mean_or_nan(values)
    clean = [Float64(value) for value in values if isfinite(Float64(value))]
    isempty(clean) && return NaN
    return Statistics.mean(clean)
end

function std_or_nan(values)
    clean = [Float64(value) for value in values if isfinite(Float64(value))]
    length(clean) <= 1 && return NaN
    return Statistics.std(clean)
end

function summarize_runs(rows)
    candidates = sort(unique((string(row.candidate_name), string(row.schedule_name), string(row.phase)) for row in rows))
    summary = NamedTuple[]
    for (candidate, schedule, phase) in candidates
        candidate_rows = [
            row for row in rows if
            string(row.candidate_name) == candidate &&
            string(row.schedule_name) == schedule &&
            string(row.phase) == phase
        ]
        run_ids = sort(unique(string(row.run_id) for row in candidate_rows))
        ok_rows = NamedTuple[]
        failed_run_count = 0
        for run_id in run_ids
            run_rows = [row for row in candidate_rows if string(row.run_id) == run_id]
            run_ok_rows = [row for row in run_rows if string(row.status) == "ok"]
            if isempty(run_ok_rows)
                failed_run_count += 1
            else
                push!(ok_rows, last(run_ok_rows))
            end
        end
        values = [row.test_relative_regret_mean for row in ok_rows]
        push!(
            summary,
            (;
                phase=phase,
                candidate_name=candidate,
                schedule_name=schedule,
                run_count=length(run_ids),
                ok_count=length(ok_rows),
                failed_count=failed_run_count,
                mean_test_relative_regret_mean=mean_or_nan(values),
                std_test_relative_regret_mean=std_or_nan(values),
                min_test_relative_regret_mean=isempty(values) ? NaN : minimum(Float64.(values)),
                max_test_relative_regret_mean=isempty(values) ? NaN : maximum(Float64.(values)),
                mean_training_seconds=mean_or_nan([row.training_seconds for row in ok_rows]),
                mean_evaluation_seconds=mean_or_nan([row.evaluation_seconds for row in ok_rows]),
            ),
        )
    end
    return summary
end

function summarize_phase(phase; smoke=false)
    rows = [
        row for row in read_csv_rows(result_paths(smoke=smoke).runs) if
        string(row.phase) == string(phase)
    ]
    return summarize_runs(rows)
end

function summarize_all(; smoke=false)
    return summarize_runs(read_csv_rows(result_paths(smoke=smoke).runs))
end
