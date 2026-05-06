using ContextualDFL
using ContextualDFLExperiments
using Distributed
using Flux
using Random
using Serialization
using Sockets
using Statistics

const SUITE_DIR = @__DIR__
const REPO_ROOT = normpath(joinpath(SUITE_DIR, "..", "..", ".."))
const TRAINING_PROJECT_DIR =
    joinpath(REPO_ROOT, "src", "ContextualDFL", "ContextualDFLTraining")

const SUITE_VERSION = "training-data-depth-2026-05-05-v1"
const TEST_SEED = 20_260_505
const TRAINING_SEED_BASE = 2_026_050_500

const BASE_TRAINING_CONTEXTS = 100
const BASE_TOTAL_EPOCHS = 130
const BASE_HIDDEN_SIZE = 128
const BASE_DEPTH = 4
const BASE_ACTIVATION = :gelu
const BASE_BATCH_SIZE = 1
const BASE_LEARNING_RATE = 1e-3
const NR_SCENARIOS = 1
const CHECKPOINT_EVERY_EPOCHS = 5

const TEST_CONTEXTS = 30
const TEST_SCENARIOS_PER_CONTEXT = 100
const TEST_EVALUATION_BATCHES = 5
const SMOKE_TEST_CONTEXTS = 2
const SMOKE_TEST_SCENARIOS_PER_CONTEXT = 2
const SMOKE_TEST_EVALUATION_BATCHES = 1

const DEMAND_SIGMA = 5.0
const DEMAND_POWER = 2.0
const CONTEXT_TERMS = 3

const ATTEMPT_HEADERS = (
    :attempt_id,
    :run_id,
    :phase,
    :candidate_name,
    :replicate,
    :worker_id,
    :hostname,
    :pid,
    :started_at,
)

const CONFIG_HEADERS = (
    :version,
    :run_id,
    :smoke,
    :phase,
    :candidate_name,
    :replicate,
    :seed,
    :training_contexts,
    :epochs,
    :depth,
    :hidden_size,
    :activation,
    :batch_size,
    :learning_rate,
    :mu_in_preview,
    :mu_ref_preview,
)

const EPOCH_HEADERS = (
    :attempt_id,
    :run_id,
    :phase,
    :candidate_name,
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
    :seed,
    :worker_id,
    :hostname,
    :created_at,
)

const RUN_HEADERS = (
    :attempt_id,
    :run_id,
    :status,
    :phase,
    :candidate_name,
    :replicate,
    :seed,
    :training_contexts,
    :epochs,
    :depth,
    :hidden_size,
    :activation,
    :average_test_relative_regret,
    :mean_test_regret,
    :mean_test_policy_value,
    :mean_test_optimal_value,
    :test_sample_count,
    :test_evaluation_batches,
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
    :phase,
    :candidate_name,
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
    :phase,
    :candidate_name,
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
    :run_count,
    :ok_count,
    :failed_count,
    :mean_average_test_relative_regret,
    :std_average_test_relative_regret,
    :min_average_test_relative_regret,
    :max_average_test_relative_regret,
)

const TEST_SCENARIO_HEADERS = (
    :sample_index,
    :scenario_index,
    :context_1,
    :context_2,
    :context_3,
    :demand_values,
)

const TEST_OPTIMUM_HEADERS = (
    :sample_index,
    :evaluation_batches,
    :objective_value,
    :objective_values,
)

const TEST_METADATA_HEADERS = (
    :version,
    :smoke,
    :seed,
    :contexts,
    :scenarios_per_context,
    :evaluation_batches,
    :solve_seconds,
    :worker_id,
    :hostname,
    :pid,
    :started_at,
    :finished_at,
)

const WORKER_TEST_CACHE = Ref{Any}(nothing)

unix_milliseconds() = round(Int64, time() * 1000)

result_paths(; smoke=false) = begin
    dir = joinpath(SUITE_DIR, smoke ? "smoke_results" : "results")
    (;
        dir=dir,
        attempts=joinpath(dir, "run_attempts.csv"),
        configs=joinpath(dir, "configs.csv"),
        epochs=joinpath(dir, "epochs.csv"),
        runs=joinpath(dir, "runs.csv"),
        test_samples=joinpath(dir, "test_per_sample.csv"),
        checkpoints=joinpath(dir, "checkpoints.csv"),
        checkpoints_dir=joinpath(dir, "checkpoints"),
    )
end

test_cache_paths(; smoke=false) = begin
    dir = joinpath(SUITE_DIR, "artifacts", smoke ? "smoke_test_data" : "test_data")
    (;
        dir=dir,
        scenarios=joinpath(dir, "test_scenarios.csv"),
        optima=joinpath(dir, "test_optima.csv"),
        metadata=joinpath(dir, "metadata.csv"),
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

function activation_function(name)
    symbol = Symbol(name)
    symbol == :gelu && return Flux.gelu
    symbol == :relu && return Flux.relu
    symbol in (:silu, :swish) && return Flux.swish
    throw(ArgumentError("unsupported activation: $(name)"))
end

softplus_output(x) = Flux.softplus.(x)

function build_model(config, problem)
    depth = Int(config.depth)
    depth > 0 || throw(ArgumentError("depth must be positive."))
    hidden_size = Int(config.hidden_size)
    output_dimension = demand_count(problem) * NR_SCENARIOS
    activation = activation_function(config.activation)
    init = Flux.glorot_uniform(Random.MersenneTwister(Int(config.seed)))

    layers = Any[Flux.Dense(3 => hidden_size, activation; init=init)]
    for _ in 2:depth
        push!(layers, Flux.Dense(hidden_size => hidden_size, activation; init=init))
    end
    push!(layers, Flux.Dense(hidden_size => output_dimension; init=init))
    push!(layers, softplus_output)
    return Flux.Chain(layers...) |> Flux.f64
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

generate_training_dataset(config) =
    generate_resource_allocation_dataset(
        seed=Int(config.seed),
        context_count=Int(config.training_contexts),
        scenarios_per_context=1,
    )

function test_shape(; smoke=false)
    if smoke
        return (;
            contexts=SMOKE_TEST_CONTEXTS,
            scenarios_per_context=SMOKE_TEST_SCENARIOS_PER_CONTEXT,
            evaluation_batches=SMOKE_TEST_EVALUATION_BATCHES,
        )
    end
    return (;
        contexts=TEST_CONTEXTS,
        scenarios_per_context=TEST_SCENARIOS_PER_CONTEXT,
        evaluation_batches=TEST_EVALUATION_BATCHES,
    )
end

function test_cache_exists(; smoke=false)
    paths = test_cache_paths(smoke=smoke)
    return isfile(paths.scenarios) && isfile(paths.optima) && isfile(paths.metadata)
end

function scenario_rows_from_dataset(dataset)
    rows = NamedTuple[]
    for (sample_index, data_point) in enumerate(dataset)
        context = Float64.(data_point.context)
        for (scenario_index, scenario) in enumerate(data_point.scenario_parameters)
            push!(
                rows,
                (;
                    sample_index=sample_index,
                    scenario_index=scenario_index,
                    context_1=context[1],
                    context_2=context[2],
                    context_3=context[3],
                    demand_values=Float64.(collect(scenario.h_eq_xi)),
                ),
            )
        end
    end
    return rows
end

function optimum_rows_from_results(optimal_results)
    return [
        (;
            sample_index=index,
            evaluation_batches=Int(result.evaluation_batches),
            objective_value=Float64(result.objective_value),
            objective_values=Float64.(collect(result.objective_values)),
        ) for (index, result) in enumerate(optimal_results)
    ]
end

function precompute_test_cache(; smoke=false)
    shape = test_shape(smoke=smoke)
    started_at = unix_milliseconds()
    dataset = generate_resource_allocation_dataset(
        seed=TEST_SEED,
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
            evaluation_batches=shape.evaluation_batches,
        )
    end

    metadata = (;
        version=SUITE_VERSION,
        smoke=Bool(smoke),
        seed=TEST_SEED,
        contexts=shape.contexts,
        scenarios_per_context=shape.scenarios_per_context,
        evaluation_batches=shape.evaluation_batches,
        solve_seconds=solve_seconds,
        worker_id=Distributed.myid(),
        hostname=Sockets.gethostname(),
        pid=getpid(),
        started_at=started_at,
        finished_at=unix_milliseconds(),
    )

    return (;
        dataset=dataset,
        optimal_results=optimal_results,
        scenario_rows=scenario_rows_from_dataset(dataset),
        optimum_rows=optimum_rows_from_results(optimal_results),
        metadata=metadata,
    )
end

function write_test_cache_csv!(cache; smoke=false)
    paths = test_cache_paths(smoke=smoke)
    write_csv_file(paths.scenarios, TEST_SCENARIO_HEADERS, cache.scenario_rows)
    write_csv_file(paths.optima, TEST_OPTIMUM_HEADERS, cache.optimum_rows)
    write_csv_file(paths.metadata, TEST_METADATA_HEADERS, [cache.metadata])
    return paths
end

function load_test_cache_from_csv(; smoke=false)
    paths = test_cache_paths(smoke=smoke)
    test_cache_exists(smoke=smoke) ||
        error("Missing test cache CSV files in $(paths.dir).")

    metadata_rows = read_csv_rows(paths.metadata)
    metadata = isempty(metadata_rows) ? (;) : first(metadata_rows)
    grouped = Dict{Int,Vector{Any}}()
    for row in read_csv_rows(paths.scenarios)
        sample_index = Int(row.sample_index)
        push!(get!(grouped, sample_index, Any[]), row)
    end

    dataset = ContextualDFL.ContextualDataPoint[]
    for sample_index in sort(collect(keys(grouped)))
        rows = sort(grouped[sample_index]; by=row -> Int(row.scenario_index))
        first_row = first(rows)
        context = Float64[first_row.context_1, first_row.context_2, first_row.context_3]
        scenarios = [
            ContextualDFL.ParametricScenario(;
                W_eq_xi=Float64[],
                W_ineq_xi=Float64[],
                T_eq_xi=Float64[],
                T_ineq_xi=Float64[],
                h_eq_xi=parse_vector_cell(row.demand_values),
                h_ineq_xi=Float64[],
                q_xi=Float64[],
            ) for row in rows
        ]
        push!(dataset, ContextualDFL.ContextualDataPoint(context, scenarios))
    end

    optima_rows = sort(read_csv_rows(paths.optima); by=row -> Int(row.sample_index))
    optimal_results = [
        (;
            evaluation_batches=Int(row.evaluation_batches),
            objective_values=parse_vector_cell(row.objective_values),
            objective_value=Float64(row.objective_value),
        ) for row in optima_rows
    ]

    return (; dataset=dataset, optimal_results=optimal_results, metadata=metadata)
end

function set_worker_test_cache!(cache)
    WORKER_TEST_CACHE[] = cache
    return (;
        worker_id=Distributed.myid(),
        hostname=Sockets.gethostname(),
        pid=getpid(),
        test_contexts=length(cache.dataset),
        optimal_results=length(cache.optimal_results),
    )
end

function replicate_seed(replicate)
    return TRAINING_SEED_BASE + Int(replicate)
end

function annealing_segments()
    values = Float64[1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.08, 0.06, 0.04, 0.02, 0.01]
    segments = Tuple{Float64,Float64,Int}[]
    for (index, value) in enumerate(values)
        push!(segments, (value, value, index == 1 ? 20 : 10))
    end
    push!(segments, (last(values), 0.0, 10))
    return segments
end

function scaled_segment_lengths(total_epochs)
    total_epochs > 0 || return Int[]
    segments = annealing_segments()
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
        throw(ArgumentError("could not scale annealing schedule to $total_epochs epochs."))
    return lengths
end

function mu_schedules_for_epochs(total_epochs)
    total_epochs <= 0 && return Float64[], Float64[]
    segments = annealing_segments()
    lengths = scaled_segment_lengths(Int(total_epochs))
    mu_in = Float64[]
    mu_ref = Float64[]
    for ((mu_value, ref_value, _), length) in zip(segments, lengths)
        append!(mu_in, fill(mu_value, length))
        append!(mu_ref, fill(ref_value, length))
    end
    return mu_in, mu_ref
end

mu_schedules_for_config(config) = mu_schedules_for_epochs(Int(config.epochs))

function schedule_preview(values; n=6)
    isempty(values) && return ""
    keep = min(Int(n), length(values))
    head = join(string.(round.(values[1:keep]; digits=6)), "|")
    length(values) <= keep && return head
    return head * "|...|" * string(round(last(values); digits=6))
end

function run_id(phase, candidate_name, replicate; smoke=false)
    return join(
        (
            smoke ? "smoke" : "full",
            string(phase),
            safe_path_part(candidate_name),
            "rep" * lpad(string(Int(replicate)), 2, "0"),
        ),
        "_",
    )
end

function base_config(; phase, candidate_name, replicate, smoke=false, overrides...)
    epoch_count = smoke ? 2 : BASE_TOTAL_EPOCHS
    config = merge(
        (;
            version=SUITE_VERSION,
            smoke=Bool(smoke),
            phase=Symbol(phase),
            candidate_name=String(candidate_name),
            replicate=Int(replicate),
            seed=replicate_seed(replicate),
            training_contexts=BASE_TRAINING_CONTEXTS,
            epochs=epoch_count,
            depth=BASE_DEPTH,
            hidden_size=BASE_HIDDEN_SIZE,
            activation=BASE_ACTIVATION,
            batch_size=BASE_BATCH_SIZE,
            learning_rate=BASE_LEARNING_RATE,
            checkpoint_every_epochs=smoke ? 1 : CHECKPOINT_EVERY_EPOCHS,
        ),
        NamedTuple(overrides),
    )
    return merge(
        config,
        (;
            run_id=run_id(
                Symbol(config.phase),
                String(config.candidate_name),
                Int(config.replicate);
                smoke=Bool(config.smoke),
            ),
        ),
    )
end

normalized_epochs(training_contexts; smoke=false) =
    smoke ? 2 : max(1, round(Int, BASE_TOTAL_EPOCHS * BASE_TRAINING_CONTEXTS / Int(training_contexts)))

function baseline_configs(; smoke=false)
    reps = 1:(smoke ? 1 : 10)
    return [
        base_config(;
            phase=:baseline,
            candidate_name="standard_n100_depth4_gelu",
            replicate=replicate,
            smoke=smoke,
        ) for replicate in reps
    ]
end

function data_amount_configs(; smoke=false)
    reps = 1:(smoke ? 1 : 5)
    amounts = smoke ? (5, 10) : (500, 1000)
    return [
        base_config(;
            phase=:data_amount,
            candidate_name="n$(amount)",
            replicate=replicate,
            smoke=smoke,
            training_contexts=amount,
            epochs=normalized_epochs(amount; smoke=smoke),
        ) for amount in amounts for replicate in reps
    ]
end

function depth_configs(; smoke=false)
    reps = 1:(smoke ? 1 : 6)
    depths = smoke ? (5, 6) : (5, 6, 10, 20, 40)
    return [
        base_config(;
            phase=:depth,
            candidate_name="depth$(depth)",
            replicate=replicate,
            smoke=smoke,
            depth=depth,
        ) for depth in depths for replicate in reps
    ]
end

function all_configs(; smoke=false)
    return vcat(
        baseline_configs(smoke=smoke),
        data_amount_configs(smoke=smoke),
        depth_configs(smoke=smoke),
    )
end

function config_row(config)
    mu_in, mu_ref = mu_schedules_for_config(config)
    return (;
        version=config.version,
        run_id=config.run_id,
        smoke=config.smoke,
        phase=String(config.phase),
        candidate_name=config.candidate_name,
        replicate=config.replicate,
        seed=config.seed,
        training_contexts=config.training_contexts,
        epochs=config.epochs,
        depth=config.depth,
        hidden_size=config.hidden_size,
        activation=String(config.activation),
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        mu_in_preview=schedule_preview(mu_in),
        mu_ref_preview=schedule_preview(mu_ref),
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

function prediction_metrics(model, dataset)
    isempty(dataset) && return (; mse=NaN, mae=NaN)
    target = target_matrix(dataset)
    prediction = Array(model(context_matrix(dataset)))
    if ndims(prediction) == 1
        prediction = reshape(prediction, :, 1)
    end
    size(prediction) == size(target) ||
        throw(DimensionMismatch("prediction size $(size(prediction)) != target size $(size(target))"))
    errors = prediction .- target
    return (; mse=Statistics.mean(abs2, errors), mae=Statistics.mean(abs.(errors)))
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

function put_log!(logger, kind::Symbol, row)
    logger === nothing && return nothing
    put!(logger, (; kind=kind, row=row))
    return nothing
end

function epoch_row(config, attempt_id, epoch, history_row, train_metrics, test_metrics)
    return (;
        attempt_id=attempt_id,
        run_id=config.run_id,
        phase=String(config.phase),
        candidate_name=config.candidate_name,
        replicate=Int(config.replicate),
        epoch=Int(epoch),
        mu_in=Float64(history_row.mu_in),
        mu_ref=Float64(history_row.mu_ref),
        rho_in=Float64(history_row.rho_in),
        rho_ref=Float64(history_row.rho_ref),
        iterations=Int(history_row.iterations),
        epoch_seconds=Float64(history_row.epoch_seconds),
        training_loss=Float64(history_row.loss),
        display_loss=Float64(history_row.display_loss),
        train_target_mse=Float64(train_metrics.mse),
        train_target_mae=Float64(train_metrics.mae),
        test_target_mse=Float64(test_metrics.mse),
        test_target_mae=Float64(test_metrics.mae),
        training_contexts=Int(config.training_contexts),
        epochs=Int(config.epochs),
        depth=Int(config.depth),
        hidden_size=Int(config.hidden_size),
        activation=String(config.activation),
        seed=Int(config.seed),
        worker_id=Distributed.myid(),
        hostname=Sockets.gethostname(),
        created_at=unix_milliseconds(),
    )
end

function metric_value(metrics, key, default=NaN)
    key in propertynames(metrics) || return default
    return Float64(getproperty(metrics, key))
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
    mu_in, _ = mu_schedules_for_config(config)
    policy = ContextualDFLExperiments.ScenarioGenerationPolicy(
        scenario_generator,
        objects.solver,
        objects.program;
        mu=isempty(mu_in) ? 0.0 : Float64(last(mu_in)),
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
            phase=String(config.phase),
            candidate_name=config.candidate_name,
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
        average_test_relative_regret=metric_value(metrics, :test_relative_regret_mean),
        mean_test_regret=metric_value(metrics, :test_regret_mean),
        mean_test_policy_value=metric_value(metrics, :test_policy_value_mean),
        mean_test_optimal_value=metric_value(metrics, :test_optimal_value_mean),
        test_sample_count=metric_value(metrics, :test_sample_count),
        test_evaluation_batches=metric_value(metrics, :test_evaluation_batches),
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
        phase=String(config.phase),
        candidate_name=config.candidate_name,
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
        phase=String(config.phase),
        candidate_name=config.candidate_name,
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
        phase=String(config.phase),
        candidate_name=config.candidate_name,
        replicate=Int(config.replicate),
        seed=Int(config.seed),
        training_contexts=Int(config.training_contexts),
        epochs=Int(config.epochs),
        depth=Int(config.depth),
        hidden_size=Int(config.hidden_size),
        activation=String(config.activation),
        average_test_relative_regret=row_value(metrics, :average_test_relative_regret, Inf),
        mean_test_regret=row_value(metrics, :mean_test_regret, Inf),
        mean_test_policy_value=row_value(metrics, :mean_test_policy_value, NaN),
        mean_test_optimal_value=row_value(metrics, :mean_test_optimal_value, NaN),
        test_sample_count=row_value(metrics, :test_sample_count, 0),
        test_evaluation_batches=row_value(metrics, :test_evaluation_batches, 0),
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
        test_dataset = WORKER_TEST_CACHE[].dataset
        loss = build_loss(objects)
        mu_in, mu_ref = mu_schedules_for_config(config)
        optimizer = Flux.Adam(Float64(config.learning_rate))

        for epoch in (completed_epoch + 1):Int(config.epochs)
            result = nothing
            elapsed = @elapsed begin
                result = ContextualDFL.train!(
                    model,
                    loss,
                    [mu_in[epoch]],
                    [mu_ref[epoch]],
                    data_set_training;
                    opt=optimizer,
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

            final_train_metrics = prediction_metrics(model, data_set_training)
            final_test_metrics = prediction_metrics(model, test_dataset)
            history_row = only(result.history)
            put_log!(
                logger,
                :epoch,
                epoch_row(
                    config,
                    attempt_id,
                    epoch,
                    history_row,
                    final_train_metrics,
                    final_test_metrics,
                ),
            )

            checkpoint_every = Int(config.checkpoint_every_epochs)
            if epoch == Int(config.epochs) || epoch % checkpoint_every == 0
                kind = epoch == Int(config.epochs) ? :final : :latest
                bytes = serialize_to_bytes(
                    checkpoint_payload(
                        config,
                        attempt_id,
                        model,
                        completed_epoch,
                        training_seconds_total,
                    ),
                )
                put_log!(logger, :checkpoint, checkpoint_row(config, attempt_id, epoch, kind, bytes))
            end
        end

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

function attach_resume_checkpoint(config)
    final_path = final_checkpoint_path(config.run_id; smoke=Bool(config.smoke))
    latest_path = latest_checkpoint_path(config.run_id; smoke=Bool(config.smoke))
    path = isfile(final_path) ? final_path : latest_path
    isfile(path) || return config
    return merge(config, (; resume_checkpoint_bytes=read(path)))
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

function summarize_phase(phase; smoke=false)
    rows = [row for row in read_csv_rows(result_paths(smoke=smoke).runs) if string(row.phase) == string(phase)]
    candidates = sort(unique(string(row.candidate_name) for row in rows))
    summary = NamedTuple[]
    for candidate in candidates
        candidate_rows = [row for row in rows if string(row.candidate_name) == candidate]
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
        values = [row.average_test_relative_regret for row in ok_rows]
        push!(
            summary,
            (;
                phase=string(phase),
                candidate_name=candidate,
                run_count=length(run_ids),
                ok_count=length(ok_rows),
                failed_count=failed_run_count,
                mean_average_test_relative_regret=mean_or_nan(values),
                std_average_test_relative_regret=std_or_nan(values),
                min_average_test_relative_regret=isempty(values) ? NaN : minimum(Float64.(values)),
                max_average_test_relative_regret=isempty(values) ? NaN : maximum(Float64.(values)),
            ),
        )
    end
    return summary
end
