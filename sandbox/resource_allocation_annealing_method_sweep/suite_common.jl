using ContextualDFL
using ContextualDFLExperiments
using Distributed
using Flux
using Random
using Sockets
using Statistics

const SUITE_DIR = @__DIR__
const REPO_ROOT = normpath(joinpath(SUITE_DIR, "..", ".."))
const TRAINING_PROJECT_DIR =
    joinpath(REPO_ROOT, "src", "ContextualDFL", "ContextualDFLTraining")

const SUITE_VERSION = "resource-allocation-annealing-method-sweep-v1"
const TEST_SEED = 20_260_505
const TRAINING_SEED_BASE = 95_000
const REPLICATES = 6
const SCREENING_REPLICATES = 2
const FINALIST_REPLICATES = 6
const FINALIST_COUNT = 2
const SCREENING_BASE_TOTAL_EPOCHS = 50
const EARLY_STOP_CHECK_EPOCH = 25
const EARLY_STOP_DOMINANCE_FACTOR = 3.0
const QUICK_VALIDATION_CONTEXTS = 20
const QUICK_VALIDATION_SEED_BASE = 80_000

const BASE_TRAINING_CONTEXTS = 100
const BASE_TEST_CONTEXTS = 100
const BASE_TEST_SCENARIOS_PER_CONTEXT = 100
const SMOKE_TEST_CONTEXTS = 2
const SMOKE_TEST_SCENARIOS_PER_CONTEXT = 2

const DEMAND_SIGMA = 5.0
const DEMAND_POWER = 2.0
const CONTEXT_TERMS = 3
const NR_SCENARIOS = 1
const BASE_HIDDEN_SIZE = 128
const BASE_DEPTH = 3
const BASE_ACTIVATION = :relu
const BASE_BATCH_SIZE = 1
const BASE_LEARNING_RATE = 1e-3
const BASE_TOTAL_EPOCHS = 130
const BASE_FINE_TUNING_EPOCHS = 10

const DEFAULT_STARTING_MU = 1.0
const DEFAULT_ENDING_MU = 0.01
const DEFAULT_PIECE_LENGTH = 10
const DEFAULT_NR_PIECES = 11
const DEFAULT_STARTING_PHASE_LENGTH = 20
const DEFAULT_FINE_TUNING_PHASE_LENGTH = 10

const PHASES = (
    :data_amount,
    :depth,
    :width,
    :schedule_shape,
    :piecewise_linear,
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
    :solve_seconds,
    :worker_id,
    :hostname,
    :pid,
    :started_at,
    :finished_at,
)

const WORKER_TEST_CACHE = Ref{Any}(nothing)

function unix_milliseconds()
    return round(Int64, time() * 1000)
end

suite_path(parts...) = joinpath(SUITE_DIR, parts...)
artifact_path(parts...) = suite_path("artifacts", parts...)
phase_dir(phase; smoke=false) = suite_path("phases", smoke ? "smoke" : "full", String(phase))
phase_path(phase, filename; smoke=false) = joinpath(phase_dir(phase; smoke=smoke), filename)

function test_cache_paths(; smoke=false)
    dir = artifact_path(smoke ? "smoke_test_data" : "test_data")
    return (;
        dir=dir,
        scenarios=joinpath(dir, "test_scenarios.csv"),
        optima=joinpath(dir, "test_optima.csv"),
        metadata=joinpath(dir, "metadata.csv"),
    )
end

function result_paths(; smoke=false)
    dir = suite_path(smoke ? "smoke_results" : "results")
    return (;
        dir=dir,
        attempts=joinpath(dir, "run_attempts.csv"),
        epochs=joinpath(dir, "epochs.csv"),
        runs=joinpath(dir, "runs.csv"),
        test_samples=joinpath(dir, "test_per_sample.csv"),
        early_checks=joinpath(dir, "early_checks.csv"),
        decisions=joinpath(dir, "decisions.csv"),
        final_selection=joinpath(dir, "final_selection.csv"),
    )
end

function csv_escape(value)
    if value === nothing || value === missing
        return ""
    elseif value isa Symbol
        value = String(value)
    elseif value isa AbstractVector
        value = join(string.(value), "|")
    elseif value isa Tuple
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
        haskey(row, String(key)) && return row[String(key)]
        return default
    else
        key in propertynames(row) && return getproperty(row, key)
        return default
    end
end

function append_csv_row(path, headers, row)
    mkpath(dirname(path))
    needs_header = !isfile(path) || filesize(path) == 0
    open(path, "a") do io
        if needs_header
            println(io, join(String.(headers), ","))
        end
        println(io, join((csv_escape(row_value(row, header)) for header in headers), ","))
        flush(io)
    end
    return path
end

function write_csv_file(path, headers, rows)
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, join(String.(headers), ","))
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
        else
            continue
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
    isfile(path) || return Any[]
    records = csv_records(readlines(path))
    isempty(records) && return Any[]
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
    symbol == :relu && return Flux.relu
    symbol == :gelu && return Flux.gelu
    symbol in (:silu, :swish) && return Flux.swish
    throw(ArgumentError("unsupported activation: $(name)"))
end

function build_model(config, problem)
    depth = Int(config.depth)
    depth > 0 || throw(ArgumentError("depth must be positive."))
    hidden_size = Int(config.hidden_size)
    output_dimension = demand_count(problem) * Int(config.nr_scenarios)
    activation = activation_function(config.activation)

    layers = Any[Flux.Dense(3 => hidden_size, activation)]
    for _ in 2:depth
        push!(layers, Flux.Dense(hidden_size => hidden_size, activation))
    end
    push!(layers, Flux.Dense(hidden_size => output_dimension, Flux.relu))
    return Flux.Chain(layers...) |> Flux.f64
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

function display_reference_input(point)
    return reduce(vcat, (scenario.h_eq_xi for scenario in point.scenario_parameters))
end

function test_shape(; smoke=false)
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
        )
    end

    metadata = (;
        version=SUITE_VERSION,
        smoke=Bool(smoke),
        seed=TEST_SEED,
        contexts=shape.contexts,
        scenarios_per_context=shape.scenarios_per_context,
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

function parse_vector_cell(value)
    text = string(value)
    isempty(text) && return Float64[]
    return [parse(Float64, item) for item in split(text, "|") if !isempty(item)]
end

function load_test_cache_from_csv(; smoke=false)
    paths = test_cache_paths(smoke=smoke)
    test_cache_exists(smoke=smoke) ||
        error("Missing test cache CSV files in $(paths.dir).")

    metadata_rows = read_csv_rows(paths.metadata)
    metadata = isempty(metadata_rows) ? (;) : first(metadata_rows)
    scenario_rows = read_csv_rows(paths.scenarios)
    legacy_scenarios =
        !isempty(scenario_rows) && !(:demand_values in propertynames(first(scenario_rows)))

    dataset = if legacy_scenarios
        seed = Int(row_value(metadata, :seed, TEST_SEED))
        context_count = Int(row_value(metadata, :contexts, test_shape(smoke=smoke).contexts))
        scenarios_per_context = Int(
            row_value(
                metadata,
                :scenarios_per_context,
                test_shape(smoke=smoke).scenarios_per_context,
            ),
        )
        regenerated = generate_resource_allocation_dataset(
            seed=seed,
            context_count=context_count,
            scenarios_per_context=scenarios_per_context,
        )
        write_csv_file(paths.scenarios, TEST_SCENARIO_HEADERS, scenario_rows_from_dataset(regenerated))
        regenerated
    else
        grouped = Dict{Int,Vector{Any}}()
        for row in scenario_rows
            sample_index = Int(row.sample_index)
            push!(get!(grouped, sample_index, Any[]), row)
        end

        parsed_dataset = ContextualDFL.ContextualDataPoint[]
        for sample_index in sort(collect(keys(grouped)))
            rows = sort(grouped[sample_index]; by=row -> Int(row.scenario_index))
            first_row = first(rows)
            context = Float64[
                first_row.context_1,
                first_row.context_2,
                first_row.context_3,
            ]
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
            push!(parsed_dataset, ContextualDFL.ContextualDataPoint(context, scenarios))
        end
        parsed_dataset
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
        test_contexts=length(cache.dataset),
        optimal_results=length(cache.optimal_results),
    )
end

function replicate_seed(replicate)
    return TRAINING_SEED_BASE + Int(replicate)
end

function normalized_epochs(training_contexts; smoke=false, base_epochs=BASE_TOTAL_EPOCHS)
    smoke && return 1
    return max(1, round(Int, Int(base_epochs) * BASE_TRAINING_CONTEXTS / Int(training_contexts)))
end

screening_epochs(training_contexts; smoke=false) =
    normalized_epochs(training_contexts; smoke=smoke, base_epochs=SCREENING_BASE_TOTAL_EPOCHS)

function fine_tuning_epochs(total_epochs; smoke=false)
    smoke && return 0
    total_epochs <= 1 && return 0
    return clamp(round(Int, total_epochs * BASE_FINE_TUNING_EPOCHS / BASE_TOTAL_EPOCHS), 1, total_epochs - 1)
end

function default_selection(; smoke=false)
    epochs = normalized_epochs(BASE_TRAINING_CONTEXTS; smoke=smoke)
    return (;
        training_contexts=BASE_TRAINING_CONTEXTS,
        epochs=epochs,
        final_epochs=fine_tuning_epochs(epochs; smoke=smoke),
        depth=BASE_DEPTH,
        hidden_size=BASE_HIDDEN_SIZE,
        activation=BASE_ACTIVATION,
        schedule_kind=:piecewise_linear,
        starting_mu=DEFAULT_STARTING_MU,
        ending_mu=DEFAULT_ENDING_MU,
        piece_length=DEFAULT_PIECE_LENGTH,
        nr_pieces=DEFAULT_NR_PIECES,
        starting_phase_length=DEFAULT_STARTING_PHASE_LENGTH,
        fine_tuning_phase_length=DEFAULT_FINE_TUNING_PHASE_LENGTH,
    )
end

function safe_path_part(text)
    return replace(String(text), r"[^A-Za-z0-9_.=-]" => "_")
end

function run_id(phase, candidate_name, replicate; smoke=false, stage=:exhaustive)
    return join(
        (
            smoke ? "smoke" : "full",
            string(phase),
            safe_path_part(stage),
            safe_path_part(candidate_name),
            "rep" * lpad(string(Int(replicate)), 2, "0"),
        ),
        "_",
    )
end

function early_stop_overrides(thresholds=nothing)
    thresholds === nothing && return (;)
    return (;
        early_stop_enabled=true,
        early_stop_epoch=Int(row_value(thresholds, :epoch, EARLY_STOP_CHECK_EPOCH)),
        early_stop_training_loss_threshold=Float64(
            row_value(thresholds, :training_loss_threshold, Inf),
        ),
        early_stop_display_loss_threshold=Float64(
            row_value(thresholds, :display_loss_threshold, Inf),
        ),
        early_stop_quick_validation_loss_threshold=Float64(
            row_value(thresholds, :quick_validation_loss_threshold, Inf),
        ),
    )
end

function base_run_config(; phase, candidate_name, replicate, smoke=false, overrides...)
    selection = default_selection(smoke=smoke)
    config = merge(
        (;
            version=SUITE_VERSION,
            smoke=Bool(smoke),
            stage=:exhaustive,
            phase=Symbol(phase),
            candidate_name=String(candidate_name),
            replicate=Int(replicate),
            seed=replicate_seed(replicate),
            training_contexts=selection.training_contexts,
            epochs=selection.epochs,
            final_epochs=selection.final_epochs,
            depth=selection.depth,
            hidden_size=selection.hidden_size,
            activation=selection.activation,
            schedule_kind=selection.schedule_kind,
            starting_mu=selection.starting_mu,
            ending_mu=selection.ending_mu,
            piece_length=selection.piece_length,
            nr_pieces=selection.nr_pieces,
            starting_phase_length=selection.starting_phase_length,
            fine_tuning_phase_length=selection.fine_tuning_phase_length,
            nr_scenarios=NR_SCENARIOS,
            batch_size=BASE_BATCH_SIZE,
            learning_rate=BASE_LEARNING_RATE,
            reset_optimizer_each_epoch=true,
            early_stop_enabled=false,
            early_stop_epoch=EARLY_STOP_CHECK_EPOCH,
            early_stop_training_loss_threshold=Inf,
            early_stop_display_loss_threshold=Inf,
            early_stop_quick_validation_loss_threshold=Inf,
            quick_validation_contexts=QUICK_VALIDATION_CONTEXTS,
            quick_validation_seed=QUICK_VALIDATION_SEED_BASE + Int(replicate),
        ),
        NamedTuple(overrides),
    )
    return merge(
        config,
        (;
            run_id=run_id(
                Symbol(phase),
                String(candidate_name),
                replicate;
                smoke=smoke,
                stage=Symbol(config.stage),
            ),
        ),
    )
end

function config_from_selection(; phase, candidate_name, replicate, selection, smoke=false, overrides...)
    return base_run_config(;
        phase=phase,
        candidate_name=candidate_name,
        replicate=replicate,
        smoke=smoke,
        training_contexts=selection.training_contexts,
        epochs=selection.epochs,
        final_epochs=selection.final_epochs,
        depth=selection.depth,
        hidden_size=selection.hidden_size,
        activation=selection.activation,
        schedule_kind=selection.schedule_kind,
        starting_mu=selection.starting_mu,
        ending_mu=selection.ending_mu,
        piece_length=selection.piece_length,
        nr_pieces=selection.nr_pieces,
        starting_phase_length=selection.starting_phase_length,
        fine_tuning_phase_length=selection.fine_tuning_phase_length,
        overrides...,
    )
end

function data_amount_configs(
    selection;
    smoke=false,
    replicates=nothing,
    stage=:exhaustive,
    screening=false,
    candidate_names=nothing,
    early_stop_thresholds=nothing,
)
    reps = replicates === nothing ? (1:(smoke ? 1 : REPLICATES)) : replicates
    amounts = smoke ? (2, 3, 4) : (100, 500, 1000)
    if candidate_names !== nothing
        keep = Set(String.(candidate_names))
        amounts = tuple([amount for amount in amounts if "n$(amount)" in keep]...)
    end
    return [
        base_run_config(;
            phase=:data_amount,
            candidate_name="n$(amount)",
            replicate=replicate,
            smoke=smoke,
            stage=stage,
            training_contexts=amount,
            epochs=screening ? screening_epochs(amount; smoke=smoke) :
                   normalized_epochs(amount; smoke=smoke),
            final_epochs=fine_tuning_epochs(
                screening ? screening_epochs(amount; smoke=smoke) :
                normalized_epochs(amount; smoke=smoke);
                smoke=smoke,
            ),
            early_stop_overrides(early_stop_thresholds)...,
        ) for amount in amounts for replicate in reps
    ]
end

function depth_configs(
    selection;
    smoke=false,
    replicates=nothing,
    stage=:exhaustive,
    screening=false,
    candidate_names=nothing,
    early_stop_thresholds=nothing,
)
    reps = replicates === nothing ? (1:(smoke ? 1 : REPLICATES)) : replicates
    depths = (3, 4, 5, 10, 20)
    if candidate_names !== nothing
        keep = Set(String.(candidate_names))
        depths = tuple([depth for depth in depths if "depth$(depth)" in keep]...)
    end
    return [
        config_from_selection(;
            phase=:depth,
            candidate_name="depth$(depth)",
            replicate=replicate,
            selection=selection,
            smoke=smoke,
            stage=stage,
            epochs=screening ? screening_epochs(selection.training_contexts; smoke=smoke) :
                   selection.epochs,
            final_epochs=screening ? fine_tuning_epochs(
                screening_epochs(selection.training_contexts; smoke=smoke);
                smoke=smoke,
            ) : selection.final_epochs,
            depth=depth,
            early_stop_overrides(early_stop_thresholds)...,
        ) for depth in depths for replicate in reps
    ]
end

function width_configs(
    selection;
    smoke=false,
    replicates=nothing,
    stage=:exhaustive,
    screening=false,
    candidate_names=nothing,
    early_stop_thresholds=nothing,
)
    reps = replicates === nothing ? (1:(smoke ? 1 : REPLICATES)) : replicates
    widths = (32, 64, 128, 256, 512)
    if candidate_names !== nothing
        keep = Set(String.(candidate_names))
        widths = tuple([width for width in widths if "width$(width)" in keep]...)
    end
    return [
        config_from_selection(;
            phase=:width,
            candidate_name="width$(width)",
            replicate=replicate,
            selection=selection,
            smoke=smoke,
            stage=stage,
            epochs=screening ? screening_epochs(selection.training_contexts; smoke=smoke) :
                   selection.epochs,
            final_epochs=screening ? fine_tuning_epochs(
                screening_epochs(selection.training_contexts; smoke=smoke);
                smoke=smoke,
            ) : selection.final_epochs,
            hidden_size=width,
            early_stop_overrides(early_stop_thresholds)...,
        ) for width in widths for replicate in reps
    ]
end

function schedule_shape_configs(
    selection;
    smoke=false,
    replicates=nothing,
    stage=:exhaustive,
    screening=false,
    candidate_names=nothing,
    early_stop_thresholds=nothing,
)
    reps = replicates === nothing ? (1:(smoke ? 1 : REPLICATES)) : replicates
    shapes = (
        :piecewise_linear,
        :linear,
        :geometric,
        :cosine,
        :delayed_quadratic,
        :early_quadratic,
    )
    if candidate_names !== nothing
        keep = Set(String.(candidate_names))
        shapes = tuple([shape for shape in shapes if String(shape) in keep]...)
    end
    return [
        config_from_selection(;
            phase=:schedule_shape,
            candidate_name=String(shape),
            replicate=replicate,
            selection=selection,
            smoke=smoke,
            stage=stage,
            epochs=screening ? screening_epochs(selection.training_contexts; smoke=smoke) :
                   selection.epochs,
            final_epochs=screening ? fine_tuning_epochs(
                screening_epochs(selection.training_contexts; smoke=smoke);
                smoke=smoke,
            ) : selection.final_epochs,
            schedule_kind=shape,
            early_stop_overrides(early_stop_thresholds)...,
        ) for shape in shapes for replicate in reps
    ]
end

function piecewise_candidate_specs()
    return (
        (;
            name="default",
            starting_mu=1.0,
            ending_mu=0.01,
            piece_length=10,
            nr_pieces=11,
            starting_phase_length=20,
            fine_tuning_phase_length=10,
        ),
        (;
            name="lower_end_mu",
            starting_mu=1.0,
            ending_mu=0.001,
            piece_length=10,
            nr_pieces=11,
            starting_phase_length=20,
            fine_tuning_phase_length=10,
        ),
        (;
            name="higher_end_mu",
            starting_mu=1.0,
            ending_mu=0.05,
            piece_length=10,
            nr_pieces=11,
            starting_phase_length=20,
            fine_tuning_phase_length=10,
        ),
        (;
            name="short_start",
            starting_mu=1.0,
            ending_mu=0.01,
            piece_length=11,
            nr_pieces=11,
            starting_phase_length=10,
            fine_tuning_phase_length=10,
        ),
        (;
            name="long_start",
            starting_mu=1.0,
            ending_mu=0.01,
            piece_length=9,
            nr_pieces=11,
            starting_phase_length=30,
            fine_tuning_phase_length=10,
        ),
        (;
            name="more_pieces",
            starting_mu=1.0,
            ending_mu=0.01,
            piece_length=5,
            nr_pieces=21,
            starting_phase_length=20,
            fine_tuning_phase_length=10,
        ),
        (;
            name="fewer_pieces",
            starting_mu=1.0,
            ending_mu=0.01,
            piece_length=20,
            nr_pieces=6,
            starting_phase_length=20,
            fine_tuning_phase_length=10,
        ),
        (;
            name="long_finetune",
            starting_mu=1.0,
            ending_mu=0.01,
            piece_length=9,
            nr_pieces=11,
            starting_phase_length=20,
            fine_tuning_phase_length=20,
        ),
    )
end

function piecewise_linear_configs(
    selection;
    smoke=false,
    replicates=nothing,
    stage=:exhaustive,
    screening=false,
    candidate_names=nothing,
    early_stop_thresholds=nothing,
)
    reps = replicates === nothing ? (1:(smoke ? 1 : REPLICATES)) : replicates
    keep = candidate_names === nothing ? nothing : Set(String.(candidate_names))
    configs = NamedTuple[]
    for spec in piecewise_candidate_specs(), replicate in reps
        keep === nothing || spec.name in keep || continue
        epoch_count = screening ? screening_epochs(selection.training_contexts; smoke=smoke) :
                      selection.epochs
        push!(
            configs,
            config_from_selection(;
                phase=:piecewise_linear,
                candidate_name=spec.name,
                replicate=replicate,
                selection=selection,
                smoke=smoke,
                stage=stage,
                epochs=epoch_count,
                final_epochs=screening ? fine_tuning_epochs(epoch_count; smoke=smoke) :
                             selection.final_epochs,
                schedule_kind=:piecewise_linear,
                starting_mu=spec.starting_mu,
                ending_mu=spec.ending_mu,
                piece_length=spec.piece_length,
                nr_pieces=spec.nr_pieces,
                starting_phase_length=spec.starting_phase_length,
                fine_tuning_phase_length=spec.fine_tuning_phase_length,
                early_stop_overrides(early_stop_thresholds)...,
            ),
        )
    end
    return configs
end

function phase_configs(
    phase,
    selection;
    smoke=false,
    replicates=nothing,
    stage=:exhaustive,
    screening=false,
    candidate_names=nothing,
    early_stop_thresholds=nothing,
)
    kwargs = (;
        smoke=smoke,
        replicates=replicates,
        stage=stage,
        screening=screening,
        candidate_names=candidate_names,
        early_stop_thresholds=early_stop_thresholds,
    )
    phase == :data_amount && return data_amount_configs(selection; kwargs...)
    phase == :depth && return depth_configs(selection; kwargs...)
    phase == :width && return width_configs(selection; kwargs...)
    phase == :schedule_shape && return schedule_shape_configs(selection; kwargs...)
    phase == :piecewise_linear && return piecewise_linear_configs(selection; kwargs...)
    throw(ArgumentError("unsupported phase: $(phase)"))
end

function interpolate_resample(values, target_length)
    source_length = length(values)
    target_length == source_length && return Float64.(copy(values))
    target_length <= 0 && return Float64[]
    source_length == 0 && return zeros(Float64, target_length)
    target_length == 1 && return [Float64(first(values))]
    source_length == 1 && return fill(Float64(first(values)), target_length)

    output = Float64[]
    for index in 1:target_length
        position = 1 + (index - 1) * (source_length - 1) / (target_length - 1)
        lower = floor(Int, position)
        upper = ceil(Int, position)
        if lower == upper
            push!(output, Float64(values[lower]))
        else
            fraction = position - lower
            push!(output, (1 - fraction) * Float64(values[lower]) + fraction * Float64(values[upper]))
        end
    end
    return output
end

function piecewise_raw_mu_schedules(config)
    starting_mu = Float64(config.starting_mu)
    ending_mu = Float64(config.ending_mu)
    piece_length = Int(config.piece_length)
    nr_pieces = Int(config.nr_pieces)
    starting_phase_length = Int(config.starting_phase_length)
    fine_tuning_phase_length = Int(config.fine_tuning_phase_length)
    piece_length > 0 || throw(ArgumentError("piece_length must be positive."))
    nr_pieces >= 2 || throw(ArgumentError("nr_pieces must be at least 2."))
    starting_phase_length >= 0 || throw(ArgumentError("starting_phase_length must be non-negative."))
    fine_tuning_phase_length >= 0 || throw(ArgumentError("fine_tuning_phase_length must be non-negative."))

    anchors = collect(range(starting_mu, ending_mu; length=nr_pieces))
    mu_in = fill(starting_mu, starting_phase_length)
    for segment in 1:(nr_pieces - 1)
        left = anchors[segment]
        right = anchors[segment + 1]
        for step in 1:piece_length
            t = step / piece_length
            push!(mu_in, (1 - t) * left + t * right)
        end
    end
    append!(mu_in, fill(ending_mu, fine_tuning_phase_length))

    mu_ref = copy(mu_in)
    if fine_tuning_phase_length > 0
        mu_ref[(end - fine_tuning_phase_length + 1):end] .= 0.0
    end
    return Float64.(mu_in), Float64.(mu_ref)
end

function shaped_mu_main(kind, main_epochs, starting_mu, ending_mu)
    main_epochs <= 0 && return Float64[]
    main_epochs == 1 && return [Float64(starting_mu)]
    t_values = collect(range(0.0, 1.0; length=main_epochs))
    kind = Symbol(kind)

    if kind == :linear
        return collect(range(starting_mu, ending_mu; length=main_epochs))
    elseif kind == :geometric
        starting_mu > 0 && ending_mu > 0 ||
            throw(ArgumentError("geometric schedule requires positive start/end mu."))
        return exp.(range(log(starting_mu), log(ending_mu); length=main_epochs))
    elseif kind == :cosine
        return [
            ending_mu + (starting_mu - ending_mu) * 0.5 * (1 + cos(pi * t)) for
            t in t_values
        ]
    elseif kind == :delayed_quadratic
        return [starting_mu + (ending_mu - starting_mu) * t^2 for t in t_values]
    elseif kind == :early_quadratic
        return [starting_mu + (ending_mu - starting_mu) * (1 - (1 - t)^2) for t in t_values]
    end

    throw(ArgumentError("unsupported shaped schedule: $(kind)"))
end

function mu_schedules_for_config(config)
    total_epochs = Int(config.epochs)
    final_epochs = Int(config.final_epochs)
    total_epochs >= 0 || throw(ArgumentError("epochs must be non-negative."))
    final_epochs >= 0 || throw(ArgumentError("final_epochs must be non-negative."))
    final_epochs <= total_epochs ||
        throw(ArgumentError("final_epochs cannot exceed epochs."))
    total_epochs == 0 && return Float64[], Float64[]

    kind = Symbol(config.schedule_kind)
    if kind == :piecewise_linear
        raw_in, raw_ref = piecewise_raw_mu_schedules(config)
        return interpolate_resample(raw_in, total_epochs), interpolate_resample(raw_ref, total_epochs)
    end

    main_epochs = total_epochs - final_epochs
    main = shaped_mu_main(kind, main_epochs, Float64(config.starting_mu), Float64(config.ending_mu))
    mu_in = vcat(main, fill(Float64(config.ending_mu), final_epochs))
    mu_ref = vcat(copy(main), zeros(Float64, final_epochs))
    return mu_in, mu_ref
end

function schedule_preview(values; n=6)
    isempty(values) && return ""
    keep = min(Int(n), length(values))
    head = join(string.(round.(values[1:keep]; digits=6)), "|")
    length(values) <= keep && return head
    return head * "|...|" * string(round(last(values); digits=6))
end

function piecewise_raw_epoch_count(config)
    if Symbol(config.schedule_kind) != :piecewise_linear
        return missing
    end
    return Int(config.starting_phase_length) +
           Int(config.piece_length) * (Int(config.nr_pieces) - 1) +
           Int(config.fine_tuning_phase_length)
end

function config_row(config)
    mu_in, mu_ref = mu_schedules_for_config(config)
    return (;
        version=config.version,
        run_id=config.run_id,
        smoke=config.smoke,
        stage=String(config.stage),
        phase=String(config.phase),
        candidate_name=config.candidate_name,
        replicate=config.replicate,
        seed=config.seed,
        training_contexts=config.training_contexts,
        epochs=config.epochs,
        final_epochs=config.final_epochs,
        depth=config.depth,
        hidden_size=config.hidden_size,
        activation=String(config.activation),
        schedule_kind=String(config.schedule_kind),
        starting_mu=config.starting_mu,
        ending_mu=config.ending_mu,
        piece_length=config.piece_length,
        nr_pieces=config.nr_pieces,
        starting_phase_length=config.starting_phase_length,
        fine_tuning_phase_length=config.fine_tuning_phase_length,
        piecewise_raw_epochs=piecewise_raw_epoch_count(config),
        schedule_resampled=piecewise_raw_epoch_count(config) !== missing &&
                           piecewise_raw_epoch_count(config) != Int(config.epochs),
        nr_scenarios=config.nr_scenarios,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        reset_optimizer_each_epoch=config.reset_optimizer_each_epoch,
        early_stop_enabled=config.early_stop_enabled,
        early_stop_epoch=config.early_stop_epoch,
        early_stop_training_loss_threshold=config.early_stop_training_loss_threshold,
        early_stop_display_loss_threshold=config.early_stop_display_loss_threshold,
        early_stop_quick_validation_loss_threshold=
            config.early_stop_quick_validation_loss_threshold,
        quick_validation_contexts=config.quick_validation_contexts,
        mu_in_preview=schedule_preview(mu_in),
        mu_ref_preview=schedule_preview(mu_ref),
    )
end

const CONFIG_HEADERS = (
    :version,
    :run_id,
    :smoke,
    :stage,
    :phase,
    :candidate_name,
    :replicate,
    :seed,
    :training_contexts,
    :epochs,
    :final_epochs,
    :depth,
    :hidden_size,
    :activation,
    :schedule_kind,
    :starting_mu,
    :ending_mu,
    :piece_length,
    :nr_pieces,
    :starting_phase_length,
    :fine_tuning_phase_length,
    :piecewise_raw_epochs,
    :schedule_resampled,
    :nr_scenarios,
    :batch_size,
    :learning_rate,
    :reset_optimizer_each_epoch,
    :early_stop_enabled,
    :early_stop_epoch,
    :early_stop_training_loss_threshold,
    :early_stop_display_loss_threshold,
    :early_stop_quick_validation_loss_threshold,
    :quick_validation_contexts,
    :mu_in_preview,
    :mu_ref_preview,
)

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

const EPOCH_HEADERS = (
    :attempt_id,
    :run_id,
    :phase,
    :candidate_name,
    :replicate,
    :epoch,
    :mu,
    :mu_in,
    :mu_ref,
    :rho_in,
    :rho_ref,
    :iterations,
    :epoch_seconds,
    :training_loss,
    :display_loss,
    :training_contexts,
    :epochs,
    :depth,
    :hidden_size,
    :schedule_kind,
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
    :final_epochs,
    :depth,
    :hidden_size,
    :activation,
    :schedule_kind,
    :starting_mu,
    :ending_mu,
    :piece_length,
    :nr_pieces,
    :starting_phase_length,
    :fine_tuning_phase_length,
    :average_test_loss,
    :mean_test_relative_regret,
    :mean_test_regret,
    :mean_test_policy_value,
    :mean_test_optimal_value,
    :test_sample_count,
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

const EARLY_CHECK_HEADERS = (
    :attempt_id,
    :run_id,
    :phase,
    :candidate_name,
    :replicate,
    :epoch,
    :training_loss,
    :display_loss,
    :quick_validation_loss,
    :training_loss_threshold,
    :display_loss_threshold,
    :quick_validation_loss_threshold,
    :stopped,
    :reason,
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
    :mean_average_test_loss,
    :std_average_test_loss,
    :min_average_test_loss,
    :max_average_test_loss,
)

const DECISION_HEADERS = (
    :phase,
    :selected_candidate,
    :selected_mean_average_test_loss,
    :selected_ok_count,
    :selected_run_count,
    :training_contexts,
    :epochs,
    :final_epochs,
    :depth,
    :hidden_size,
    :activation,
    :schedule_kind,
    :starting_mu,
    :ending_mu,
    :piece_length,
    :nr_pieces,
    :starting_phase_length,
    :fine_tuning_phase_length,
    :decided_at,
)

const FINALIST_HEADERS = (
    :phase,
    :rank,
    :candidate_name,
    :screening_run_count,
    :screening_ok_count,
    :screening_failed_count,
    :screening_mean_average_test_loss,
    :screening_std_average_test_loss,
    :selected_at,
)

function put_log!(logger, kind::Symbol, row)
    logger === nothing && return nothing
    put!(logger, (; kind=kind, row=row))
    return nothing
end

struct EarlyStopException <: Exception
    message::String
end

Base.showerror(io::IO, error::EarlyStopException) = print(io, error.message)

function epoch_row(config, attempt_id, local_epoch, loss_value, display_loss, metadata)
    return (;
        attempt_id=attempt_id,
        run_id=config.run_id,
        phase=String(config.phase),
        candidate_name=config.candidate_name,
        replicate=Int(config.replicate),
        epoch=Int(local_epoch),
        mu=Float64(metadata.mu),
        mu_in=Float64(metadata.mu_in),
        mu_ref=Float64(metadata.mu_ref),
        rho_in=Float64(metadata.rho_in),
        rho_ref=Float64(metadata.rho_ref),
        iterations=Int(metadata.iterations),
        epoch_seconds=Float64(metadata.epoch_seconds),
        training_loss=Float64(loss_value),
        display_loss=Float64(display_loss),
        training_contexts=Int(config.training_contexts),
        epochs=Int(config.epochs),
        depth=Int(config.depth),
        hidden_size=Int(config.hidden_size),
        schedule_kind=String(config.schedule_kind),
        seed=Int(config.seed),
        worker_id=Distributed.myid(),
        hostname=Sockets.gethostname(),
        created_at=unix_milliseconds(),
    )
end

function quick_validation_dataset(config)
    context_count = Int(config.quick_validation_contexts)
    context_count <= 0 && return nothing
    return generate_resource_allocation_dataset(
        seed=Int(config.quick_validation_seed),
        context_count=context_count,
        scenarios_per_context=1,
    )
end

function quick_validation_loss(model, loss, validation_dataset, config, mu_in, mu_ref)
    validation_dataset === nothing && return NaN
    values = Float64[]
    for point in validation_dataset
        value = loss(
            model(point.context),
            point.scenario_parameters,
            mu_in,
            mu_ref;
            nr_scenarios=Int(config.nr_scenarios),
        )
        push!(values, Float64(value))
    end
    isempty(values) && return NaN
    return Statistics.mean(values)
end

function early_stop_threshold(value)
    parsed = numeric_or_inf(value)
    return isfinite(parsed) ? parsed : Inf
end

function early_stop_decision(config, training_loss, display_loss, quick_loss)
    training_threshold = early_stop_threshold(config.early_stop_training_loss_threshold)
    display_threshold = early_stop_threshold(config.early_stop_display_loss_threshold)
    quick_threshold = early_stop_threshold(config.early_stop_quick_validation_loss_threshold)

    reasons = String[]
    if isfinite(training_threshold) && Float64(training_loss) > training_threshold
        push!(
            reasons,
            "training_loss $(Float64(training_loss)) > threshold $(training_threshold)",
        )
    end
    if isfinite(display_threshold) && Float64(display_loss) > display_threshold
        push!(
            reasons,
            "display_loss $(Float64(display_loss)) > threshold $(display_threshold)",
        )
    end
    if isfinite(quick_threshold) && isfinite(Float64(quick_loss)) &&
       Float64(quick_loss) > quick_threshold
        push!(
            reasons,
            "quick_validation_loss $(Float64(quick_loss)) > threshold $(quick_threshold)",
        )
    end

    return (; stopped=!isempty(reasons), reason=join(reasons, "; "))
end

function early_check_row(config, attempt_id, epoch, training_loss, display_loss, quick_loss, decision)
    return (;
        attempt_id=attempt_id,
        run_id=config.run_id,
        phase=String(config.phase),
        candidate_name=config.candidate_name,
        replicate=Int(config.replicate),
        epoch=Int(epoch),
        training_loss=Float64(training_loss),
        display_loss=Float64(display_loss),
        quick_validation_loss=Float64(quick_loss),
        training_loss_threshold=config.early_stop_training_loss_threshold,
        display_loss_threshold=config.early_stop_display_loss_threshold,
        quick_validation_loss_threshold=config.early_stop_quick_validation_loss_threshold,
        stopped=Bool(decision.stopped),
        reason=decision.reason,
        worker_id=Distributed.myid(),
        hostname=Sockets.gethostname(),
        created_at=unix_milliseconds(),
    )
end

function build_loss(objects, config)
    return ContextualDFL.DflScenLoss(
        objects.scenario_decoder,
        objects.reference_scenario_decoder,
        objects.solver,
        objects.program;
        nr_scenarios=Int(config.nr_scenarios),
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
    mean_relative_regret = metric_value(metrics, :test_relative_regret_mean)
    return (;
        metrics=metrics,
        per_sample_rows=per_sample_rows,
        average_test_loss=mean_relative_regret,
        mean_test_relative_regret=mean_relative_regret,
        mean_test_regret=metric_value(metrics, :test_regret_mean),
        mean_test_policy_value=metric_value(metrics, :test_policy_value_mean),
        mean_test_optimal_value=metric_value(metrics, :test_optimal_value_mean),
        test_sample_count=metric_value(metrics, :test_sample_count),
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

function run_result_row(config, attempt_id, status, started_at, finished_at; metrics=nothing, timings=nothing, error="")
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
        final_epochs=Int(config.final_epochs),
        depth=Int(config.depth),
        hidden_size=Int(config.hidden_size),
        activation=String(config.activation),
        schedule_kind=String(config.schedule_kind),
        starting_mu=Float64(config.starting_mu),
        ending_mu=Float64(config.ending_mu),
        piece_length=Int(config.piece_length),
        nr_pieces=Int(config.nr_pieces),
        starting_phase_length=Int(config.starting_phase_length),
        fine_tuning_phase_length=Int(config.fine_tuning_phase_length),
        average_test_loss=row_value(metrics, :average_test_loss, Inf),
        mean_test_relative_regret=row_value(metrics, :mean_test_relative_regret, Inf),
        mean_test_regret=row_value(metrics, :mean_test_regret, Inf),
        mean_test_policy_value=row_value(metrics, :mean_test_policy_value, NaN),
        mean_test_optimal_value=row_value(metrics, :mean_test_optimal_value, NaN),
        test_sample_count=row_value(metrics, :test_sample_count, 0),
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

function run_sweep_config(config, logger)
    started_at = unix_milliseconds()
    attempt_id = config.run_id * "_attempt" * string(started_at) * "_w" * string(Distributed.myid())
    put_log!(logger, :attempt, run_attempt_row(config, attempt_id, started_at))

    try
        objects = problem_objects()
        model = build_model(config, objects.problem)
        data_set_training = generate_training_dataset(config)
        loss = build_loss(objects, config)
        mu_in, mu_ref = mu_schedules_for_config(config)
        validation_dataset = Bool(config.early_stop_enabled) ?
                             quick_validation_dataset(config) : nothing

        training_seconds = @elapsed begin
            ContextualDFL.train!(
                model,
                loss,
                mu_in,
                mu_ref,
                data_set_training;
                opt=Flux.Adam(Float64(config.learning_rate)),
                epochs=Int(config.epochs),
                batchsize=Int(config.batch_size),
                shuffle=false,
                display_iterations=false,
                verbose=false,
                display_plot=false,
                save_model=false,
                reset_optimizer_each_epoch=Bool(config.reset_optimizer_each_epoch),
                nr_scenarios=Int(config.nr_scenarios),
                display_smooth=false,
                display_reference_input=display_reference_input,
                on_epoch_end=(epoch, loss_value, display_loss, metadata) -> begin
                    put_log!(
                        logger,
                        :epoch,
                            epoch_row(config, attempt_id, epoch, loss_value, display_loss, metadata),
                    )
                    if Bool(config.early_stop_enabled) &&
                       Int(epoch) == Int(config.early_stop_epoch)
                        quick_loss = quick_validation_loss(
                            model,
                            loss,
                            validation_dataset,
                            config,
                            metadata.mu_in,
                            metadata.mu_ref,
                        )
                        decision = early_stop_decision(
                            config,
                            loss_value,
                            display_loss,
                            quick_loss,
                        )
                        put_log!(
                            logger,
                            :early_check,
                            early_check_row(
                                config,
                                attempt_id,
                                epoch,
                                loss_value,
                                display_loss,
                                quick_loss,
                                decision,
                            ),
                        )
                        if decision.stopped
                            throw(EarlyStopException(decision.reason))
                        end
                    end
                end,
            )
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
            timings=(; training_seconds=training_seconds, evaluation_seconds=evaluation_seconds),
        )
        put_log!(logger, :run, result)
        return result
    catch error
        finished_at = unix_milliseconds()
        text = sprint(showerror, error, catch_backtrace())
        status = error isa EarlyStopException ? "early_stopped" : "failed"
        result = run_result_row(
            config,
            attempt_id,
            status,
            started_at,
            finished_at;
            error=text,
        )
        put_log!(logger, :run, result)
        return result
    end
end

function numeric_or_inf(value)
    if value === missing || value === nothing
        return Inf
    end
    parsed = tryparse(Float64, string(value))
    parsed === nothing && return Inf
    return parsed
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

function latest_ok_runs_by_id(run_rows)
    by_id = Dict{String,Any}()
    for row in run_rows
        status = string(row_value(row, :status, ""))
        status == "ok" || continue
        by_id[string(row_value(row, :run_id, ""))] = row
    end
    return by_id
end

function terminal_runs_by_id(run_rows)
    by_id = Dict{String,Any}()
    for row in run_rows
        status = string(row_value(row, :status, ""))
        status in ("ok", "early_stopped") || continue
        by_id[string(row_value(row, :run_id, ""))] = row
    end
    return by_id
end

function completed_run_ids(; smoke=false)
    rows = read_csv_rows(result_paths(smoke=smoke).runs)
    return Set(keys(terminal_runs_by_id(rows)))
end

function summarize_phase(configs; smoke=false)
    rows_by_id = latest_ok_runs_by_id(read_csv_rows(result_paths(smoke=smoke).runs))
    candidates = sort(unique(config.candidate_name for config in configs))
    summary_rows = NamedTuple[]
    for candidate in candidates
        candidate_configs = [config for config in configs if config.candidate_name == candidate]
        ok_rows = Any[]
        for config in candidate_configs
            haskey(rows_by_id, config.run_id) && push!(ok_rows, rows_by_id[config.run_id])
        end
        metrics = [numeric_or_inf(row_value(row, :average_test_loss, Inf)) for row in ok_rows]
        push!(
            summary_rows,
            (;
                phase=String(first(candidate_configs).phase),
                candidate_name=candidate,
                run_count=length(candidate_configs),
                ok_count=length(ok_rows),
                failed_count=length(candidate_configs) - length(ok_rows),
                mean_average_test_loss=mean_or_nan(metrics),
                std_average_test_loss=std_or_nan(metrics),
                min_average_test_loss=isempty(metrics) ? NaN : minimum(metrics),
                max_average_test_loss=isempty(metrics) ? NaN : maximum(metrics),
            ),
        )
    end
    return summary_rows
end

function choose_candidate(summary_rows)
    complete_rows = [
        row for row in summary_rows if Int(row.ok_count) == Int(row.run_count) && Int(row.run_count) > 0
    ]
    isempty(complete_rows) &&
        error("No candidate has all required replicates completed successfully.")
    return first(sort(complete_rows; by=row -> Float64(row.mean_average_test_loss)))
end

function choose_finalists(summary_rows; count=FINALIST_COUNT)
    usable_rows = [
        row for row in summary_rows if Int(row.ok_count) > 0 &&
                                    isfinite(Float64(row.mean_average_test_loss))
    ]
    isempty(usable_rows) &&
        error("No screening candidate has a completed run.")
    return first(
        sort(usable_rows; by=row -> Float64(row.mean_average_test_loss)),
        min(Int(count), length(usable_rows)),
    )
end

function finalist_rows(phase, finalists)
    return [
        (;
            phase=String(phase),
            rank=index,
            candidate_name=row.candidate_name,
            screening_run_count=row.run_count,
            screening_ok_count=row.ok_count,
            screening_failed_count=row.failed_count,
            screening_mean_average_test_loss=row.mean_average_test_loss,
            screening_std_average_test_loss=row.std_average_test_loss,
            selected_at=unix_milliseconds(),
        ) for (index, row) in enumerate(finalists)
    ]
end

function early_stop_thresholds_from_checks(configs; smoke=false)
    check_rows = read_csv_rows(result_paths(smoke=smoke).early_checks)
    isempty(check_rows) && return nothing

    run_ids = Set(config.run_id for config in configs)
    rows = [
        row for row in check_rows if string(row_value(row, :run_id, "")) in run_ids &&
                                  Int(row_value(row, :epoch, 0)) == EARLY_STOP_CHECK_EPOCH
    ]
    isempty(rows) && return nothing

    training_values = [
        numeric_or_inf(row_value(row, :training_loss, Inf)) for row in rows
    ]
    display_values = [
        numeric_or_inf(row_value(row, :display_loss, Inf)) for row in rows
    ]
    quick_values = [
        numeric_or_inf(row_value(row, :quick_validation_loss, Inf)) for row in rows
    ]

    finite_training = [value for value in training_values if isfinite(value)]
    finite_display = [value for value in display_values if isfinite(value)]
    finite_quick = [value for value in quick_values if isfinite(value)]
    isempty(finite_training) && isempty(finite_display) && isempty(finite_quick) &&
        return nothing

    factor = EARLY_STOP_DOMINANCE_FACTOR
    return (;
        epoch=EARLY_STOP_CHECK_EPOCH,
        training_loss_threshold=isempty(finite_training) ? Inf : minimum(finite_training) * factor,
        display_loss_threshold=isempty(finite_display) ? Inf : minimum(finite_display) * factor,
        quick_validation_loss_threshold=isempty(finite_quick) ? Inf : minimum(finite_quick) * factor,
    )
end

function config_for_candidate(configs, candidate_name)
    matches = [config for config in configs if config.candidate_name == String(candidate_name)]
    isempty(matches) && error("No config found for candidate $(candidate_name).")
    return first(sort(matches; by=config -> Int(config.replicate)))
end

function update_selection(selection, phase, decision, selected_config)
    if phase == :data_amount
        return merge(
            selection,
            (;
                training_contexts=Int(selected_config.training_contexts),
                epochs=Int(selected_config.epochs),
                final_epochs=Int(selected_config.final_epochs),
            ),
        )
    elseif phase == :depth
        return merge(selection, (; depth=Int(selected_config.depth)))
    elseif phase == :width
        return merge(selection, (; hidden_size=Int(selected_config.hidden_size)))
    elseif phase == :schedule_shape
        return merge(selection, (; schedule_kind=Symbol(selected_config.schedule_kind)))
    elseif phase == :piecewise_linear
        return merge(
            selection,
            (;
                schedule_kind=:piecewise_linear,
                starting_mu=Float64(selected_config.starting_mu),
                ending_mu=Float64(selected_config.ending_mu),
                piece_length=Int(selected_config.piece_length),
                nr_pieces=Int(selected_config.nr_pieces),
                starting_phase_length=Int(selected_config.starting_phase_length),
                fine_tuning_phase_length=Int(selected_config.fine_tuning_phase_length),
            ),
        )
    end
    return selection
end

function decision_row(phase, decision, selected_config)
    return (;
        phase=String(phase),
        selected_candidate=decision.candidate_name,
        selected_mean_average_test_loss=decision.mean_average_test_loss,
        selected_ok_count=decision.ok_count,
        selected_run_count=decision.run_count,
        training_contexts=Int(selected_config.training_contexts),
        epochs=Int(selected_config.epochs),
        final_epochs=Int(selected_config.final_epochs),
        depth=Int(selected_config.depth),
        hidden_size=Int(selected_config.hidden_size),
        activation=String(selected_config.activation),
        schedule_kind=String(selected_config.schedule_kind),
        starting_mu=Float64(selected_config.starting_mu),
        ending_mu=Float64(selected_config.ending_mu),
        piece_length=Int(selected_config.piece_length),
        nr_pieces=Int(selected_config.nr_pieces),
        starting_phase_length=Int(selected_config.starting_phase_length),
        fine_tuning_phase_length=Int(selected_config.fine_tuning_phase_length),
        decided_at=unix_milliseconds(),
    )
end
