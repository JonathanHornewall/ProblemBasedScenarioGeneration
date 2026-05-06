#!/usr/bin/env julia

const SCRIPT_PATH = abspath(@__FILE__)
const PROJECT_DIR = normpath(joinpath(@__DIR__, "..", ".."))
const DEFAULT_REMOTE_JULIA = get(ENV, "REMOTE_JULIA", "/home/rwl/.juliaup/bin/julia")
const CACHE_VERSION = "v3"
const RESOURCE_ALLOCATION_EXPERIMENT_ID = "resource_allocation/experiment_1_tiny"
const RESOURCE_ALLOCATION_TEST_ARTIFACT_DIR = normpath(
    joinpath(
        PROJECT_DIR,
        "..",
        "ContextualDFLTraining",
        "src",
        "experiments",
        "resource_allocation",
        "experiment_1_tiny",
        "artifacts",
        "test_data",
    ),
)

import Pkg
Pkg.activate(PROJECT_DIR)

using ContextualDFL
using ContextualDFLExperiments
using Dates
using Distributed
using LinearAlgebra
using Printf
using Random
using Serialization
using Sockets

const SMOKE_CONFIG = (;
    profile="smoke",
    train_contexts=5,
    train_scenarios_per_context=1,
    test_contexts=1,
    test_scenarios_per_context=1,
    evaluation_batches=1,
    seed=20260505,
    knn_k=1,
)

const PROPER_CONFIG = (;
    profile="proper",
    train_contexts=100,
    train_scenarios_per_context=1,
    test_contexts=30,
    test_scenarios_per_context=1000,
    evaluation_batches=20,
    seed=20260505,
    knn_k=10,
)

const BENCHMARK_NAMES = (
    "resource_allocation",
    "shipment_planning",
    "transshipment_q",
    "transshipment_h",
    "transshipment_h_and_q",
    "random_yield",
    "unreliable_newsvendor",
)

const POLICY_NAMES = ("saa", "knn", "least_squares", "er_saa")

const CSV_COLUMNS = (
    :timestamp,
    :profile,
    :benchmark,
    :variant,
    :policy,
    :status,
    :worker_id,
    :hostname,
    :train_contexts,
    :train_scenarios_per_context,
    :test_contexts,
    :test_scenarios_per_context,
    :evaluation_batches,
    :seed,
    :cache_key,
    :loaded_from_cache,
    :data_seconds,
    :fit_seconds,
    :eval_seconds,
    :sample_count,
    :policy_value_mean,
    :optimal_value_mean,
    :regret_mean,
    :relative_regret_mean,
    :gap_stderr_mean,
    :policy_eval_seconds,
    :error,
)

function main(args=ARGS)
    options = parse_options(args)
    mkpath(options.output_dir)

    configure_workers!(options)

    timestamp = Dates.format(now(), dateformat"yyyymmdd_HHMMSS")
    timestamped_path = joinpath(options.output_dir, "baseline_results_$(timestamp).csv")
    latest_path = joinpath(options.output_dir, "baseline_results_latest.csv")
    write_csv(timestamped_path, NamedTuple[])
    write_csv(latest_path, NamedTuple[])

    benchmarks = selected_names(BENCHMARK_NAMES, options.benchmark_names, "benchmark")
    policies = selected_names(POLICY_NAMES, options.policy_names, "policy")

    println("Running $(options.config.profile) baseline benchmark")
    println("Coordinator: $(Sockets.gethostname()) pid=$(getpid())")
    println("Workers: $(workers())")
    println("Benchmarks: $(join(benchmarks, ", "))")
    println("Policies: $(join(policies, ", "))")
    println("Cache dir: $(options.cache_dir)")
    println("Results: $(latest_path)")

    precompute_jobs = [
        (; benchmark=name, config=options.config, cache_dir=options.cache_dir,
           refresh_cache=options.refresh_cache)
        for name in benchmarks
    ]
    precompute_results = pmap_or_map(ensure_data_bundle_job, precompute_jobs)
    for result in precompute_results
        if result.status == "ok"
            println(
                "cache[$(result.benchmark)] $(result.loaded_from_cache ? "loaded" : "generated") " *
                "on $(result.hostname) in $(round(result.data_seconds; digits=2))s",
            )
        else
            println("cache[$(result.benchmark)] ERROR on $(result.hostname): $(result.error)")
        end
    end
    if any(result -> result.status != "ok", precompute_results)
        error("Aborting policy evaluation because one or more benchmark caches failed.")
    end

    jobs = [
        (; benchmark=benchmark, policy=policy, config=options.config,
           cache_dir=options.cache_dir, refresh_cache=false, timestamp=timestamp)
        for benchmark in benchmarks for policy in policies
    ]

    rows = NamedTuple[]
    for row in pmap_or_map(run_policy_job, jobs)
        push!(rows, row)
        write_csv(timestamped_path, rows)
        write_csv(latest_path, rows)
        print_row_summary(row)
    end

    println("Wrote $(length(rows)) policy rows")
    println("Latest results: $(latest_path)")
    any(row -> row.status != "ok", rows) && exit(1)
    return rows
end

function parse_options(args)
    profile = "smoke"
    refresh_cache = false
    overrides = Dict{Symbol,Int}()
    benchmark_names = nothing
    policy_names = nothing
    worker_hosts = String[]
    workers_per_host = 1
    remote_julia = DEFAULT_REMOTE_JULIA
    output_dir = joinpath(@__DIR__, "results")
    cache_dir = joinpath(@__DIR__, "cache")

    index = 1
    while index <= length(args)
        arg = args[index]
        if arg in ("--refresh-cache", "--fresh")
            refresh_cache = true
        elseif arg == "--profile"
            index += 1
            profile = args[index]
        elseif startswith(arg, "--profile=")
            profile = split(arg, "=", limit=2)[2]
        elseif arg == "--benchmarks"
            index += 1
            benchmark_names = split_names(args[index])
        elseif startswith(arg, "--benchmarks=")
            benchmark_names = split_names(split(arg, "=", limit=2)[2])
        elseif arg == "--policies"
            index += 1
            policy_names = split_names(args[index])
        elseif startswith(arg, "--policies=")
            policy_names = split_names(split(arg, "=", limit=2)[2])
        elseif arg == "--worker-hosts"
            index += 1
            worker_hosts = collect(split_names(args[index]))
        elseif startswith(arg, "--worker-hosts=")
            worker_hosts = collect(split_names(split(arg, "=", limit=2)[2]))
        elseif arg == "--workers-per-host"
            index += 1
            workers_per_host = parse(Int, args[index])
        elseif startswith(arg, "--workers-per-host=")
            workers_per_host = parse(Int, split(arg, "=", limit=2)[2])
        elseif arg == "--remote-julia"
            index += 1
            remote_julia = args[index]
        elseif startswith(arg, "--remote-julia=")
            remote_julia = split(arg, "=", limit=2)[2]
        elseif arg == "--output-dir"
            index += 1
            output_dir = abspath(args[index])
        elseif startswith(arg, "--output-dir=")
            output_dir = abspath(split(arg, "=", limit=2)[2])
        elseif arg == "--cache-dir"
            index += 1
            cache_dir = abspath(args[index])
        elseif startswith(arg, "--cache-dir=")
            cache_dir = abspath(split(arg, "=", limit=2)[2])
        elseif arg == "--train-contexts"
            index += 1
            overrides[:train_contexts] = parse(Int, args[index])
        elseif startswith(arg, "--train-contexts=")
            overrides[:train_contexts] = parse(Int, split(arg, "=", limit=2)[2])
        elseif arg == "--train-scenarios"
            index += 1
            overrides[:train_scenarios_per_context] = parse(Int, args[index])
        elseif startswith(arg, "--train-scenarios=")
            overrides[:train_scenarios_per_context] = parse(Int, split(arg, "=", limit=2)[2])
        elseif arg == "--test-contexts"
            index += 1
            overrides[:test_contexts] = parse(Int, args[index])
        elseif startswith(arg, "--test-contexts=")
            overrides[:test_contexts] = parse(Int, split(arg, "=", limit=2)[2])
        elseif arg == "--test-scenarios"
            index += 1
            overrides[:test_scenarios_per_context] = parse(Int, args[index])
        elseif startswith(arg, "--test-scenarios=")
            overrides[:test_scenarios_per_context] = parse(Int, split(arg, "=", limit=2)[2])
        elseif arg == "--evaluation-batches"
            index += 1
            overrides[:evaluation_batches] = parse(Int, args[index])
        elseif startswith(arg, "--evaluation-batches=")
            overrides[:evaluation_batches] = parse(Int, split(arg, "=", limit=2)[2])
        elseif arg == "--seed"
            index += 1
            overrides[:seed] = parse(Int, args[index])
        elseif startswith(arg, "--seed=")
            overrides[:seed] = parse(Int, split(arg, "=", limit=2)[2])
        elseif arg == "--knn-k"
            index += 1
            overrides[:knn_k] = parse(Int, args[index])
        elseif startswith(arg, "--knn-k=")
            overrides[:knn_k] = parse(Int, split(arg, "=", limit=2)[2])
        else
            throw(ArgumentError("Unknown argument: $(arg)"))
        end
        index += 1
    end

    base_config = profile == "smoke" ? SMOKE_CONFIG :
                  profile == "proper" ? PROPER_CONFIG :
                  throw(ArgumentError("Unknown profile $(repr(profile)); expected smoke or proper."))

    config = (;
        profile=profile,
        train_contexts=get(overrides, :train_contexts, base_config.train_contexts),
        train_scenarios_per_context=get(overrides, :train_scenarios_per_context, base_config.train_scenarios_per_context),
        test_contexts=get(overrides, :test_contexts, base_config.test_contexts),
        test_scenarios_per_context=get(overrides, :test_scenarios_per_context, base_config.test_scenarios_per_context),
        evaluation_batches=get(overrides, :evaluation_batches, base_config.evaluation_batches),
        seed=get(overrides, :seed, base_config.seed),
        knn_k=get(overrides, :knn_k, base_config.knn_k),
    )
    config.test_scenarios_per_context % config.evaluation_batches == 0 ||
        throw(ArgumentError("test_scenarios_per_context must be divisible by evaluation_batches."))
    workers_per_host > 0 || throw(ArgumentError("workers_per_host must be positive."))

    return (;
        config=config,
        refresh_cache=refresh_cache,
        benchmark_names=benchmark_names,
        policy_names=policy_names,
        worker_hosts=worker_hosts,
        workers_per_host=workers_per_host,
        remote_julia=remote_julia,
        output_dir=abspath(output_dir),
        cache_dir=abspath(cache_dir),
    )
end

function split_names(value)
    names = Set{String}()
    for raw_name in split(value, ",")
        name = strip(raw_name)
        isempty(name) || push!(names, name)
    end
    isempty(names) && throw(ArgumentError("Name filters must not be empty."))
    return names
end

function selected_names(all_names, requested_names, label)
    requested_names === nothing && return collect(all_names)
    selected = [name for name in all_names if name in requested_names]
    found = Set(selected)
    missing = setdiff(requested_names, found)
    isempty(missing) ||
        throw(ArgumentError("Unknown $(label)(s): $(join(sort(collect(missing)), ", "))."))
    return selected
end

function configure_workers!(options)
    isempty(options.worker_hosts) && return nothing

    machines = [(host, options.workers_per_host) for host in options.worker_hosts]
    println("Adding compute workers: $(machines)")
    addprocs(
        machines;
        exename=options.remote_julia,
        exeflags="--project=$(PROJECT_DIR)",
        dir=PROJECT_DIR,
        tunnel=true,
        max_parallel=length(machines),
    )

    script_path = SCRIPT_PATH
    for worker in workers()
        remotecall_wait(worker, script_path) do script
            include(script)
            return nothing
        end
    end
    return nothing
end

pmap_or_map(f, jobs) = nworkers() == 0 ? map(f, jobs) : pmap(f, jobs)

function benchmark_variant(name)
    if startswith(name, "transshipment_")
        return replace(name, "transshipment_" => "")
    end
    return name
end

function make_problem(name)
    name == "resource_allocation" && return ResourceAllocationProblem()
    name == "shipment_planning" && return ShipmentPlanningProblem()
    name == "transshipment_q" && return TransShipmentExperimentProblem(; variant=:q_only)
    name == "transshipment_h" && return TransShipmentExperimentProblem(; variant=:h_only)
    name == "transshipment_h_and_q" && return TransShipmentExperimentProblem(; variant=:h_and_q)
    name == "random_yield" && return RandomYieldProblem(; r=5, a=10, K_support=5)
    name == "unreliable_newsvendor" && return UnreliableNewsvendorProblem()
    throw(ArgumentError("Unknown benchmark $(repr(name))."))
end

function make_decoder(name, problem)
    name == "resource_allocation" && return ResourceAllocationDemandParametricDecoder(problem)
    name == "shipment_planning" && return compact_shipment_decoder(problem)
    startswith(name, "transshipment_") && return transshipment_decoder(problem)
    name == "random_yield" && return RandomYieldParametricDecoder(problem)
    name == "unreliable_newsvendor" && return UnreliableNewsvendorParametricDecoder(problem)
    throw(ArgumentError("Unknown benchmark $(repr(name))."))
end

function compact_shipment_decoder(problem::ShipmentPlanningProblem)
    base = base_scenario(problem)
    return ContextualDFL.ParametricDecoder(
        (:h_eq,);
        base_W_eq=base.W_eq,
        base_W_ineq=base.W_ineq,
        base_T_eq=base.T_eq,
        base_T_ineq=base.T_ineq,
        base_h_ineq=base.h_ineq,
        base_q=base.q,
    )
end

function generate_dataset_for_benchmark(name, problem; n_contexts, scenarios_per_context, seed)
    name == "shipment_planning" && return generate_compact_shipment_dataset(
        problem;
        n_contexts=n_contexts,
        scenarios_per_context=scenarios_per_context,
        seed=seed,
    )

    return generate_benchmark_dataset(
        problem;
        n_contexts=n_contexts,
        scenarios_per_context=scenarios_per_context,
        seed=seed,
    )
end

function generate_compact_shipment_dataset(
    problem::ShipmentPlanningProblem;
    n_contexts,
    scenarios_per_context,
    seed,
)
    rng = Random.MersenneTwister(seed)
    contexts = [abs.(randn(rng, problem.context_dim)) for _ in 1:n_contexts]
    scenario_collections = [
        [
            ContextualDFL.ParametricScenario(;
                h_eq_xi=compact_shipment_h_eq(problem, context, rng),
            )
            for _ in 1:scenarios_per_context
        ]
        for context in contexts
    ]
    return generate_contextual_data_set(contexts, scenario_collections)
end

function compact_shipment_h_eq(problem::ShipmentPlanningProblem, context, rng::Random.AbstractRNG)
    features = Float64.(context) .^ problem.p
    h_eq = zeros(Float64, problem.demand_count + problem.warehouse_count)
    for j in 1:problem.demand_count
        signal = problem.demand_intercepts[j] +
                 sum(problem.demand_slopes[j, term] * features[term] for term in 1:problem.context_dim)
        h_eq[j] = max(1e-6, signal + problem.sigma * randn(rng))
    end
    return h_eq
end

function benchmark_solver()
    return ContextualDFL.Solver(ContextualDFL.IpoptSolver(), ContextualDFL.HiGHSSolver())
end

function resource_artifact_cache_tag(name)
    name == "resource_allocation" || return "generated_test_optima"
    isdir(RESOURCE_ALLOCATION_TEST_ARTIFACT_DIR) || return "generated_test_optima"
    return "resource_artifacts_optional"
end

function cache_key(name, config)
    return join(
        (
            CACHE_VERSION,
            config.profile,
            name,
            benchmark_variant(name),
            "train$(config.train_contexts)x$(config.train_scenarios_per_context)",
            "test$(config.test_contexts)x$(config.test_scenarios_per_context)",
            "batches$(config.evaluation_batches)",
            "seed$(config.seed)",
            "solver_ipopt_highs",
            resource_artifact_cache_tag(name),
        ),
        "__",
    )
end

function cache_path(cache_dir, name, config)
    return joinpath(cache_dir, cache_key(name, config) * ".jls")
end

function resource_allocation_artifact_bundle(name, config, problem)
    name == "resource_allocation" || return nothing, ""
    if !isdir(RESOURCE_ALLOCATION_TEST_ARTIFACT_DIR)
        return nothing, "artifact directory not found: $(RESOURCE_ALLOCATION_TEST_ARTIFACT_DIR)"
    end

    pairs = resource_allocation_artifact_pairs()
    isempty(pairs) && return nothing, "no test_data_seed*.jls artifacts found"

    datasets = Any[]
    optimal_result_sets = Any[]
    seeds = Int[]
    test_paths = String[]
    optimal_paths = String[]

    try
        for (seed, test_path, optimal_path) in pairs
            test_payload = open(Serialization.deserialize, test_path)
            dataset = artifact_dataset(test_payload, test_path)
            optimal_payload = open(Serialization.deserialize, optimal_path)
            optimal_results = artifact_optimal_results(optimal_payload)

            validate_resource_artifact_test_payload!(
                test_payload,
                dataset,
                test_path,
                seed,
            )
            validate_resource_artifact_optimal_payload!(
                optimal_payload,
                optimal_results,
                optimal_path,
                seed,
                test_payload,
                dataset,
            )

            push!(datasets, dataset)
            push!(optimal_result_sets, optimal_results)
            push!(seeds, seed)
            push!(test_paths, test_path)
            push!(optimal_paths, optimal_path)
        end

        test_data = vcat(datasets...)
        optimal_results = vcat(optimal_result_sets...)
        length(test_data) == length(optimal_results) ||
            throw(DimensionMismatch(
                "artifact test data has $(length(test_data)) rows but optima have $(length(optimal_results)) rows.",
            ))

        source_contexts = length(test_data)
        if source_contexts < config.test_contexts
            return nothing, "artifacts cover $source_contexts contexts, requested $(config.test_contexts)"
        end

        if source_contexts > config.test_contexts
            test_data = test_data[1:config.test_contexts]
            optimal_results = optimal_results[1:config.test_contexts]
        end

        scenarios_per_context = scenario_count_per_context(test_data)
        if scenarios_per_context != config.test_scenarios_per_context
            return nothing, "artifacts have $scenarios_per_context scenarios/context, requested $(config.test_scenarios_per_context)"
        end

        batch_count = optimal_results_batch_count(optimal_results)
        if scenarios_per_context % batch_count != 0
            return nothing, "artifacts have $scenarios_per_context scenarios/context but $batch_count evaluation batch(es)"
        end

        validate_resource_artifact_dataset!(problem, test_data)

        return (;
            test_data=test_data,
            optimal_results=optimal_results,
            metadata=(;
                source=:resource_allocation_test_artifacts,
                experiment_id=RESOURCE_ALLOCATION_EXPERIMENT_ID,
                artifact_dir=RESOURCE_ALLOCATION_TEST_ARTIFACT_DIR,
                test_paths=test_paths,
                optimal_paths=optimal_paths,
                seeds=seeds,
                source_contexts=source_contexts,
                selected_contexts=length(test_data),
                scenarios_per_context=scenarios_per_context,
                evaluation_batches=batch_count,
                requested_evaluation_batches=config.evaluation_batches,
                scenarios_per_batch=scenarios_per_context ÷ batch_count,
            ),
        ), ""
    catch error
        return nothing, sprint(showerror, error, catch_backtrace())
    end
end

function resource_allocation_artifact_pairs()
    pattern = r"^test_data_seed([0-9]+)\.jls$"
    pairs = Tuple{Int,String,String}[]
    for name in readdir(RESOURCE_ALLOCATION_TEST_ARTIFACT_DIR)
        match_result = match(pattern, name)
        match_result === nothing && continue
        seed = parse(Int, only(match_result.captures))
        test_path = joinpath(RESOURCE_ALLOCATION_TEST_ARTIFACT_DIR, name)
        optimal_path = joinpath(
            RESOURCE_ALLOCATION_TEST_ARTIFACT_DIR,
            "optimal_solutions_seed$(seed).jls",
        )
        isfile(optimal_path) || throw(ArgumentError(
            "missing matching optimal_solutions_seed$(seed).jls for $(basename(test_path)).",
        ))
        push!(pairs, (seed, test_path, optimal_path))
    end
    return sort!(pairs; by=first)
end

function artifact_dataset(payload, path)
    payload isa NamedTuple && hasproperty(payload, :dataset) && return payload.dataset
    throw(ArgumentError("test artifact $(path) does not contain a dataset field."))
end

function artifact_optimal_results(payload)
    payload isa NamedTuple && hasproperty(payload, :optimal_results) &&
        return payload.optimal_results
    payload isa AbstractVector && return payload
    throw(ArgumentError("optimal artifact does not contain optimal_results."))
end

function payload_value(payload, field, default)
    payload isa NamedTuple || return default
    return hasproperty(payload, field) ? getproperty(payload, field) : default
end

function validate_resource_artifact_test_payload!(payload, dataset, path, seed)
    payload isa NamedTuple ||
        throw(ArgumentError("test artifact $(path) must contain a metadata payload."))
    String(payload_value(payload, :experiment_id, RESOURCE_ALLOCATION_EXPERIMENT_ID)) ==
        RESOURCE_ALLOCATION_EXPERIMENT_ID ||
        throw(ArgumentError("test artifact $(path) has experiment_id=$(payload.experiment_id)."))
    Symbol(payload_value(payload, :split_name, :test)) == :test ||
        throw(ArgumentError("test artifact $(path) is for split $(payload.split_name)."))
    Int(payload_value(payload, :test_data_seed, seed)) == seed ||
        throw(ArgumentError("test artifact $(path) has test_data_seed=$(payload.test_data_seed)."))
    length(dataset) == Int(payload_value(payload, :data_set_size, length(dataset))) ||
        throw(ArgumentError("test artifact $(path) data_set_size does not match dataset length."))
    dataset isa AbstractVector ||
        throw(ArgumentError("test artifact $(path) dataset is not a vector."))
    scenario_count = scenario_count_per_context(dataset)
    Int(payload_value(payload, :scenarios_per_context, scenario_count)) == scenario_count ||
        throw(ArgumentError("test artifact $(path) scenarios_per_context metadata is stale."))
    context_dimension = context_dimension_per_point(dataset)
    Int(payload_value(payload, :context_dimension, context_dimension)) == context_dimension ||
        throw(ArgumentError("test artifact $(path) context_dimension metadata is stale."))
    return nothing
end

function validate_resource_artifact_optimal_payload!(
    payload,
    optimal_results,
    path,
    seed,
    test_payload,
    dataset,
)
    payload isa NamedTuple ||
        throw(ArgumentError("optimal artifact $(path) must contain a metadata payload."))
    String(payload_value(payload, :experiment_id, RESOURCE_ALLOCATION_EXPERIMENT_ID)) ==
        RESOURCE_ALLOCATION_EXPERIMENT_ID ||
        throw(ArgumentError("optimal artifact $(path) has experiment_id=$(payload.experiment_id)."))
    Symbol(payload_value(payload, :split_name, :test)) == :test ||
        throw(ArgumentError("optimal artifact $(path) is for split $(payload.split_name)."))
    Int(payload_value(payload, :test_data_seed, seed)) == seed ||
        throw(ArgumentError("optimal artifact $(path) has test_data_seed=$(payload.test_data_seed)."))
    length(optimal_results) == length(dataset) ||
        throw(DimensionMismatch(
            "optimal artifact $(path) has $(length(optimal_results)) rows, expected $(length(dataset)).",
        ))
    if hasproperty(test_payload, :dataset_digest) && hasproperty(payload, :dataset_digest)
        String(test_payload.dataset_digest) == String(payload.dataset_digest) ||
            throw(ArgumentError("optimal artifact $(path) dataset_digest does not match test data."))
    end
    batch_count = optimal_results_batch_count(optimal_results)
    Int(payload_value(payload, :evaluation_batches, batch_count)) == batch_count ||
        throw(ArgumentError("optimal artifact $(path) evaluation_batches metadata is stale."))
    return nothing
end

function validate_resource_artifact_dataset!(problem::ResourceAllocationProblem, dataset)
    _, demand_count = size(problem.problem_data.service_rate_parameters)
    for (data_index, data_point) in enumerate(dataset)
        data_point isa ContextualDFL.ContextualDataPoint ||
            throw(ArgumentError("artifact row $data_index is not a ContextualDataPoint."))
        length(data_point.context) == 3 ||
            throw(DimensionMismatch("artifact row $data_index has context dimension $(length(data_point.context)), expected 3."))
        for (scenario_index, scenario) in enumerate(data_point.scenario_parameters)
            scenario isa ContextualDFL.ParametricScenario ||
                throw(ArgumentError("artifact row $data_index scenario $scenario_index is not a ParametricScenario."))
            length(scenario.h_eq_xi) == demand_count ||
                throw(DimensionMismatch(
                    "artifact row $data_index scenario $scenario_index has demand length $(length(scenario.h_eq_xi)), expected $demand_count.",
                ))
            all(isfinite, Float64.(scenario.h_eq_xi)) ||
                throw(DomainError(scenario.h_eq_xi, "artifact demand contains non-finite values."))
            resource_allocation_unused_fields_empty(scenario) || throw(ArgumentError(
                "artifact row $data_index scenario $scenario_index contains non-empty non-demand scenario fields.",
            ))
        end
    end
    return nothing
end

function resource_allocation_unused_fields_empty(scenario)
    return artifact_field_empty(scenario.W_eq_xi) &&
           artifact_field_empty(scenario.W_ineq_xi) &&
           artifact_field_empty(scenario.T_eq_xi) &&
           artifact_field_empty(scenario.T_ineq_xi) &&
           artifact_field_empty(scenario.h_ineq_xi) &&
           artifact_field_empty(scenario.q_xi)
end

artifact_field_empty(value::Number) = iszero(value)
artifact_field_empty(value) = isempty(value)

function scenario_count_per_context(dataset)
    isempty(dataset) && return 0
    scenario_count = length(first(dataset).scenario_parameters)
    for (index, data_point) in enumerate(dataset)
        length(data_point.scenario_parameters) == scenario_count ||
            throw(ArgumentError(
                "dataset row $index has $(length(data_point.scenario_parameters)) scenarios, expected $scenario_count.",
            ))
    end
    return scenario_count
end

function context_dimension_per_point(dataset)
    isempty(dataset) && return 0
    context_dimension = length(first(dataset).context)
    for (index, data_point) in enumerate(dataset)
        length(data_point.context) == context_dimension ||
            throw(ArgumentError(
                "dataset row $index has context dimension $(length(data_point.context)), expected $context_dimension.",
            ))
    end
    return context_dimension
end

function optimal_results_batch_count(optimal_results)
    isempty(optimal_results) && return 0
    batch_count = length(artifact_objective_values(first(optimal_results)))
    for (index, result) in enumerate(optimal_results)
        length(artifact_objective_values(result)) == batch_count ||
            throw(ArgumentError(
                "optimal result row $index has a different evaluation batch count.",
            ))
    end
    return batch_count
end

function artifact_objective_values(result)
    if hasproperty(result, :objective_values)
        values = Float64.(collect(result.objective_values))
    elseif hasproperty(result, :batch_objective_values)
        throw(ArgumentError(
            "optimal artifact uses old batch_objective_values protocol; regenerate it.",
        ))
    elseif hasproperty(result, :objective_value)
        values = [Float64(result.objective_value)]
    else
        throw(ArgumentError("optimal artifact result is missing objective_values."))
    end
    isempty(values) &&
        throw(ArgumentError("optimal artifact result has no objective values."))
    all(isfinite, values) ||
        throw(DomainError(values, "optimal artifact result has non-finite objective values."))
    if hasproperty(result, :objective_value)
        mean_value = sum(values) / length(values)
        isapprox(Float64(result.objective_value), mean_value; rtol=1e-10, atol=1e-10) ||
            throw(ArgumentError("optimal artifact objective_value is not mean(objective_values)."))
    end
    if hasproperty(result, :evaluation_batches)
        Int(result.evaluation_batches) == length(values) ||
            throw(ArgumentError("optimal artifact result evaluation_batches metadata is stale."))
    end
    return values
end

function ensure_data_bundle_job(job)
    worker = worker_metadata()
    try
        bundle, loaded, seconds = load_or_create_data_bundle(
            job.benchmark,
            job.config,
            job.cache_dir;
            refresh_cache=job.refresh_cache,
        )
        return merge(
            worker,
            (;
                benchmark=job.benchmark,
                status="ok",
                cache_key=bundle.cache_key,
                loaded_from_cache=loaded,
                data_seconds=seconds,
                error="",
            ),
        )
    catch error
        return merge(
            worker,
            (;
                benchmark=job.benchmark,
                status="error",
                cache_key=cache_key(job.benchmark, job.config),
                loaded_from_cache=false,
                data_seconds=0.0,
                error=sprint(showerror, error, catch_backtrace()),
            ),
        )
    end
end

function load_or_create_data_bundle(name, config, cache_dir; refresh_cache=false)
    mkpath(cache_dir)
    path = cache_path(cache_dir, name, config)
    if isfile(path) && !refresh_cache
        elapsed = @elapsed bundle = open(Serialization.deserialize, path)
        return bundle, true, elapsed
    end

    bundle = nothing
    elapsed = @elapsed begin
        problem = make_problem(name)
        decoder = make_decoder(name, problem)
        solver = benchmark_solver()
        program = stochastic_program(problem)

        train_data = generate_dataset_for_benchmark(
            name,
            problem;
            n_contexts=config.train_contexts,
            scenarios_per_context=config.train_scenarios_per_context,
            seed=config.seed,
        )

        artifact_bundle, artifact_reason =
            resource_allocation_artifact_bundle(name, config, problem)
        bundle_cache_key = cache_key(name, config)
        test_data = nothing
        optimal_results = nothing
        artifact_metadata = NamedTuple()
        if artifact_bundle === nothing
            if name == "resource_allocation" && !isempty(artifact_reason)
                println("resource_allocation artifact fallback: $(artifact_reason)")
            end
            test_data = generate_dataset_for_benchmark(
                name,
                problem;
                n_contexts=config.test_contexts,
                scenarios_per_context=config.test_scenarios_per_context,
                seed=config.seed + 10_000,
            )
            optimal_results = solve_dataset_to_optimality(
                test_data,
                program,
                decoder,
                solver;
                evaluation_batches=config.evaluation_batches,
                progress_io=stdout,
                progress_label=name,
            )
            artifact_metadata = (; source=:generated, reason=artifact_reason)
        else
            test_data = artifact_bundle.test_data
            optimal_results = artifact_bundle.optimal_results
            artifact_metadata = artifact_bundle.metadata
            bundle_cache_key = join(
                (
                    bundle_cache_key,
                    "artifact_contexts$(artifact_metadata.selected_contexts)",
                    "artifact_scenarios$(artifact_metadata.scenarios_per_context)",
                    "artifact_batches$(artifact_metadata.evaluation_batches)",
                ),
                "__",
            )
            println(
                "resource_allocation artifacts loaded: " *
                "$(length(test_data)) contexts, " *
                "$(artifact_metadata.scenarios_per_context) scenarios/context, " *
                "$(artifact_metadata.evaluation_batches) evaluation batch(es)",
            )
        end

        bundle = (;
            cache_version=CACHE_VERSION,
            cache_key=bundle_cache_key,
            benchmark=name,
            variant=benchmark_variant(name),
            config=config,
            train_data=train_data,
            test_data=test_data,
            optimal_results=optimal_results,
            artifact_metadata=artifact_metadata,
            test_contexts=length(test_data),
            test_scenarios_per_context=scenario_count_per_context(test_data),
            evaluation_batches=optimal_results_batch_count(optimal_results),
        )
        open(path, "w") do io
            Serialization.serialize(io, bundle)
        end
    end

    return bundle, false, elapsed
end

function run_policy_job(job)
    worker = worker_metadata()
    data_seconds = 0.0
    fit_seconds = 0.0
    eval_seconds = 0.0
    loaded_from_cache = false
    key = cache_key(job.benchmark, job.config)

    try
        data_bundle, loaded_from_cache, data_seconds = load_or_create_data_bundle(
            job.benchmark,
            job.config,
            job.cache_dir;
            refresh_cache=job.refresh_cache,
        )
        key = data_bundle.cache_key
        problem = make_problem(job.benchmark)
        decoder = make_decoder(job.benchmark, problem)
        solver = benchmark_solver()
        program = stochastic_program(problem)

        policy = nothing
        fit_seconds = @elapsed policy = make_policy(
            job.policy,
            job.benchmark,
            problem,
            data_bundle.train_data,
            solver,
            program,
            decoder,
            job.config,
        )

        comparison = nothing
        eval_seconds = @elapsed comparison = evaluate_policy_against_optimum(
            policy,
            data_bundle.test_data,
            program,
            decoder,
            solver;
            optimal_results=data_bundle.optimal_results,
        )
        metrics = comparison.metrics

        return result_row(
            job,
            worker,
            key,
            loaded_from_cache,
            data_seconds,
            fit_seconds,
            eval_seconds,
            "ok",
            "";
            sample_count=metrics.test_sample_count,
            policy_value_mean=metrics.test_policy_value_mean,
            optimal_value_mean=metrics.test_optimal_value_mean,
            regret_mean=metrics.test_regret_mean,
            relative_regret_mean=metrics.test_relative_regret_mean,
            gap_stderr_mean=metrics.test_gap_stderr_mean,
            policy_eval_seconds=metrics.test_policy_eval_seconds,
            test_contexts=length(data_bundle.test_data),
            test_scenarios_per_context=scenario_count_per_context(data_bundle.test_data),
            evaluation_batches=metrics.test_evaluation_batches,
        )
    catch error
        return result_row(
            job,
            worker,
            key,
            loaded_from_cache,
            data_seconds,
            fit_seconds,
            eval_seconds,
            "error",
            sprint(showerror, error, catch_backtrace()),
        )
    finally
        GC.gc()
    end
end

function make_policy(policy_name, benchmark, problem, train_data, solver, program, decoder, config)
    policy_name == "saa" && return SampleAverageApproximationPolicy(
        train_data,
        solver,
        program,
        decoder,
    )

    policy_name == "knn" && return KNearestNeighborsPolicy(
        train_data,
        solver,
        program,
        decoder;
        k=min(config.knn_k, length(train_data)),
    )

    target_component, postprocess_prediction = regression_settings(benchmark, problem)

    policy_name == "least_squares" && return LeastSquaresPolicy(
        train_data,
        solver,
        program,
        decoder;
        target_component=target_component,
        postprocess_prediction=postprocess_prediction,
    )

    policy_name == "er_saa" && return ResidualSampleAverageApproximationPolicy(
        train_data,
        solver,
        program,
        decoder;
        target_component=target_component,
        postprocess_prediction=postprocess_prediction,
    )

    throw(ArgumentError("Unknown policy $(repr(policy_name))."))
end

function regression_settings(benchmark, problem)
    benchmark == "resource_allocation" &&
        return :h_eq_xi, target -> max.(Float64.(target), 1e-6)

    if benchmark == "shipment_planning"
        return :h_eq_xi, target -> begin
            values = Float64.(target)
            values[1:problem.demand_count] = max.(values[1:problem.demand_count], 1e-6)
            values[(problem.demand_count + 1):end] .= 0.0
            values
        end
    end

    benchmark == "transshipment_q" &&
        return :q_xi, target -> max.(Float64.(target), 1e-4)
    benchmark == "transshipment_h" &&
        return :h_eq_xi, target -> max.(Float64.(target), 1e-4)

    if benchmark == "transshipment_h_and_q"
        mean_parameters = ContextualDFL.transshipment_mean_parameters(problem.core_problem)
        h_len = length(mean_parameters.rhs)
        q_len = length(mean_parameters.q)
        return (:h_eq_xi, :q_xi), target -> begin
            values = max.(Float64.(target), 1e-4)
            length(values) == h_len + q_len ||
                throw(DimensionMismatch("expected $(h_len + q_len) h/q values."))
            values
        end
    end

    if benchmark == "random_yield"
        base = base_scenario(problem)
        return :W_eq_xi, target -> begin
            values = reshape(Float64.(target), size(base.W_eq))
            values[:, 1:problem.activity_count] =
                max.(values[:, 1:problem.activity_count], 0.0)
            values[:, (problem.activity_count + 1):end] =
                base.W_eq[:, (problem.activity_count + 1):end]
            values
        end
    end

    if benchmark == "unreliable_newsvendor"
        return :h_eq_xi, target -> begin
            values = Float64.(target)
            length(values) == 2 ||
                throw(DimensionMismatch("expected two newsvendor parameters."))
            values[1] = clamp(values[1], 0.0, problem.demand_upper_bound)
            values[2] = clamp(values[2], 0.0, 1.0)
            values
        end
    end

    throw(ArgumentError("No regression settings for benchmark $(repr(benchmark))."))
end

function worker_metadata()
    return (;
        worker_id=Distributed.myid(),
        hostname=Sockets.gethostname(),
    )
end

function result_row(
    job,
    worker,
    cache_key_value,
    loaded_from_cache,
    data_seconds,
    fit_seconds,
    eval_seconds,
    status,
    error;
    sample_count="",
    policy_value_mean="",
    optimal_value_mean="",
    regret_mean="",
    relative_regret_mean="",
    gap_stderr_mean="",
    policy_eval_seconds="",
    test_contexts=job.config.test_contexts,
    test_scenarios_per_context=job.config.test_scenarios_per_context,
    evaluation_batches=job.config.evaluation_batches,
)
    return (;
        timestamp=job.timestamp,
        profile=job.config.profile,
        benchmark=job.benchmark,
        variant=benchmark_variant(job.benchmark),
        policy=job.policy,
        status=status,
        worker_id=worker.worker_id,
        hostname=worker.hostname,
        train_contexts=job.config.train_contexts,
        train_scenarios_per_context=job.config.train_scenarios_per_context,
        test_contexts=test_contexts,
        test_scenarios_per_context=test_scenarios_per_context,
        evaluation_batches=evaluation_batches,
        seed=job.config.seed,
        cache_key=cache_key_value,
        loaded_from_cache=loaded_from_cache,
        data_seconds=data_seconds,
        fit_seconds=fit_seconds,
        eval_seconds=eval_seconds,
        sample_count=sample_count,
        policy_value_mean=policy_value_mean,
        optimal_value_mean=optimal_value_mean,
        regret_mean=regret_mean,
        relative_regret_mean=relative_regret_mean,
        gap_stderr_mean=gap_stderr_mean,
        policy_eval_seconds=policy_eval_seconds,
        error=error,
    )
end

function write_csv(path, rows)
    open(path, "w") do io
        println(io, join(String.(CSV_COLUMNS), ","))
        for row in rows
            println(io, join((csv_cell(getproperty(row, column)) for column in CSV_COLUMNS), ","))
        end
    end
    return path
end

function csv_cell(value)
    text = string(value)
    if any(contains(text, needle) for needle in (",", "\"", "\n", "\r"))
        return "\"" * replace(text, "\"" => "\"\"") * "\""
    end
    return text
end

function print_row_summary(row)
    if row.status == "ok"
        @printf(
            "ok %-24s %-14s regret=%12.6g rel=%12.6g fit=%8.3fs eval=%8.3fs worker=%s\n",
            row.benchmark,
            row.policy,
            Float64(row.regret_mean),
            Float64(row.relative_regret_mean),
            Float64(row.fit_seconds),
            Float64(row.eval_seconds),
            row.hostname,
        )
    else
        println("error $(row.benchmark) $(row.policy) worker=$(row.hostname): $(row.error)")
    end
end

if abspath(PROGRAM_FILE) == SCRIPT_PATH && Distributed.myid() == 1
    main()
end
