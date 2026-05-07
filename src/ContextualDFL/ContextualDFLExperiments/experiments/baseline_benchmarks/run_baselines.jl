#!/usr/bin/env julia

const SCRIPT_PATH = abspath(@__FILE__)
const PROJECT_DIR = normpath(joinpath(@__DIR__, "..", ".."))
const DEFAULT_REMOTE_JULIA = get(ENV, "REMOTE_JULIA", "/home/rwl/.juliaup/bin/julia")
const CACHE_VERSION = "v3"
const DATA_ARTIFACT_VERSION = "baseline_data_bundle_v1"
const TINY_ARTIFACT_DIR = normpath(joinpath(
    @__DIR__,
    "artifacts",
    "tiny_30ctx_5x100_seed20260505",
))
const TINY_FULL_RESULTS_DIR = normpath(joinpath(
    @__DIR__,
    "results",
    "tiny_30ctx_5x100_full_baselines_20260507",
))
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

haskey(ENV, "PYTHONNOUSERSITE") || (ENV["PYTHONNOUSERSITE"] = "1")

import Pkg
Pkg.activate(PROJECT_DIR)

using ContextualDFL
using ContextualDFLExperiments
using Dates
using Distributed
using LinearAlgebra
using Optim
using Printf
using Random
using Serialization
using SHA
using Sockets

const Flux = ContextualDFL.Flux

const SMOKE_CONFIG = (;
    profile="smoke",
    train_contexts=5,
    train_scenarios_per_context=1,
    test_contexts=1,
    test_scenarios_per_context=1,
    evaluation_batches=1,
    seed=20260505,
    knn_k=1,
    ad_tree_depth=1,
    ad_tree_min_leaf=1,
    resource_allocation_test_source="auto",
)

const PROPER_CONFIG = (;
    profile="proper",
    train_contexts=100,
    train_scenarios_per_context=1,
    test_contexts=30,
    test_scenarios_per_context=1000,
    evaluation_batches=1,
    seed=20260505,
    knn_k=10,
    ad_tree_depth=2,
    ad_tree_min_leaf=2,
    resource_allocation_test_source="auto",
)

const TINY_CONFIG = (;
    profile="tiny",
    train_contexts=100,
    train_scenarios_per_context=1,
    test_contexts=30,
    test_scenarios_per_context=500,
    evaluation_batches=5,
    seed=20260505,
    knn_k=10,
    ad_tree_depth=2,
    ad_tree_min_leaf=2,
    resource_allocation_test_source="generated",
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

const DETERMINISTIC_POLICY_NAMES = ("saa", "knn", "least_squares", "er_saa")
const REPLICATED_POLICY_NAMES = ("cart", "nn", "ad", "ad_tree", "m5_ad")
const DFL_RHO_VALUES = (0.1, 0.01, 0.001)
const DFL_POLICY_NAMES = Tuple("dfl_mu0_rho$(rho)" for rho in DFL_RHO_VALUES)
const POLICY_NAMES = (
    DETERMINISTIC_POLICY_NAMES...,
    REPLICATED_POLICY_NAMES...,
    DFL_POLICY_NAMES...,
)
const DEFAULT_POLICY_NAMES = (
    DETERMINISTIC_POLICY_NAMES...,
    REPLICATED_POLICY_NAMES...,
)
const FULL_BASELINE_POLICY_NAMES = POLICY_NAMES
const NN_BASELINE_VERSION = "nnv1"
const DFL_BASELINE_VERSION = "dflrho_v1"
const DEFAULT_REPLICA_SEEDS = (20260505, 20260506, 20260507)

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
    :replica_index,
    :replica_seed,
    :source_artifact_path,
    :cache_key,
    :loaded_from_cache,
    :loaded_policy_from_cache,
    :data_seconds,
    :fit_seconds,
    :eval_seconds,
    :mu_train,
    :rho_train,
    :mu_eval,
    :rho_eval,
    :policy_history_path,
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
    default_policies = options.full_baseline_grid ?
        FULL_BASELINE_POLICY_NAMES :
        DEFAULT_POLICY_NAMES
    policies = options.policy_names === nothing ?
        collect(default_policies) :
        selected_names(POLICY_NAMES, options.policy_names, "policy")

    println("Running $(options.config.profile) baseline benchmark")
    println("Coordinator: $(Sockets.gethostname()) pid=$(getpid())")
    println("Workers: $(workers())")
    println("Benchmarks: $(join(benchmarks, ", "))")
    println("Policies: $(join(policies, ", "))")
    println(
        "Grid mode: " *
        (options.full_baseline_grid ? "full baseline replicas" : "single policy row"),
    )
    println("Replica seeds: $(join(options.replica_seeds, ", "))")
    println("Cache dir: $(options.cache_dir)")
    options.data_artifact_dir !== nothing &&
        println("Data artifact dir: $(options.data_artifact_dir)")
    options.export_data_artifact_dir !== nothing &&
        println("Export data artifacts: $(options.export_data_artifact_dir)")
    !isempty(options.context_source_artifact_dirs) &&
        println("Context source artifact dirs: $(join(options.context_source_artifact_dirs, ", "))")
    println("Results: $(latest_path)")

    if options.validate_data_artifacts
        validate_data_artifact_dir!(
            options.data_artifact_dir,
            benchmarks,
            options.config;
            context_source_artifact_dirs=options.context_source_artifact_dirs,
        )
        println("Validated data artifacts; exiting.")
        return NamedTuple[]
    end

    if options.export_data_artifact_dir !== nothing &&
       options.policy_names === nothing &&
       !options.full_baseline_grid
        export_data_bundles!(
            options.export_data_artifact_dir,
            benchmarks,
            options.config,
            options.cache_dir;
            context_source_artifact_dirs=options.context_source_artifact_dirs,
            write_cache=false,
            parallel_optima=options.local_workers > 0,
        )
        println("Exported data artifacts; no explicit policies were requested, exiting.")
        return NamedTuple[]
    end

    parallel_optima = options.local_workers > 0 && options.data_artifact_dir === nothing
    precompute_jobs = [
        (; benchmark=name, config=options.config, cache_dir=options.cache_dir,
           refresh_cache=options.refresh_cache,
           data_artifact_dir=options.data_artifact_dir,
           parallel_optima=parallel_optima)
        for name in benchmarks
    ]
    precompute_results = parallel_optima ?
        map(ensure_data_bundle_job, precompute_jobs) :
        pmap_or_map(ensure_data_bundle_job, precompute_jobs)
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

    if options.export_data_artifact_dir !== nothing
        export_data_bundles!(
            options.export_data_artifact_dir,
            benchmarks,
            options.config,
            options.cache_dir;
            context_source_artifact_dirs=options.context_source_artifact_dirs,
            write_cache=true,
            parallel_optima=false,
        )
        if options.policy_names === nothing && !options.full_baseline_grid
            println("Exported data artifacts; no explicit policies were requested, exiting.")
            return NamedTuple[]
        end
    end

    jobs = policy_jobs(benchmarks, policies, options, timestamp)

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
    local_workers = 0
    remote_julia = DEFAULT_REMOTE_JULIA
    output_dir = joinpath(@__DIR__, "results")
    output_dir_was_set = false
    cache_dir = joinpath(@__DIR__, "cache")
    export_data_artifact_dir = nothing
    data_artifact_dir = nothing
    data_artifact_dir_was_set = false
    export_data_artifact_dir_was_set = false
    tiny_artifact_dir = TINY_ARTIFACT_DIR
    context_source_artifact_dirs = String[]
    validate_data_artifacts = false
    resource_allocation_test_source = nothing
    full_baseline_grid = false
    tiny_full_baselines = false
    export_tiny_data_artifacts = false
    use_tiny_data_artifacts = false
    validate_tiny_data_artifacts = false
    replica_count = length(DEFAULT_REPLICA_SEEDS)
    replica_seeds = collect(DEFAULT_REPLICA_SEEDS)
    replica_seeds_were_set = false

    index = 1
    while index <= length(args)
        arg = args[index]
        if arg in ("--refresh-cache", "--fresh")
            refresh_cache = true
        elseif arg == "--full-baseline-grid"
            full_baseline_grid = true
        elseif arg == "--tiny-full-baselines"
            profile = "tiny"
            full_baseline_grid = true
            tiny_full_baselines = true
            use_tiny_data_artifacts = true
        elseif arg == "--export-tiny-data-artifacts"
            profile = "tiny"
            export_tiny_data_artifacts = true
        elseif arg == "--use-tiny-data-artifacts"
            profile = "tiny"
            use_tiny_data_artifacts = true
        elseif arg == "--validate-tiny-data-artifacts"
            profile = "tiny"
            validate_tiny_data_artifacts = true
            use_tiny_data_artifacts = true
            validate_data_artifacts = true
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
        elseif arg == "--local-workers"
            index += 1
            local_workers = parse(Int, args[index])
        elseif startswith(arg, "--local-workers=")
            local_workers = parse(Int, split(arg, "=", limit=2)[2])
        elseif arg == "--remote-julia"
            index += 1
            remote_julia = args[index]
        elseif startswith(arg, "--remote-julia=")
            remote_julia = split(arg, "=", limit=2)[2]
        elseif arg == "--output-dir"
            index += 1
            output_dir = abspath(args[index])
            output_dir_was_set = true
        elseif startswith(arg, "--output-dir=")
            output_dir = abspath(split(arg, "=", limit=2)[2])
            output_dir_was_set = true
        elseif arg == "--cache-dir"
            index += 1
            cache_dir = abspath(args[index])
        elseif startswith(arg, "--cache-dir=")
            cache_dir = abspath(split(arg, "=", limit=2)[2])
        elseif arg == "--export-data-artifacts"
            index += 1
            export_data_artifact_dir = abspath(args[index])
            export_data_artifact_dir_was_set = true
        elseif startswith(arg, "--export-data-artifacts=")
            export_data_artifact_dir = abspath(split(arg, "=", limit=2)[2])
            export_data_artifact_dir_was_set = true
        elseif arg == "--data-artifact-dir"
            index += 1
            data_artifact_dir = abspath(args[index])
            data_artifact_dir_was_set = true
        elseif startswith(arg, "--data-artifact-dir=")
            data_artifact_dir = abspath(split(arg, "=", limit=2)[2])
            data_artifact_dir_was_set = true
        elseif arg == "--tiny-artifact-dir"
            index += 1
            tiny_artifact_dir = abspath(args[index])
        elseif startswith(arg, "--tiny-artifact-dir=")
            tiny_artifact_dir = abspath(split(arg, "=", limit=2)[2])
        elseif arg in ("--context-source-artifact-dir", "--source-data-artifact-dir")
            index += 1
            append!(context_source_artifact_dirs, split_paths(args[index]))
        elseif startswith(arg, "--context-source-artifact-dir=")
            append!(
                context_source_artifact_dirs,
                split_paths(split(arg, "=", limit=2)[2]),
            )
        elseif startswith(arg, "--source-data-artifact-dir=")
            append!(
                context_source_artifact_dirs,
                split_paths(split(arg, "=", limit=2)[2]),
            )
        elseif arg == "--validate-data-artifacts"
            validate_data_artifacts = true
        elseif arg == "--resource-allocation-test-source"
            index += 1
            resource_allocation_test_source = args[index]
        elseif startswith(arg, "--resource-allocation-test-source=")
            resource_allocation_test_source = split(arg, "=", limit=2)[2]
        elseif arg == "--force-generated-resource-allocation"
            resource_allocation_test_source = "generated"
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
        elseif arg == "--ad-tree-depth"
            index += 1
            overrides[:ad_tree_depth] = parse(Int, args[index])
        elseif startswith(arg, "--ad-tree-depth=")
            overrides[:ad_tree_depth] = parse(Int, split(arg, "=", limit=2)[2])
        elseif arg == "--ad-tree-min-leaf"
            index += 1
            overrides[:ad_tree_min_leaf] = parse(Int, args[index])
        elseif startswith(arg, "--ad-tree-min-leaf=")
            overrides[:ad_tree_min_leaf] = parse(Int, split(arg, "=", limit=2)[2])
        elseif arg == "--replicas"
            index += 1
            replica_count = parse(Int, args[index])
        elseif startswith(arg, "--replicas=")
            replica_count = parse(Int, split(arg, "=", limit=2)[2])
        elseif arg == "--replica-seeds"
            index += 1
            replica_seeds = collect(parse_ints(args[index]))
            replica_seeds_were_set = true
        elseif startswith(arg, "--replica-seeds=")
            replica_seeds = collect(parse_ints(split(arg, "=", limit=2)[2]))
            replica_seeds_were_set = true
        else
            throw(ArgumentError("Unknown argument: $(arg)"))
        end
        index += 1
    end

    base_config = profile == "smoke" ? SMOKE_CONFIG :
                  profile == "proper" ? PROPER_CONFIG :
                  profile == "tiny" ? TINY_CONFIG :
                  throw(ArgumentError("Unknown profile $(repr(profile)); expected smoke, proper, or tiny."))
    active_resource_allocation_test_source =
        isnothing(resource_allocation_test_source) ?
        base_config.resource_allocation_test_source :
        resource_allocation_test_source

    config = (;
        profile=profile,
        train_contexts=get(overrides, :train_contexts, base_config.train_contexts),
        train_scenarios_per_context=get(overrides, :train_scenarios_per_context, base_config.train_scenarios_per_context),
        test_contexts=get(overrides, :test_contexts, base_config.test_contexts),
        test_scenarios_per_context=get(overrides, :test_scenarios_per_context, base_config.test_scenarios_per_context),
        evaluation_batches=get(overrides, :evaluation_batches, base_config.evaluation_batches),
        seed=get(overrides, :seed, base_config.seed),
        knn_k=get(overrides, :knn_k, base_config.knn_k),
        ad_tree_depth=get(overrides, :ad_tree_depth, base_config.ad_tree_depth),
        ad_tree_min_leaf=get(overrides, :ad_tree_min_leaf, base_config.ad_tree_min_leaf),
        resource_allocation_test_source=active_resource_allocation_test_source,
    )
    config.test_scenarios_per_context % config.evaluation_batches == 0 ||
        throw(ArgumentError("test_scenarios_per_context must be divisible by evaluation_batches."))
    config.resource_allocation_test_source in ("auto", "generated") ||
        throw(ArgumentError(
            "resource_allocation_test_source must be auto or generated, got " *
            repr(config.resource_allocation_test_source),
        ))
    workers_per_host > 0 || throw(ArgumentError("workers_per_host must be positive."))
    local_workers >= 0 || throw(ArgumentError("local_workers must be non-negative."))
    config.ad_tree_depth > 0 || throw(ArgumentError("ad_tree_depth must be positive."))
    config.ad_tree_min_leaf > 0 || throw(ArgumentError("ad_tree_min_leaf must be positive."))
    replica_count > 0 || throw(ArgumentError("replicas must be positive."))
    if !replica_seeds_were_set
        replica_seeds = [Int(config.seed) + offset for offset in 0:(replica_count - 1)]
    end
    isempty(replica_seeds) && throw(ArgumentError("replica-seeds must not be empty."))
    if export_tiny_data_artifacts && !export_data_artifact_dir_was_set
        export_data_artifact_dir = abspath(tiny_artifact_dir)
    end
    if use_tiny_data_artifacts && !data_artifact_dir_was_set
        data_artifact_dir = abspath(tiny_artifact_dir)
    end
    if validate_tiny_data_artifacts && !data_artifact_dir_was_set
        data_artifact_dir = abspath(tiny_artifact_dir)
    end
    if tiny_full_baselines && !output_dir_was_set
        output_dir = abspath(TINY_FULL_RESULTS_DIR)
    end
    if export_data_artifact_dir !== nothing && data_artifact_dir !== nothing
        throw(ArgumentError("--export-data-artifacts and --data-artifact-dir are mutually exclusive."))
    end
    if validate_data_artifacts && data_artifact_dir === nothing
        throw(ArgumentError("--validate-data-artifacts requires --data-artifact-dir."))
    end

    return (;
        config=config,
        refresh_cache=refresh_cache,
        benchmark_names=benchmark_names,
        policy_names=policy_names,
        worker_hosts=worker_hosts,
        workers_per_host=workers_per_host,
        local_workers=local_workers,
        remote_julia=remote_julia,
        output_dir=abspath(output_dir),
        cache_dir=abspath(cache_dir),
        export_data_artifact_dir=export_data_artifact_dir,
        data_artifact_dir=data_artifact_dir,
        context_source_artifact_dirs=unique(context_source_artifact_dirs),
        validate_data_artifacts=validate_data_artifacts,
        full_baseline_grid=full_baseline_grid,
        replica_seeds=Int.(replica_seeds),
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

function split_paths(value)
    paths = String[]
    for raw_path in split(value, ",")
        path = strip(raw_path)
        isempty(path) || push!(paths, abspath(path))
    end
    isempty(paths) && throw(ArgumentError("Path filters must not be empty."))
    return paths
end

function parse_ints(value)
    parsed = Int[]
    for raw_value in split(value, ",")
        text = strip(raw_value)
        isempty(text) || push!(parsed, parse(Int, text))
    end
    isempty(parsed) && throw(ArgumentError("Integer lists must not be empty."))
    return parsed
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
    if options.local_workers > 0
        println("Adding local workers: $(options.local_workers)")
        addprocs(
            options.local_workers;
            exeflags="--project=$(PROJECT_DIR)",
            dir=PROJECT_DIR,
        )
    end

    if !isempty(options.worker_hosts)
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
    end

    script_path = SCRIPT_PATH
    for worker in workers()
        worker == 1 && continue
        remotecall_wait(worker, script_path) do script
            include(script)
            return nothing
        end
    end
    return nothing
end

pmap_or_map(f, jobs) = nworkers() == 0 ? map(f, jobs) : pmap(f, jobs)

function policy_jobs(benchmarks, policies, options, timestamp)
    jobs = NamedTuple[]
    history_dir = joinpath(options.output_dir, "policy_histories")
    for benchmark in benchmarks
        for policy in policies
            replica_specs = replica_specs_for_policy(policy, options)
            for replica_spec in replica_specs
                push!(
                    jobs,
                    (;
                        benchmark=benchmark,
                        policy=policy,
                        config=options.config,
                        cache_dir=options.cache_dir,
                        refresh_cache=false,
                        refresh_policy_cache=options.refresh_cache,
                        timestamp=timestamp,
                        data_artifact_dir=options.data_artifact_dir,
                        output_dir=options.output_dir,
                        history_dir=history_dir,
                        replica_index=replica_spec.index,
                        replica_seed=replica_spec.seed,
                    ),
                )
            end
        end
    end
    return jobs
end

function replica_specs_for_policy(policy, options)
    if options.full_baseline_grid && is_replicated_policy(policy)
        return [
            (; index=replica_index, seed=seed)
            for (replica_index, seed) in enumerate(options.replica_seeds)
        ]
    end

    seed = first(options.replica_seeds)
    return [(; index=1, seed=seed)]
end

is_replicated_policy(policy_name) =
    policy_name in REPLICATED_POLICY_NAMES || is_dfl_policy(policy_name)

is_dfl_policy(policy_name) = policy_name in DFL_POLICY_NAMES

function dfl_rho_for_policy(policy_name)
    for rho in DFL_RHO_VALUES
        policy_name == "dfl_mu0_rho$(rho)" && return Float64(rho)
    end
    throw(ArgumentError("Policy $(repr(policy_name)) is not a rho-DFL policy."))
end

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

function resource_artifact_cache_tag(name, config)
    name == "resource_allocation" || return "generated_test_optima"
    config.resource_allocation_test_source == "generated" && return "generated_test_optima"
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
            resource_artifact_cache_tag(name, config),
        ),
        "__",
    )
end

function policy_result_cache_key(policy_name, data_cache_key, config, replica_seed=config.seed)
    replica_tag = "replica_seed$(Int(replica_seed))"
    policy_name == "ad" &&
        return join((data_cache_key, "policy_ad", ad_cache_tag(), replica_tag), "__")
    policy_name == "m5_ad" &&
        return join((data_cache_key, "policy_m5_ad", m5ad_cache_tag(; random_state=replica_seed), replica_tag), "__")
    policy_name == "nn" &&
        return join((data_cache_key, "policy_nn", neural_network_cache_tag(config; seed=replica_seed)), "__")
    is_dfl_policy(policy_name) &&
        return join((data_cache_key, "policy_dfl", dfl_cache_tag(policy_name, config, replica_seed)), "__")
    policy_name in REPLICATED_POLICY_NAMES &&
        return join((data_cache_key, "policy_$(policy_name)", replica_tag), "__")
    return data_cache_key
end

function cache_path(cache_dir, name, config)
    return joinpath(cache_dir, cache_key(name, config) * ".jls")
end

function resource_allocation_artifact_bundle(name, config, problem)
    name == "resource_allocation" || return nothing, ""
    config.resource_allocation_test_source == "generated" &&
        return nothing, "resource allocation test artifacts disabled by config"
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

function solve_dataset_to_optimality_for_runner(
    name,
    test_data,
    program,
    decoder,
    solver,
    config;
    parallel_optima=false,
)
    if parallel_optima && nworkers() > 1 && Distributed.myid() == 1
        return solve_dataset_to_optimality_parallel(name, test_data, config)
    end

    return solve_dataset_to_optimality(
        test_data,
        program,
        decoder,
        solver;
        evaluation_batches=config.evaluation_batches,
        progress_io=stdout,
        progress_label=name,
    )
end

function solve_dataset_to_optimality_parallel(name, test_data, config)
    batch_count = Int(config.evaluation_batches)
    jobs = NamedTuple[]
    for (data_point_index, data_point) in enumerate(test_data)
        scenario_count = length(data_point.scenario_parameters)
        scenario_count > 0 ||
            throw(ArgumentError("$(name) context $(data_point_index) has no scenarios."))
        scenario_count % batch_count == 0 ||
            throw(ArgumentError(
                "$(name) context $(data_point_index) has $scenario_count scenarios, " *
                "not divisible by evaluation_batches=$batch_count.",
            ))
        batch_size = scenario_count ÷ batch_count
        for batch_index in 1:batch_count
            scenario_range = ((batch_index - 1) * batch_size + 1):(batch_index * batch_size)
            push!(
                jobs,
                (;
                    benchmark=name,
                    data_point_index=data_point_index,
                    batch_index=batch_index,
                    scenarios=collect(data_point.scenario_parameters[scenario_range]),
                ),
            )
        end
    end

    println(
        "optimality[$(name)] parallel start contexts=$(length(test_data)) " *
        "batches=$(batch_count) jobs=$(length(jobs)) workers=$(workers())",
    )
    started = time()
    batch_results = pmap(optimality_batch_job, jobs)
    println(
        "optimality[$(name)] parallel finish jobs=$(length(jobs)) " *
        "seconds=$(round(time() - started; digits=2))",
    )

    objective_values = [zeros(Float64, batch_count) for _ in eachindex(test_data)]
    for result in batch_results
        objective_values[result.data_point_index][result.batch_index] = result.objective_value
    end

    return [
        (;
            evaluation_batches=batch_count,
            objective_values=values,
            objective_value=sum(values) / length(values),
        )
        for values in objective_values
    ]
end

function optimality_batch_job(job)
    problem = make_problem(job.benchmark)
    decoder = make_decoder(job.benchmark, problem)
    solver = benchmark_solver()
    program = stochastic_program(problem)

    objective_value = Ref{Float64}()
    batch_seconds = @elapsed begin
        W_eq, W_ineq, T_eq, T_ineq, h_eq, h_ineq, q =
            ContextualDFL.decode_scenario_collection(decoder, job.scenarios)

        _, _, _, solve_result = ContextualDFL._solve_stochastic_extensive(
            solver,
            program,
            W_eq,
            W_ineq,
            T_eq,
            T_ineq,
            h_eq,
            h_ineq,
            q;
            μ=0,
            ρ=0,
        )
        objective_value[] = ContextualDFLExperiments._checked_solve_result_objective(
            solve_result;
            data_point_index=job.data_point_index,
            batch_index=job.batch_index,
        )
    end

    return (;
        benchmark=job.benchmark,
        data_point_index=job.data_point_index,
        batch_index=job.batch_index,
        objective_value=objective_value[],
        batch_seconds=batch_seconds,
        worker_id=Distributed.myid(),
        hostname=Sockets.gethostname(),
    )
end

data_bundle_artifact_path(artifact_dir, name) = joinpath(artifact_dir, "$(name).jls")

function export_data_bundles!(
    artifact_dir,
    benchmarks,
    config,
    cache_dir;
    context_source_artifact_dirs=String[],
    write_cache=true,
    parallel_optima=false,
)
    mkpath(artifact_dir)
    for benchmark in benchmarks
        bundle = nothing
        seconds = @elapsed begin
            if write_cache && isempty(context_source_artifact_dirs)
                bundle, _, _ = load_or_create_data_bundle(
                    benchmark,
                    config,
                    cache_dir;
                    refresh_cache=false,
                    data_artifact_dir=nothing,
                    parallel_optima=parallel_optima,
                )
            else
                bundle = create_data_bundle(
                    benchmark,
                    config;
                    parallel_optima=parallel_optima,
                    context_source_artifact_dirs=context_source_artifact_dirs,
                )
            end
        end
        bundle = ensure_bundle_metadata(bundle)
        validate_data_bundle!(bundle, benchmark, config; source="cache before export")
        path = data_bundle_artifact_path(artifact_dir, benchmark)
        payload = (;
            artifact_version=DATA_ARTIFACT_VERSION,
            created_at=Dates.format(now(), dateformat"yyyy-mm-ddTHH:MM:SS"),
            created_by=Sockets.gethostname(),
            bundle=bundle,
        )
        open(path, "w") do io
            Serialization.serialize(io, payload)
        end
        println(
            "export[$(benchmark)] $(path) " *
            "seconds=$(round(seconds; digits=2)) " *
            "context_digest=$(bundle.context_digest)",
        )
        bundle = nothing
        GC.gc()
    end
    return artifact_dir
end

function load_data_bundle_artifact(artifact_dir, name, config)
    path = data_bundle_artifact_path(artifact_dir, name)
    isfile(path) || throw(ArgumentError("missing data artifact for $(name): $(path)"))
    payload = open(Serialization.deserialize, path)
    if payload isa NamedTuple && hasproperty(payload, :bundle)
        String(get(payload, :artifact_version, DATA_ARTIFACT_VERSION)) == DATA_ARTIFACT_VERSION ||
            throw(ArgumentError("unsupported data artifact version in $(path)."))
        bundle = payload.bundle
    else
        bundle = payload
    end
    bundle = ensure_bundle_metadata(bundle)
    validate_data_bundle!(bundle, name, config; source=path)
    return bundle
end

function ensure_bundle_metadata(bundle)
    scenario_count = scenario_count_per_context(bundle.test_data)
    batch_count = optimal_results_batch_count(bundle.optimal_results)
    derived = (;
        test_contexts=length(bundle.test_data),
        test_scenarios_per_context=scenario_count,
        evaluation_batches=batch_count,
        scenarios_per_batch=scenario_count ÷ batch_count,
        context_digest=dataset_context_digest(bundle.test_data),
    )
    return merge(bundle, derived)
end

function validate_data_artifact_dir!(
    artifact_dir,
    benchmarks,
    config;
    context_source_artifact_dirs=String[],
)
    for benchmark in benchmarks
        bundle = load_data_bundle_artifact(artifact_dir, benchmark, config)
        if !isempty(context_source_artifact_dirs)
            validate_context_preserving_bundle!(
                bundle,
                benchmark,
                context_source_artifact_dirs,
            )
        end
        println(
            "validated[$(benchmark)] " *
            "contexts=$(length(bundle.test_data)) " *
            "scenarios_per_context=$(scenario_count_per_context(bundle.test_data)) " *
            "evaluation_batches=$(optimal_results_batch_count(bundle.optimal_results)) " *
            "scenarios_per_batch=$(bundle.scenarios_per_batch) " *
            "context_digest=$(bundle.context_digest)",
        )
    end
    return nothing
end

function artifact_payload_bundle(payload, path)
    if payload isa NamedTuple && hasproperty(payload, :bundle)
        String(get(payload, :artifact_version, DATA_ARTIFACT_VERSION)) == DATA_ARTIFACT_VERSION ||
            throw(ArgumentError("unsupported data artifact version in $(path)."))
        return payload.bundle
    end
    payload isa NamedTuple && hasproperty(payload, :test_data) && return payload
    throw(ArgumentError("source data artifact $(path) does not contain a baseline data bundle."))
end

function source_context_artifact_path(source_dirs, name)
    for source_dir in source_dirs
        path = data_bundle_artifact_path(source_dir, name)
        isfile(path) && return path
    end
    throw(ArgumentError(
        "missing source context artifact for $(name) in " *
        join(source_dirs, ", "),
    ))
end

function load_source_context_bundle(source_dirs, name)
    path = source_context_artifact_path(source_dirs, name)
    payload = open(Serialization.deserialize, path)
    bundle = artifact_payload_bundle(payload, path)
    validate_source_context_bundle!(bundle, name, path)
    return bundle, path
end

function validate_source_context_bundle!(bundle, name, path)
    bundle isa NamedTuple ||
        throw(ArgumentError("source context artifact $(path) is not a NamedTuple."))
    hasproperty(bundle, :benchmark) && String(bundle.benchmark) == String(name) ||
        throw(ArgumentError("source context artifact $(path) benchmark mismatch."))
    hasproperty(bundle, :test_data) ||
        throw(ArgumentError("source context artifact $(path) is missing test_data."))
    hasproperty(bundle, :optimal_results) ||
        throw(ArgumentError("source context artifact $(path) is missing optimal_results."))

    length(bundle.test_data) == 30 ||
        throw(DimensionMismatch(
            "source context artifact $(path) has $(length(bundle.test_data)) contexts, expected 30.",
        ))
    scenario_count_per_context(bundle.test_data) == 1000 ||
        throw(DimensionMismatch(
            "source context artifact $(path) must have 1000 scenarios/context.",
        ))
    optimal_results_batch_count(bundle.optimal_results) == 1 ||
        throw(DimensionMismatch(
            "source context artifact $(path) must have one evaluation batch.",
        ))
    return bundle
end

function validate_context_preserving_bundle!(bundle, name, source_dirs)
    source_bundle, source_path = load_source_context_bundle(source_dirs, name)
    source_contexts = context_vectors_from_dataset(source_bundle.test_data)
    bundle_contexts = context_vectors_from_dataset(bundle.test_data)
    contexts_equal(source_contexts, bundle_contexts) ||
        throw(ArgumentError(
            "data artifact $(name) contexts do not exactly match source $(source_path).",
        ))
    source_digest = context_digest(source_contexts)
    String(bundle.context_digest) == source_digest ||
        throw(ArgumentError(
            "data artifact $(name) context_digest=$(bundle.context_digest), " *
            "expected source digest $(source_digest).",
        ))
    if hasproperty(bundle, :artifact_metadata) &&
       hasproperty(bundle.artifact_metadata, :source_context_digest)
        String(bundle.artifact_metadata.source_context_digest) == source_digest ||
            throw(ArgumentError("data artifact $(name) source context digest mismatch."))
    end
    return bundle
end

function context_preserving_test_data(name, problem, config, source_dirs)
    source_bundle, source_path = load_source_context_bundle(source_dirs, name)
    contexts = context_vectors_from_dataset(source_bundle.test_data)
    length(contexts) == Int(config.test_contexts) ||
        throw(DimensionMismatch(
            "source context artifact $(source_path) has $(length(contexts)) contexts, " *
            "requested $(config.test_contexts).",
        ))

    source_digest = context_digest(contexts)
    test_data = generate_context_preserving_dataset(
        name,
        problem,
        contexts;
        scenarios_per_context=config.test_scenarios_per_context,
        seed=config.seed + 10_000,
    )
    contexts_equal(contexts, context_vectors_from_dataset(test_data)) ||
        throw(ArgumentError("context-preserving generation changed contexts for $(name)."))

    scenario_count = scenario_count_per_context(test_data)
    scenario_count == Int(config.test_scenarios_per_context) ||
        throw(DimensionMismatch(
            "context-preserving generation produced $scenario_count scenarios/context, " *
            "expected $(config.test_scenarios_per_context).",
        ))

    println(
        "context_source[$(name)] $(source_path) " *
        "context_digest=$(source_digest) " *
        "new_scenarios_per_context=$(scenario_count)",
    )

    return test_data, (;
        source=:context_preserving_generated,
        source_artifact_path=source_path,
        source_cache_key=hasproperty(source_bundle, :cache_key) ? source_bundle.cache_key : "",
        source_test_contexts=length(source_bundle.test_data),
        source_test_scenarios_per_context=scenario_count_per_context(source_bundle.test_data),
        source_evaluation_batches=optimal_results_batch_count(source_bundle.optimal_results),
        source_context_digest=source_digest,
        seed=Int(config.seed),
        generation_seed=Int(config.seed) + 10_000,
        test_contexts=length(test_data),
        test_scenarios_per_context=scenario_count,
        evaluation_batches=Int(config.evaluation_batches),
        scenarios_per_batch=scenario_count ÷ Int(config.evaluation_batches),
        context_digest=source_digest,
    )
end

function generate_context_preserving_dataset(
    name,
    problem,
    contexts;
    scenarios_per_context,
    seed,
)
    scenario_count = ContextualDFLExperiments._checked_positive_integer(
        scenarios_per_context,
        :scenarios_per_context,
    )
    rng = Random.MersenneTwister(seed)

    if name == "resource_allocation"
        return generate_resource_allocation_context_preserving_dataset(
            problem,
            contexts;
            scenarios_per_context=scenario_count,
            rng=rng,
        )
    end

    if name == "shipment_planning"
        return generate_compact_shipment_context_preserving_dataset(
            problem,
            contexts;
            scenarios_per_context=scenario_count,
            rng=rng,
        )
    end

    generate_benchmark_contexts(problem; n_contexts=length(contexts), rng=rng)
    scenario_collections = [
        generate_benchmark_scenarios(
            problem,
            context;
            n_scenarios=scenario_count,
            rng=rng,
        )
        for context in contexts
    ]
    return generate_contextual_data_set(contexts, scenario_collections)
end

function generate_resource_allocation_context_preserving_dataset(
    problem::ResourceAllocationProblem,
    contexts;
    scenarios_per_context,
    rng,
)
    context_generator = ResourceAllocationContextDataGenerator(rng=rng)
    scenario_generator = ResourceAllocationScenarioDataGenerator(
        problem;
        sigma=5.0,
        p=2.0,
        L=3,
        rng=rng,
    )
    for _ in 1:length(contexts)
        context_generator()
    end
    scenario_collections = [
        [
            scenario_generator(ContextualDFLExperiments._checked_context_vector(context, 3))
            for _ in 1:scenarios_per_context
        ]
        for context in contexts
    ]
    return generate_contextual_data_set(contexts, scenario_collections)
end

function generate_compact_shipment_context_preserving_dataset(
    problem::ShipmentPlanningProblem,
    contexts;
    scenarios_per_context,
    rng,
)
    for _ in 1:length(contexts)
        abs.(randn(rng, problem.context_dim))
    end
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

function context_vectors_from_dataset(dataset)
    return [Float64.(collect(data_point.context)) for data_point in dataset]
end

function dataset_context_digest(dataset)
    return context_digest(context_vectors_from_dataset(dataset))
end

function context_digest(contexts)
    io = IOBuffer()
    println(io, length(contexts))
    for context in contexts
        println(io, length(context))
        for value in context
            @printf(io, "%.17g\n", Float64(value))
        end
    end
    return bytes2hex(SHA.sha1(take!(io)))
end

function contexts_equal(left, right)
    length(left) == length(right) || return false
    for index in eachindex(left)
        Float64.(left[index]) == Float64.(right[index]) || return false
    end
    return true
end

function validate_data_bundle!(bundle, name, config; source)
    bundle isa NamedTuple ||
        throw(ArgumentError("data bundle $(source) is not a NamedTuple."))
    hasproperty(bundle, :benchmark) && String(bundle.benchmark) == String(name) ||
        throw(ArgumentError("data bundle $(source) benchmark mismatch."))
    hasproperty(bundle, :config) ||
        throw(ArgumentError("data bundle $(source) is missing config."))

    fields = Symbol[
        :profile,
        :train_contexts,
        :train_scenarios_per_context,
        :test_contexts,
        :test_scenarios_per_context,
        :evaluation_batches,
        :seed,
    ]
    name == "resource_allocation" && push!(fields, :resource_allocation_test_source)

    for field in fields
        requested = getproperty(config, field)
        stored = getproperty(bundle.config, field)
        requested == stored ||
            throw(ArgumentError(
                "data bundle $(source) config mismatch for $(field): " *
                "stored=$(stored), requested=$(requested).",
            ))
    end

    length(bundle.train_data) == Int(config.train_contexts) ||
        throw(DimensionMismatch("data bundle $(source) train_data length mismatch."))
    length(bundle.test_data) == Int(config.test_contexts) ||
        throw(DimensionMismatch("data bundle $(source) test_data length mismatch."))
    scenario_count_per_context(bundle.test_data) == Int(config.test_scenarios_per_context) ||
        throw(DimensionMismatch("data bundle $(source) test scenario count mismatch."))
    optimal_results_batch_count(bundle.optimal_results) == Int(config.evaluation_batches) ||
        throw(DimensionMismatch("data bundle $(source) evaluation batch count mismatch."))
    length(bundle.optimal_results) == length(bundle.test_data) ||
        throw(DimensionMismatch("data bundle $(source) optimal result length mismatch."))
    scenario_count = scenario_count_per_context(bundle.test_data)
    batch_count = optimal_results_batch_count(bundle.optimal_results)
    scenario_count % batch_count == 0 ||
        throw(DimensionMismatch("data bundle $(source) scenarios are not divisible into batches."))
    computed_context_digest = dataset_context_digest(bundle.test_data)
    if hasproperty(bundle, :context_digest)
        String(bundle.context_digest) == computed_context_digest ||
            throw(ArgumentError("data bundle $(source) context_digest mismatch."))
    end
    if hasproperty(bundle, :artifact_metadata)
        metadata = bundle.artifact_metadata
        if metadata isa NamedTuple
            if hasproperty(metadata, :context_digest)
                String(metadata.context_digest) == computed_context_digest ||
                    throw(ArgumentError("data bundle $(source) artifact context_digest mismatch."))
            end
            if hasproperty(metadata, :test_contexts)
                Int(metadata.test_contexts) == length(bundle.test_data) ||
                    throw(ArgumentError("data bundle $(source) metadata test_contexts mismatch."))
            end
            if hasproperty(metadata, :test_scenarios_per_context)
                Int(metadata.test_scenarios_per_context) == scenario_count ||
                    throw(ArgumentError("data bundle $(source) metadata scenario count mismatch."))
            end
            if hasproperty(metadata, :evaluation_batches)
                Int(metadata.evaluation_batches) == batch_count ||
                    throw(ArgumentError("data bundle $(source) metadata evaluation_batches mismatch."))
            end
            if hasproperty(metadata, :scenarios_per_batch)
                Int(metadata.scenarios_per_batch) == scenario_count ÷ batch_count ||
                    throw(ArgumentError("data bundle $(source) metadata scenarios_per_batch mismatch."))
            end
            if hasproperty(metadata, :seed)
                Int(metadata.seed) == Int(config.seed) ||
                    throw(ArgumentError("data bundle $(source) metadata seed mismatch."))
            end
            if config.profile == "tiny" &&
               hasproperty(metadata, :source) &&
               Symbol(metadata.source) == :generated
                required_metadata = (
                    :test_contexts,
                    :test_scenarios_per_context,
                    :evaluation_batches,
                    :scenarios_per_batch,
                    :seed,
                    :context_digest,
                )
                for field in required_metadata
                    hasproperty(metadata, field) ||
                        throw(ArgumentError(
                            "generated data bundle $(source) metadata is missing $(field).",
                        ))
                end
            end
        end
    end
    return bundle
end

function ensure_data_bundle_job(job)
    worker = worker_metadata()
    try
        bundle, loaded, seconds = load_or_create_data_bundle(
            job.benchmark,
            job.config,
            job.cache_dir;
            refresh_cache=job.refresh_cache,
            data_artifact_dir=job.data_artifact_dir,
            parallel_optima=job.parallel_optima,
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

function load_or_create_data_bundle(
    name,
    config,
    cache_dir;
    refresh_cache=false,
    data_artifact_dir=nothing,
    parallel_optima=false,
)
    if data_artifact_dir !== nothing
        elapsed = @elapsed bundle = load_data_bundle_artifact(data_artifact_dir, name, config)
        return bundle, true, elapsed
    end

    mkpath(cache_dir)
    path = cache_path(cache_dir, name, config)
    if isfile(path) && !refresh_cache
        elapsed = @elapsed bundle = open(Serialization.deserialize, path)
        validate_data_bundle!(bundle, name, config; source=path)
        return bundle, true, elapsed
    end

    bundle = nothing
    elapsed = @elapsed begin
        bundle = create_data_bundle(name, config; parallel_optima=parallel_optima)
        open(path, "w") do io
            Serialization.serialize(io, bundle)
        end
        validate_data_bundle!(bundle, name, config; source=path)
    end

    return bundle, false, elapsed
end

function create_data_bundle(
    name,
    config;
    parallel_optima=false,
    context_source_artifact_dirs=String[],
)
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

    bundle_cache_key = cache_key(name, config)
    test_data = nothing
    optimal_results = nothing
    artifact_metadata = NamedTuple()

    if !isempty(context_source_artifact_dirs)
        test_data, artifact_metadata = context_preserving_test_data(
            name,
            problem,
            config,
            context_source_artifact_dirs,
        )
        optimal_results = solve_dataset_to_optimality_for_runner(
            name,
            test_data,
            program,
            decoder,
            solver,
            config;
            parallel_optima=parallel_optima,
        )
        bundle_cache_key = join(
            (
                bundle_cache_key,
                "context_digest$(artifact_metadata.context_digest)",
                "source_scenarios$(artifact_metadata.source_test_scenarios_per_context)",
                "source_batches$(artifact_metadata.source_evaluation_batches)",
            ),
            "__",
        )
    else
        artifact_bundle, artifact_reason =
            resource_allocation_artifact_bundle(name, config, problem)
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
            optimal_results = solve_dataset_to_optimality_for_runner(
                name,
                test_data,
                program,
                decoder,
                solver,
                config;
                parallel_optima=parallel_optima,
            )
            artifact_metadata = (;
                source=:generated,
                reason=artifact_reason,
                seed=Int(config.seed),
                generation_seed=Int(config.seed) + 10_000,
                test_contexts=length(test_data),
                test_scenarios_per_context=scenario_count_per_context(test_data),
                evaluation_batches=optimal_results_batch_count(optimal_results),
                scenarios_per_batch=scenario_count_per_context(test_data) ÷
                                    optimal_results_batch_count(optimal_results),
                context_digest=dataset_context_digest(test_data),
            )
        else
            test_data = artifact_bundle.test_data
            optimal_results = artifact_bundle.optimal_results
            artifact_metadata = merge(
                artifact_bundle.metadata,
                (;
                    test_contexts=length(test_data),
                    test_scenarios_per_context=scenario_count_per_context(test_data),
                    evaluation_batches=optimal_results_batch_count(optimal_results),
                    context_digest=dataset_context_digest(test_data),
                ),
            )
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
    end

    scenario_count = scenario_count_per_context(test_data)
    batch_count = optimal_results_batch_count(optimal_results)
    return (;
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
        test_scenarios_per_context=scenario_count,
        evaluation_batches=batch_count,
        scenarios_per_batch=scenario_count ÷ batch_count,
        context_digest=dataset_context_digest(test_data),
    )
end

function run_policy_job(job)
    worker = worker_metadata()
    data_seconds = 0.0
    fit_seconds = 0.0
    eval_seconds = 0.0
    loaded_from_cache = false
    loaded_policy_from_cache = false
    key = cache_key(job.benchmark, job.config)
    source_artifact_path = job_data_artifact_path(job)

    try
        data_bundle, loaded_from_cache, data_seconds = load_or_create_data_bundle(
            job.benchmark,
            job.config,
            job.cache_dir;
            refresh_cache=job.refresh_cache,
            data_artifact_dir=job.data_artifact_dir,
            parallel_optima=false,
        )
        source_artifact_path = source_artifact_path_for_row(data_bundle, job)
        key = policy_result_cache_key(
            job.policy,
            data_bundle.cache_key,
            job.config,
            job.replica_seed,
        )
        problem = make_problem(job.benchmark)
        decoder = make_decoder(job.benchmark, problem)
        solver = benchmark_solver()
        program = stochastic_program(problem)

        policy_fit = nothing
        fit_seconds = @elapsed policy_fit = make_policy(
            job.policy,
            job.benchmark,
            problem,
            data_bundle.train_data,
            solver,
            program,
            decoder,
            job.config;
            cache_dir=job.cache_dir,
            data_cache_key=data_bundle.cache_key,
            refresh_policy_cache=job.refresh_policy_cache,
            policy_seed=job.replica_seed,
            history_dir=job.history_dir,
            timestamp=job.timestamp,
            replica_index=job.replica_index,
        )
        policy = fitted_policy(policy_fit)
        policy_metadata = fitted_policy_metadata(policy_fit)
        loaded_policy_from_cache = metadata_value(policy_metadata, :loaded_policy_from_cache, false)
        mu_eval = metadata_value(policy_metadata, :mu_eval, 0.0)
        rho_eval = metadata_value(policy_metadata, :rho_eval, 0.0)

        comparison = nothing
        eval_seconds = @elapsed comparison = evaluate_policy_against_optimum(
            policy,
            data_bundle.test_data,
            program,
            decoder,
            solver;
            optimal_results=data_bundle.optimal_results,
            mu=mu_eval,
            rho=rho_eval,
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
            loaded_policy_from_cache=loaded_policy_from_cache,
            source_artifact_path=source_artifact_path,
            mu_train=metadata_value(policy_metadata, :mu_train, ""),
            rho_train=metadata_value(policy_metadata, :rho_train, ""),
            mu_eval=mu_eval,
            rho_eval=rho_eval,
            policy_history_path=metadata_value(policy_metadata, :policy_history_path, ""),
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
            loaded_policy_from_cache=loaded_policy_from_cache,
            source_artifact_path=source_artifact_path,
        )
    finally
        GC.gc()
    end
end

fitted_policy(policy) = policy

function fitted_policy(fit::NamedTuple)
    hasproperty(fit, :policy) && return fit.policy
    return fit
end

fitted_policy_metadata(policy) = NamedTuple()

function fitted_policy_metadata(fit::NamedTuple)
    hasproperty(fit, :metadata) && return fit.metadata
    return NamedTuple()
end

function metadata_value(metadata, field, default)
    metadata isa NamedTuple || return default
    return hasproperty(metadata, field) ? getproperty(metadata, field) : default
end

function job_data_artifact_path(job)
    job.data_artifact_dir === nothing && return ""
    return data_bundle_artifact_path(job.data_artifact_dir, job.benchmark)
end

function source_artifact_path_for_row(data_bundle, job)
    job.data_artifact_dir !== nothing &&
        return data_bundle_artifact_path(job.data_artifact_dir, job.benchmark)
    if hasproperty(data_bundle, :artifact_metadata)
        metadata = data_bundle.artifact_metadata
        metadata isa NamedTuple || return ""
        hasproperty(metadata, :source_artifact_path) && return String(metadata.source_artifact_path)
        hasproperty(metadata, :artifact_dir) && return String(metadata.artifact_dir)
    end
    return ""
end

function make_policy(
    policy_name,
    benchmark,
    problem,
    train_data,
    solver,
    program,
    decoder,
    config;
    cache_dir=nothing,
    data_cache_key=cache_key(benchmark, config),
    refresh_policy_cache=false,
    policy_seed=Int(config.seed),
    history_dir=nothing,
    timestamp="",
    replica_index=1,
)
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

    policy_name == "nn" && return neural_network_policy(
        benchmark,
        problem,
        train_data,
        solver,
        program,
        decoder,
        target_component,
        postprocess_prediction,
        config;
        cache_dir=cache_dir,
        data_cache_key=data_cache_key,
        refresh_policy_cache=refresh_policy_cache,
        policy_seed=policy_seed,
    )

    is_dfl_policy(policy_name) && return dfl_policy(
        policy_name,
        benchmark,
        problem,
        train_data,
        solver,
        program,
        decoder,
        config;
        cache_dir=cache_dir,
        data_cache_key=data_cache_key,
        refresh_policy_cache=refresh_policy_cache,
        policy_seed=policy_seed,
        history_dir=history_dir,
        timestamp=timestamp,
        replica_index=replica_index,
    )

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

    policy_name == "cart" && return CARTPolicy(
        train_data,
        solver,
        program,
        decoder;
        target_component=target_component,
        postprocess_prediction=postprocess_prediction,
        random_state=policy_seed,
    )

    if policy_name == "m5_ad"
        ad_postprocess_prediction, training_prediction_transform =
            decision_focused_settings(benchmark, problem, postprocess_prediction)
        settings = m5ad_baseline_settings(; random_state=policy_seed)
        return M5ADPolicy(
            train_data,
            solver,
            program,
            decoder;
            target_component=target_component,
            postprocess_prediction=ad_postprocess_prediction,
            training_prediction_transform=training_prediction_transform,
            optimize=settings.iterations > 0,
            optimizer_options=Optim.Options(
                f_reltol=settings.f_reltol,
                f_abstol=settings.f_tol,
                iterations=settings.iterations,
            ),
            min_samples_leaf=settings.min_samples_leaf,
            test_size=settings.test_size,
            random_state=settings.random_state,
        )
    end

    if policy_name == "ad_tree"
        Random.seed!(policy_seed)
        settings = ad_tree_baseline_settings()
        return AdaptiveDecisionTreePolicy(
            train_data,
            solver,
            program,
            decoder;
            target_component=target_component,
            postprocess_prediction=postprocess_prediction,
            depth=config.ad_tree_depth,
            min_leaf=config.ad_tree_min_leaf,
            mip_optimizer_attributes=(;
                threads=settings.threads,
                time_limit=settings.time_limit,
            ),
        )
    end

    if policy_name == "ad"
        Random.seed!(policy_seed)
        ad_postprocess_prediction, training_prediction_transform =
            decision_focused_settings(benchmark, problem, postprocess_prediction)
        settings = ad_baseline_settings()
        return DecisionFocusedLinearPolicy(
            train_data,
            solver,
            program,
            decoder;
            target_component=target_component,
            postprocess_prediction=ad_postprocess_prediction,
            training_prediction_transform=training_prediction_transform,
            optimize=settings.iterations > 0,
            optimizer_options=Optim.Options(
                f_reltol=settings.f_reltol,
                f_abstol=settings.f_tol,
                iterations=settings.iterations,
            ),
        )
    end

    throw(ArgumentError("Unknown policy $(repr(policy_name))."))
end

function ad_baseline_settings()
    return (;
        iterations=env_int("CDFL_BASELINE_AD_ITERATIONS", 1000),
        f_reltol=env_float("CDFL_BASELINE_AD_F_RELTOL", 1e-4),
        f_tol=env_float("CDFL_BASELINE_AD_F_TOL", 1e-4),
    )
end

function ad_tree_baseline_settings()
    return (;
        threads=env_int("CDFL_BASELINE_AD_TREE_THREADS", 1),
        time_limit=env_float("CDFL_BASELINE_AD_TREE_TIME_LIMIT", 300.0),
    )
end

function ad_cache_tag()
    settings = ad_baseline_settings()
    return join(
        (
            "iters$(settings.iterations)",
            "frtol$(settings.f_reltol)",
            "ftol$(settings.f_tol)",
        ),
        "_",
    )
end

function m5ad_baseline_settings(; random_state=nothing)
    default_random_state = env_int("CDFL_BASELINE_M5AD_RANDOM_STATE", 42)
    return (;
        min_samples_leaf=env_int("CDFL_BASELINE_M5AD_MIN_SAMPLES_LEAF", 25),
        test_size=env_float("CDFL_BASELINE_M5AD_TEST_SIZE", 0.2),
        random_state=isnothing(random_state) ? default_random_state : Int(random_state),
        iterations=env_int("CDFL_BASELINE_M5AD_ITERATIONS", 1000),
        f_reltol=env_float("CDFL_BASELINE_M5AD_F_RELTOL", 1e-4),
        f_tol=env_float("CDFL_BASELINE_M5AD_F_TOL", 1e-4),
    )
end

function m5ad_cache_tag(; random_state=nothing)
    settings = m5ad_baseline_settings(; random_state=random_state)
    return join(
        (
            "minleaf$(settings.min_samples_leaf)",
            "test$(settings.test_size)",
            "random$(settings.random_state)",
            "iters$(settings.iterations)",
            "frtol$(settings.f_reltol)",
            "ftol$(settings.f_tol)",
        ),
        "_",
    )
end

function neural_network_policy(
    benchmark,
    problem,
    train_data,
    solver,
    program,
    decoder,
    target_component,
    postprocess_prediction,
    config;
    cache_dir=nothing,
    data_cache_key,
    refresh_policy_cache=false,
    policy_seed=Int(config.seed),
)
    settings = neural_network_baseline_settings(config; seed=policy_seed)
    if benchmark != "resource_allocation"
        return generic_neural_network_policy(
            benchmark,
            train_data,
            solver,
            program,
            decoder,
            target_component,
            postprocess_prediction,
            settings;
            cache_dir=cache_dir,
            data_cache_key=data_cache_key,
            refresh_policy_cache=refresh_policy_cache,
        )
    end

    neural_net = load_or_train_resource_allocation_nn(
        problem,
        train_data,
        solver,
        program,
        settings;
        cache_dir=cache_dir,
        data_cache_key=data_cache_key,
        refresh_cache=refresh_policy_cache,
    )
    scenario_generator = ContextualDFL.ScenarioGenerator(
        neural_net=neural_net,
        scenario_decoder=ResourceAllocationDemandVectorDecoder(problem),
    )
    return ScenarioGenerationPolicy(
        scenario_generator,
        solver,
        program;
        mu=settings.mu,
        rho=settings.rho,
        nr_scenarios=settings.scenarios,
    )
end

struct GenericNeuralNetworkPolicy{
    TModel,
    TTemplate,
    TTargetComponent,
    TSolver,
    TProgram,
    TDecoder,
    TPostprocess,
    TKwargs,
} <: ContextualDFLExperiments.Policy
    neural_net::TModel
    scenario_template::TTemplate
    target_component::TTargetComponent
    target_length::Int
    solver::TSolver
    program::TProgram
    parametric_decoder::TDecoder
    postprocess_prediction::TPostprocess
    solve_kwargs::TKwargs
end

function ContextualDFLExperiments.infer(policy::GenericNeuralNetworkPolicy, context)
    raw_prediction = vec(policy.neural_net(Float64.(collect(context))))
    target_vector = ContextualDFLExperiments._processed_prediction(
        policy.postprocess_prediction,
        raw_prediction,
        policy.target_length,
    )
    scenario = ContextualDFLExperiments._scenario_from_target_vector(
        policy.scenario_template,
        policy.target_component,
        target_vector,
    )

    return ContextualDFLExperiments._solve_scenario_collection(
        policy.solver,
        policy.program,
        policy.parametric_decoder,
        [scenario];
        policy.solve_kwargs...,
    )
end

function generic_neural_network_policy(
    benchmark,
    train_data,
    solver,
    program,
    decoder,
    target_component,
    postprocess_prediction,
    settings;
    cache_dir,
    data_cache_key,
    refresh_policy_cache=false,
)
    regression = ContextualDFLExperiments._fit_scenario_target_regression(
        train_data,
        target_component;
        validate_fixed_components=true,
    )
    neural_net = load_or_train_generic_nn(
        benchmark,
        regression.contexts,
        regression.targets,
        settings;
        cache_dir=cache_dir,
        data_cache_key=data_cache_key,
        refresh_cache=refresh_policy_cache,
    )
    return GenericNeuralNetworkPolicy(
        neural_net,
        first(regression.scenario_templates),
        regression.target_component,
        regression.target_length,
        solver,
        program,
        decoder,
        postprocess_prediction,
        NamedTuple(),
    )
end

function neural_network_baseline_settings(config; seed=Int(config.seed))
    return (;
        version=NN_BASELINE_VERSION,
        seed=Int(seed),
        epochs=env_int("CDFL_BASELINE_NN_EPOCHS", config_value(config, :nn_epochs, 20)),
        scenarios=env_int("CDFL_BASELINE_NN_SCENARIOS", config_value(config, :nn_scenarios, 3)),
        batchsize=env_int("CDFL_BASELINE_NN_BATCHSIZE", 1),
        hidden_dim=env_int("CDFL_BASELINE_NN_HIDDEN_DIM", 128),
        depth=env_int("CDFL_BASELINE_NN_DEPTH", 3),
        learning_rate=env_float("CDFL_BASELINE_NN_LEARNING_RATE", 1e-3),
        mu=env_float("CDFL_BASELINE_NN_MU", 0.01),
        rho=env_float("CDFL_BASELINE_NN_RHO", 0.0),
    )
end

function config_value(config, name::Symbol, default)
    return hasproperty(config, name) ? getproperty(config, name) : default
end

function env_int(name, default)
    value = get(ENV, name, "")
    isempty(value) && return Int(default)
    return parse(Int, value)
end

function env_float(name, default)
    value = get(ENV, name, "")
    isempty(value) && return Float64(default)
    return parse(Float64, value)
end

function neural_network_cache_tag(config; seed=Int(config.seed))
    settings = neural_network_baseline_settings(config; seed=seed)
    return join(
        (
            settings.version,
            "seed$(settings.seed)",
            "epochs$(settings.epochs)",
            "scenarios$(settings.scenarios)",
            "batch$(settings.batchsize)",
            "hidden$(settings.hidden_dim)",
            "depth$(settings.depth)",
            "lr$(settings.learning_rate)",
            "mu$(settings.mu)",
            "rho$(settings.rho)",
        ),
        "_",
    )
end

function load_or_train_resource_allocation_nn(
    problem::ResourceAllocationProblem,
    train_data,
    solver,
    program,
    settings;
    cache_dir,
    data_cache_key,
    refresh_cache=false,
)
    isnothing(cache_dir) &&
        throw(ArgumentError("cache_dir must be provided for the NN baseline."))
    path = neural_network_model_cache_path(
        cache_dir,
        data_cache_key,
        settings;
        benchmark="resource_allocation",
    )
    if isfile(path) && !refresh_cache
        return open(Serialization.deserialize, path)
    end

    Random.seed!(settings.seed)
    model = build_resource_allocation_nn(
        problem;
        nr_scenarios=settings.scenarios,
        hidden_dim=settings.hidden_dim,
        depth=settings.depth,
    )
    train_resource_allocation_nn!(
        model,
        problem,
        train_data,
        solver,
        program,
        settings,
    )
    mkpath(dirname(path))
    open(path, "w") do io
        Serialization.serialize(io, model)
    end
    return model
end

function load_or_train_generic_nn(
    benchmark,
    contexts::AbstractMatrix,
    targets::AbstractMatrix,
    settings;
    cache_dir,
    data_cache_key,
    refresh_cache=false,
)
    isnothing(cache_dir) &&
        throw(ArgumentError("cache_dir must be provided for the NN baseline."))
    path = neural_network_model_cache_path(
        cache_dir,
        data_cache_key,
        settings;
        benchmark=benchmark,
    )
    if isfile(path) && !refresh_cache
        return open(Serialization.deserialize, path)
    end

    Random.seed!(settings.seed)
    model = build_generic_nn(
        size(contexts, 2),
        size(targets, 2);
        hidden_dim=settings.hidden_dim,
        depth=settings.depth,
    )
    train_generic_nn!(model, contexts, targets, settings)
    mkpath(dirname(path))
    open(path, "w") do io
        Serialization.serialize(io, model)
    end
    return model
end

function neural_network_model_cache_path(cache_dir, data_cache_key, settings; benchmark)
    key_material = join(
        (
            data_cache_key,
            settings.version,
            "seed$(settings.seed)",
            "epochs$(settings.epochs)",
            "scenarios$(settings.scenarios)",
            "batch$(settings.batchsize)",
            "hidden$(settings.hidden_dim)",
            "depth$(settings.depth)",
            "lr$(settings.learning_rate)",
            "mu$(settings.mu)",
            "rho$(settings.rho)",
        ),
        "__",
    )
    digest = bytes2hex(SHA.sha1(Vector{UInt8}(codeunits(key_material))))
    safe_benchmark = replace(String(benchmark), r"[^A-Za-z0-9_]+" => "_")
    return joinpath(cache_dir, "models", "$(safe_benchmark)_nn_$(digest).jls")
end

function build_generic_nn(input_dim, output_dim; hidden_dim=128, depth=3)
    input_dim > 0 || throw(ArgumentError("input_dim must be positive."))
    output_dim > 0 || throw(ArgumentError("output_dim must be positive."))
    depth > 0 || throw(ArgumentError("depth must be positive."))

    layers = Any[Flux.Dense(input_dim, hidden_dim, Flux.relu)]
    for _ in 2:depth
        push!(layers, Flux.Dense(hidden_dim, hidden_dim, Flux.relu))
    end
    push!(layers, Flux.Dense(hidden_dim, output_dim))
    return Flux.f64(Flux.Chain(layers...))
end

function train_generic_nn!(model, contexts::AbstractMatrix, targets::AbstractMatrix, settings)
    settings.epochs == 0 && return model
    size(contexts, 1) == size(targets, 1) ||
        throw(DimensionMismatch("NN contexts and targets must have the same row count."))
    sample_count = size(contexts, 1)
    target_dim = size(targets, 2)
    batchsize = max(1, Int(settings.batchsize))
    rng = Random.MersenneTwister(settings.seed)
    optimizer = Flux.Adam(settings.learning_rate)
    state = Flux.setup(optimizer, model)
    indices = collect(1:sample_count)

    for _ in 1:settings.epochs
        Random.shuffle!(rng, indices)
        for batch_indices in Iterators.partition(indices, batchsize)
            batch = collect(batch_indices)
            loss_value, gradients = Flux.withgradient(model) do trainable_model
                total = 0.0
                for index in batch
                    prediction = vec(trainable_model(collect(view(contexts, index, :))))
                    target = vec(view(targets, index, :))
                    total += sum(abs2, prediction .- target) / target_dim
                end
                total / length(batch)
            end
            isfinite(Float64(loss_value)) ||
                throw(DomainError(loss_value, "generic NN training loss is not finite."))
            Flux.update!(state, model, gradients[1])
        end
    end
    return model
end

function build_resource_allocation_nn(
    problem::ResourceAllocationProblem;
    nr_scenarios=3,
    input_dim=3,
    hidden_dim=128,
    depth=3,
)
    nr_scenarios > 0 || throw(ArgumentError("nr_scenarios must be positive."))
    depth > 0 || throw(ArgumentError("depth must be positive."))
    demand_count = size(problem.problem_data.service_rate_parameters, 2)
    output_dim = demand_count * nr_scenarios

    layers = Any[Flux.Dense(input_dim, hidden_dim, Flux.relu)]
    for _ in 2:depth
        push!(layers, Flux.Dense(hidden_dim, hidden_dim, Flux.relu))
    end
    push!(layers, Flux.Dense(hidden_dim, output_dim, Flux.relu))
    return Flux.f64(Flux.Chain(layers...))
end

function train_resource_allocation_nn!(
    model,
    problem::ResourceAllocationProblem,
    train_data,
    solver,
    program,
    settings,
)
    settings.epochs == 0 && return model

    loss = ContextualDFL.DflScenLoss(
        ResourceAllocationDemandVectorDecoder(problem),
        ResourceAllocationDemandParametricDecoder(problem),
        solver,
        program;
        nr_scenarios=settings.scenarios,
    )
    rng = Random.MersenneTwister(settings.seed)
    ContextualDFL.train!(
        model,
        loss,
        nothing,
        fill(settings.mu, settings.epochs),
        fill(0.0, settings.epochs),
        train_data;
        optimizer_type=Flux.Adam,
        learning_rate=settings.learning_rate,
        epochs=settings.epochs,
        batchsize=settings.batchsize,
        display_iterations=false,
        verbose=false,
        display_plot=false,
        shuffle=false,
        rng=rng,
        reset_optimizer_each_epoch=true,
        nr_scenarios=settings.scenarios,
    )
    return model
end

function dfl_policy(
    policy_name,
    benchmark,
    problem,
    train_data,
    solver,
    program,
    reference_decoder,
    config;
    cache_dir,
    data_cache_key,
    refresh_policy_cache=false,
    policy_seed=Int(config.seed),
    history_dir=nothing,
    timestamp="",
    replica_index=1,
)
    rho = dfl_rho_for_policy(policy_name)
    settings = dfl_baseline_settings(; seed=policy_seed, rho=rho)
    input_spec = dfl_input_decoder(benchmark, problem)
    model, history, loaded_from_cache, policy_cache_path = load_or_train_dfl_model(
        policy_name,
        benchmark,
        problem,
        train_data,
        solver,
        program,
        reference_decoder,
        input_spec.decoder,
        input_spec.width,
        settings;
        cache_dir=cache_dir,
        data_cache_key=data_cache_key,
        refresh_cache=refresh_policy_cache,
    )
    policy_history_path = write_dfl_policy_history(
        history_dir,
        timestamp,
        benchmark,
        policy_name,
        replica_index,
        policy_seed,
        history,
    )
    scenario_generator = ContextualDFL.ScenarioGenerator(
        neural_net=model,
        scenario_decoder=input_spec.decoder,
    )
    policy = ScenarioGenerationPolicy(
        scenario_generator,
        solver,
        program;
        mu=0.0,
        rho=settings.rho,
        nr_scenarios=settings.scenarios,
    )
    return (;
        policy=policy,
        metadata=(;
            loaded_policy_from_cache=loaded_from_cache,
            policy_cache_path=policy_cache_path,
            policy_history_path=policy_history_path,
            mu_train=0.0,
            rho_train=settings.rho,
            mu_eval=0.0,
            rho_eval=0.0,
        ),
    )
end

function dfl_baseline_settings(; seed, rho)
    return (;
        version=DFL_BASELINE_VERSION,
        seed=Int(seed),
        rho=Float64(rho),
        epochs=env_int("CDFL_BASELINE_DFL_EPOCHS", 130),
        scenarios=env_int("CDFL_BASELINE_DFL_SCENARIOS", 1),
        batchsize=env_int("CDFL_BASELINE_DFL_BATCHSIZE", 1),
        hidden_dim=env_int("CDFL_BASELINE_DFL_HIDDEN_DIM", 128),
        depth=env_int("CDFL_BASELINE_DFL_DEPTH", 3),
        learning_rate=env_float("CDFL_BASELINE_DFL_LEARNING_RATE", 1e-3),
        mu_train=0.0,
        mu_eval=0.0,
        rho_eval=0.0,
    )
end

function dfl_cache_tag(policy_name, config, replica_seed)
    settings = dfl_baseline_settings(
        seed=replica_seed,
        rho=dfl_rho_for_policy(policy_name),
    )
    return join(
        (
            settings.version,
            policy_name,
            "seed$(settings.seed)",
            "epochs$(settings.epochs)",
            "scenarios$(settings.scenarios)",
            "batch$(settings.batchsize)",
            "hidden$(settings.hidden_dim)",
            "depth$(settings.depth)",
            "lr$(settings.learning_rate)",
            "mu$(settings.mu_train)",
            "rho$(settings.rho)",
        ),
        "_",
    )
end

function dfl_input_decoder(benchmark, problem)
    if benchmark == "resource_allocation"
        return (;
            decoder=ResourceAllocationDemandVectorDecoder(problem),
            width=size(problem.problem_data.service_rate_parameters, 2),
        )
    elseif benchmark == "shipment_planning"
        return (;
            decoder=ShipmentPlanningDemandVectorDecoder(problem),
            width=problem.demand_count,
        )
    elseif benchmark == "transshipment_q"
        mean_parameters = ContextualDFL.transshipment_mean_parameters(problem.core_problem)
        return (;
            decoder=TransShipmentPositiveQVectorDecoder(problem),
            width=length(mean_parameters.q),
        )
    elseif benchmark == "transshipment_h"
        mean_parameters = ContextualDFL.transshipment_mean_parameters(problem.core_problem)
        return (;
            decoder=TransShipmentPositiveHVectorDecoder(problem),
            width=length(mean_parameters.rhs),
        )
    elseif benchmark == "transshipment_h_and_q"
        mean_parameters = ContextualDFL.transshipment_mean_parameters(problem.core_problem)
        return (;
            decoder=TransShipmentPositiveHQVectorDecoder(problem),
            width=length(mean_parameters.rhs) + length(mean_parameters.q),
        )
    elseif benchmark == "random_yield"
        return (;
            decoder=RandomYieldPositiveQVectorDecoder(problem),
            width=length(base_scenario(problem).q),
        )
    elseif benchmark == "unreliable_newsvendor"
        return (;
            decoder=UnreliableNewsvendorParameterVectorDecoder(problem),
            width=2,
        )
    end

    throw(ArgumentError("No DFL vector decoder for benchmark $(repr(benchmark))."))
end

function load_or_train_dfl_model(
    policy_name,
    benchmark,
    problem,
    train_data,
    solver,
    program,
    reference_decoder,
    input_decoder,
    scenario_width,
    settings;
    cache_dir,
    data_cache_key,
    refresh_cache=false,
)
    isnothing(cache_dir) &&
        throw(ArgumentError("cache_dir must be provided for rho-DFL baselines."))
    path = dfl_model_cache_path(cache_dir, data_cache_key, settings; benchmark=benchmark, policy_name=policy_name)
    if isfile(path) && !refresh_cache
        payload = open(Serialization.deserialize, path)
        return payload.model,
               hasproperty(payload, :history) ? payload.history : NamedTuple[],
               true,
               path
    end

    settings.scenarios == 1 ||
        throw(ArgumentError("rho-DFL baseline currently expects nr_scenarios=1."))
    Random.seed!(settings.seed)
    model = build_generic_nn(
        context_dimension_per_point(train_data),
        scenario_width * settings.scenarios;
        hidden_dim=settings.hidden_dim,
        depth=settings.depth,
    )
    history = train_dfl_model!(
        model,
        input_decoder,
        reference_decoder,
        solver,
        program,
        train_data,
        settings,
    )
    mkpath(dirname(path))
    open(path, "w") do io
        Serialization.serialize(
            io,
            (;
                model=model,
                history=history,
                settings=settings,
                benchmark=benchmark,
                policy_name=policy_name,
            ),
        )
    end
    return model, history, false, path
end

function dfl_model_cache_path(cache_dir, data_cache_key, settings; benchmark, policy_name)
    key_material = join(
        (
            data_cache_key,
            settings.version,
            policy_name,
            "seed$(settings.seed)",
            "epochs$(settings.epochs)",
            "scenarios$(settings.scenarios)",
            "batch$(settings.batchsize)",
            "hidden$(settings.hidden_dim)",
            "depth$(settings.depth)",
            "lr$(settings.learning_rate)",
            "rho$(settings.rho)",
        ),
        "__",
    )
    digest = bytes2hex(SHA.sha1(Vector{UInt8}(codeunits(key_material))))
    safe_benchmark = replace(String(benchmark), r"[^A-Za-z0-9_]+" => "_")
    safe_policy = replace(String(policy_name), r"[^A-Za-z0-9_]+" => "_")
    return joinpath(cache_dir, "models", "$(safe_benchmark)_$(safe_policy)_$(digest).jls")
end

function train_dfl_model!(
    model,
    input_decoder,
    reference_decoder,
    solver,
    program,
    train_data,
    settings,
)
    settings.epochs == 0 && return NamedTuple[]
    loss = ContextualDFL.DflScenLoss(
        input_decoder,
        reference_decoder,
        solver,
        program;
        nr_scenarios=settings.scenarios,
    )
    rng = Random.MersenneTwister(settings.seed)
    result = ContextualDFL.train!(
        model,
        loss,
        nothing,
        fill(0.0, settings.epochs),
        fill(0.0, settings.epochs),
        train_data;
        optimizer_type=Flux.Adam,
        learning_rate=settings.learning_rate,
        epochs=settings.epochs,
        batchsize=settings.batchsize,
        display_iterations=false,
        verbose=false,
        display_plot=false,
        shuffle=false,
        rng=rng,
        reset_optimizer_each_epoch=true,
        nr_scenarios=settings.scenarios,
        rho_in_schedule=fill(settings.rho, settings.epochs),
        rho_ref_schedule=fill(settings.rho, settings.epochs),
    )
    return result.history
end

function write_dfl_policy_history(
    history_dir,
    timestamp,
    benchmark,
    policy_name,
    replica_index,
    replica_seed,
    history,
)
    history_dir === nothing && return ""
    mkpath(history_dir)
    safe_policy = replace(String(policy_name), r"[^A-Za-z0-9_]+" => "_")
    path = joinpath(
        history_dir,
        "$(timestamp)_$(benchmark)_$(safe_policy)_replica$(replica_index)_seed$(replica_seed)_history.csv",
    )
    write_history_csv(path, history)
    return path
end

function write_history_csv(path, history)
    rows = collect(history)
    columns = history_columns(rows)
    open(path, "w") do io
        println(io, join(String.(columns), ","))
        for row in rows
            values = history_row_values(row)
            println(io, join((csv_cell(get(values, column, "")) for column in columns), ","))
        end
    end
    return path
end

function history_columns(rows)
    preferred = [
        :epoch,
        :mu,
        :mu_in,
        :mu_ref,
        :rho_in,
        :rho_ref,
        :loss,
        :display_loss,
        :real_display_loss,
        :iterations,
        :seconds,
    ]
    found = Symbol[]
    for row in rows
        for key in keys(history_row_values(row))
            key in found || push!(found, key)
        end
    end
    columns = [key for key in preferred if key in found]
    append!(columns, [key for key in found if !(key in columns)])
    return columns
end

function history_row_values(row)
    if row isa NamedTuple
        return Dict{Symbol,Any}(pairs(row))
    elseif row isa AbstractDict
        return Dict{Symbol,Any}(Symbol(key) => value for (key, value) in row)
    end
    return Dict{Symbol,Any}(:value => row)
end

function decision_focused_settings(benchmark, problem, postprocess_prediction)
    if benchmark == "resource_allocation"
        return (
            target -> max.(Float64.(target), 0.0),
            nonnegative_prediction_penalty_transform(
                lower_bound=0.0,
                penalty_weight=1000.0,
            ),
        )
    end

    return postprocess_prediction, target -> (;
        target=postprocess_prediction(target),
        penalty=0.0,
    )
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
    source_artifact_path=job_data_artifact_path(job),
    loaded_policy_from_cache=false,
    mu_train="",
    rho_train="",
    mu_eval=0.0,
    rho_eval=0.0,
    policy_history_path="",
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
        replica_index=job.replica_index,
        replica_seed=job.replica_seed,
        source_artifact_path=source_artifact_path,
        cache_key=cache_key_value,
        loaded_from_cache=loaded_from_cache,
        loaded_policy_from_cache=loaded_policy_from_cache,
        data_seconds=data_seconds,
        fit_seconds=fit_seconds,
        eval_seconds=eval_seconds,
        mu_train=mu_train,
        rho_train=rho_train,
        mu_eval=mu_eval,
        rho_eval=rho_eval,
        policy_history_path=policy_history_path,
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
