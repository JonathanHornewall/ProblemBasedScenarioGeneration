#!/usr/bin/env julia

# Smoke benchmark runner for the new ContextualDFLExperiments benchmark instances.
#
# Everything this script writes stays below this directory:
#   experiments/temp/Testing_implementation_accuracy/
#
# The generated train/test data and precomputed test optima are serialized into
# cache/*.jls so the expensive optimum solve can be reused across reruns.

const LOCAL_PROJECT = normpath(
    joinpath(@__DIR__, "..", "..", "..", "src", "ContextualDFL", "ContextualDFLExperiments"),
)
if isfile(joinpath(LOCAL_PROJECT, "Project.toml")) && !(LOCAL_PROJECT in LOAD_PATH)
    pushfirst!(LOAD_PATH, LOCAL_PROJECT)
end

using ContextualDFL
using ContextualDFLExperiments
using Dates
using Printf
using Random
using Serialization

const RUN_ROOT = @__DIR__
const CACHE_DIR = joinpath(RUN_ROOT, "cache")
const RESULTS_DIR = joinpath(RUN_ROOT, "results")
const CACHE_VERSION = "v1"

const SMOKE_CONFIG = (;
    profile="smoke",
    train_contexts=5,
    train_scenarios_per_context=1,
    test_contexts=1,
    test_scenarios_per_context=1,
    seed=20260505,
    knn_k=1,
)

const PROPER_CONFIG = (;
    profile="proper",
    train_contexts=100,
    train_scenarios_per_context=1,
    test_contexts=30,
    test_scenarios_per_context=1000,
    seed=20260505,
    knn_k=10,
)

const CSV_COLUMNS = (
    :timestamp,
    :profile,
    :benchmark,
    :variant,
    :policy,
    :status,
    :train_contexts,
    :train_scenarios_per_context,
    :test_contexts,
    :test_scenarios_per_context,
    :seed,
    :cache_key,
    :loaded_from_cache,
    :data_seconds,
    :fit_seconds,
    :eval_seconds,
    :sample_count,
    :evaluation_batches,
    :policy_value_mean,
    :optimal_value_mean,
    :regret_mean,
    :relative_regret_mean,
    :policy_eval_seconds,
    :error,
)

function main(args=ARGS)
    mkpath(CACHE_DIR)
    mkpath(RESULTS_DIR)

    options = parse_options(args)
    config = options.config
    refresh_cache = options.refresh_cache
    solver = ContextualDFL.Solver(ContextualDFL.IpoptSolver(), ContextualDFL.HiGHSSolver())
    timestamp = Dates.format(now(), dateformat"yyyymmdd_HHMMSS")

    rows = NamedTuple[]
    timestamped_path = joinpath(RESULTS_DIR, "baseline_results_$(timestamp).csv")
    latest_path = joinpath(RESULTS_DIR, "baseline_results_latest.csv")
    write_csv(timestamped_path, rows)
    write_csv(latest_path, rows)

    function record_row(row)
        push!(rows, row)
        write_csv(timestamped_path, rows)
        write_csv(latest_path, rows)
        return nothing
    end

    println("Running $(config.profile) benchmark baseline check")
    println("Cache directory: $(CACHE_DIR)")
    println("Results directory: $(RESULTS_DIR)")
    refresh_cache && println("Refreshing cached test data and optima")

    for spec in selected_benchmark_specs(options.benchmark_names)
        run_benchmark(
            spec,
            solver,
            config,
            timestamp;
            refresh_cache,
            policy_names=options.policy_names,
            on_row=record_row,
        )
    end

    println("Wrote $(length(rows)) policy rows")
    println("Latest results: $(latest_path)")
    print_summary(rows)
    any(row -> row.status != "ok", rows) && exit(1)
    return rows
end

function parse_options(args)
    profile = "smoke"
    refresh_cache = false
    overrides = Dict{Symbol,Int}()
    benchmark_names = nothing
    policy_names = nothing

    index = 1
    while index <= length(args)
        arg = args[index]
        if arg in ("--refresh-cache", "--fresh")
            refresh_cache = true
        elseif arg == "--profile"
            index += 1
            index <= length(args) || throw(ArgumentError("--profile requires a value."))
            profile = args[index]
        elseif startswith(arg, "--profile=")
            profile = split(arg, "=", limit=2)[2]
        elseif arg in ("--train-contexts", "--train-data")
            index += 1
            overrides[:train_contexts] = parse(Int, args[index])
        elseif startswith(arg, "--train-contexts=")
            overrides[:train_contexts] = parse(Int, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--train-data=")
            overrides[:train_contexts] = parse(Int, split(arg, "=", limit=2)[2])
        elseif arg == "--train-scenarios"
            index += 1
            overrides[:train_scenarios_per_context] = parse(Int, args[index])
        elseif startswith(arg, "--train-scenarios=")
            overrides[:train_scenarios_per_context] = parse(Int, split(arg, "=", limit=2)[2])
        elseif arg in ("--test-contexts", "--validation-contexts")
            index += 1
            overrides[:test_contexts] = parse(Int, args[index])
        elseif startswith(arg, "--test-contexts=")
            overrides[:test_contexts] = parse(Int, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--validation-contexts=")
            overrides[:test_contexts] = parse(Int, split(arg, "=", limit=2)[2])
        elseif arg in ("--test-scenarios", "--validation-scenarios")
            index += 1
            overrides[:test_scenarios_per_context] = parse(Int, args[index])
        elseif startswith(arg, "--test-scenarios=")
            overrides[:test_scenarios_per_context] = parse(Int, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--validation-scenarios=")
            overrides[:test_scenarios_per_context] = parse(Int, split(arg, "=", limit=2)[2])
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
        else
            throw(ArgumentError("Unknown argument: $(arg)"))
        end
        index += 1
    end

    base_config = if profile == "smoke"
        SMOKE_CONFIG
    elseif profile == "proper"
        PROPER_CONFIG
    else
        throw(ArgumentError("Unknown profile $(repr(profile)); expected smoke or proper."))
    end

    config = (;
        profile=profile,
        train_contexts=get(overrides, :train_contexts, base_config.train_contexts),
        train_scenarios_per_context=get(
            overrides,
            :train_scenarios_per_context,
            base_config.train_scenarios_per_context,
        ),
        test_contexts=get(overrides, :test_contexts, base_config.test_contexts),
        test_scenarios_per_context=get(
            overrides,
            :test_scenarios_per_context,
            base_config.test_scenarios_per_context,
        ),
        seed=get(overrides, :seed, base_config.seed),
        knn_k=get(overrides, :knn_k, base_config.knn_k),
    )

    return (;
        config=config,
        refresh_cache=refresh_cache,
        benchmark_names=benchmark_names,
        policy_names=policy_names,
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

function benchmark_specs()
    return [
        (;
            name="shipment_h",
            variant="h_only",
            make_problem=() -> ShipmentPlanningProblem(),
            make_decoder=problem -> compact_shipment_decoder(problem),
        ),
        (;
            name="transshipment_q",
            variant="q_only",
            make_problem=() -> TransShipmentExperimentProblem(; variant=:q_only),
            make_decoder=problem -> transshipment_decoder(problem),
        ),
        (;
            name="transshipment_h",
            variant="h_only",
            make_problem=() -> TransShipmentExperimentProblem(; variant=:h_only),
            make_decoder=problem -> transshipment_decoder(problem),
        ),
        (;
            name="transshipment_h_and_q",
            variant="h_and_q",
            make_problem=() -> TransShipmentExperimentProblem(; variant=:h_and_q),
            make_decoder=problem -> transshipment_decoder(problem),
        ),
        (;
            name="random_yield_W_small",
            variant="W_support",
            make_problem=() -> RandomYieldProblem(; r=5, a=10, K_support=5),
            make_decoder=problem -> RandomYieldParametricDecoder(problem),
        ),
    ]
end

function selected_benchmark_specs(benchmark_names)
    specs = benchmark_specs()
    isnothing(benchmark_names) && return specs

    selected = [spec for spec in specs if spec.name in benchmark_names]
    found = Set(spec.name for spec in selected)
    missing = setdiff(benchmark_names, found)
    isempty(missing) ||
        throw(ArgumentError("Unknown benchmark(s): $(join(sort(collect(missing)), ", "))."))
    return selected
end

function run_benchmark(
    spec,
    solver,
    config,
    timestamp;
    refresh_cache=false,
    policy_names=nothing,
    on_row=row -> nothing,
)
    println()
    println("== $(spec.name) ==")
    problem = spec.make_problem()
    decoder = spec.make_decoder(problem)
    program = stochastic_program(problem)

    data_bundle, loaded_from_cache, data_seconds = cached_data_bundle(
        spec,
        problem,
        decoder,
        solver,
        program,
        config;
        refresh_cache,
    )
    println(
        loaded_from_cache ?
        "Loaded cached test data and optima: $(data_bundle.cache_key)" :
        "Generated and cached test data and optima: $(data_bundle.cache_key)",
    )

    rows = NamedTuple[]
    policies = selected_policy_specs(
        policy_specs(spec, problem, data_bundle.train_data, solver, program, decoder, config),
        policy_names,
        spec.name,
    )
    for policy_spec in policies
        println("  policy: $(policy_spec.name)")
        row = run_policy(
            spec,
            policy_spec,
            data_bundle,
            solver,
            program,
            decoder,
            config,
            timestamp,
            loaded_from_cache,
            data_seconds,
        )
        push!(rows, row)
        on_row(row)
    end
    return rows
end

function selected_policy_specs(policies, policy_names, benchmark_name)
    isnothing(policy_names) && return policies

    selected = [policy for policy in policies if policy.name in policy_names]
    found = Set(policy.name for policy in selected)
    missing = setdiff(policy_names, found)
    isempty(missing) ||
        throw(ArgumentError(
            "Benchmark $(benchmark_name) does not have policy/policies: $(join(sort(collect(missing)), ", ")).",
        ))
    return selected
end

function cached_data_bundle(
    spec,
    problem,
    decoder,
    solver,
    program,
    config;
    refresh_cache=false,
)
    key = cache_key(spec, config)
    path = joinpath(CACHE_DIR, key * ".jls")
    if isfile(path) && !refresh_cache
        elapsed = @elapsed bundle = deserialize(path)
        return bundle, true, elapsed
    end

    bundle = nothing
    elapsed = @elapsed begin
        train_data = generate_dataset_for_spec(
            spec,
            problem;
            n_contexts=config.train_contexts,
            scenarios_per_context=config.train_scenarios_per_context,
            seed=config.seed,
        )
        test_data = generate_dataset_for_spec(
            spec,
            problem;
            n_contexts=config.test_contexts,
            scenarios_per_context=config.test_scenarios_per_context,
            seed=config.seed + 10_000,
        )
        optimal_results = solve_dataset_to_optimality(test_data, program, decoder, solver)
        bundle = (;
            cache_version=CACHE_VERSION,
            cache_key=key,
            benchmark=spec.name,
            variant=spec.variant,
            config=config,
            train_data=train_data,
            test_data=test_data,
            optimal_results=optimal_results,
        )
        serialize(path, bundle)
    end
    return bundle, false, elapsed
end

function generate_dataset_for_spec(
    spec,
    problem;
    n_contexts,
    scenarios_per_context,
    seed,
)
    if spec.name == "shipment_h"
        return generate_compact_shipment_dataset(
            problem;
            n_contexts,
            scenarios_per_context,
            seed,
        )
    end

    return generate_benchmark_dataset(
        problem;
        n_contexts,
        scenarios_per_context,
        seed,
    )
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

function generate_compact_shipment_dataset(
    problem::ShipmentPlanningProblem;
    n_contexts,
    scenarios_per_context,
    seed,
)
    rng = Random.MersenneTwister(seed)
    contexts = [randn(rng, problem.context_dim) for _ in 1:n_contexts]
    scenario_collections = [
        [
            ContextualDFL.ParametricScenario(;
                h_eq_xi=compact_shipment_h_eq(problem, context, rng),
            ) for _ in 1:scenarios_per_context
        ] for context in contexts
    ]
    return generate_contextual_data_set(contexts, scenario_collections)
end

function compact_shipment_h_eq(
    problem::ShipmentPlanningProblem,
    context,
    rng::Random.AbstractRNG,
)
    features = Float64.(context) .^ problem.p
    h_eq = zeros(Float64, problem.demand_count + problem.warehouse_count)
    for j in 1:problem.demand_count
        signal = problem.demand_intercepts[j] +
                 sum(problem.demand_slopes[j, term] * features[term] for term in 1:problem.context_dim)
        h_eq[j] = max(1e-6, signal + problem.sigma * randn(rng))
    end
    return h_eq
end

function cache_key(spec, config)
    return join(
        (
            CACHE_VERSION,
            config.profile,
            spec.name,
            spec.variant,
            "train$(config.train_contexts)x$(config.train_scenarios_per_context)",
            "test$(config.test_contexts)x$(config.test_scenarios_per_context)",
            "seed$(config.seed)",
            "solver_ipopt_highs",
        ),
        "__",
    )
end

function policy_specs(spec, problem, train_data, solver, program, decoder, config)
    policies = [
        (;
            name="saa",
            build=() -> SampleAverageApproximationPolicy(train_data, solver, program, decoder),
        ),
        (;
            name="knn",
            build=() -> KNearestNeighborsPolicy(
                train_data,
                solver,
                program,
                decoder;
                k=min(config.knn_k, length(train_data)),
            ),
        ),
    ]

    if spec.name == "shipment_h"
        shipment_postprocess = target -> begin
            values = Float64.(target)
            values[1:problem.demand_count] = max.(values[1:problem.demand_count], 1e-6)
            values[(problem.demand_count + 1):end] .= 0.0
            values
        end
        append!(
            policies,
            [
                (;
                    name="least_squares",
                    build=() -> LeastSquaresPolicy(
                        train_data,
                        solver,
                        program,
                        decoder;
                        target_component=:h_eq_xi,
                        postprocess_prediction=shipment_postprocess,
                    ),
                ),
                (;
                    name="er_saa",
                    build=() -> ResidualSampleAverageApproximationPolicy(
                        train_data,
                        solver,
                        program,
                        decoder;
                        target_component=:h_eq_xi,
                        postprocess_prediction=shipment_postprocess,
                    ),
                ),
            ],
        )
    elseif spec.name == "transshipment_q"
        append!(
            policies,
            regression_policy_specs(
                train_data,
                solver,
                program,
                decoder,
                :q_xi,
                target -> max.(Float64.(target), 1e-4),
            ),
        )
    elseif spec.name == "transshipment_h"
        append!(
            policies,
            regression_policy_specs(
                train_data,
                solver,
                program,
                decoder,
                :h_eq_xi,
                target -> max.(Float64.(target), 1e-4),
            ),
        )
    end

    return policies
end

function regression_policy_specs(
    train_data,
    solver,
    program,
    decoder,
    target_component,
    postprocess_prediction,
)
    return [
        (;
            name="least_squares",
            build=() -> LeastSquaresPolicy(
                train_data,
                solver,
                program,
                decoder;
                target_component,
                postprocess_prediction,
            ),
        ),
        (;
            name="er_saa",
            build=() -> ResidualSampleAverageApproximationPolicy(
                train_data,
                solver,
                program,
                decoder;
                target_component,
                postprocess_prediction,
            ),
        ),
    ]
end

function run_policy(
    spec,
    policy_spec,
    data_bundle,
    solver,
    program,
    decoder,
    config,
    timestamp,
    loaded_from_cache,
    data_seconds,
)
    fit_seconds = 0.0
    eval_seconds = 0.0
    try
        policy = nothing
        fit_seconds = @elapsed policy = policy_spec.build()
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
            spec,
            policy_spec.name,
            config,
            timestamp,
            data_bundle.cache_key,
            loaded_from_cache,
            data_seconds,
            fit_seconds,
            eval_seconds,
            "ok",
            "";
            sample_count=metrics.test_sample_count,
            evaluation_batches=metrics.test_evaluation_batches,
            policy_value_mean=metrics.test_policy_value_mean,
            optimal_value_mean=metrics.test_optimal_value_mean,
            regret_mean=metrics.test_regret_mean,
            relative_regret_mean=metrics.test_relative_regret_mean,
            policy_eval_seconds=metrics.test_policy_eval_seconds,
        )
    catch error
        showerror(stderr, error)
        println(stderr)
        return result_row(
            spec,
            policy_spec.name,
            config,
            timestamp,
            data_bundle.cache_key,
            loaded_from_cache,
            data_seconds,
            fit_seconds,
            eval_seconds,
            "error",
            sprint(showerror, error),
        )
    end
end

function result_row(
    spec,
    policy,
    config,
    timestamp,
    cache_key,
    loaded_from_cache,
    data_seconds,
    fit_seconds,
    eval_seconds,
    status,
    error;
    sample_count="",
    evaluation_batches="",
    policy_value_mean="",
    optimal_value_mean="",
    regret_mean="",
    relative_regret_mean="",
    policy_eval_seconds="",
)
    return (;
        timestamp=timestamp,
        profile=config.profile,
        benchmark=spec.name,
        variant=spec.variant,
        policy=policy,
        status=status,
        train_contexts=config.train_contexts,
        train_scenarios_per_context=config.train_scenarios_per_context,
        test_contexts=config.test_contexts,
        test_scenarios_per_context=config.test_scenarios_per_context,
        seed=config.seed,
        cache_key=cache_key,
        loaded_from_cache=loaded_from_cache,
        data_seconds=data_seconds,
        fit_seconds=fit_seconds,
        eval_seconds=eval_seconds,
        sample_count=sample_count,
        evaluation_batches=evaluation_batches,
        policy_value_mean=policy_value_mean,
        optimal_value_mean=optimal_value_mean,
        regret_mean=regret_mean,
        relative_regret_mean=relative_regret_mean,
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

function print_summary(rows)
    println()
    println("Summary")
    for row in rows
        if row.status == "ok"
            @printf(
                "  %-26s %-14s regret=%12.6g rel=%12.6g fit=%7.3fs eval=%7.3fs\n",
                row.benchmark,
                row.policy,
                Float64(row.regret_mean),
                Float64(row.relative_regret_mean),
                Float64(row.fit_seconds),
                Float64(row.eval_seconds),
            )
        else
            println("  $(row.benchmark) $(row.policy) ERROR: $(row.error)")
        end
    end
end

main()
