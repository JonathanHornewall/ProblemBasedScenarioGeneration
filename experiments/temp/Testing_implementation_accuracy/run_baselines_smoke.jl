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

    refresh_cache = any(arg -> arg in ("--refresh-cache", "--fresh"), args)
    config = SMOKE_CONFIG
    solver = ContextualDFL.Solver(ContextualDFL.IpoptSolver(), ContextualDFL.HiGHSSolver())
    timestamp = Dates.format(now(), dateformat"yyyymmdd_HHMMSS")

    rows = NamedTuple[]
    println("Running $(config.profile) benchmark baseline check")
    println("Cache directory: $(CACHE_DIR)")
    println("Results directory: $(RESULTS_DIR)")
    refresh_cache && println("Refreshing cached test data and optima")

    for spec in benchmark_specs()
        append!(rows, run_benchmark(spec, solver, config, timestamp; refresh_cache))
    end

    timestamped_path = joinpath(RESULTS_DIR, "baseline_results_$(timestamp).csv")
    latest_path = joinpath(RESULTS_DIR, "baseline_results_latest.csv")
    write_csv(timestamped_path, rows)
    write_csv(latest_path, rows)

    println("Wrote $(length(rows)) policy rows")
    println("Latest results: $(latest_path)")
    print_summary(rows)
    any(row -> row.status != "ok", rows) && exit(1)
    return rows
end

function benchmark_specs()
    return [
        (;
            name="shipment_h",
            variant="h_only",
            make_problem=() -> ShipmentPlanningProblem(),
            make_decoder=problem -> ShipmentPlanningParametricDecoder(problem),
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

function run_benchmark(spec, solver, config, timestamp; refresh_cache=false)
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
    for policy_spec in policy_specs(spec, problem, data_bundle.train_data, solver, program, decoder, config)
        println("  policy: $(policy_spec.name)")
        push!(
            rows,
            run_policy(
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
            ),
        )
    end
    return rows
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
        train_data = generate_benchmark_dataset(
            problem;
            n_contexts=config.train_contexts,
            scenarios_per_context=config.train_scenarios_per_context,
            seed=config.seed,
        )
        test_data = generate_benchmark_dataset(
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
