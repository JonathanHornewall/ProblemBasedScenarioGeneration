#!/usr/bin/env julia

import Pkg

const EXPERIMENT_DIR = @__DIR__
const REPO_ROOT = normpath(joinpath(EXPERIMENT_DIR, ".."))
const TRAINING_PROJECT_DIR =
    joinpath(REPO_ROOT, "src", "ContextualDFL", "ContextualDFLTraining")

Pkg.activate(TRAINING_PROJECT_DIR; io=devnull)
if get(ENV, "DFL_BASELINES_INSTANTIATE", "0") == "1"
    Pkg.instantiate()
end

using ContextualDFL
using ContextualDFLExperiments
using Dates
using Random
using Serialization
using SHA
using Statistics

const ARTIFACT_DIR = joinpath(EXPERIMENT_DIR, "artifacts")
const RESULT_DIR = joinpath(EXPERIMENT_DIR, "results")
const CACHE_DIR = joinpath(REPO_ROOT, "temp_experiments", "dfl_suite", "artifacts", "test_cache")
const TRAINING_SEED = 202615051
const TRAINING_CONTEXTS = 100
const TRAINING_SCENARIOS_PER_CONTEXT = 1
const EXPECTED_TEST_CONTEXTS = 30
const EXPECTED_TEST_SCENARIOS_PER_CONTEXT = 100
const DEMAND_SIGMA = 5.0
const DEMAND_POWER = 2.0
const CONTEXT_TERMS = 3
const EVAL_MU = 0.0
const EVAL_RHO = 0.0

artifact_path(parts...) = joinpath(ARTIFACT_DIR, parts...)
result_path(parts...) = joinpath(RESULT_DIR, parts...)

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

    text = string(value)
    if occursin(",", text) || occursin("\"", text) || occursin("\n", text) || occursin("\r", text)
        return "\"" * replace(text, "\"" => "\"\"") * "\""
    end
    return text
end

function row_value(row, key::Symbol, default=missing)
    if row isa NamedTuple
        key in keys(row) && return getproperty(row, key)
    elseif row isa AbstractDict
        haskey(row, key) && return row[key]
        haskey(row, String(key)) && return row[String(key)]
    elseif key in propertynames(row)
        return getproperty(row, key)
    end
    return default
end

function write_rows_csv(path, rows; headers=nothing)
    mkpath(dirname(path))
    if headers === nothing
        discovered = Symbol[]
        seen = Set{Symbol}()
        for row in rows
            for key in propertynames(row)
                symbol = Symbol(key)
                symbol in seen && continue
                push!(seen, symbol)
                push!(discovered, symbol)
            end
        end
        headers = discovered
    else
        headers = Symbol.(headers)
    end

    open(path, "w") do io
        println(io, join(String.(headers), ","))
        for row in rows
            println(io, join((csv_escape(row_value(row, header)) for header in headers), ","))
        end
    end
    return path
end

function file_sha1(path)
    return open(path, "r") do io
        bytes2hex(sha1(read(io)))
    end
end

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
        parametric_decoder=ContextualDFLExperiments.ResourceAllocationDemandParametricDecoder(problem),
    )
end

function generate_training_dataset(objects)
    rng = Random.MersenneTwister(TRAINING_SEED)
    context_generator = ContextualDFLExperiments.ResourceAllocationContextDataGenerator(rng=rng)
    scenario_generator = ContextualDFLExperiments.ResourceAllocationScenarioDataGenerator(
        objects.problem;
        sigma=DEMAND_SIGMA,
        p=DEMAND_POWER,
        L=CONTEXT_TERMS,
        rng=rng,
    )

    contexts = [Vector{Float64}(context_generator()) for _ in 1:TRAINING_CONTEXTS]
    scenario_collections = [
        [scenario_generator(context) for _ in 1:TRAINING_SCENARIOS_PER_CONTEXT] for
        context in contexts
    ]
    return ContextualDFLExperiments.generate_contextual_data_set(contexts, scenario_collections)
end

function copy_required_cache_artifacts!()
    mkpath(ARTIFACT_DIR)
    source_dataset = joinpath(CACHE_DIR, "test_dataset.jls")
    source_optima = joinpath(CACHE_DIR, "test_optimal_results.jls")
    source_metadata = joinpath(CACHE_DIR, "metadata.csv")

    for path in (source_dataset, source_optima, source_metadata)
        isfile(path) || error("missing required cached artifact: $path")
    end

    copied = [
        (source=source_dataset, target=artifact_path("test_dataset.jls")),
        (source=source_optima, target=artifact_path("test_optimal_results.jls")),
        (source=source_metadata, target=artifact_path("source_test_cache_metadata.csv")),
    ]
    for item in copied
        cp(item.source, item.target; force=true)
    end

    rows = [
        (;
            artifact=basename(item.target),
            source_path=abspath(item.source),
            artifact_path=abspath(item.target),
            bytes=filesize(item.target),
            sha1=file_sha1(item.target),
        ) for item in copied
    ]
    write_rows_csv(artifact_path("input_manifest.csv"), rows)
    return rows
end

function load_test_artifacts()
    dataset_path = artifact_path("test_dataset.jls")
    optima_path = artifact_path("test_optimal_results.jls")
    return (;
        dataset=Serialization.deserialize(dataset_path),
        optimal_results=Serialization.deserialize(optima_path),
    )
end

function validate_data!(training_dataset, test_dataset, optimal_results)
    length(training_dataset) == TRAINING_CONTEXTS ||
        error("training dataset has $(length(training_dataset)) rows; expected $TRAINING_CONTEXTS")
    all(point -> length(point.scenario_parameters) == TRAINING_SCENARIOS_PER_CONTEXT, training_dataset) ||
        error("training dataset must have one scenario per context")

    length(test_dataset) == EXPECTED_TEST_CONTEXTS ||
        error("test dataset has $(length(test_dataset)) rows; expected $EXPECTED_TEST_CONTEXTS")
    all(point -> length(point.scenario_parameters) == EXPECTED_TEST_SCENARIOS_PER_CONTEXT, test_dataset) ||
        error("test dataset must have 100 scenarios per context")
    length(optimal_results) == length(test_dataset) ||
        error("optimal results length $(length(optimal_results)) does not match test length $(length(test_dataset))")
    return nothing
end

function baseline_specs(training_dataset, objects)
    return [
        (;
            method=:saa,
            label="SAA",
            build=() -> ContextualDFLExperiments.SampleAverageApproximationPolicy(
                training_dataset,
                objects.solver,
                objects.program,
                objects.parametric_decoder;
                mu=EVAL_MU,
                rho=EVAL_RHO,
            ),
            extra=(; k=missing),
        ),
        (;
            method=:least_squares,
            label="Least Squares",
            build=() -> ContextualDFLExperiments.LeastSquaresPolicy(
                training_dataset,
                objects.solver,
                objects.program,
                objects.parametric_decoder;
                mu=EVAL_MU,
                rho=EVAL_RHO,
            ),
            extra=(; k=missing),
        ),
        (;
            method=:residual_saa,
            label="Residual SAA",
            build=() -> ContextualDFLExperiments.ResidualSampleAverageApproximationPolicy(
                training_dataset,
                objects.solver,
                objects.program,
                objects.parametric_decoder;
                mu=EVAL_MU,
                rho=EVAL_RHO,
            ),
            extra=(; k=missing),
        ),
        (;
            method=:knn_saa,
            label="kNN-SAA",
            build=() -> ContextualDFLExperiments.KNearestNeighborsPolicy(
                training_dataset,
                objects.solver,
                objects.program,
                objects.parametric_decoder;
                mu=EVAL_MU,
                rho=EVAL_RHO,
            ),
            extra=(; k=ContextualDFLExperiments.default_knn_k(length(training_dataset))),
        ),
    ]
end

function all_finite(matrix)
    return all(isfinite, Float64.(matrix))
end

function same_columns(matrix; atol=1e-8)
    size(matrix, 2) <= 1 && return true
    first_column = view(matrix, :, 1)
    return all(
        maximum(abs.(view(matrix, :, index) .- first_column)) <= atol for
        index in 2:size(matrix, 2)
    )
end

function metric(metrics, name::Symbol, default=NaN)
    hasproperty(metrics, name) || return default
    return getproperty(metrics, name)
end

function median_relative_regret(per_sample)
    values = sort!(Float64[row.relative_regret for row in per_sample])
    isempty(values) && return NaN
    midpoint = length(values) ÷ 2
    return isodd(length(values)) ? values[midpoint + 1] : (values[midpoint] + values[midpoint + 1]) / 2
end

function worst_relative_regret(per_sample)
    values = Float64[row.relative_regret for row in per_sample]
    isempty(values) && return NaN
    return maximum(values)
end

function decision_rows(method, label, decision_set)
    rows = NamedTuple[]
    for sample_index in axes(decision_set, 2)
        z_fields = (;
            (
                Symbol("z_", index) => Float64(decision_set[index, sample_index]) for
                index in axes(decision_set, 1)
            )...,
        )
        push!(
            rows,
            merge(
                (;
                    method=String(method),
                    label=label,
                    sample_index=sample_index,
                ),
                z_fields,
            ),
        )
    end
    return rows
end

function per_sample_rows(method, label, evaluation)
    return [
        (;
            method=String(method),
            label=label,
            sample_index=row.sample_index,
            policy_value=Float64(row.policy_value),
            optimal_value=Float64(row.optimal_value),
            regret=Float64(row.regret),
            relative_regret=Float64(row.relative_regret),
            optimality_gap_percent=100 * Float64(row.relative_regret),
            policy_collection_values=Float64.(row.policy_collection_values),
            optimal_collection_values=Float64.(row.optimal_collection_values),
            gap_values=Float64.(row.gap_values),
        ) for row in evaluation.per_sample
    ]
end

function summary_row(spec, build_seconds, decision_seconds, value_eval_seconds, policy, decision_set, evaluation)
    metrics = evaluation.metrics
    finite_decisions = all_finite(decision_set)
    decision_dimension = size(decision_set, 1)
    validation_passed =
        finite_decisions &&
        decision_dimension == 20 &&
        (spec.method != :saa || same_columns(decision_set))

    return (;
        method=String(spec.method),
        label=spec.label,
        k=spec.extra.k,
        train_contexts=TRAINING_CONTEXTS,
        train_scenarios_per_context=TRAINING_SCENARIOS_PER_CONTEXT,
        test_contexts=EXPECTED_TEST_CONTEXTS,
        test_scenarios_per_context=EXPECTED_TEST_SCENARIOS_PER_CONTEXT,
        build_seconds=build_seconds,
        decision_seconds=decision_seconds,
        value_eval_seconds=value_eval_seconds,
        policy_eval_seconds=decision_seconds + value_eval_seconds,
        mean_policy_value=Float64(metric(metrics, :test_policy_value_mean)),
        mean_optimal_value=Float64(metric(metrics, :test_optimal_value_mean)),
        mean_regret=Float64(metric(metrics, :test_regret_mean)),
        mean_relative_regret=Float64(metric(metrics, :test_relative_regret_mean)),
        optimality_gap_percent=100 * Float64(metric(metrics, :test_relative_regret_mean)),
        median_relative_regret=median_relative_regret(evaluation.per_sample),
        worst_relative_regret=worst_relative_regret(evaluation.per_sample),
        decision_dimension=decision_dimension,
        expected_decision_dimension=20,
        decisions_finite=finite_decisions,
        saa_context_independent=spec.method == :saa ? same_columns(decision_set) : missing,
        validation_passed=validation_passed,
    )
end

function evaluate_baselines(training_dataset, test_dataset, optimal_results, objects)
    summary_rows = NamedTuple[]
    sample_rows = NamedTuple[]
    all_decision_rows = NamedTuple[]

    for spec in baseline_specs(training_dataset, objects)
        println("Building $(spec.label)...")
        policy_ref = Ref{Any}()
        build_seconds = @elapsed begin
            policy_ref[] = spec.build()
        end
        policy = policy_ref[]

        println("Generating decisions for $(spec.label)...")
        decision_ref = Ref{Any}()
        decision_seconds = @elapsed begin
            decision_ref[] = ContextualDFLExperiments.generate_decision_set(policy, test_dataset)
        end
        decision_set = decision_ref[]

        size(decision_set, 1) == 20 ||
            error("$(spec.label) produced decision dimension $(size(decision_set, 1)); expected 20")
        all_finite(decision_set) ||
            error("$(spec.label) produced non-finite decisions")
        spec.method == :saa && same_columns(decision_set) ||
            spec.method != :saa ||
            error("SAA baseline should return one context-independent decision")

        println("Evaluating $(spec.label)...")
        evaluation_ref = Ref{Any}()
        value_eval_seconds = @elapsed begin
            evaluation_ref[] = ContextualDFLExperiments.evaluate_policy_against_optimum(
                decision_set,
                test_dataset,
                objects.program,
                objects.parametric_decoder,
                objects.solver;
                optimal_results=optimal_results,
                split_name=:test,
                mu=EVAL_MU,
                rho=EVAL_RHO,
            )
        end
        evaluation = evaluation_ref[]

        push!(
            summary_rows,
            summary_row(spec, build_seconds, decision_seconds, value_eval_seconds, policy, decision_set, evaluation),
        )
        append!(sample_rows, per_sample_rows(spec.method, spec.label, evaluation))
        append!(all_decision_rows, decision_rows(spec.method, spec.label, decision_set))
    end

    return (; summary_rows=summary_rows, per_sample_rows=sample_rows, decision_rows=all_decision_rows)
end

function format_float(value; digits=4)
    value === missing && return ""
    x = Float64(value)
    isfinite(x) || return string(x)
    return string(round(x; digits=digits))
end

function write_report(summary_rows, training_dataset, test_dataset, optimal_results, manifest_rows)
    sorted = sort(summary_rows; by=row -> Float64(row.mean_relative_regret))
    timestamp = string(now(UTC)) * "Z"
    saa_check = only(row.saa_context_independent for row in summary_rows if row.method == "saa")
    lines = String[]
    push!(lines, "# Deterministic Baseline Validation")
    push!(lines, "")
    push!(lines, "Generated at: $(timestamp)")
    push!(lines, "")
    push!(lines, "## Data")
    push!(lines, "")
    push!(lines, "- Training: $(length(training_dataset)) context-scenario pairs, seed $(TRAINING_SEED).")
    push!(lines, "- Test: $(length(test_dataset)) cached contexts with $(length(first(test_dataset).scenario_parameters)) scenarios per context.")
    push!(lines, "- Optimality results: $(length(optimal_results)) cached rows reused from `temp_experiments/dfl_suite/artifacts/test_cache`.")
    push!(lines, "- Local artifact digest: training dataset $(serialized_digest(training_dataset)).")
    push!(lines, "")
    push!(lines, "## Ranking")
    push!(lines, "")
    push!(lines, "| Rank | Method | Gap % | Mean Relative Regret | Mean Regret | Policy Eval Seconds |")
    push!(lines, "|---:|---|---:|---:|---:|---:|")
    for (rank, row) in enumerate(sorted)
        push!(
            lines,
            "| $(rank) | $(row.label) | $(format_float(row.optimality_gap_percent; digits=3)) | $(format_float(row.mean_relative_regret; digits=5)) | $(format_float(row.mean_regret; digits=3)) | $(format_float(row.policy_eval_seconds; digits=3)) |",
        )
    end
    push!(lines, "")
    push!(lines, "## Validation")
    push!(lines, "")
    push!(lines, "- Training dataset length and per-context scenario count validated.")
    push!(lines, "- Test dataset length, per-context scenario count, and optimal-result count validated.")
    push!(lines, "- All baseline decisions are finite 20-dimensional first-stage vectors.")
    push!(lines, "- SAA context-independence check: $(saa_check).")
    push!(lines, "")
    push!(lines, "## Artifacts")
    push!(lines, "")
    for row in manifest_rows
        push!(lines, "- `artifacts/$(row.artifact)` copied from `$(row.source_path)`.")
    end
    push!(lines, "- `results/summary.csv`")
    push!(lines, "- `results/per_sample.csv`")
    push!(lines, "- `results/decisions.csv`")

    write(joinpath(EXPERIMENT_DIR, "report.md"), join(lines, "\n") * "\n")
end

function main()
    mkpath(ARTIFACT_DIR)
    mkpath(RESULT_DIR)

    objects = problem_objects()
    manifest_rows = copy_required_cache_artifacts!()

    println("Generating deterministic training data...")
    training_dataset = generate_training_dataset(objects)
    Serialization.serialize(artifact_path("training_dataset.jls"), training_dataset)

    test_artifacts = load_test_artifacts()
    validate_data!(training_dataset, test_artifacts.dataset, test_artifacts.optimal_results)

    data_manifest = [
        (;
            artifact="training_dataset.jls",
            source_path="generated",
            artifact_path=abspath(artifact_path("training_dataset.jls")),
            bytes=filesize(artifact_path("training_dataset.jls")),
            sha1=file_sha1(artifact_path("training_dataset.jls")),
        ),
    ]
    write_rows_csv(artifact_path("generated_manifest.csv"), data_manifest)

    results = evaluate_baselines(
        training_dataset,
        test_artifacts.dataset,
        test_artifacts.optimal_results,
        objects,
    )

    write_rows_csv(result_path("summary.csv"), results.summary_rows)
    write_rows_csv(result_path("per_sample.csv"), results.per_sample_rows)
    write_rows_csv(result_path("decisions.csv"), results.decision_rows)
    write_report(
        results.summary_rows,
        training_dataset,
        test_artifacts.dataset,
        test_artifacts.optimal_results,
        manifest_rows,
    )

    println("Wrote deterministic baseline validation to $(EXPERIMENT_DIR)")
    return results
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
