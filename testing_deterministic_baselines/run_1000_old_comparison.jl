#!/usr/bin/env julia

import Pkg

const EXPERIMENT_DIR = @__DIR__
const REPO_ROOT = normpath(joinpath(EXPERIMENT_DIR, ".."))
const TRAINING_PROJECT_DIR =
    joinpath(REPO_ROOT, "src", "ContextualDFL", "ContextualDFLTraining")
const OLD_PACKAGE_LOAD_PATH = joinpath(REPO_ROOT, "src", "ProblemBasedScenarioGeneration")

Pkg.activate(TRAINING_PROJECT_DIR; io=devnull)
if get(ENV, "DFL_BASELINES_INSTANTIATE", "0") == "1"
    Pkg.instantiate()
end

push!(LOAD_PATH, OLD_PACKAGE_LOAD_PATH)

using ContextualDFL
using ContextualDFLExperiments
using Dates
using LinearAlgebra
import ProblemBasedScenarioGeneration
const OldPBSG = ProblemBasedScenarioGeneration
using Random
using Serialization
using SHA
using Statistics

const RUN_NAME = "benchmark_1000_p1_old_comparison"
const RUN_DIR = joinpath(EXPERIMENT_DIR, RUN_NAME)
const ARTIFACT_DIR = joinpath(RUN_DIR, "artifacts")
const RESULT_DIR = joinpath(RUN_DIR, "results")
const SOURCE_CACHE_ROOT =
    joinpath(REPO_ROOT, "test_value_stochastic_solution_resource_allocation", "p_sweep", "p_1p0")

const SOURCE_SEEDS = (1, 2, 3)
const TRAINING_SEED = 202615051
const TRAINING_CONTEXTS = 100
const TRAINING_SCENARIOS_PER_CONTEXT = 1
const TEST_CONTEXTS_PER_SEED = 10
const EXPECTED_TEST_CONTEXTS = 30
const EXPECTED_TEST_SCENARIOS_PER_CONTEXT = 1000
const EXPECTED_EVALUATION_BATCHES = 20
const DEMAND_SIGMA = 5.0
const DEMAND_POWER = 1.0
const CONTEXT_TERMS = 3
const EVAL_MU = 0.0
const EVAL_RHO = 0.0
const AGREEMENT_GAP_PERCENT_TOL = 2.0
const AGREEMENT_VALUE_REL_TOL = 0.01

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

payload_value(payload, property::Symbol) =
    hasproperty(payload, property) ? getproperty(payload, property) : payload

function contextual_objects()
    problem = ContextualDFLExperiments.ResourceAllocationProblem(
        ContextualDFLExperiments.default_resource_allocation_problem_data(),
    )
    solver = ContextualDFL.Solver(ContextualDFL.IpoptSolver(), ContextualDFL.HiGHSSolver())
    return (;
        problem=problem,
        solver=solver,
        program=ContextualDFLExperiments.stochastic_program(problem),
        decoder=ContextualDFLExperiments.ResourceAllocationDemandParametricDecoder(problem),
    )
end

function old_objects(new_problem)
    data = new_problem.problem_data
    old_data = OldPBSG.ResourceAllocationProblemData(
        Matrix{Float64}(data.service_rate_parameters),
        Vector{Float64}(data.first_stage_costs),
        Vector{Float64}(data.second_stage_costs),
        Vector{Float64}(data.yield_parameters),
    )
    problem = OldPBSG.ResourceAllocationProblem(old_data)
    return (;
        problem=problem,
        A=problem.s1_constraint_matrix,
        b=problem.s1_constraint_vector,
        c=problem.s1_cost_vector,
        W=problem.s2_constraint_matrix,
        T=problem.s2_coupling_matrix,
        q=problem.s2_cost_vector,
        resource_count=size(problem.s2_coupling_matrix, 2),
        demand_count=size(problem.problem_data.service_rate_parameters, 2),
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

function copy_source_cache!()
    mkpath(ARTIFACT_DIR)
    rows = NamedTuple[]
    for seed in SOURCE_SEEDS
        src_dir = joinpath(SOURCE_CACHE_ROOT, "seed_$(seed)")
        target_dir = artifact_path("source_cache", "seed_$(seed)")
        mkpath(target_dir)
        for name in ("test_dataset.jls", "optimal_results.jls", "summary.csv", "per_context.csv")
            source = joinpath(src_dir, name)
            isfile(source) || error("missing required source cache artifact: $source")
            target = joinpath(target_dir, name)
            cp(source, target; force=true)
            push!(
                rows,
                (;
                    seed=seed,
                    artifact=name,
                    source_path=abspath(source),
                    artifact_path=abspath(target),
                    bytes=filesize(target),
                    sha1=file_sha1(target),
                ),
            )
        end
    end
    write_rows_csv(artifact_path("input_manifest.csv"), rows)
    return rows
end

function load_source_artifacts()
    datasets = Any[]
    optimal_results = Any[]
    source_keys = NamedTuple[]

    for seed in SOURCE_SEEDS
        dir = artifact_path("source_cache", "seed_$(seed)")
        dataset_payload = Serialization.deserialize(joinpath(dir, "test_dataset.jls"))
        optimal_payload = Serialization.deserialize(joinpath(dir, "optimal_results.jls"))
        dataset = payload_value(dataset_payload, :dataset)
        results = payload_value(optimal_payload, :optimal_results)

        length(dataset) == TEST_CONTEXTS_PER_SEED ||
            error("seed $seed dataset has $(length(dataset)) contexts; expected $TEST_CONTEXTS_PER_SEED")
        all(point -> length(point.scenario_parameters) == EXPECTED_TEST_SCENARIOS_PER_CONTEXT, dataset) ||
            error("seed $seed dataset does not have $EXPECTED_TEST_SCENARIOS_PER_CONTEXT scenarios per context")
        length(results) == length(dataset) ||
            error("seed $seed optimal results length $(length(results)) does not match dataset length $(length(dataset))")
        all(result -> length(result.objective_values) == EXPECTED_EVALUATION_BATCHES, results) ||
            error("seed $seed optimal results do not have $EXPECTED_EVALUATION_BATCHES objective values")

        append!(datasets, dataset)
        append!(optimal_results, results)
        append!(source_keys, [(; seed=seed, local_index=index) for index in eachindex(dataset)])
    end

    return (; dataset=datasets, optimal_results=optimal_results, source_keys=source_keys)
end

function validate_training!(training_dataset)
    length(training_dataset) == TRAINING_CONTEXTS ||
        error("training dataset has $(length(training_dataset)) rows; expected $TRAINING_CONTEXTS")
    all(point -> length(point.scenario_parameters) == TRAINING_SCENARIOS_PER_CONTEXT, training_dataset) ||
        error("training dataset must have one scenario per context")
end

function fit_regression(training_dataset)
    contexts = Matrix{Float64}(undef, length(training_dataset), length(first(training_dataset).context))
    targets = Matrix{Float64}(undef, length(training_dataset), length(first(first(training_dataset).scenario_parameters).h_eq_xi))
    templates = ContextualDFL.ParametricScenario[]
    for (index, point) in enumerate(training_dataset)
        contexts[index, :] = Float64.(point.context)
        scenario = only(point.scenario_parameters)
        targets[index, :] = Float64.(scenario.h_eq_xi)
        push!(templates, scenario)
    end
    design = hcat(contexts, ones(Float64, size(contexts, 1)))
    coefficients = design \ targets
    fitted = design * coefficients
    return (;
        contexts=contexts,
        targets=targets,
        coefficients=coefficients,
        residuals=targets - fitted,
        templates=templates,
    )
end

function predict_target(coefficients, context)
    context_vector = Float64.(collect(context))
    return vec(transpose(vcat(context_vector, 1.0)) * coefficients)
end

function scenario_with_demand(template, demand)
    return ContextualDFL.ParametricScenario(;
        W_eq_xi=copy(template.W_eq_xi),
        W_ineq_xi=copy(template.W_ineq_xi),
        T_eq_xi=copy(template.T_eq_xi),
        T_ineq_xi=copy(template.T_ineq_xi),
        h_eq_xi=Vector{Float64}(demand),
        h_ineq_xi=copy(template.h_ineq_xi),
        q_xi=copy(template.q_xi),
    )
end

function squared_distance(a, b)
    sum(abs2, Float64.(a) .- Float64.(b))
end

function nearest_indices(training_dataset, context, k)
    distances = [
        (index, squared_distance(point.context, context)) for
        (index, point) in enumerate(training_dataset)
    ]
    sort!(distances; by=item -> item[2])
    return [item[1] for item in distances[1:k]]
end

function flatten_scenarios(dataset)
    scenarios = ContextualDFL.ParametricScenario[]
    for point in dataset
        append!(scenarios, point.scenario_parameters)
    end
    return scenarios
end

function new_policy_specs(training_dataset, objects)
    return [
        (;
            method=:saa,
            label="SAA",
            k=missing,
            build=() -> ContextualDFLExperiments.SampleAverageApproximationPolicy(
                training_dataset,
                objects.solver,
                objects.program,
                objects.decoder;
                mu=EVAL_MU,
                rho=EVAL_RHO,
            ),
        ),
        (;
            method=:least_squares,
            label="Least Squares",
            k=missing,
            build=() -> ContextualDFLExperiments.LeastSquaresPolicy(
                training_dataset,
                objects.solver,
                objects.program,
                objects.decoder;
                mu=EVAL_MU,
                rho=EVAL_RHO,
            ),
        ),
        (;
            method=:residual_saa,
            label="Residual SAA",
            k=missing,
            build=() -> ContextualDFLExperiments.ResidualSampleAverageApproximationPolicy(
                training_dataset,
                objects.solver,
                objects.program,
                objects.decoder;
                mu=EVAL_MU,
                rho=EVAL_RHO,
            ),
        ),
        (;
            method=:knn_saa,
            label="kNN-SAA",
            k=ContextualDFLExperiments.default_knn_k(length(training_dataset)),
            build=() -> ContextualDFLExperiments.KNearestNeighborsPolicy(
                training_dataset,
                objects.solver,
                objects.program,
                objects.decoder;
                mu=EVAL_MU,
                rho=EVAL_RHO,
            ),
        ),
    ]
end

function old_twostage(old, scenario_parameters)
    scenarios = collect(scenario_parameters)
    scenario_count = length(scenarios)
    scenario_count > 0 || error("scenario collection must not be empty")

    h_columns = [
        vcat(zeros(Float64, old.resource_count), Float64.(scenario.h_eq_xi)) for
        scenario in scenarios
    ]
    Ws = repeat(reshape(old.W, size(old.W, 1), size(old.W, 2), 1), 1, 1, scenario_count)
    Ts = repeat(reshape(old.T, size(old.T, 1), size(old.T, 2), 1), 1, 1, scenario_count)
    hs = hcat(h_columns...)
    qs = repeat(reshape(old.q, length(old.q), 1), 1, scenario_count)
    return OldPBSG.TwoStageSLP(old.A, old.b, old.c, Ws, Ts, hs, qs)
end

function old_solve_scenario_collection(old, scenario_parameters)
    twoslp = old_twostage(old, scenario_parameters)
    solution, _ = OldPBSG.solve_canonical_lp(OldPBSG.CanLP(twoslp))
    return Float64.(solution[1:length(old.c)])
end

function old_cost_scenario_collection(old, z, scenario_parameters)
    twoslp = old_twostage(old, scenario_parameters)
    return Float64(OldPBSG.s1_cost(twoslp, Float64.(z), 0.0))
end

function old_decision_set(method, test_dataset, training_dataset, old, regression)
    if method == :saa
        decision = old_solve_scenario_collection(old, flatten_scenarios(training_dataset))
        return reduce(hcat, [decision for _ in eachindex(test_dataset)])
    elseif method == :least_squares
        template = first(regression.templates)
        decisions = [
            old_solve_scenario_collection(
                old,
                [scenario_with_demand(template, predict_target(regression.coefficients, point.context))],
            ) for point in test_dataset
        ]
        return reduce(hcat, decisions)
    elseif method == :residual_saa
        decisions = Vector{Float64}[]
        for point in test_dataset
            base = predict_target(regression.coefficients, point.context)
            scenarios = [
                scenario_with_demand(
                    regression.templates[index],
                    base .+ view(regression.residuals, index, :),
                ) for index in axes(regression.residuals, 1)
            ]
            push!(decisions, old_solve_scenario_collection(old, scenarios))
        end
        return reduce(hcat, decisions)
    elseif method == :knn_saa
        k = ContextualDFLExperiments.default_knn_k(length(training_dataset))
        decisions = Vector{Float64}[]
        for point in test_dataset
            indices = nearest_indices(training_dataset, point.context, k)
            scenarios = flatten_scenarios(training_dataset[indices])
            push!(decisions, old_solve_scenario_collection(old, scenarios))
        end
        return reduce(hcat, decisions)
    end
    error("unsupported old method: $method")
end

function scenario_collection_ranges(point, batch_count)
    scenario_count = length(point.scenario_parameters)
    scenario_count % batch_count == 0 ||
        error("scenario count $scenario_count is not divisible by $batch_count")
    batch_size = scenario_count ÷ batch_count
    return [
        ((batch - 1) * batch_size + 1):(batch * batch_size) for
        batch in 1:batch_count
    ]
end

function mean_value(values)
    isempty(values) && return NaN
    return sum(Float64.(values)) / length(values)
end

function median_value(values)
    isempty(values) && return NaN
    sorted = sort(Float64.(values))
    midpoint = length(sorted) ÷ 2
    isodd(length(sorted)) && return sorted[midpoint + 1]
    return (sorted[midpoint] + sorted[midpoint + 1]) / 2
end

function std_value(values)
    length(values) <= 1 && return length(values) == 0 ? NaN : 0.0
    return Statistics.std(Float64.(values))
end

function summarize_values(values; prefix)
    values = Float64.(values)
    return (;
        Symbol(prefix, :_mean) => mean_value(values),
        Symbol(prefix, :_median) => median_value(values),
        Symbol(prefix, :_std) => std_value(values),
        Symbol(prefix, :_min) => isempty(values) ? NaN : minimum(values),
        Symbol(prefix, :_max) => isempty(values) ? NaN : maximum(values),
    )
end

function old_native_evaluate(method, decision_set, test_dataset, optimal_results, old)
    rows = NamedTuple[]
    seconds = @elapsed begin
        for index in eachindex(test_dataset)
            z = view(decision_set, :, index)
            collection_values = [
                old_cost_scenario_collection(
                    old,
                    z,
                    view(test_dataset[index].scenario_parameters, scenario_range),
                ) for scenario_range in scenario_collection_ranges(
                    test_dataset[index],
                    EXPECTED_EVALUATION_BATCHES,
                )
            ]
            optimal_values = Float64.(optimal_results[index].objective_values)
            policy_value = mean_value(collection_values)
            optimal_value = mean_value(optimal_values)
            regret = policy_value - optimal_value
            relative_regret = regret / max(abs(optimal_value), eps(Float64))
            push!(
                rows,
                (;
                    method=String(method),
                    sample_index=index,
                    policy_value=policy_value,
                    optimal_value=optimal_value,
                    regret=regret,
                    relative_regret=relative_regret,
                    policy_collection_values=collection_values,
                    optimal_collection_values=optimal_values,
                    gap_values=Float64.(collection_values) .- optimal_values,
                ),
            )
        end
    end
    policy_values = [row.policy_value for row in rows]
    optimal_values = [row.optimal_value for row in rows]
    regrets = [row.regret for row in rows]
    relative_regrets = [row.relative_regret for row in rows]
    metrics = merge(
        summarize_values(policy_values; prefix=:test_policy_value),
        summarize_values(optimal_values; prefix=:test_optimal_value),
        summarize_values(regrets; prefix=:test_regret),
        summarize_values(relative_regrets; prefix=:test_relative_regret),
        (; test_sample_count=length(rows), test_policy_eval_seconds=seconds),
    )
    return (; metrics=metrics, per_sample=rows, seconds=seconds)
end

function metric(metrics, name::Symbol, default=NaN)
    hasproperty(metrics, name) || return default
    return Float64(getproperty(metrics, name))
end

function evaluation_summary(method, label, implementation, decision_seconds, evaluation)
    metrics = evaluation.metrics
    return (;
        method=String(method),
        label=label,
        implementation=implementation,
        decision_seconds=decision_seconds,
        policy_eval_seconds=metric(metrics, :test_policy_eval_seconds),
        mean_policy_value=metric(metrics, :test_policy_value_mean),
        mean_optimal_value=metric(metrics, :test_optimal_value_mean),
        mean_regret=metric(metrics, :test_regret_mean),
        mean_relative_regret=metric(metrics, :test_relative_regret_mean),
        optimality_gap_percent=100 * metric(metrics, :test_relative_regret_mean),
        median_relative_regret=metric(metrics, :test_relative_regret_median),
        worst_relative_regret=metric(metrics, :test_relative_regret_max),
    )
end

function per_sample_rows(method, label, implementation, evaluation)
    return [
        (;
            method=String(method),
            label=label,
            implementation=implementation,
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

function decision_rows(method, label, implementation, decision_set)
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
                    implementation=implementation,
                    sample_index=sample_index,
                ),
                z_fields,
            ),
        )
    end
    return rows
end

function max_abs_diff(a, b)
    maximum(abs.(Float64.(a) .- Float64.(b)))
end

function mean_l2_column_diff(a, b)
    mean(norm(Float64.(view(a, :, index)) .- Float64.(view(b, :, index))) for index in axes(a, 2))
end

function compare_rows(method, label, new_decisions, old_decisions, new_eval, old_contextual_eval, old_native_eval)
    new_gap = 100 * metric(new_eval.metrics, :test_relative_regret_mean)
    old_contextual_gap = 100 * metric(old_contextual_eval.metrics, :test_relative_regret_mean)
    old_native_gap = 100 * metric(old_native_eval.metrics, :test_relative_regret_mean)
    old_value = metric(old_native_eval.metrics, :test_policy_value_mean)
    contextual_old_value = metric(old_contextual_eval.metrics, :test_policy_value_mean)
    value_rel_delta = abs(old_value - contextual_old_value) / max(abs(contextual_old_value), eps(Float64))
    gap_delta = abs(old_native_gap - old_contextual_gap)
    return (;
        method=String(method),
        label=label,
        new_gap_percent=new_gap,
        old_decision_contextual_gap_percent=old_contextual_gap,
        old_native_gap_percent=old_native_gap,
        new_vs_old_contextual_gap_delta_percent=abs(new_gap - old_contextual_gap),
        old_native_vs_contextual_gap_delta_percent=gap_delta,
        old_native_vs_contextual_policy_value_relative_delta=value_rel_delta,
        decision_max_abs_diff=max_abs_diff(new_decisions, old_decisions),
        decision_mean_l2_diff=mean_l2_column_diff(new_decisions, old_decisions),
        roughly_agrees=(gap_delta <= AGREEMENT_GAP_PERCENT_TOL) ||
                        (value_rel_delta <= AGREEMENT_VALUE_REL_TOL),
    )
end

function evaluate_all(training_dataset, test_dataset, optimal_results, objects, old)
    regression = fit_regression(training_dataset)
    summaries = NamedTuple[]
    samples = NamedTuple[]
    decisions = NamedTuple[]
    comparisons = NamedTuple[]

    for spec in new_policy_specs(training_dataset, objects)
        println("Building ContextualDFL $(spec.label)...")
        policy_ref = Ref{Any}()
        build_seconds = @elapsed begin
            policy_ref[] = spec.build()
        end
        policy = policy_ref[]

        println("Generating ContextualDFL decisions for $(spec.label)...")
        new_decisions_ref = Ref{Any}()
        new_decision_seconds = @elapsed begin
            new_decisions_ref[] = ContextualDFLExperiments.generate_decision_set(policy, test_dataset)
        end
        new_decisions = new_decisions_ref[]

        println("Evaluating ContextualDFL decisions for $(spec.label)...")
        new_eval = ContextualDFLExperiments.evaluate_policy_against_optimum(
            new_decisions,
            test_dataset,
            objects.program,
            objects.decoder,
            objects.solver;
            optimal_results=optimal_results,
            split_name=:test,
            mu=EVAL_MU,
            rho=EVAL_RHO,
        )

        println("Generating old-implementation decisions for $(spec.label)...")
        old_decisions_ref = Ref{Any}()
        old_decision_seconds = @elapsed begin
            old_decisions_ref[] = old_decision_set(
                spec.method,
                test_dataset,
                training_dataset,
                old,
                regression,
            )
        end
        old_decisions = old_decisions_ref[]

        println("Evaluating old decisions with ContextualDFL evaluator for $(spec.label)...")
        old_contextual_eval = ContextualDFLExperiments.evaluate_policy_against_optimum(
            old_decisions,
            test_dataset,
            objects.program,
            objects.decoder,
            objects.solver;
            optimal_results=optimal_results,
            split_name=:test,
            mu=EVAL_MU,
            rho=EVAL_RHO,
        )

        println("Evaluating old decisions with old native cost for $(spec.label)...")
        old_native_eval = old_native_evaluate(
            spec.method,
            old_decisions,
            test_dataset,
            optimal_results,
            old,
        )

        push!(
            summaries,
            merge(
                evaluation_summary(
                    spec.method,
                    spec.label,
                    "contextualdfl",
                    build_seconds + new_decision_seconds,
                    new_eval,
                ),
                (; k=spec.k),
            ),
        )
        push!(
            summaries,
            merge(
                evaluation_summary(
                    spec.method,
                    spec.label,
                    "old_decision_contextual_eval",
                    old_decision_seconds,
                    old_contextual_eval,
                ),
                (; k=spec.k),
            ),
        )
        push!(
            summaries,
            merge(
                evaluation_summary(
                    spec.method,
                    spec.label,
                    "old_native_eval",
                    old_decision_seconds,
                    old_native_eval,
                ),
                (; k=spec.k),
            ),
        )
        append!(samples, per_sample_rows(spec.method, spec.label, "contextualdfl", new_eval))
        append!(
            samples,
            per_sample_rows(spec.method, spec.label, "old_decision_contextual_eval", old_contextual_eval),
        )
        append!(samples, per_sample_rows(spec.method, spec.label, "old_native_eval", old_native_eval))
        append!(decisions, decision_rows(spec.method, spec.label, "contextualdfl", new_decisions))
        append!(decisions, decision_rows(spec.method, spec.label, "old", old_decisions))
        push!(
            comparisons,
            compare_rows(
                spec.method,
                spec.label,
                new_decisions,
                old_decisions,
                new_eval,
                old_contextual_eval,
                old_native_eval,
            ),
        )
    end

    return (; summaries=summaries, samples=samples, decisions=decisions, comparisons=comparisons)
end

function format_float(value; digits=4)
    x = Float64(value)
    isfinite(x) || return string(x)
    return string(round(x; digits=digits))
end

function write_report(results, training_dataset, test_dataset, optimal_results, manifest_rows)
    contextual_rows = [
        row for row in results.summaries if row.implementation == "contextualdfl"
    ]
    sorted = sort(contextual_rows; by=row -> Float64(row.mean_relative_regret))
    timestamp = string(now(UTC)) * "Z"
    all_agree = all(row.roughly_agrees for row in results.comparisons)

    lines = String[]
    push!(lines, "# 1000-Scenario Deterministic Baseline Validation")
    push!(lines, "")
    push!(lines, "Generated at: $(timestamp)")
    push!(lines, "")
    push!(lines, "## Sandbox")
    push!(lines, "")
    push!(lines, "- All generated files are under `testing_deterministic_baselines/$(RUN_NAME)`.")
    push!(lines, "- Source packages and source experiment artifacts were read-only inputs.")
    push!(lines, "")
    push!(lines, "## Data")
    push!(lines, "")
    push!(lines, "- Training: $(length(training_dataset)) context-scenario pairs, demand power $(DEMAND_POWER), seed $(TRAINING_SEED).")
    push!(lines, "- Test: $(length(test_dataset)) cached contexts with $(length(first(test_dataset).scenario_parameters)) scenarios per context.")
    push!(lines, "- Optimality results: $(length(optimal_results)) cached rows, each with $(length(first(optimal_results).objective_values)) evaluation batches.")
    push!(lines, "- Local training digest: $(serialized_digest(training_dataset)).")
    push!(lines, "")
    push!(lines, "## ContextualDFL Ranking")
    push!(lines, "")
    push!(lines, "| Rank | Method | Gap % | Mean Relative Regret | Mean Regret | Eval Seconds |")
    push!(lines, "|---:|---|---:|---:|---:|---:|")
    for (rank, row) in enumerate(sorted)
        push!(
            lines,
            "| $(rank) | $(row.label) | $(format_float(row.optimality_gap_percent; digits=3)) | $(format_float(row.mean_relative_regret; digits=5)) | $(format_float(row.mean_regret; digits=3)) | $(format_float(row.policy_eval_seconds; digits=3)) |",
        )
    end
    push!(lines, "")
    push!(lines, "## Old Implementation Agreement")
    push!(lines, "")
    push!(lines, "| Method | New Gap % | Old Decision Gap % | Old Native Gap % | Native-vs-NewEval Delta pp | Roughly Agrees |")
    push!(lines, "|---|---:|---:|---:|---:|:---:|")
    for row in results.comparisons
        push!(
            lines,
            "| $(row.label) | $(format_float(row.new_gap_percent; digits=3)) | $(format_float(row.old_decision_contextual_gap_percent; digits=3)) | $(format_float(row.old_native_gap_percent; digits=3)) | $(format_float(row.old_native_vs_contextual_gap_delta_percent; digits=3)) | $(row.roughly_agrees) |",
        )
    end
    push!(lines, "")
    push!(lines, "Overall agreement check: $(all_agree).")
    push!(lines, "")
    push!(lines, "## Artifacts")
    push!(lines, "")
    for row in manifest_rows
        push!(lines, "- `artifacts/source_cache/seed_$(row.seed)/$(row.artifact)` copied from `$(row.source_path)`.")
    end
    push!(lines, "- `results/summary.csv`")
    push!(lines, "- `results/per_sample.csv`")
    push!(lines, "- `results/decisions.csv`")
    push!(lines, "- `results/comparison.csv`")

    write(joinpath(RUN_DIR, "report.md"), join(lines, "\n") * "\n")
end

function main()
    mkpath(ARTIFACT_DIR)
    mkpath(RESULT_DIR)

    objects = contextual_objects()
    old = old_objects(objects.problem)

    manifest_rows = copy_source_cache!()
    source = load_source_artifacts()
    length(source.dataset) == EXPECTED_TEST_CONTEXTS ||
        error("combined test dataset has $(length(source.dataset)) rows; expected $EXPECTED_TEST_CONTEXTS")

    println("Generating sandboxed deterministic training data...")
    training_dataset = generate_training_dataset(objects)
    validate_training!(training_dataset)
    Serialization.serialize(artifact_path("training_dataset.jls"), training_dataset)
    write_rows_csv(
        artifact_path("generated_manifest.csv"),
        [
            (;
                artifact="training_dataset.jls",
                source_path="generated",
                artifact_path=abspath(artifact_path("training_dataset.jls")),
                bytes=filesize(artifact_path("training_dataset.jls")),
                sha1=file_sha1(artifact_path("training_dataset.jls")),
                demand_power=DEMAND_POWER,
                training_seed=TRAINING_SEED,
            ),
        ],
    )

    println("Starting 1000-scenario baseline comparison...")
    results = evaluate_all(training_dataset, source.dataset, source.optimal_results, objects, old)

    write_rows_csv(result_path("summary.csv"), results.summaries)
    write_rows_csv(result_path("per_sample.csv"), results.samples)
    write_rows_csv(result_path("decisions.csv"), results.decisions)
    write_rows_csv(result_path("comparison.csv"), results.comparisons)
    write_report(results, training_dataset, source.dataset, source.optimal_results, manifest_rows)

    println("Wrote 1000-scenario old/new comparison to $(RUN_DIR)")
    return results
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
