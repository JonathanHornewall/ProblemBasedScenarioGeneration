#!/usr/bin/env julia

import Pkg

const EXPERIMENT_DIR = @__DIR__
const REPO_ROOT = dirname(EXPERIMENT_DIR)
const TRAINING_PROJECT_DIR =
    joinpath(REPO_ROOT, "src", "ContextualDFL", "ContextualDFLTraining")

Pkg.activate(TRAINING_PROJECT_DIR; io=devnull)

using CSV
using ContextualDFL
using ContextualDFLExperiments
using Dates
using Serialization
using Statistics

const ARTIFACT_DIR = joinpath(
    REPO_ROOT,
    "src",
    "ContextualDFL",
    "ContextualDFLTraining",
    "src",
    "experiments",
    "resource_allocation",
    "experiment_1",
    "artifacts",
    "test_data",
    "test_data",
)
const SEEDS = (1, 2, 3)
const MU = 0.0
const RHO = 0.0

function artifact_paths(seed::Integer)
    return (;
        dataset=joinpath(ARTIFACT_DIR, "test_data_seed$(seed).jls"),
        optimal_results=joinpath(ARTIFACT_DIR, "optimal_solutions_seed$(seed).jls"),
    )
end

function payload_value(payload, property::Symbol)
    return hasproperty(payload, property) ? getproperty(payload, property) : payload
end

function load_seed_artifact(seed::Integer)
    paths = artifact_paths(seed)
    isfile(paths.dataset) ||
        throw(ArgumentError("missing test dataset artifact: $(paths.dataset)"))
    isfile(paths.optimal_results) ||
        throw(ArgumentError("missing optimal-results artifact: $(paths.optimal_results)"))

    dataset_payload = Serialization.deserialize(paths.dataset)
    optimal_payload = Serialization.deserialize(paths.optimal_results)
    dataset = payload_value(dataset_payload, :dataset)
    optimal_results = payload_value(optimal_payload, :optimal_results)

    length(dataset) == length(optimal_results) ||
        throw(DimensionMismatch("seed $seed has $(length(dataset)) data rows but $(length(optimal_results)) optimal rows."))
    all(result -> hasproperty(result, :objective_values), optimal_results) ||
        throw(ArgumentError("seed $seed optimal results are not in the current objective_values format."))

    return (;
        seed=Int(seed),
        paths=paths,
        dataset=dataset,
        optimal_results=optimal_results,
        dataset_payload=dataset_payload,
        optimal_payload=optimal_payload,
    )
end

function average_component(scenarios, field::Symbol)
    first_value = getproperty(first(scenarios), field)

    if first_value isa Number
        return mean(Float64(getproperty(scenario, field)) for scenario in scenarios)
    end

    first_value isa AbstractArray ||
        throw(ArgumentError("cannot average non-numeric scenario component $field of type $(typeof(first_value))."))

    accumulator = zeros(Float64, size(first_value))
    for scenario in scenarios
        value = getproperty(scenario, field)
        size(value) == size(first_value) ||
            throw(DimensionMismatch("scenario component $field has inconsistent sizes."))
        accumulator .+= Float64.(value)
    end
    return accumulator ./ length(scenarios)
end

function average_scenario(scenarios)
    isempty(scenarios) && throw(ArgumentError("cannot average an empty scenario collection."))
    return ContextualDFL.ParametricScenario(;
        W_eq_xi=average_component(scenarios, :W_eq_xi),
        W_ineq_xi=average_component(scenarios, :W_ineq_xi),
        T_eq_xi=average_component(scenarios, :T_eq_xi),
        T_ineq_xi=average_component(scenarios, :T_ineq_xi),
        h_eq_xi=average_component(scenarios, :h_eq_xi),
        h_ineq_xi=average_component(scenarios, :h_ineq_xi),
        q_xi=average_component(scenarios, :q_xi),
    )
end

function problem_objects()
    problem = ResourceAllocationProblem(default_resource_allocation_problem_data())
    solver = ContextualDFL.Solver(ContextualDFL.IpoptSolver(), ContextualDFL.HiGHSSolver())
    return (;
        problem=problem,
        solver=solver,
        program=stochastic_program(problem),
        decoder=ResourceAllocationDemandParametricDecoder(problem),
    )
end

function solve_average_scenario_decision(data_point, objects)
    scenario = average_scenario(data_point.scenario_parameters)
    arrays = ContextualDFL.decode_scenario_collection(objects.decoder, [scenario])
    solution = ContextualDFL.solve(
        objects.solver,
        objects.program,
        arrays...;
        μ=MU,
        ρ=RHO,
    )
    return collect(solution[1]), scenario
end

function joined(values)
    return join(string.(Float64.(collect(values))), "|")
end

function metric_row(metrics, decision_solve_seconds, artifacts, data_set)
    row = Dict{Symbol,Any}()
    for key in keys(metrics)
        row[key] = getproperty(metrics, key)
    end
    row[:decision_solve_seconds] = decision_solve_seconds
    row[:total_eval_seconds] = decision_solve_seconds + metrics.test_policy_eval_seconds
    row[:artifact_dir] = ARTIFACT_DIR
    row[:seeds] = join(SEEDS, "|")
    row[:contexts] = length(data_set)
    row[:scenarios_per_context] = length(first(data_set).scenario_parameters)
    row[:generated_at] = Dates.format(now(UTC), dateformat"yyyy-mm-ddTHH:MM:SS.sZ")
    row[:source_dataset_digests] = join(
        [
            hasproperty(artifact.dataset_payload, :dataset_digest) ?
            artifact.dataset_payload.dataset_digest :
            missing for artifact in artifacts
        ],
        "|",
    )
    row[:source_optimal_digests] = join(
        [
            hasproperty(artifact.optimal_payload, :dataset_digest) ?
            artifact.optimal_payload.dataset_digest :
            missing for artifact in artifacts
        ],
        "|",
    )
    return (; (key => row[key] for key in sort!(collect(keys(row))))...)
end

function write_report(path, summary)
    open(path, "w") do io
        println(io, "# Average-Scenario Baseline")
        println(io)
        println(io, "This run solves one deterministic average-scenario resource-allocation decision per test context, then evaluates those fixed decisions on the original stochastic scenario collections.")
        println(io)
        println(io, "- Contexts: $(summary.contexts)")
        println(io, "- Scenarios per context: $(summary.scenarios_per_context)")
        println(io, "- Evaluation batches: $(summary.test_evaluation_batches)")
        println(io, "- Mean average-scenario policy value: $(summary.test_policy_value_mean)")
        println(io, "- Mean stochastic optimum value: $(summary.test_optimal_value_mean)")
        println(io, "- Mean gap: $(summary.test_regret_mean)")
        println(io, "- Mean relative gap: $(summary.test_relative_regret_mean)")
        println(io, "- Median gap: $(summary.test_regret_median)")
        println(io, "- 95th percentile gap: $(summary.test_regret_p95)")
        println(io)
        println(io, "Generated at $(summary.generated_at).")
    end
end

function main()
    artifacts = [load_seed_artifact(seed) for seed in SEEDS]
    data_set = reduce(vcat, [artifact.dataset for artifact in artifacts])
    optimal_results = reduce(vcat, [artifact.optimal_results for artifact in artifacts])
    source_keys = [
        (seed=artifact.seed, local_index=index)
        for artifact in artifacts
        for index in eachindex(artifact.dataset)
    ]

    objects = problem_objects()
    average_scenarios = ContextualDFL.ParametricScenario[]
    decision_columns = Vector{Float64}[]
    decision_solve_seconds = @elapsed begin
        for data_point in data_set
            decision, scenario = solve_average_scenario_decision(data_point, objects)
            push!(decision_columns, decision)
            push!(average_scenarios, scenario)
        end
    end

    decision_set = hcat(decision_columns...)
    comparison = evaluate_policy_against_optimum(
        decision_set,
        data_set,
        objects.program,
        objects.decoder,
        objects.solver;
        optimal_results=optimal_results,
        split_name=:test,
        mu=MU,
        rho=RHO,
    )

    per_context_rows = [
        (;
            sample_index=row.sample_index,
            source_seed=source_keys[row.sample_index].seed,
            source_index=source_keys[row.sample_index].local_index,
            context=joined(data_set[row.sample_index].context),
            average_demand=joined(average_scenarios[row.sample_index].h_eq_xi),
            average_scenario_decision=joined(view(decision_set, :, row.sample_index)),
            average_scenario_policy_value=row.policy_value,
            stochastic_optimal_value=row.optimal_value,
            gap=row.regret,
            relative_gap=row.relative_regret,
            policy_collection_values=joined(row.policy_collection_values),
            optimal_collection_values=joined(row.optimal_collection_values),
            gap_values=joined(row.gap_values),
            gap_std=row.gap_std,
            gap_stderr=row.gap_stderr,
        ) for row in comparison.per_sample
    ]
    summary = metric_row(comparison.metrics, decision_solve_seconds, artifacts, data_set)

    CSV.write(joinpath(EXPERIMENT_DIR, "per_context.csv"), per_context_rows)
    CSV.write(joinpath(EXPERIMENT_DIR, "summary.csv"), [summary])
    write_report(joinpath(EXPERIMENT_DIR, "report.md"), summary)

    println("Wrote:")
    println("  ", joinpath(EXPERIMENT_DIR, "per_context.csv"))
    println("  ", joinpath(EXPERIMENT_DIR, "summary.csv"))
    println("  ", joinpath(EXPERIMENT_DIR, "report.md"))
    println()
    println("Mean gap: ", summary.test_regret_mean)
    println("Mean relative gap: ", summary.test_relative_regret_mean)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
