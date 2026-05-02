using ContextualDFL
using ContextualDFLExperiments
using Random
using Test

import ChainRulesCore
import ContextualDFLExperiments: infer

struct ConstantPolicy <: Policy
    z::Vector{Float64}
end

infer(policy::ConstantPolicy, context) = policy.z

struct TinyVectorDecoder <: ContextualDFL.VectorDecoder end

function (::TinyVectorDecoder)(vector::AbstractVector)
    return (
        reshape([1.0], 1, 1),
        zeros(0, 1),
        reshape([1.0], 1, 1),
        zeros(0, 1),
        [only(vector)],
        Float64[],
        [3.0],
    )
end

function tiny_program()
    return ContextualDFL.StochasticProgram(
        A_eq=reshape([1.0], 1, 1),
        A_ineq=zeros(0, 1),
        b_eq=[1.0],
        b_ineq=Float64[],
        c=[2.0],
    )
end

function tiny_scenario(h)
    return ContextualDFL.ParametricScenario(;
        W_eq_xi=reshape([1.0], 1, 1),
        W_ineq_xi=zeros(0, 1),
        T_eq_xi=reshape([1.0], 1, 1),
        T_ineq_xi=zeros(0, 1),
        h_eq_xi=[h],
        h_ineq_xi=Float64[],
        q_xi=[3.0],
    )
end

@testset "ContextualDFLExperiments" begin
    contexts = [[1.0], [2.0]]
    scenarios = [[tiny_scenario(5.0)], [tiny_scenario(6.0)]]
    data_set = generate_contextual_data_set(contexts, scenarios)

    @test length(data_set) == 2
    @test data_set[1] isa ContextualDFL.ContextualDataPoint
    @test data_set[1].context == [1.0]
    @test data_set[2].scenario_parameters[1].h_eq_xi == [6.0]

    decision_set = generate_decision_set(ConstantPolicy([1.0]), data_set)
    @test size(decision_set) == (1, 2)
    @test decision_set == reshape([1.0, 1.0], 1, 2)

    solver = ContextualDFL.Solver(ContextualDFL.IpoptSolver(), ContextualDFL.HiGHSSolver())
    program = tiny_program()
    decoder = ContextualDFL.ParametricDecoder()

    optimal_results = solve_dataset_to_optimality(data_set, program, decoder, solver)
    @test [result.z for result in optimal_results] == [[1.0], [1.0]]
    @test [result.objective_value for result in optimal_results] ≈ [14.0, 17.0]

    policy_values = evaluate_policy(decision_set, data_set, program, decoder, solver)
    @test policy_values ≈ [14.0, 17.0]

    value_summary = summarize_values([1.0, 2.0, 3.0]; prefix=:toy)
    @test value_summary.toy_count == 3
    @test value_summary.toy_mean ≈ 2.0
    @test value_summary.toy_median ≈ 2.0
    @test value_summary.toy_std ≈ 1.0
    @test value_summary.toy_min ≈ 1.0
    @test value_summary.toy_max ≈ 3.0
    @test value_summary.toy_p95 ≈ 3.0

    regret_summary = summarize_regret([15.0, 19.0], [14.0, 17.0]; prefix=:test)
    @test regret_summary.test_regret_mean ≈ 1.5
    @test regret_summary.test_relative_regret_mean ≈ ((1.0 / 14.0) + (2.0 / 17.0)) / 2

    comparison = evaluate_policy_against_optimum(
        decision_set,
        data_set,
        program,
        decoder,
        solver;
        optimal_results=optimal_results,
        split_name=:test,
    )
    @test comparison.optimal_results === optimal_results
    @test length(comparison.per_sample) == 2
    @test comparison.metrics.test_sample_count == 2
    @test comparison.metrics.test_policy_value_mean ≈ 15.5
    @test comparison.metrics.test_optimal_value_mean ≈ 15.5
    @test comparison.metrics.test_regret_mean ≈ 0.0
    @test comparison.metrics.test_relative_regret_mean ≈ 0.0
    @test comparison.metrics.test_optimal_solve_seconds == 0.0
    @test comparison.metrics.test_policy_eval_seconds >= 0.0

    generator = ContextualDFL.ScenarioGenerator(
        neural_net=context -> [context[1] + 4.0],
        scenario_decoder=TinyVectorDecoder(),
    )
    scenario_policy = ScenarioGenerationPolicy(generator, solver, program)
    @test infer(scenario_policy, [1.0]) ≈ [1.0]

    resource_data = default_resource_allocation_problem_data()
    @test size(resource_data.service_rate_parameters) == (20, 30)
    @test length(resource_data.first_stage_costs) == 20
    @test length(resource_data.second_stage_costs) == 30
    @test length(resource_data.yield_parameters) == 20

    small_resource_data = ResourceAllocationProblemData(
        resource_data.service_rate_parameters[1:2, 1:3],
        resource_data.first_stage_costs[1:2],
        resource_data.second_stage_costs[1:3],
        resource_data.yield_parameters[1:2],
    )
    resource_problem = ResourceAllocationProblem(small_resource_data)
    resource_base_scenario = base_scenario(resource_problem)
    @test size(resource_base_scenario.W_eq) == (5, 14)
    @test size(resource_base_scenario.W_ineq) == (14, 14)
    @test size(resource_base_scenario.T_eq) == (5, 2)
    @test resource_base_scenario.h_eq == zeros(5)
    @test resource_base_scenario.h_ineq == zeros(14)

    context_generator = ResourceAllocationContextDataGenerator(
        rng=Random.MersenneTwister(1),
    )
    resource_context = context_generator()
    @test length(resource_context) == 3
    @test all(>=(0.0), resource_context)

    scenario_generator = ResourceAllocationScenarioDataGenerator(
        resource_problem;
        sigma=0.0,
        p=1.0,
        L=3,
        rng=Random.MersenneTwister(2),
    )
    resource_scenario = scenario_generator(resource_context)
    @test resource_scenario.h_eq_xi isa Vector{Float64}
    @test length(resource_scenario.h_eq_xi) == 3
    @test isempty(resource_scenario.W_eq_xi)
    @test isempty(resource_scenario.h_ineq_xi)

    resource_vector_decoder = ResourceAllocationDemandVectorDecoder(resource_problem)
    _, _, _, _, vector_h_eq, _, _ = resource_vector_decoder(resource_scenario.h_eq_xi)
    @test vector_h_eq[1:2] == zeros(2)
    @test vector_h_eq[3:5] == resource_scenario.h_eq_xi

    resource_parametric_decoder = ResourceAllocationDemandParametricDecoder(resource_problem)
    _, _, _, _, h_eq, h_ineq, q = resource_parametric_decoder(resource_scenario)
    @test h_eq[1:2] == zeros(2)
    @test h_eq[3:5] == resource_scenario.h_eq_xi
    @test h_ineq == zeros(14)
    @test length(q) == 14

    second_resource_scenario = ContextualDFL.ParametricScenario(;
        W_eq_xi=Float64[],
        W_ineq_xi=Float64[],
        T_eq_xi=Float64[],
        T_ineq_xi=Float64[],
        h_eq_xi=2 .* resource_scenario.h_eq_xi,
        h_ineq_xi=Float64[],
        q_xi=Float64[],
    )
    resource_scenario_collection = [resource_scenario, second_resource_scenario]
    parametric_decoded = ContextualDFL.decode_scenario_collection(
        resource_parametric_decoder,
        resource_scenario_collection,
    )
    _, parametric_pullback = ChainRulesCore.rrule(
        ContextualDFL.decode_scenario_collection,
        resource_parametric_decoder,
        resource_scenario_collection,
    )
    parametric_dh_eq_cotangent = zeros(size(parametric_decoded[5]))
    parametric_dh_eq_cotangent[3:5, :] = [10.0 40.0; 20.0 50.0; 30.0 60.0]
    parametric_output_cotangent = ntuple(
        index -> index == 5 ? parametric_dh_eq_cotangent : zeros(size(parametric_decoded[index])),
        length(parametric_decoded),
    )
    parametric_tangents = parametric_pullback(parametric_output_cotangent)
    parametric_scenario_tangents = parametric_tangents[3]
    @test parametric_scenario_tangents[1].h_eq_xi == parametric_dh_eq_cotangent[3:5, 1]
    @test parametric_scenario_tangents[2].h_eq_xi == parametric_dh_eq_cotangent[3:5, 2]
    for scenario_tangent in parametric_scenario_tangents
        @test scenario_tangent.W_eq_xi isa ChainRulesCore.NoTangent
        @test scenario_tangent.W_ineq_xi isa ChainRulesCore.NoTangent
        @test scenario_tangent.T_eq_xi isa ChainRulesCore.NoTangent
        @test scenario_tangent.T_ineq_xi isa ChainRulesCore.NoTangent
        @test scenario_tangent.h_ineq_xi isa ChainRulesCore.NoTangent
        @test scenario_tangent.q_xi isa ChainRulesCore.NoTangent
    end

    resource_data_set = generate_contextual_data_set(
        [resource_context],
        [[resource_scenario]],
    )
    resource_results = solve_dataset_to_optimality(
        resource_data_set,
        stochastic_program(resource_problem),
        resource_parametric_decoder,
        solver,
    )
    @test length(resource_results) == 1
    @test length(only(resource_results).z) == 2
    @test isfinite(only(resource_results).objective_value)
end
