using ContextualDFL
using ContextualDFLExperiments
using Flux
using LinearAlgebra
using Random
using Test

import ChainRulesCore
import ContextualDFLExperiments: infer
import MathOptInterface

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

function shortage_program()
    return ContextualDFL.StochasticProgram(
        A_eq=zeros(0, 1),
        A_ineq=reshape([-1.0], 1, 1),
        b_eq=Float64[],
        b_ineq=[0.0],
        c=[1.0],
    )
end

function shortage_scenario(demand)
    return ContextualDFL.ParametricScenario(;
        W_eq_xi=zeros(0, 1),
        W_ineq_xi=reshape([-1.0, -1.0], 2, 1),
        T_eq_xi=zeros(0, 1),
        T_ineq_xi=reshape([-1.0, 0.0], 2, 1),
        h_eq_xi=Float64[],
        h_ineq_xi=[-Float64(demand), 0.0],
        q_xi=[10.0],
    )
end

function shortage_scenario_with_q(demand, q)
    return ContextualDFL.ParametricScenario(;
        W_eq_xi=zeros(0, 1),
        W_ineq_xi=reshape([-1.0, -1.0], 2, 1),
        T_eq_xi=zeros(0, 1),
        T_ineq_xi=reshape([-1.0, 0.0], 2, 1),
        h_eq_xi=Float64[],
        h_ineq_xi=[-Float64(demand), 0.0],
        q_xi=[Float64(q)],
    )
end

function lex_tie_program()
    return ContextualDFL.StochasticProgram(
        A_eq=zeros(0, 1),
        A_ineq=reshape([-1.0, 1.0], 2, 1),
        b_eq=Float64[],
        b_ineq=[0.0, 1.0],
        c=[0.0],
    )
end

function lex_tie_scenario()
    return ContextualDFL.ParametricScenario(;
        W_eq_xi=reshape([1.0], 1, 1),
        W_ineq_xi=zeros(0, 1),
        T_eq_xi=zeros(1, 1),
        T_ineq_xi=zeros(0, 1),
        h_eq_xi=[0.0],
        h_ineq_xi=Float64[],
        q_xi=[0.0],
    )
end

function fixed_two_recourse_program()
    return ContextualDFL.StochasticProgram(
        A_eq=reshape([1.0], 1, 1),
        A_ineq=zeros(0, 1),
        b_eq=[1.0],
        b_ineq=Float64[],
        c=[2.0],
    )
end

function fixed_two_recourse_scenario()
    return ContextualDFL.ParametricScenario(;
        W_eq_xi=Matrix{Float64}(I, 2, 2),
        W_ineq_xi=-Matrix{Float64}(I, 2, 2),
        T_eq_xi=reshape([1.0, 0.0], 2, 1),
        T_ineq_xi=zeros(2, 1),
        h_eq_xi=[4.0, 5.0],
        h_ineq_xi=zeros(2),
        q_xi=[7.0, 11.0],
    )
end

function small_resource_allocation_ad_problem()
    data = ResourceAllocationProblemData(
        [1.0 0.8 1.2; 0.7 1.1 0.9],
        [1.0, 1.2],
        [3.0, 4.0, 5.0],
        [1.0, 1.0],
    )
    return data, ResourceAllocationProblem(data)
end

function assert_decoded_shapes_match_base(arrays, base)
    @test size(arrays[1]) == (size(base.W_eq)..., 1)
    @test size(arrays[2]) == (size(base.W_ineq)..., 1)
    @test size(arrays[3]) == (size(base.T_eq)..., 1)
    @test size(arrays[4]) == (size(base.T_ineq)..., 1)
    @test size(arrays[5]) == (length(base.h_eq), 1)
    @test size(arrays[6]) == (length(base.h_ineq), 1)
    @test size(arrays[7]) == (length(base.q), 1)
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
    @test [result.objective_values for result in optimal_results] ≈ [[14.0], [17.0]]
    @test [result.evaluation_batches for result in optimal_results] == [1, 1]
    @test [result.objective_value for result in optimal_results] ≈ [14.0, 17.0]

    policy_values = evaluate_policy(decision_set, data_set, program, decoder, solver)
    @test policy_values ≈ [14.0, 17.0]

    rho_optimal_results = solve_dataset_to_optimality(
        data_set,
        program,
        decoder,
        solver;
        rho=0.5,
    )
    rho_policy_values = evaluate_policy(
        decision_set,
        data_set,
        program,
        decoder,
        solver;
        rho=0.5,
    )
    @test [result.objective_value for result in rho_optimal_results] ≈ [18.25, 23.5]
    @test rho_policy_values ≈ [18.25, 23.5]

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
    @test comparison.metrics.test_evaluation_batches == 1
    @test comparison.metrics.test_policy_value_mean ≈ 15.5
    @test comparison.metrics.test_optimal_value_mean ≈ 15.5
    @test comparison.metrics.test_regret_mean ≈ 0.0
    @test comparison.metrics.test_relative_regret_mean ≈ 0.0
    @test comparison.metrics.test_gap_std_mean ≈ 0.0
    @test comparison.metrics.test_policy_eval_seconds >= 0.0

    split_data_set = generate_contextual_data_set(
        [[1.0]],
        [[tiny_scenario(5.0), tiny_scenario(6.0)]],
    )
    split_decision_set = reshape([1.0], 1, 1)
    split_optimal_results = solve_dataset_to_optimality(
        split_data_set,
        program,
        decoder,
        solver;
        evaluation_batches=2,
    )
    split_result = only(split_optimal_results)
    @test split_result.objective_values ≈ [14.0, 17.0]
    @test split_result.objective_value ≈ 15.5
    @test split_result.evaluation_batches == 2

    split_policy_values = evaluate_policy(
        split_decision_set,
        split_data_set,
        program,
        decoder,
        solver;
        evaluation_batches=2,
    )
    @test split_policy_values ≈ [15.5]

    split_comparison = evaluate_policy_against_optimum(
        split_decision_set,
        split_data_set,
        program,
        decoder,
        solver;
        optimal_results=split_optimal_results,
        split_name=:test,
    )
    @test split_comparison.metrics.test_evaluation_batches == 2
    @test only(split_comparison.per_sample).policy_collection_values ≈ [14.0, 17.0]
    @test only(split_comparison.per_sample).optimal_collection_values ≈ [14.0, 17.0]
    @test only(split_comparison.per_sample).gap_values ≈ [0.0, 0.0]
    @test only(split_comparison.per_sample).gap_std ≈ 0.0
    @test split_comparison.metrics.test_regret_mean ≈ 0.0

    partition_data_set = generate_contextual_data_set(
        [[0.0]],
        [[tiny_scenario(Float64(index)) for index in 1:10000]],
    )
    partition_ranges =
        ContextualDFLExperiments._scenario_collection_ranges(first(partition_data_set), 50)
    partition_indices = collect(Iterators.flatten(partition_ranges))
    @test length(partition_ranges) == 50
    @test all(==(200), length.(partition_ranges))
    @test first(partition_ranges) == 1:200
    @test last(partition_ranges) == 9801:10000
    @test partition_indices == collect(1:10000)
    @test length(unique(partition_indices)) == 10000

    replication_data_set = generate_contextual_data_set(
        [[0.0]],
        [[shortage_scenario(2.0), shortage_scenario(8.0)]],
    )
    replication_decision_set = reshape([8.0], 1, 1)
    replication_optimal_results = solve_dataset_to_optimality(
        replication_data_set,
        shortage_program(),
        ContextualDFL.ParametricDecoder(),
        solver;
        evaluation_batches=2,
    )
    @test only(replication_optimal_results).objective_values ≈ [2.0, 8.0] atol = 1e-6
    @test only(replication_optimal_results).objective_value ≈ 5.0 atol = 1e-6

    replication_comparison = evaluate_policy_against_optimum(
        replication_decision_set,
        replication_data_set,
        shortage_program(),
        ContextualDFL.ParametricDecoder(),
        solver;
        optimal_results=replication_optimal_results,
        split_name=:test,
    )
    @test only(replication_comparison.per_sample).policy_collection_values ≈
        [8.0, 8.0] atol = 1e-6
    @test only(replication_comparison.per_sample).gap_values ≈ [6.0, 0.0] atol = 1e-6
    @test only(replication_comparison.per_sample).regret ≈ 3.0 atol = 1e-6
    @test only(replication_comparison.per_sample).gap_stderr ≈ 3.0 atol = 1e-6
    @test minimum(only(replication_comparison.per_sample).gap_values) >= -1e-5

    replication_full_optimal_results = solve_dataset_to_optimality(
        replication_data_set,
        shortage_program(),
        ContextualDFL.ParametricDecoder(),
        solver;
        evaluation_batches=1,
    )
    @test only(replication_optimal_results).objective_value <=
          only(replication_full_optimal_results).objective_value + 1e-6
    @test evaluate_policy(
        replication_decision_set,
        replication_data_set,
        shortage_program(),
        ContextualDFL.ParametricDecoder(),
        solver;
        evaluation_batches=2,
    ) ≈ evaluate_policy(
        replication_decision_set,
        replication_data_set,
        shortage_program(),
        ContextualDFL.ParametricDecoder(),
        solver;
        evaluation_batches=1,
    ) atol = 1e-6

    @test_throws ArgumentError evaluate_policy_against_optimum(
        replication_decision_set,
        replication_data_set,
        shortage_program(),
        ContextualDFL.ParametricDecoder(),
        solver;
        optimal_results=[(; objective_values=[9.0, 9.0], objective_value=9.0)],
        split_name=:test,
    )

    @test_throws ArgumentError solve_dataset_to_optimality(
        split_data_set,
        program,
        decoder,
        solver;
        evaluation_batches=3,
    )
    @test_throws ArgumentError evaluate_policy_against_optimum(
        decision_set,
        data_set,
        program,
        decoder,
        solver;
        optimal_results=[
            (; objective_values=[1.0, 2.0], objective_value=1.5),
            (; objective_values=[1.0], objective_value=1.0),
        ],
        split_name=:test,
    )

    @test_throws ArgumentError evaluate_policy_against_optimum(
        decision_set,
        data_set,
        program,
        decoder,
        solver;
        optimal_results=[
            (; objective_values=[14.0], objective_value=15.0),
            (; objective_values=[17.0], objective_value=17.0),
        ],
        split_name=:test,
    )

    @test_throws UndefKeywordError evaluate_policy_against_optimum(
        decision_set,
        data_set,
        program,
        decoder,
        solver;
        split_name=:test,
    )

    generator = ContextualDFL.ScenarioGenerator(
        neural_net=context -> [context[1] + 4.0],
        scenario_decoder=TinyVectorDecoder(),
    )
    scenario_policy = ScenarioGenerationPolicy(generator, solver, program)
    rho_scenario_policy = ScenarioGenerationPolicy(generator, solver, program; mu=0.1, rho=0.2)
    @test infer(scenario_policy, [1.0]) ≈ [1.0]
    @test rho_scenario_policy.mu == 0.1
    @test rho_scenario_policy.rho == 0.2
    @test infer(rho_scenario_policy, [1.0]) ≈ [1.0]

    shortage_data_set = generate_contextual_data_set(
        [[0.0], [10.0]],
        [[shortage_scenario(2.0)], [shortage_scenario(8.0)]],
    )
    shortage_decoder = ContextualDFL.ParametricDecoder()
    shortage_policy = SampleAverageApproximationPolicy(
        shortage_data_set,
        solver,
        shortage_program(),
        shortage_decoder,
    )
    @test infer(shortage_policy, [100.0]) ≈ [8.0] atol = 1e-6
    @test generate_decision_set(shortage_policy, shortage_data_set) ≈
        reshape([8.0, 8.0], 1, 2) atol = 1e-6

    direct_shortage_policy = SampleAverageApproximationPolicy(
        [shortage_scenario(2.0)],
        solver,
        shortage_program(),
        shortage_decoder,
    )
    @test infer(direct_shortage_policy, [100.0]) ≈ [2.0] atol = 1e-6

    @test default_knn_k(100) == 32
    knn_policy = KNearestNeighborsPolicy(
        shortage_data_set,
        solver,
        shortage_program(),
        shortage_decoder;
        k=1,
    )
    @test infer(knn_policy, [0.1]) ≈ [2.0] atol = 1e-6
    @test infer(knn_policy, [9.9]) ≈ [8.0] atol = 1e-6
    @test_throws ArgumentError KNearestNeighborsPolicy(
        shortage_data_set,
        solver,
        shortage_program(),
        shortage_decoder;
        k=0,
    )
    @test_throws DimensionMismatch infer(knn_policy, [1.0, 2.0])

    ad_tree_data_set = generate_contextual_data_set(
        [[-2.0], [-1.0], [1.0], [2.0]],
        [
            [shortage_scenario(2.0)],
            [shortage_scenario(2.0)],
            [shortage_scenario(8.0)],
            [shortage_scenario(8.0)],
        ],
    )
    ad_tree_policy = AdaptiveDecisionTreePolicy(
        ad_tree_data_set,
        solver,
        shortage_program(),
        shortage_decoder;
        depth=1,
        min_leaf=1,
    )
    @test ad_tree_policy.metadata.depth == 1
    @test ad_tree_policy.metadata.min_leaf == 1
    @test ad_tree_policy.metadata.termination_status == MathOptInterface.OPTIMAL
    @test infer(ad_tree_policy, [-2.0]) ≈ [2.0] atol = 1e-6
    @test infer(ad_tree_policy, [2.0]) ≈ [8.0] atol = 1e-6
    @test generate_decision_set(ad_tree_policy, ad_tree_data_set) ≈
        reshape([2.0, 2.0, 8.0, 8.0], 1, 4) atol = 1e-6
    @test_throws ArgumentError AdaptiveDecisionTreePolicy(
        ad_tree_data_set,
        solver,
        shortage_program(),
        shortage_decoder;
        depth=2,
        min_leaf=2,
    )
    @test_throws DimensionMismatch infer(ad_tree_policy, [1.0, 2.0])

    residual_data_set = generate_contextual_data_set(
        [[-1.0], [1.0]],
        [
            [shortage_scenario(3.0), shortage_scenario(5.0)],
            [shortage_scenario(5.0), shortage_scenario(7.0)],
        ],
    )
    least_squares_policy = LeastSquaresPolicy(
        residual_data_set,
        solver,
        shortage_program(),
        shortage_decoder;
        target_component=:h_ineq_xi,
    )
    @test size(least_squares_policy.coefficients) == (2, 2)
    @test least_squares_policy.coefficients ≈ [-1.0 0.0; -5.0 0.0]
    @test infer(least_squares_policy, [2.0]) ≈ [7.0] atol = 1e-6
    @test generate_decision_set(least_squares_policy, residual_data_set) ≈
        reshape([4.0, 6.0], 1, 2) atol = 1e-6

    residual_policy = ResidualSampleAverageApproximationPolicy(
        residual_data_set,
        solver,
        shortage_program(),
        shortage_decoder;
        target_component=:h_ineq_xi,
    )
    @test size(residual_policy.residuals) == (4, 2)
    @test residual_policy.residuals[:, 1] ≈ [1.0, -1.0, 1.0, -1.0]
    @test residual_policy.residuals[:, 2] ≈ zeros(4)
    @test infer(residual_policy, [2.0]) ≈ [8.0] atol = 1e-6

    ad_no_opt_policy = DecisionFocusedLinearPolicy(
        residual_data_set,
        solver,
        shortage_program(),
        shortage_decoder;
        target_component=:h_ineq_xi,
        optimize=false,
    )
    @test ad_no_opt_policy.initial_coefficients ≈ least_squares_policy.coefficients
    @test ad_no_opt_policy.coefficients ≈ least_squares_policy.coefficients
    @test isnothing(ad_no_opt_policy.optimization_result)
    @test infer(ad_no_opt_policy, [2.0]) ≈ infer(least_squares_policy, [2.0]) atol = 1e-6

    lex_policy = LexSPOLinearPolicy(
        residual_data_set,
        solver,
        shortage_program(),
        shortage_decoder;
        target_component=:h_ineq_xi,
        optimize=false,
    )
    @test lex_policy.initial_coefficients ≈ least_squares_policy.coefficients
    @test lex_policy.coefficients ≈ least_squares_policy.coefficients
    @test isnothing(lex_policy.optimization_result)
    @test infer(lex_policy, [2.0]) ≈ infer(least_squares_policy, [2.0]) atol = 1e-6

    @test_throws ArgumentError LexSPOLinearPolicy(
        residual_data_set,
        solver,
        shortage_program(),
        shortage_decoder;
        target_component=:q_xi,
    )
    @test_throws ArgumentError LexSPOLinearPolicy(
        residual_data_set,
        solver,
        shortage_program(),
        shortage_decoder;
        target_component=:h_ineq_xi,
        mu=0.1,
    )
    @test_throws ArgumentError LexSPOLinearPolicy(
        residual_data_set,
        solver,
        shortage_program(),
        shortage_decoder;
        target_component=:h_ineq_xi,
        rho=0.1,
    )

    z_lex = ContextualDFLExperiments._lex_solve_scenario_collection(
        solver,
        lex_tie_program(),
        ContextualDFL.ParametricDecoder(),
        [lex_tie_scenario()],
    )
    @test only(z_lex) ≈ 0.0 atol = 1e-6

    penalty_transform = nonnegative_prediction_penalty_transform(
        lower_bound=0.0,
        penalty_weight=10.0,
    )
    penalty_result = penalty_transform([-2.0, 3.0])
    @test penalty_result.target ≈ [0.0, 3.0]
    @test penalty_result.penalty ≈ 20.0

    clipped_policy = LeastSquaresPolicy(
        residual_data_set,
        solver,
        shortage_program(),
        shortage_decoder;
        target_component=:h_ineq_xi,
        postprocess_prediction=target -> max.(target, [-6.0, 0.0]),
    )
    @test infer(clipped_policy, [2.0]) ≈ [6.0] atol = 1e-6

    @test_throws ArgumentError LeastSquaresPolicy(
        residual_data_set,
        solver,
        shortage_program(),
        shortage_decoder;
        target_component=:not_a_component,
    )
    bad_postprocess_policy = LeastSquaresPolicy(
        residual_data_set,
        solver,
        shortage_program(),
        shortage_decoder;
        target_component=:h_ineq_xi,
        postprocess_prediction=target -> target[1:1],
    )
    @test_throws DimensionMismatch infer(bad_postprocess_policy, [2.0])
    @test_throws DimensionMismatch infer(least_squares_policy, [1.0, 2.0])

    varying_structure_data_set = generate_contextual_data_set(
        [[0.0], [1.0]],
        [[shortage_scenario_with_q(2.0, 10.0)], [shortage_scenario_with_q(3.0, 12.0)]],
    )
    @test_throws ArgumentError LeastSquaresPolicy(
        varying_structure_data_set,
        solver,
        shortage_program(),
        shortage_decoder;
        target_component=:h_ineq_xi,
    )

    cart_policy = CARTPolicy(
        residual_data_set,
        solver,
        shortage_program(),
        shortage_decoder;
        target_component=:h_ineq_xi,
        min_samples_leaf=1,
        test_size=nothing,
    )
    @test cart_policy.metadata.min_samples_leaf == 1
    @test cart_policy.metadata.effective_min_samples_leaf == 1
    @test isnothing(cart_policy.metadata.score)
    @test infer(cart_policy, [-1.0]) ≈ [4.0] atol = 1e-6
    @test infer(cart_policy, [1.0]) ≈ [6.0] atol = 1e-6

    cart_split_policy = CARTPolicy(
        residual_data_set,
        solver,
        shortage_program(),
        shortage_decoder;
        target_component=:h_ineq_xi,
        min_samples_leaf=25,
        test_size=0.25,
    )
    @test cart_split_policy.metadata.effective_min_samples_leaf <=
          cart_split_policy.metadata.min_samples_leaf
    @test !isnothing(cart_split_policy.metadata.score)

    cart_clipped_policy = CARTPolicy(
        residual_data_set,
        solver,
        shortage_program(),
        shortage_decoder;
        target_component=:h_ineq_xi,
        min_samples_leaf=1,
        test_size=nothing,
        postprocess_prediction=target -> max.(target, [-3.0, 0.0]),
    )
    @test infer(cart_clipped_policy, [-1.0]) ≈ [3.0] atol = 1e-6

    m5ad_policy = M5ADPolicy(
        residual_data_set,
        solver,
        shortage_program(),
        shortage_decoder;
        target_component=:h_ineq_xi,
        min_samples_leaf=1,
        test_size=nothing,
        optimize=false,
    )
    @test m5ad_policy.metadata.leaf_count >= 1
    @test all(>(0), values(m5ad_policy.metadata.leaf_sample_counts))
    @test infer(m5ad_policy, [-1.0]) ≈ infer(least_squares_policy, [-1.0]) atol = 1e-6
    @test infer(m5ad_policy, [1.0]) ≈ infer(least_squares_policy, [1.0]) atol = 1e-6
    @test generate_decision_set(m5ad_policy, residual_data_set) ≈
        generate_decision_set(least_squares_policy, residual_data_set) atol = 1e-6
    @test_throws DimensionMismatch infer(m5ad_policy, [1.0, 2.0])

    @test_throws ArgumentError CARTPolicy(
        residual_data_set,
        solver,
        shortage_program(),
        shortage_decoder;
        target_component=:h_ineq_xi,
        min_samples_leaf=0,
    )
    @test_throws ArgumentError CARTPolicy(
        residual_data_set,
        solver,
        shortage_program(),
        shortage_decoder;
        target_component=:h_ineq_xi,
        test_size=1.5,
    )
    @test_throws ArgumentError CARTPolicy(
        varying_structure_data_set,
        solver,
        shortage_program(),
        shortage_decoder;
        target_component=:h_ineq_xi,
    )

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

    predicted_demand = vcat(resource_scenario.h_eq_xi, 2 .* resource_scenario.h_eq_xi)
    @test_throws ArgumentError ChainRulesCore.rrule(
        ContextualDFL.decode_scenario_collection,
        resource_vector_decoder,
        predicted_demand,
    )
    @test_throws DimensionMismatch ChainRulesCore.rrule(
        ContextualDFL.decode_scenario_collection,
        resource_vector_decoder,
        predicted_demand[1:(end - 1)];
        nr_scenarios=2,
    )

    decoded = ContextualDFL.decode_scenario_collection(
        resource_vector_decoder,
        predicted_demand;
        nr_scenarios=2,
    )
    _, vector_pullback = ChainRulesCore.rrule(
        ContextualDFL.decode_scenario_collection,
        resource_vector_decoder,
        predicted_demand;
        nr_scenarios=2,
    )
    dh_eq_cotangent = zeros(size(decoded[5]))
    dh_eq_cotangent[3:5, :] = [1.0 4.0; 2.0 5.0; 3.0 6.0]
    output_cotangent = ntuple(
        index -> index == 5 ? dh_eq_cotangent : zeros(size(decoded[index])),
        length(decoded),
    )
    vector_tangents =
        vector_pullback(ChainRulesCore.Tangent{typeof(decoded)}(output_cotangent...))
    @test vector_tangents[3] == vec(dh_eq_cotangent[3:5, :])
    @test vector_tangents[3][1] == dh_eq_cotangent[3, 1]
    @test vector_tangents[3][4] == dh_eq_cotangent[3, 2]

    zero_vector_tangents = vector_pullback(ChainRulesCore.ZeroTangent())
    @test zero_vector_tangents[3] == zeros(length(predicted_demand))
    @test zero_vector_tangents[3] isa Vector{Float64}

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

    @testset "first-pass vector decoders" begin
        @testset "resource allocation q decoders" begin
            data, problem = small_resource_allocation_ad_problem()
            base = base_scenario(problem)
            I, J = size(data.service_rate_parameters)

            original_epsilon = 0.25
            original_decoder = ResourceAllocationOriginalCostVectorDecoder(
                problem;
                epsilon=original_epsilon,
                scale=2.0,
            )
            original_raw = collect(range(-1.0, 1.0; length=J))
            original_arrays = ContextualDFL.decode_scenario_collection(
                original_decoder,
                original_raw;
                nr_scenarios=1,
            )
            assert_decoded_shapes_match_base(original_arrays, base)
            original_q = original_arrays[7][:, 1]
            @test all(original_q[1:J] .>= original_epsilon)
            @test original_q[(J + 1):end] == base.q[(J + 1):end]
            @test original_arrays[5][:, 1] == base.h_eq

            economic_epsilon = 0.1
            economic_decoder = ResourceAllocationEconomicCostVectorDecoder(
                problem;
                epsilon=economic_epsilon,
                allocation_scale=1.5,
                unmet_scale=2.0,
            )
            economic_raw = collect(range(-2.0, 2.0; length=J + I * J))
            economic_arrays = ContextualDFL.decode_scenario_collection(
                economic_decoder,
                economic_raw;
                nr_scenarios=1,
            )
            assert_decoded_shapes_match_base(economic_arrays, base)
            economic_q = economic_arrays[7][:, 1]
            @test all(economic_q[1:J] .>= economic_epsilon)
            for i in 1:I, j in 1:J
                q_index = J + J * (i - 1) + j
                lower = -data.first_stage_costs[i] / data.yield_parameters[i]
                @test economic_q[q_index] >= lower + economic_epsilon
            end
            slack_start = J + I * J + 1
            @test economic_q[slack_start:end] == base.q[slack_start:end]
            @test economic_arrays[5][:, 1] == base.h_eq
        end

        @testset "shipment planning decoders" begin
            problem = ShipmentPlanningProblem()
            base = base_scenario(problem)
            I = problem.warehouse_count
            J = problem.demand_count
            shipment_range = (I + 1):(I + I * J)
            demand_slack_range = (I + I * J + 1):(I + I * J + J)
            supply_slack_range = (I + I * J + J + 1):(I + I * J + J + I)

            demand_decoder = ShipmentPlanningDemandVectorDecoder(problem)
            demand_raw = collect(range(-1.0, 1.0; length=J))
            demand_arrays = ContextualDFL.decode_scenario_collection(
                demand_decoder,
                demand_raw;
                nr_scenarios=1,
            )
            assert_decoded_shapes_match_base(demand_arrays, base)
            @test demand_arrays[5][1:J, 1] ≈ problem.demand_intercepts .+ 10.0 .* demand_raw
            @test demand_arrays[5][(J + 1):end, 1] == base.h_eq[(J + 1):end]
            @test demand_arrays[7][:, 1] == base.q

            positive_demand_epsilon = 0.2
            positive_demand_decoder = ShipmentPlanningPositiveDemandVectorDecoder(
                problem;
                epsilon=positive_demand_epsilon,
            )
            positive_demand_arrays = ContextualDFL.decode_scenario_collection(
                positive_demand_decoder,
                demand_raw;
                nr_scenarios=1,
            )
            assert_decoded_shapes_match_base(positive_demand_arrays, base)
            @test all(positive_demand_arrays[5][1:J, 1] .>= positive_demand_epsilon)
            @test positive_demand_arrays[5][(J + 1):end, 1] == base.h_eq[(J + 1):end]
            @test positive_demand_arrays[7][:, 1] == base.q

            shipping_raw = collect(range(-1.0, 1.0; length=I * J))
            positive_shipping_epsilon = 0.3
            positive_shipping_decoder = ShipmentPlanningPositiveShippingCostVectorDecoder(
                problem;
                epsilon=positive_shipping_epsilon,
                scale=2.0,
            )
            positive_shipping_arrays = ContextualDFL.decode_scenario_collection(
                positive_shipping_decoder,
                shipping_raw;
                nr_scenarios=1,
            )
            assert_decoded_shapes_match_base(positive_shipping_arrays, base)
            positive_shipping_q = positive_shipping_arrays[7][:, 1]
            @test all(positive_shipping_q[shipment_range] .>= positive_shipping_epsilon)
            @test positive_shipping_q[1:I] == base.q[1:I]
            @test positive_shipping_q[demand_slack_range] == base.q[demand_slack_range]
            @test positive_shipping_q[supply_slack_range] == base.q[supply_slack_range]
            @test positive_shipping_arrays[5][:, 1] == base.h_eq

            economic_shipping_epsilon = 0.4
            economic_shipping_decoder = ShipmentPlanningEconomicShippingCostVectorDecoder(
                problem;
                epsilon=economic_shipping_epsilon,
                scale=1.5,
            )
            economic_shipping_arrays = ContextualDFL.decode_scenario_collection(
                economic_shipping_decoder,
                shipping_raw;
                nr_scenarios=1,
            )
            assert_decoded_shapes_match_base(economic_shipping_arrays, base)
            economic_shipping_q = economic_shipping_arrays[7][:, 1]
            for j in 1:J, i in 1:I
                q_index = I + (j - 1) * I + i
                lower = -min(stochastic_program(problem).c[i], base.q[i])
                @test economic_shipping_q[q_index] >= lower + economic_shipping_epsilon
            end
            @test economic_shipping_q[1:I] == base.q[1:I]
            @test economic_shipping_q[demand_slack_range] == base.q[demand_slack_range]
            @test economic_shipping_q[supply_slack_range] == base.q[supply_slack_range]
            @test economic_shipping_arrays[5][:, 1] == base.h_eq
        end

        @testset "transshipment positive decoders" begin
            problem = TransShipmentExperimentProblem()
            base = base_scenario(problem)
            mean_parameters = ContextualDFL.transshipment_mean_parameters(problem.core_problem)
            h_indices = [entry.index for entry in problem.core_problem.data.random_rhs_entries]
            q_indices = [entry.index for entry in problem.core_problem.data.random_objective_entries]

            q_decoder = TransShipmentPositiveQVectorDecoder(problem; epsilon=0.2, scale=1.5)
            q_raw = collect(range(-1.0, 1.0; length=length(mean_parameters.q)))
            q_arrays = ContextualDFL.decode_scenario_collection(
                q_decoder,
                q_raw;
                nr_scenarios=1,
            )
            assert_decoded_shapes_match_base(q_arrays, base)
            q = q_arrays[7][:, 1]
            @test all(q[q_indices] .> 0.0)
            @test q[setdiff(1:length(q), q_indices)] == base.q[setdiff(1:length(q), q_indices)]
            @test q_arrays[5][:, 1] == base.h_eq

            h_decoder = TransShipmentPositiveHVectorDecoder(problem; epsilon=0.2, scale=1.5)
            h_raw = collect(range(-1.0, 1.0; length=length(mean_parameters.rhs)))
            h_arrays = ContextualDFL.decode_scenario_collection(
                h_decoder,
                h_raw;
                nr_scenarios=1,
            )
            assert_decoded_shapes_match_base(h_arrays, base)
            h = h_arrays[5][:, 1]
            @test all(h[h_indices] .> 0.0)
            @test h[setdiff(1:length(h), h_indices)] == base.h_eq[setdiff(1:length(h), h_indices)]
            @test h_arrays[7][:, 1] == base.q

            hq_decoder = TransShipmentPositiveHQVectorDecoder(problem; epsilon_h=0.2, epsilon_q=0.3)
            hq_raw = vcat(h_raw, q_raw)
            hq_arrays = ContextualDFL.decode_scenario_collection(
                hq_decoder,
                hq_raw;
                nr_scenarios=1,
            )
            assert_decoded_shapes_match_base(hq_arrays, base)
            @test all(hq_arrays[5][h_indices, 1] .> 0.0)
            @test all(hq_arrays[7][q_indices, 1] .> 0.0)
        end

        @testset "random yield decoders" begin
            problem = RandomYieldProblem(; r=5, a=10, K_support=5)
            base = base_scenario(problem)

            q_decoder = RandomYieldPositiveQVectorDecoder(problem)
            q_raw = collect(range(-1.0, 1.0; length=length(base.q)))
            q_arrays = ContextualDFL.decode_scenario_collection(
                q_decoder,
                q_raw;
                nr_scenarios=1,
            )
            assert_decoded_shapes_match_base(q_arrays, base)
            @test q_arrays[1][:, :, 1] == base.W_eq
            @test q_arrays[5][:, 1] == base.h_eq
            @test all(q_arrays[7][:, 1] .> 0.0)

            h_decoder = RandomYieldHVectorDecoder(problem)
            h_raw = collect(range(-1.0, 1.0; length=length(base.h_eq)))
            h_arrays = ContextualDFL.decode_scenario_collection(
                h_decoder,
                h_raw;
                nr_scenarios=1,
            )
            assert_decoded_shapes_match_base(h_arrays, base)
            @test h_arrays[1][:, :, 1] == base.W_eq
            @test h_arrays[5][:, 1] ≈ base.h_eq .+ h_raw
            @test h_arrays[7][:, 1] == base.q
        end

        @testset "unreliable newsvendor decoder" begin
            problem = UnreliableNewsvendorProblem()
            base = base_scenario(problem)
            decoder = UnreliableNewsvendorParameterVectorDecoder(problem)
            arrays = ContextualDFL.decode_scenario_collection(
                decoder,
                [0.0, 0.0];
                nr_scenarios=1,
            )
            assert_decoded_shapes_match_base(arrays, base)
            @test vec(arrays[3][:, :, 1]) ≈ [0.0, -0.5]
            @test arrays[5][:, 1] ≈ [-0.5, 0.0]
            @test arrays[7][:, 1] == base.q
            @test_throws DimensionMismatch ContextualDFL.decode_scenario_collection(
                decoder,
                [0.0, 0.0, 0.0];
                nr_scenarios=1,
            )
        end
    end

    @testset "ResourceAllocationDemandVectorDecoder real AD" begin
        data, problem = small_resource_allocation_ad_problem()
        decoder = ResourceAllocationDemandVectorDecoder(problem)

        resource_count = size(data.service_rate_parameters, 1)
        demand_count = size(data.service_rate_parameters, 2)
        K = 2

        demand = collect(1.0:(demand_count * K))
        H = reshape(
            collect(1.0:((resource_count + demand_count) * K)),
            resource_count + demand_count,
            K,
        )

        f(d) = begin
            _, _, _, _, h_eq_array, _, _ =
                ContextualDFL.decode_scenario_collection(decoder, d; nr_scenarios=K)
            return sum(h_eq_array .* H)
        end

        g = only(Flux.gradient(f, demand))
        expected = vec(H[(resource_count + 1):(resource_count + demand_count), :])

        @test g ≈ expected atol = 1e-10 rtol = 1e-10
        @test !all(iszero, g)
    end

    @testset "solve rrule real AD matches finite difference wrt h_eq_array" begin
        data, problem = small_resource_allocation_ad_problem()
        solver = ContextualDFL.Solver(ContextualDFL.IpoptSolver(), ContextualDFL.HiGHSSolver())
        decoder = ResourceAllocationDemandVectorDecoder(problem)

        K = 1
        demand = [5.0, 6.0, 7.0]
        W_eq, W_ineq, T_eq, T_ineq, h_eq, h_ineq, q =
            ContextualDFL.decode_scenario_collection(decoder, demand; nr_scenarios=K)

        μ = 0.25
        z_weight = [0.3, -0.7]

        f(h_candidate) = begin
            z, _, _, _, _, _ = ContextualDFL.solve(
                solver,
                stochastic_program(problem),
                W_eq,
                W_ineq,
                T_eq,
                T_ineq,
                h_candidate,
                h_ineq,
                q;
                μ=μ,
                tol=1e-9,
            )
            return dot(z_weight, z)
        end

        g = only(Flux.gradient(f, h_eq))

        direction = zeros(size(h_eq))
        direction[(size(data.service_rate_parameters, 1) + 1):end, :] .= [0.4, -0.2, 0.3]

        ϵ = 1e-4
        fd = (f(h_eq .+ ϵ .* direction) - f(h_eq .- ϵ .* direction)) / (2ϵ)

        @test abs(fd) > 1e-8
        @test sum(g .* direction) ≈ fd atol = 3e-3 rtol = 3e-2
    end

    @testset "predicted demand gradient through decode and solve is nonzero and correct" begin
        _, problem = small_resource_allocation_ad_problem()
        solver = ContextualDFL.Solver(ContextualDFL.IpoptSolver(), ContextualDFL.HiGHSSolver())
        decoder = ResourceAllocationDemandVectorDecoder(problem)

        K = 2
        demand = [5.0, 6.0, 7.0, 4.5, 6.5, 8.0]
        μ = 0.25
        z_weight = [0.3, -0.7]

        f(d) = begin
            W_eq, W_ineq, T_eq, T_ineq, h_eq, h_ineq, q =
                ContextualDFL.decode_scenario_collection(decoder, d; nr_scenarios=K)

            z, _, _, _, _, _ = ContextualDFL.solve(
                solver,
                stochastic_program(problem),
                W_eq,
                W_ineq,
                T_eq,
                T_ineq,
                h_eq,
                h_ineq,
                q;
                μ=μ,
                tol=1e-9,
            )

            return dot(z_weight, z)
        end

        g = only(Flux.gradient(f, demand))

        direction = [0.1, -0.2, 0.3, -0.4, 0.2, 0.1]
        direction ./= norm(direction)

        ϵ = 1e-4
        fd = (f(demand .+ ϵ .* direction) - f(demand .- ϵ .* direction)) / (2ϵ)

        @test abs(fd) > 1e-8
        @test !all(iszero, g)
        @test dot(g, direction) ≈ fd atol = 3e-3 rtol = 3e-2
    end

    @testset "DflScenLoss gradient wrt predicted demand matches finite difference" begin
        _, problem = small_resource_allocation_ad_problem()
        solver = ContextualDFL.Solver(ContextualDFL.IpoptSolver(), ContextualDFL.HiGHSSolver())

        input_decoder = ResourceAllocationDemandVectorDecoder(problem)
        reference_decoder = ResourceAllocationDemandParametricDecoder(problem)

        K = 2
        loss = ContextualDFL.DflScenLoss(
            input_decoder,
            reference_decoder,
            solver,
            stochastic_program(problem);
            nr_scenarios=K,
        )

        predicted_demand = [5.0, 6.0, 7.0, 4.5, 6.5, 8.0]
        reference_scenarios = [
            ContextualDFL.ParametricScenario(; h_eq_xi=[5.5, 6.0, 7.5]),
            ContextualDFL.ParametricScenario(; h_eq_xi=[4.0, 6.8, 8.2]),
        ]
        μ = 0.25

        f(d) = loss(
            d,
            reference_scenarios,
            μ,
            μ;
            tol=1e-9,
        )

        g = only(Flux.gradient(f, predicted_demand))

        direction = [0.1, -0.2, 0.3, -0.4, 0.2, 0.1]
        direction ./= norm(direction)

        ϵ = 1e-4
        fd = (f(predicted_demand .+ ϵ .* direction) -
              f(predicted_demand .- ϵ .* direction)) / (2ϵ)

        @test abs(fd) > 1e-8
        @test !all(iszero, g)
        @test dot(g, direction) ≈ fd atol = 5e-3 rtol = 5e-2
    end

    @testset "extensive objective matches cost function" begin
        _, problem = small_resource_allocation_ad_problem()
        solver = ContextualDFL.Solver(ContextualDFL.IpoptSolver(), ContextualDFL.HiGHSSolver())
        decoder = ResourceAllocationDemandParametricDecoder(problem)
        scenarios = [
            ContextualDFL.ParametricScenario(; h_eq_xi=[5.5, 6.0, 7.5]),
            ContextualDFL.ParametricScenario(; h_eq_xi=[4.0, 6.8, 8.2]),
        ]
        W_eq, W_ineq, T_eq, T_ineq, h_eq, h_ineq, q =
            ContextualDFL.decode_scenario_collection(decoder, scenarios)
        sp = stochastic_program(problem)
        for (mu, rho) in ((0.0, 0.0), (0.05, 0.02))
            _, _, _, solve_result = ContextualDFL._solve_stochastic_extensive(
                solver,
                sp,
                W_eq,
                W_ineq,
                T_eq,
                T_ineq,
                h_eq,
                h_ineq,
                q;
                μ=mu,
                ρ=rho,
            )
            z = solve_result.z[1:length(sp.first_stage_lp.c)]
            recomputed_value = ContextualDFL.cost_function(
                sp,
                solver,
                z,
                W_eq,
                W_ineq,
                T_eq,
                T_ineq,
                h_eq,
                h_ineq,
                q;
                μ=mu,
                ρ=rho,
            )
            old_z = ContextualDFL.solve(
                solver,
                sp,
                W_eq,
                W_ineq,
                T_eq,
                T_ineq,
                h_eq,
                h_ineq,
                q;
                μ=mu,
                ρ=rho,
            )[1]
            old_value = ContextualDFL.cost_function(
                sp,
                solver,
                old_z,
                W_eq,
                W_ineq,
                T_eq,
                T_ineq,
                h_eq,
                h_ineq,
                q;
                μ=mu,
                ρ=rho,
            )
            @test string(solve_result.status) in ("OPTIMAL", "LOCALLY_SOLVED")
            @test solve_result.objective_value ≈ recomputed_value atol = 1e-6 rtol = 1e-6
            @test solve_result.objective_value ≈ old_value atol = 1e-6 rtol = 1e-6
        end
    end

    @testset "general second-stage q ordering" begin
        solver = ContextualDFL.Solver(ContextualDFL.IpoptSolver(), ContextualDFL.HiGHSSolver())
        program = fixed_two_recourse_program()
        scenario = fixed_two_recourse_scenario()
        arrays = ContextualDFL.decode_scenario_collection(
            ContextualDFL.ParametricDecoder(),
            [scenario],
        )
        _, _, _, solve_result = ContextualDFL._solve_stochastic_extensive(
            solver,
            program,
            arrays...;
            μ=0.0,
            ρ=0.0,
        )
        z = solve_result.z[1:length(program.first_stage_lp.c)]
        recomputed_value = ContextualDFL.cost_function(
            program,
            solver,
            z,
            arrays...;
            μ=0.0,
            ρ=0.0,
        )
        @test arrays[7][:, 1] == [7.0, 11.0]
        @test solve_result.objective_value ≈ recomputed_value atol = 1e-6 rtol = 1e-6
        @test solve_result.objective_value ≈ 78.0 atol = 1e-6
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
    @test only(resource_results).evaluation_batches == 1
    @test length(only(resource_results).objective_values) == 1
    @test isfinite(only(resource_results).objective_value)
end

include("benchmark_instances/runtests.jl")
include("decision_optimal_q_conversion/runtests.jl")
