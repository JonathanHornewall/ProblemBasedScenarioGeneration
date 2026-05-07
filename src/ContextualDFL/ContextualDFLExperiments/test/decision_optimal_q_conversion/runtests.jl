using ContextualDFL
using ContextualDFLExperiments
using Test

function _q_conversion_solver()
    return ContextualDFL.Solver(ContextualDFL.IpoptSolver(), ContextualDFL.HiGHSSolver())
end

function _q_conversion_program()
    return ContextualDFL.StochasticProgram(
        A_eq=reshape([1.0], 1, 1),
        A_ineq=zeros(0, 1),
        b_eq=[1.0],
        b_ineq=Float64[],
        c=[2.0],
    )
end

function _q_conversion_scenario(; h=4.0, q=3.0)
    return ContextualDFL.ParametricScenario(;
        W_eq_xi=reshape([1.0], 1, 1),
        W_ineq_xi=zeros(0, 1),
        T_eq_xi=reshape([1.0], 1, 1),
        T_ineq_xi=zeros(0, 1),
        h_eq_xi=[Float64(h)],
        h_ineq_xi=Float64[],
        q_xi=[Float64(q)],
    )
end

function _q_conversion_dataset()
    return [
        ContextualDFL.ContextualDataPoint([1.0], [_q_conversion_scenario(; h=4.0, q=3.0)]),
        ContextualDFL.ContextualDataPoint([2.0], [_q_conversion_scenario(; h=5.0, q=3.0)]),
    ]
end

@testset "decision optimal q conversion" begin
    solver = _q_conversion_solver()
    program = _q_conversion_program()
    decoder = ContextualDFL.ParametricDecoder()
    base_scenario = _q_conversion_scenario(; h=4.0, q=3.0)

    identity_data = [
        ContextualDFL.ContextualDataPoint([1.0], [base_scenario]),
    ]
    identity_converted = convert_dataset_to_q(
        identity_data,
        solver,
        program,
        decoder,
        base_scenario;
        constraint_tolerance=1e-10,
    )
    @test only(identity_converted[1].scenario_parameters).q_xi ≈ base_scenario.q_xi atol = 1e-7

    converted = convert_dataset_to_q(
        _q_conversion_dataset(),
        solver,
        program,
        decoder,
        base_scenario;
        constraint_tolerance=1e-10,
    )

    for dp in converted
        s = only(dp.scenario_parameters)
        @test s.W_eq_xi ≈ base_scenario.W_eq_xi
        @test s.W_ineq_xi ≈ base_scenario.W_ineq_xi
        @test s.T_eq_xi ≈ base_scenario.T_eq_xi
        @test s.T_ineq_xi ≈ base_scenario.T_ineq_xi
        @test s.h_eq_xi ≈ base_scenario.h_eq_xi
        @test s.h_ineq_xi ≈ base_scenario.h_ineq_xi
        @test all(isfinite, s.q_xi)
    end

    q_lb = q_lower_bound_from_converted_dataset(converted; margin=1e-6)
    for dp in converted
        q = only(dp.scenario_parameters).q_xi
        @test all(q .> q_lb)
    end

    prepared = prepare_spoplus_q_dataset(
        _q_conversion_dataset(),
        solver,
        program,
        decoder,
        base_scenario;
        lower_bound_margin=1e-6,
        constraint_tolerance=1e-10,
    )
    raw_q = zeros(length(prepared.q_lower_bound))
    dp = first(prepared.converted_dataset)
    value = prepared.spo_loss(raw_q, dp.scenario_parameters, 0.0; constraint_tolerance=1e-10)

    @test isfinite(value)

    prepared_via_wrapper = prepare_decision_optimal_dataset(
        :q,
        _q_conversion_dataset(),
        solver,
        program,
        decoder,
        base_scenario;
        lower_bound_margin=1e-6,
        constraint_tolerance=1e-10,
    )
    @test prepared_via_wrapper.q_lower_bound ≈ prepared.q_lower_bound
end

@testset "decision optimal h conversion" begin
    solver = _q_conversion_solver()
    program = _q_conversion_program()
    decoder = ContextualDFL.ParametricDecoder()
    base_scenario = _q_conversion_scenario(; h=4.0, q=3.0)

    converted = convert_dataset_to_h(
        _q_conversion_dataset(),
        solver,
        program,
        decoder,
        base_scenario;
        constraint_tolerance=1e-10,
    )

    for dp in converted
        s = only(dp.scenario_parameters)
        @test s.W_eq_xi ≈ base_scenario.W_eq_xi
        @test s.W_ineq_xi ≈ base_scenario.W_ineq_xi
        @test s.T_eq_xi ≈ base_scenario.T_eq_xi
        @test s.T_ineq_xi ≈ base_scenario.T_ineq_xi
        @test s.h_eq_xi ≈ [1.0] atol = 1e-7
        @test s.h_ineq_xi ≈ base_scenario.h_ineq_xi
        @test s.q_xi ≈ base_scenario.q_xi
    end

    converted_via_wrapper = convert_dataset_to_decision_optimal(
        :h,
        _q_conversion_dataset(),
        solver,
        program,
        decoder,
        base_scenario;
        constraint_tolerance=1e-10,
    )
    @test only(first(converted_via_wrapper).scenario_parameters).h_eq_xi ≈
          only(first(converted).scenario_parameters).h_eq_xi

    prepared = prepare_decision_h_dataset(
        _q_conversion_dataset(),
        solver,
        program,
        decoder,
        base_scenario;
        constraint_tolerance=1e-10,
    )
    @test prepared.h_dimension == length(base_scenario.h_eq_xi)
    @test length(prepared.converted_dataset) == length(_q_conversion_dataset())

    raw_h = copy(only(first(prepared.converted_dataset).scenario_parameters).h_eq_xi)
    value = prepared.loss(
        raw_h,
        first(prepared.converted_dataset).scenario_parameters,
        0.0;
        constraint_tolerance=1e-10,
    )
    @test isfinite(value)

    prepared_via_wrapper = prepare_decision_optimal_dataset(
        :h,
        _q_conversion_dataset(),
        solver,
        program,
        decoder,
        base_scenario;
        constraint_tolerance=1e-10,
    )
    @test prepared_via_wrapper.h_dimension == prepared.h_dimension

    base_arrays = full_base_scenario_arrays(base_scenario)
    decision_h_decoder = DecisionOptimalHDecoder(base_arrays)
    raw = [0.25]
    decoded = ContextualDFL.decode_scenario_collection(
        decision_h_decoder,
        raw;
        nr_scenarios=1,
    )
    @test decoded[5][:, 1] ≈ raw

    @test_throws ArgumentError make_decision_h_loss(
        solver,
        program,
        base_scenario;
        loss=:spo_plus,
    )
end
