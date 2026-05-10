using ContextualDFL
using ContextualDFLExperiments
using LinearAlgebra
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

function _decision_preservation_program()
    return ContextualDFL.StochasticProgram(
        A_eq=zeros(0, 2),
        A_ineq=[
            -1.0 0.0
            0.0 -1.0
            1.0 1.0
        ],
        b_eq=Float64[],
        b_ineq=[0.0, 0.0, 1.0],
        c=zeros(2),
    )
end

function _decision_preservation_scenario(; h=[2.0, 2.0], q=[3.0, 1.0])
    return ContextualDFL.ParametricScenario(;
        W_eq_xi=[
            1.0 0.0
            0.0 1.0
        ],
        W_ineq_xi=[
            -1.0 0.0
            0.0 -1.0
        ],
        T_eq_xi=[
            1.0 0.0
            0.0 1.0
        ],
        T_ineq_xi=zeros(2, 2),
        h_eq_xi=Float64.(h),
        h_ineq_xi=zeros(2),
        q_xi=Float64.(q),
    )
end

function _decision_preservation_dataset()
    return [
        ContextualDFL.ContextualDataPoint(
            [1.0],
            [_decision_preservation_scenario(; q=[3.0, 1.0])],
        ),
        ContextualDFL.ContextualDataPoint(
            [2.0],
            [_decision_preservation_scenario(; q=[1.0, 3.0])],
        ),
    ]
end

function _general_inequality_scenario(; h=[2.0, 2.0], q=[3.0, 1.0])
    return ContextualDFL.ParametricScenario(;
        W_eq_xi=[
            1.0 0.0
            0.0 1.0
        ],
        W_ineq_xi=[
            -1.0 0.0
            0.0 -1.0
            1.0 1.0
        ],
        T_eq_xi=[
            1.0 0.0
            0.0 1.0
        ],
        T_ineq_xi=zeros(3, 2),
        h_eq_xi=Float64.(h),
        h_ineq_xi=[0.0, 0.0, 3.0],
        q_xi=Float64.(q),
    )
end

function _general_inequality_dataset()
    return [
        ContextualDFL.ContextualDataPoint(
            [1.0],
            [_general_inequality_scenario(; q=[3.0, 1.0])],
        ),
        ContextualDFL.ContextualDataPoint(
            [2.0],
            [_general_inequality_scenario(; q=[1.0, 3.0])],
        ),
    ]
end

function _solve_datapoint_decision(
    solver,
    program,
    decoder,
    datapoint;
    probabilities=nothing,
    kwargs...,
)
    arrays = ContextualDFL.decode_scenario_collection(decoder, datapoint.scenario_parameters)
    z, _, _, _, _, _ = ContextualDFL.solve(
        solver,
        program,
        arrays...;
        probabilities=probabilities,
        kwargs...,
    )
    return z
end

function _test_converted_dataset_preserves_decisions(
    original_dataset,
    converted_dataset,
    solver,
    program,
    original_decoder;
    probabilities_by_datapoint=nothing,
    atol=1e-7,
    rtol=1e-7,
    kwargs...,
)
    @test length(converted_dataset) == length(original_dataset)

    for (index, (original_dp, converted_dp)) in enumerate(zip(original_dataset, converted_dataset))
        probabilities =
            probabilities_by_datapoint === nothing ? nothing :
            probabilities_by_datapoint isa Function ? probabilities_by_datapoint(original_dp) :
            probabilities_by_datapoint[index]
        original_z = _solve_datapoint_decision(
            solver,
            program,
            original_decoder,
            original_dp;
            probabilities=probabilities,
            kwargs...,
        )
        converted_z = _solve_datapoint_decision(
            solver,
            program,
            ContextualDFL.ParametricDecoder(),
            converted_dp;
            kwargs...,
        )

        @test converted_z ≈ original_z atol = atol rtol = rtol
    end
end

@testset "equality form conversion" begin
    solver = _q_conversion_solver()
    program = _decision_preservation_program()
    decoder = ContextualDFL.ParametricDecoder()
    dataset = _general_inequality_dataset()
    base_scenario = _general_inequality_scenario(; q=[1.0, 1.0])

    equality_base = convert_base_scenario_to_equality_form(base_scenario)
    equality_dataset = convert_dataset_to_equality_form(dataset, decoder)

    base_arrays = decode_q_conversion_arrays(
        ContextualDFL.ParametricDecoder(),
        [equality_base],
    )
    @test size(base_arrays.W_eq_array[:, :, 1]) == (3, 3)
    @test base_arrays.W_ineq_array[:, :, 1] ≈ -Matrix{Float64}(I, 3, 3)
    @test base_arrays.T_ineq_array[:, :, 1] ≈ zeros(3, 2)
    @test base_arrays.h_ineq_array[:, 1] ≈ zeros(3)
    @test base_arrays.q_array[:, 1] ≈ [1.0, 1.0, 0.0]

    _test_converted_dataset_preserves_decisions(
        dataset,
        equality_dataset,
        solver,
        program,
        decoder;
        constraint_tolerance=1e-10,
    )

    q_payload = make_decision_equivalent_dataset(
        :q,
        dataset,
        solver,
        program,
        decoder,
        base_scenario;
        constraint_tolerance=1e-10,
    )
    @test q_payload.target == :q
    @test q_payload.decoder isa ContextualDFL.ParametricDecoder
    @test q_payload.diagnostics.equality_form_applied
    @test q_payload.diagnostics.base_recourse_dimension == 3
    _test_converted_dataset_preserves_decisions(
        dataset,
        q_payload.converted_dataset,
        solver,
        program,
        decoder;
        constraint_tolerance=1e-10,
    )

    h_payload = make_decision_equivalent_dataset(
        :rhs,
        dataset,
        solver,
        program,
        decoder,
        base_scenario;
        constraint_tolerance=1e-10,
    )
    @test h_payload.target == :h
    @test h_payload.decoder isa ContextualDFL.ParametricDecoder
    @test h_payload.diagnostics.equality_form_applied
    _test_converted_dataset_preserves_decisions(
        dataset,
        h_payload.converted_dataset,
        solver,
        program,
        decoder;
        constraint_tolerance=1e-10,
    )
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

    preservation_program = _decision_preservation_program()
    preservation_dataset = _decision_preservation_dataset()
    preservation_base = _decision_preservation_scenario(; q=[1.0, 1.0])
    preservation_converted = convert_dataset_to_q(
        preservation_dataset,
        solver,
        preservation_program,
        decoder,
        preservation_base;
        constraint_tolerance=1e-10,
    )
    expected_decisions = ([1.0, 0.0], [0.0, 1.0])
    for (datapoint, expected_z) in zip(preservation_dataset, expected_decisions)
        @test _solve_datapoint_decision(
            solver,
            preservation_program,
            decoder,
            datapoint;
            constraint_tolerance=1e-10,
        ) ≈ expected_z atol = 1e-7
    end
    _test_converted_dataset_preserves_decisions(
        preservation_dataset,
        preservation_converted,
        solver,
        preservation_program,
        decoder;
        constraint_tolerance=1e-10,
    )
end

@testset "implemented problem base scenarios support equality form" begin
    problems = (
        ResourceAllocationProblem(),
        ShipmentPlanningProblem(),
        RandomYieldProblem(; r=2, a=4, K_support=2),
        UnreliableNewsvendorProblem(),
        TransShipmentExperimentProblem(),
    )

    for problem in problems
        base = base_scenario(problem)
        for field in (:W_eq, :W_ineq, :T_eq, :T_ineq, :h_eq, :h_ineq, :q)
            @test hasproperty(base, field)
        end

        equality_base = convert_base_scenario_to_equality_form(base)
        arrays = decode_q_conversion_arrays(
            ContextualDFL.ParametricDecoder(),
            [equality_base],
        )
        @test size(arrays.W_ineq_array, 1) == length(equality_base.q_xi)
        @test size(arrays.W_ineq_array, 2) == length(equality_base.q_xi)
        @test arrays.W_ineq_array[:, :, 1] ≈
              -Matrix{Float64}(I, length(equality_base.q_xi), length(equality_base.q_xi))
    end
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

    preservation_program = _decision_preservation_program()
    preservation_dataset = _decision_preservation_dataset()
    preservation_base = _decision_preservation_scenario(; q=[1.0, 1.0])
    preservation_converted = convert_dataset_to_h(
        preservation_dataset,
        solver,
        preservation_program,
        decoder,
        preservation_base;
        constraint_tolerance=1e-10,
    )
    _test_converted_dataset_preserves_decisions(
        preservation_dataset,
        preservation_converted,
        solver,
        preservation_program,
        decoder;
        constraint_tolerance=1e-10,
    )
end
