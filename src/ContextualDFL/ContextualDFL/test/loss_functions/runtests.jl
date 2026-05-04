import Flux
import LinearAlgebra: dot

struct DflTestVectorDecoder <: VectorDecoder end

function (::DflTestVectorDecoder)(vector::AbstractVector)
    return (
        reshape(view(vector, 1:1), 1, 1),
        zeros(eltype(vector), 0, 1),
        reshape(view(vector, 2:2), 1, 1),
        zeros(eltype(vector), 0, 1),
        view(vector, 3:3),
        zeros(eltype(vector), 0),
        view(vector, 4:4),
    )
end

struct SPOPlusTestQDecoder <: VectorDecoder end

function (::SPOPlusTestQDecoder)(q::AbstractVector)
    return (
        zeros(eltype(q), 0, 1),
        reshape([-one(eltype(q)), one(eltype(q))], 2, 1),
        zeros(eltype(q), 0, 0),
        zeros(eltype(q), 2, 0),
        zeros(eltype(q), 0),
        [zero(eltype(q)), one(eltype(q))],
        q,
    )
end

function spo_plus_test_scenario(q)
    return ParametricScenario(;
        W_eq_xi=zeros(0, 1),
        W_ineq_xi=reshape([-1.0, 1.0], 2, 1),
        T_eq_xi=zeros(0, 0),
        T_ineq_xi=zeros(2, 0),
        h_eq_xi=Float64[],
        h_ineq_xi=[0.0, 1.0],
        q_xi=[q],
    )
end

@testset "loss_functions" begin
    input_decoder = DflTestVectorDecoder()
    reference_decoder = ParametricDecoder(
        (:h_eq,);
        base_W_eq=:base_W_eq,
        base_W_ineq=:base_W_ineq,
        base_T_eq=:base_T_eq,
        base_T_ineq=:base_T_ineq,
        base_h_ineq=:base_h_ineq,
        base_q=:base_q,
    )
    solver = Solver(IpoptSolver(), HiGHSSolver())
    program = StochasticProgram(c=[1.0])

    loss = DflScenLoss(input_decoder, reference_decoder, solver, program; nr_scenarios=2)

    @test loss.input_scenario_decoder === input_decoder
    @test loss.reference_scenario_decoder === reference_decoder
    @test loss.solver === solver
    @test loss.program === program
    @test loss.nr_scenarios == 2

    passthrough_decoder = ParametricDecoder()
    bounded_program = StochasticProgram(
        A_eq=zeros(0, 1),
        A_ineq=reshape([-1.0, 1.0], 2, 1),
        b_eq=Float64[],
        b_ineq=[0.0, 10.0],
        c=[0.0],
    )
    dfl_loss = DflScenLoss(input_decoder, passthrough_decoder, solver, bounded_program)

    input_scenario_parameter_collection = [1.0, 1.0, 5.0, 1.0]
    reference_scenario_parameter_collection = [
        ParametricScenario(;
            W_eq_xi=reshape([1.0], 1, 1),
            W_ineq_xi=reshape([1.0], 1, 1),
            T_eq_xi=reshape([1.0], 1, 1),
            T_ineq_xi=reshape([0.0], 1, 1),
            h_eq_xi=[20.0],
            h_ineq_xi=[30.0],
            q_xi=[2.0],
        ),
    ]

    @test dfl_loss(
        input_scenario_parameter_collection,
        reference_scenario_parameter_collection,
        0.0,
    ) ≈ 20.0

    positive_mu = 0.1
    default_reference_mu_loss = dfl_loss(
        input_scenario_parameter_collection,
        reference_scenario_parameter_collection,
        positive_mu,
    )
    explicit_reference_mu_loss = dfl_loss(
        input_scenario_parameter_collection,
        reference_scenario_parameter_collection,
        positive_mu,
        positive_mu,
    )
    zero_reference_mu_loss = dfl_loss(
        input_scenario_parameter_collection,
        reference_scenario_parameter_collection,
        positive_mu,
        0.0,
    )

    @test default_reference_mu_loss ≈ explicit_reference_mu_loss atol = 1e-7 rtol = 1e-7
    @test !isapprox(default_reference_mu_loss, zero_reference_mu_loss; atol=1e-4, rtol=1e-4)

    rho = 0.2
    default_reference_rho_loss = dfl_loss(
        input_scenario_parameter_collection,
        reference_scenario_parameter_collection,
        0.0,
        0.0;
        rho_in=rho,
        tol=1e-10,
    )
    explicit_reference_rho_loss = dfl_loss(
        input_scenario_parameter_collection,
        reference_scenario_parameter_collection,
        0.0,
        0.0;
        rho_in=rho,
        rho_ref=rho,
        tol=1e-10,
    )
    zero_reference_rho_loss = dfl_loss(
        input_scenario_parameter_collection,
        reference_scenario_parameter_collection,
        0.0,
        0.0;
        rho_in=rho,
        rho_ref=0.0,
    )

    @test default_reference_rho_loss ≈ explicit_reference_rho_loss atol = 1e-7 rtol = 1e-7
    @test !isapprox(default_reference_rho_loss, zero_reference_rho_loss; atol=1e-4, rtol=1e-4)

    rho_direction = [0.0, 0.0, 0.1, -0.05]
    rho_objective(v) = dfl_loss(
        v,
        reference_scenario_parameter_collection,
        0.0,
        0.0;
        rho_in=rho,
        rho_ref=rho,
        tol=1e-10,
    )
    rho_gradient = only(Flux.gradient(rho_objective, input_scenario_parameter_collection))
    ϵ = 1e-5
    rho_finite_difference = (
        rho_objective(input_scenario_parameter_collection .+ ϵ .* rho_direction) -
        rho_objective(input_scenario_parameter_collection .- ϵ .* rho_direction)
    ) / (2ϵ)

    @test dot(rho_gradient, rho_direction) ≈ rho_finite_difference atol = 1e-4 rtol = 1e-3

    @testset "SPOPlusLoss objective-vector surrogate" begin
        spo_program = StochasticProgram(
            A_eq=zeros(0, 0),
            A_ineq=zeros(0, 0),
            b_eq=Float64[],
            b_ineq=Float64[],
            c=Float64[],
        )
        q_decoder = SPOPlusTestQDecoder()
        spo_loss = SPOPlusLoss(q_decoder, ParametricDecoder(), solver, spo_program)

        @test spo_loss.input_scenario_decoder === q_decoder
        @test spo_loss.reference_scenario_decoder isa ParametricDecoder
        @test spo_loss.solver === solver
        @test spo_loss.program === spo_program
        @test spo_loss.nr_scenarios == 1
        @test_throws ArgumentError SPOPlusLoss(q_decoder, ParametricDecoder(), solver, spo_program; nr_scenarios=0)

        reference = [spo_plus_test_scenario(2.0)]
        prediction = [0.25]
        @test spo_loss(prediction, reference, 0.0) ≈ 1.5 atol = 1e-8
        @test spo_loss([2.0], reference, 0.0) ≈ 0.0 atol = 1e-8
        @test only(Flux.gradient(q -> spo_loss(q, reference, 0.0), prediction)) ≈ [-2.0] atol = 1e-8

        ϵ = 1e-6
        finite_difference_gradient =
            (
                spo_loss(prediction .+ ϵ, reference, 0.0) -
                spo_loss(prediction .- ϵ, reference, 0.0)
            ) / (2ϵ)
        @test finite_difference_gradient ≈ -2.0 atol = 1e-5
        @test_throws ArgumentError spo_loss(prediction, reference, 0.1)

        rho_spo = 4.0
        rho_spo_value = spo_loss(prediction, reference, 0.0; rho_in=rho_spo, tol=1e-10)
        explicit_rho_spo_value = spo_loss(
            prediction,
            reference,
            0.0;
            rho_in=rho_spo,
            rho_ref=rho_spo,
            tol=1e-10,
        )
        rho_spo_gradient = only(
            Flux.gradient(
                q -> spo_loss(q, reference, 0.0; rho_in=rho_spo, rho_ref=rho_spo, tol=1e-10),
                prediction,
            ),
        )
        rho_spo_finite_difference =
            (
                spo_loss(prediction .+ ϵ, reference, 0.0; rho_in=rho_spo, rho_ref=rho_spo, tol=1e-10) -
                spo_loss(prediction .- ϵ, reference, 0.0; rho_in=rho_spo, rho_ref=rho_spo, tol=1e-10)
            ) / (2ϵ)

        @test rho_spo_value ≈ explicit_rho_spo_value atol = 1e-8
        @test rho_spo_gradient[1] ≈ rho_spo_finite_difference atol = 1e-5 rtol = 1e-5

        two_scenario_loss =
            SPOPlusLoss(q_decoder, ParametricDecoder(), solver, spo_program; nr_scenarios=2)
        two_reference = [spo_plus_test_scenario(2.0), spo_plus_test_scenario(-3.0)]
        two_prediction = [0.25, -0.5]
        probabilities = [0.25, 0.75]

        @test two_scenario_loss(
            two_prediction,
            two_reference,
            0.0;
            probabilities=probabilities,
        ) ≈ 1.875 atol = 1e-8
        @test only(
            Flux.gradient(
                q -> two_scenario_loss(q, two_reference, 0.0; probabilities=probabilities),
                two_prediction,
            ),
        ) ≈ [-0.5, 1.5] atol = 1e-8

        mismatched_feasible_loss =
            SPOPlusLoss(input_decoder, passthrough_decoder, solver, bounded_program)
        @test_throws DimensionMismatch mismatched_feasible_loss(
            input_scenario_parameter_collection,
            reference_scenario_parameter_collection,
            0.0,
        )
    end
end
