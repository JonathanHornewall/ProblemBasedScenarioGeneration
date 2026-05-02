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
end
