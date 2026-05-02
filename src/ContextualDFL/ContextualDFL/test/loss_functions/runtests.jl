@testset "loss_functions" begin
    input_decoder = ComponentWiseDecoder(
        (:W_eq,);
        base_W_ineq=:base_W_ineq,
        base_T_eq=:base_T_eq,
        base_T_ineq=:base_T_ineq,
        base_h_eq=:base_h_eq,
        base_h_ineq=:base_h_ineq,
        base_q=:base_q,
    )
    reference_decoder = ComponentWiseDecoder(
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

    loss = DflScenLoss(input_decoder, reference_decoder, solver, program)

    @test loss.input_scenario_decoder === input_decoder
    @test loss.reference_scenario_decoder === reference_decoder
    @test loss.solver === solver
    @test loss.program === program

    all_components = (:W_eq, :W_ineq, :T_eq, :T_ineq, :h_eq, :h_ineq, :q)
    passthrough_decoder = ComponentWiseDecoder(all_components)
    bounded_program = StochasticProgram(
        A_eq=zeros(0, 1),
        A_ineq=reshape([-1.0, 1.0], 2, 1),
        b_eq=Float64[],
        b_ineq=[0.0, 10.0],
        c=[0.0],
    )
    dfl_loss = DflScenLoss(passthrough_decoder, passthrough_decoder, solver, bounded_program)

    input_scenario_parameter_collection = [
        (;
            W_eq=reshape([1.0], 1, 1),
            W_ineq=zeros(0, 1),
            T_eq=reshape([1.0], 1, 1),
            T_ineq=zeros(0, 1),
            h_eq=[5.0],
            h_ineq=Float64[],
            q=[1.0],
        ),
    ]
    reference_scenario_parameter_collection = [
        (;
            W_eq=reshape([1.0], 1, 1),
            W_ineq=zeros(0, 1),
            T_eq=reshape([1.0], 1, 1),
            T_ineq=zeros(0, 1),
            h_eq=[20.0],
            h_ineq=Float64[],
            q=[2.0],
        ),
    ]

    @test dfl_loss(
        bounded_program,
        input_scenario_parameter_collection,
        reference_scenario_parameter_collection,
        0.0,
    ) ≈ 20.0
    @test dfl_loss(
        input_scenario_parameter_collection,
        reference_scenario_parameter_collection,
        0.0,
    ) ≈ 20.0
end
