using ContextualDFL
using Flux
using LinearAlgebra
using Test

struct TestHDecoder <: ComponentDecoder
    n_zeros::Int
end
(decoder::TestHDecoder)(h_tail) = vcat(zeros(eltype(h_tail), decoder.n_zeros), vec(h_tail))

@testset "ContextualDFL" begin
    base = BaseScenario(
        [1.0;;],
        zeros(0, 1),
        [-1.0;;],
        zeros(0, 1),
        [0.0, 1.0],
        [2.0],
    )

    @testset "decoders" begin
        @test TrivialDecoder()([1, 2]) == [1, 2]
        @test DefaultComponentDecoder(:h)([3.0]) == [3.0]
        @test_throws ErrorException EmptyComponentDecoder(:q)([1.0])

        decoder = DataSetScenarioDecoder(
            DecoderStrategy(h_decoder=TestHDecoder(1)),
            base,
            (:h,),
        )
        row = (x=[0.0], xi_W=nothing, xi_T=nothing, xi_h=[4.0], xi_q=nothing)
        scenario = decoder(row)
        @test scenario.W_eq == base.W_eq
        @test scenario.h == [0.0, 4.0]
        @test decoder([5.0]).h == [0.0, 5.0]
    end

    @testset "lp solver" begin
        eq_lp = LP([1.0 1.0], [1.0], [1.0, 2.0])
        eq_result = solve(GLPKSolver(), eq_lp)
        @test eq_result.primal ≈ [1.0, 0.0] atol=1e-7
        @test eq_result.objective_value ≈ 1.0 atol=1e-7

        in_lp = LP(zeros(0, 1), -ones(1, 1), Float64[], [-1.0], [1.0], nothing)
        in_result = solve(GLPKSolver(), in_lp)
        @test in_result.primal[1] ≈ 1.0 atol=1e-7

        smooth_result = solve(GLPKSolver(mu=0.1), eq_lp)
        @test all(isfinite, smooth_result.primal)
        @test smooth_result.cache isa BarrierCache
    end

    @testset "stochastic programming" begin
        program = StochasticProgram(zeros(1, 1), zeros(0, 1), [0.0], [1.0])
        scenario = BaseScenario([1.0;;], zeros(0, 1), [-1.0;;], zeros(0, 1), [1.0], [2.0])
        lp_one = construct_lp(program, scenario)
        @test size(lp_one.A_eq, 2) == 2
        @test lp_one.c_eq == [1.0, 2.0]

        lp_two = construct_lp(program, [scenario, scenario])
        @test size(lp_two.A_eq, 2) == 3
        @test lp_two.c_eq == [1.0, 1.0, 1.0]

        result = solve(program, GLPKSolver(), scenario)
        @test all(isfinite, result.first_stage_decision)
        @test result.objective_value ≈ 2.0 atol=1e-7

        @test scenario_wise_cost(program, [0.0], scenario; solver=GLPKSolver()) ≈ 2.0 atol=1e-7
        @test cost_function(program, [1.0], scenario; solver=GLPKSolver()) ≈ 5.0 atol=1e-7
    end

    @testset "losses and schedules" begin
        program = StochasticProgram(zeros(1, 1), zeros(0, 1), [0.0], [1.0])
        scenario = BaseScenario([1.0;;], zeros(0, 1), [-1.0;;], zeros(0, 1), [1.0], [2.0])
        shifted = BaseScenario([1.0;;], zeros(0, 1), [-1.0;;], zeros(0, 1), [2.0], [2.0])

        @test MSEScenLoss()(program, scenario, shifted, 0.0, 0.0) > 0
        @test isfinite(DflScenLoss(GLPKSolver(), program)(program, scenario, scenario, 0.0, 0.0))
        @test isfinite(DflCLoss(GLPKSolver(), program, 0.0, 0.0)(program, scenario, scenario, 0.0, 0.0))
        @test isfinite(ProjectedZLoss(GLPKSolver(), program)(program, scenario, scenario, 0.0, 0.0))

        @test constant_schedule(3; length=2)[2] == 3
        @test linear_schedule(0.0, 10.0; steps=3)[2] ≈ 5.0
        @test geometric_schedule(1.0, 0.25; steps=3)[3] ≈ 0.25
    end

    @testset "training smoke" begin
        decoder = DataSetScenarioDecoder(
            DecoderStrategy(h_decoder=TestHDecoder(0)),
            BaseScenario([1.0;;], zeros(0, 1), [-1.0;;], zeros(0, 1), Float32[1.0], [2.0]),
            (:h,),
        )
        data_set = DataSet(reshape(Float32[1.0, 2.0], 1, 2), nothing, nothing, reshape(Float32[1.0, 1.2], 1, 2), nothing)
        model = Chain(Dense(1 => 1))
        program = StochasticProgram(zeros(1, 1), zeros(0, 1), [0.0], [1.0])
        generator = DFLScenarioGenerator(decoder, GLPKSolver(), model, program)
        result = train(
            generator,
            MSEScenLoss(),
            data_set,
            decoder,
            constant_schedule(0.0; length=1),
            constant_schedule(0.0; length=1),
            constant_schedule(1; length=1),
            constant_schedule(0.01; length=1),
        )
        @test length(result.loss_history) == 1
        @test isfinite(result.loss_history[end])
    end
end
