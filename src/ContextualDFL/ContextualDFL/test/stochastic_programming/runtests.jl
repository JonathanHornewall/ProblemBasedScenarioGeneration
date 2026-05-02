import ChainRulesCore
import SparseArrays

@testset "stochastic_programming" begin
    solver = Solver(IpoptSolver(), HiGHSSolver())

    @testset "first-stage wrapper" begin
        first_stage_lp = LP(
            A_eq=zeros(0, 1),
            A_ineq=zeros(0, 1),
            b_eq=Float64[],
            b_ineq=Float64[],
            c=[2.0],
        )
        program = StochasticProgram(first_stage_lp)

        @test program.first_stage_lp === first_stage_lp
        @test program.A_eq == first_stage_lp.A_eq
        @test program.A_ineq == first_stage_lp.A_ineq
        @test program.b_eq == first_stage_lp.b_eq
        @test program.b_ineq == first_stage_lp.b_ineq
        @test program.c == first_stage_lp.c
    end

    @testset "two-scenario equality recourse" begin
        program = StochasticProgram(
            A_eq=zeros(0, 1),
            A_ineq=zeros(0, 1),
            b_eq=Float64[],
            b_ineq=Float64[],
            c=[2.0],
        )

        W_eq_array = reshape([1.0, 1.0], 1, 1, 2)
        W_ineq_array = zeros(0, 1, 2)
        T_eq_array = reshape([1.0, 2.0], 1, 1, 2)
        T_ineq_array = zeros(0, 1, 2)
        h_eq_array = reshape([5.0, 8.0], 1, 2)
        h_ineq_array = zeros(0, 2)
        q_array = reshape([3.0, 4.0], 1, 2)
        probabilities = [0.25, 0.75]
        z = [1.0]

        extensive_lp = construct_lp(
            program,
            W_eq_array,
            W_ineq_array,
            T_eq_array,
            T_ineq_array,
            h_eq_array,
            h_ineq_array,
            q_array;
            probabilities=probabilities,
        )

        @test extensive_lp.c == [2.0, 0.75, 3.0]
        @test SparseArrays.issparse(extensive_lp.A_eq)
        @test SparseArrays.issparse(extensive_lp.A_ineq)
        @test extensive_lp.A_eq == [1.0 1.0 0.0; 2.0 0.0 1.0]
        @test size(extensive_lp.A_ineq) == (0, 3)
        @test extensive_lp.b_eq == [5.0, 8.0]
        @test isempty(extensive_lp.b_ineq)

        @test cost_function(
            program,
            solver,
            z,
            W_eq_array,
            W_ineq_array,
            T_eq_array,
            T_ineq_array,
            h_eq_array,
            h_ineq_array,
            q_array;
            probabilities=probabilities,
        ) ≈ 23.0 atol = 1e-8

        primal, dual_eq, dual_ineq = ContextualDFL.G_hat(
            solver,
            z,
            view(W_eq_array, :, :, 1),
            view(W_ineq_array, :, :, 1),
            view(T_eq_array, :, :, 1),
            view(T_ineq_array, :, :, 1),
            view(h_eq_array, :, 1),
            view(h_ineq_array, :, 1),
            view(q_array, :, 1);
            return_dual=true,
        )

        @test primal ≈ [4.0] atol = 1e-8
        @test dual_eq ≈ [3.0] atol = 1e-8
        @test isempty(dual_ineq)

        value, pullback = ChainRulesCore.rrule(
            cost_function,
            program,
            solver,
            z,
            W_eq_array,
            W_ineq_array,
            T_eq_array,
            T_ineq_array,
            h_eq_array,
            h_ineq_array,
            q_array;
            μ=0,
            probabilities=probabilities,
        )
        tangents = pullback(1.0)

        @test value ≈ 23.0 atol = 1e-8
        @test length(tangents) == 11
        @test tangents[4] ≈ [-4.75] atol = 1e-8

        ϵ = 1e-5
        finite_difference_gradient = (
            cost_function(
                program,
                solver,
                z .+ ϵ,
                W_eq_array,
                W_ineq_array,
                T_eq_array,
                T_ineq_array,
                h_eq_array,
                h_ineq_array,
                q_array;
                probabilities=probabilities,
            ) -
            cost_function(
                program,
                solver,
                z .- ϵ,
                W_eq_array,
                W_ineq_array,
                T_eq_array,
                T_ineq_array,
                h_eq_array,
                h_ineq_array,
                q_array;
                probabilities=probabilities,
            )
        ) / (2ϵ)

        @test tangents[4][1] ≈ finite_difference_gradient atol = 1e-5
    end

    @testset "single-scenario equality and inequality recourse" begin
        z = [1.0]
        W_eq = [1.0 0.0]
        W_ineq = [0.0 1.0]
        T_eq = reshape([1.0], 1, 1)
        T_ineq = reshape([-1.0], 1, 1)
        h_eq = [4.0]
        h_ineq = [3.0]
        q = [1.0, -2.0]

        @test ContextualDFL.G_hat(
            solver,
            z,
            W_eq,
            W_ineq,
            T_eq,
            T_ineq,
            h_eq,
            h_ineq,
            q,
        ) ≈ -5.0 atol = 1e-8

        primal, dual_eq, dual_ineq = ContextualDFL.G_hat(
            solver,
            z,
            W_eq,
            W_ineq,
            T_eq,
            T_ineq,
            h_eq,
            h_ineq,
            q;
            return_dual=true,
        )

        @test primal ≈ [3.0, 4.0] atol = 1e-8
        @test dual_eq ≈ [1.0] atol = 1e-8
        @test dual_ineq ≈ [2.0] atol = 1e-8
    end

    @testset "solve packages first and second stage solutions" begin
        program = StochasticProgram(
            A_eq=reshape([1.0], 1, 1),
            A_ineq=zeros(0, 1),
            b_eq=[1.0],
            b_ineq=Float64[],
            c=[0.0],
        )

        W_eq_array = reshape([1.0, 1.0], 1, 1, 2)
        W_ineq_array = zeros(0, 1, 2)
        T_eq_array = reshape([1.0, 2.0], 1, 1, 2)
        T_ineq_array = zeros(0, 1, 2)
        h_eq_array = reshape([5.0, 8.0], 1, 2)
        h_ineq_array = zeros(0, 2)
        q_array = reshape([3.0, 4.0], 1, 2)

        z, y, λ_b_eq, λ_b_ineq, λ_h_eq_array, λ_h_ineq_array = solve(
            solver,
            program,
            W_eq_array,
            W_ineq_array,
            T_eq_array,
            T_ineq_array,
            h_eq_array,
            h_ineq_array,
            q_array;
        )

        @test z ≈ [1.0] atol = 1e-8
        @test y ≈ reshape([4.0, 6.0], 1, 2) atol = 1e-8
        @test length(λ_b_eq) == 1
        @test isempty(λ_b_ineq)
        @test size(λ_h_eq_array) == (1, 2)
        @test size(λ_h_ineq_array) == (0, 2)
    end

    @testset "solve optimizes first-stage recourse tradeoff" begin
        program = StochasticProgram(
            A_eq=zeros(0, 1),
            A_ineq=reshape([1.0, -1.0], 2, 1),
            b_eq=Float64[],
            b_ineq=[3.0, 0.0],
            c=[2.0],
        )

        W_eq_array = reshape([1.0, 1.0], 1, 1, 2)
        W_ineq_array = zeros(0, 1, 2)
        T_eq_array = reshape([1.0, 1.0], 1, 1, 2)
        T_ineq_array = zeros(0, 1, 2)
        h_eq_array = reshape([4.0, 6.0], 1, 2)
        h_ineq_array = zeros(0, 2)
        q_array = reshape([3.0, 4.0], 1, 2)
        probabilities = [0.5, 0.5]

        z, y, _, _, _, _ = solve(
            solver,
            program,
            W_eq_array,
            W_ineq_array,
            T_eq_array,
            T_ineq_array,
            h_eq_array,
            h_ineq_array,
            q_array;
            probabilities=probabilities,
        )

        @test z ≈ [3.0] atol = 1e-8
        @test y ≈ reshape([1.0, 3.0], 1, 2) atol = 1e-8
        @test cost_function(
            program,
            solver,
            z,
            W_eq_array,
            W_ineq_array,
            T_eq_array,
            T_ineq_array,
            h_eq_array,
            h_ineq_array,
            q_array;
            probabilities=probabilities,
        ) ≈ 13.5 atol = 1e-8
    end

    @testset "solve rrule returns h and q tangents only" begin
        program = StochasticProgram(
            A_eq=reshape([1.0], 1, 1),
            A_ineq=zeros(0, 1),
            b_eq=[1.0],
            b_ineq=Float64[],
            c=[0.0],
        )

        W_eq_array = reshape([1.0], 1, 1, 1)
        W_ineq_array = zeros(0, 1, 1)
        T_eq_array = reshape([1.0], 1, 1, 1)
        T_ineq_array = zeros(0, 1, 1)
        h_eq_array = reshape([2.0], 1, 1)
        h_ineq_array = zeros(0, 1)
        q_array = reshape([3.0], 1, 1)

        output, pullback = ChainRulesCore.rrule(
            solve,
            solver,
            program,
            W_eq_array,
            W_ineq_array,
            T_eq_array,
            T_ineq_array,
            h_eq_array,
            h_ineq_array,
            q_array;
            μ=0,
        )
        output_tangent = (
            zeros(1),
            ones(1, 1),
            zeros(1),
            zeros(0),
            zeros(1, 1),
            zeros(0, 1),
        )
        tangents = pullback(output_tangent)

        @test output[1] ≈ [1.0] atol = 1e-8
        @test output[2] ≈ reshape([1.0], 1, 1) atol = 1e-8
        @test tangents[8] ≈ reshape([1.0], 1, 1) atol = 1e-8
        @test tangents[9] == zeros(0, 1)
        @test tangents[10] ≈ reshape([0.0], 1, 1) atol = 1e-8

        ϵ = 1e-5
        h_fd = (
            solve(
                solver,
                program,
                W_eq_array,
                W_ineq_array,
                T_eq_array,
                T_ineq_array,
                h_eq_array .+ ϵ,
                h_ineq_array,
                q_array;
                μ=0,
            )[2][1, 1] -
            solve(
                solver,
                program,
                W_eq_array,
                W_ineq_array,
                T_eq_array,
                T_ineq_array,
                h_eq_array .- ϵ,
                h_ineq_array,
                q_array;
                μ=0,
            )[2][1, 1]
        ) / (2ϵ)

        q_fd = (
            solve(
                solver,
                program,
                W_eq_array,
                W_ineq_array,
                T_eq_array,
                T_ineq_array,
                h_eq_array,
                h_ineq_array,
                q_array .+ ϵ;
                μ=0,
            )[2][1, 1] -
            solve(
                solver,
                program,
                W_eq_array,
                W_ineq_array,
                T_eq_array,
                T_ineq_array,
                h_eq_array,
                h_ineq_array,
                q_array .- ϵ;
                μ=0,
            )[2][1, 1]
        ) / (2ϵ)

        @test tangents[8][1, 1] ≈ h_fd atol = 1e-5
        @test tangents[10][1, 1] ≈ q_fd atol = 1e-5
    end

    @testset "log-barrier solve rrule q sensitivity" begin
        program = StochasticProgram(
            A_eq=zeros(0, 0),
            A_ineq=zeros(0, 0),
            b_eq=Float64[],
            b_ineq=Float64[],
            c=Float64[],
        )

        W_eq_array = zeros(0, 1, 1)
        W_ineq_array = reshape([1.0, -1.0], 2, 1, 1)
        T_eq_array = zeros(0, 0, 1)
        T_ineq_array = zeros(2, 0, 1)
        h_eq_array = zeros(0, 1)
        h_ineq_array = reshape([1.0, 1.0], 2, 1)
        q_array = reshape([0.0], 1, 1)

        output, pullback = ChainRulesCore.rrule(
            solve,
            solver,
            program,
            W_eq_array,
            W_ineq_array,
            T_eq_array,
            T_ineq_array,
            h_eq_array,
            h_ineq_array,
            q_array;
            μ=0.5,
        )
        output_tangent = (
            Float64[],
            ones(1, 1),
            Float64[],
            Float64[],
            zeros(0, 1),
            zeros(2, 1),
        )
        tangents = pullback(output_tangent)

        @test output[2] ≈ reshape([0.0], 1, 1) atol = 1e-5
        @test tangents[10] ≈ reshape([-1.0], 1, 1) atol = 1e-4

        ϵ = 1e-2
        q_fd = (
            solve(
                solver,
                program,
                W_eq_array,
                W_ineq_array,
                T_eq_array,
                T_ineq_array,
                h_eq_array,
                h_ineq_array,
                q_array .+ ϵ;
                μ=0.5,
                tol=1e-10,
            )[2][1, 1] -
            solve(
                solver,
                program,
                W_eq_array,
                W_ineq_array,
                T_eq_array,
                T_ineq_array,
                h_eq_array,
                h_ineq_array,
                q_array .- ϵ;
                μ=0.5,
                tol=1e-10,
            )[2][1, 1]
        ) / (2ϵ)

        @test tangents[10][1, 1] ≈ q_fd atol = 1e-3
    end
end
