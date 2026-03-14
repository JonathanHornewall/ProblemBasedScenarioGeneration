using Test
using ProblemBasedScenarioGeneration
using LinearAlgebra
using FiniteDiff

@testset "Implicit Differentiation" begin
    @testset "implicit_diff_h shape and finite-diff check" begin
        # Simple LP: min c'x s.t. Ax = b, x > 0
        A = [1.0 0.0; 0.0 1.0]
        b = [2.0, 3.0]
        c = [1.0, 2.0]
        mu = 1.5

        cache = solve_barrier(A, b, c, mu)
        Dh = implicit_diff_h(cache)

        # Shape check
        n, m = length(cache.x), length(cache.b)
        @test size(Dh) == (n, m)

        # Finite-difference check
        function x_opt_h(b_perturb)
            c2 = solve_barrier(A, b_perturb, c, mu)
            return c2.x
        end
        Dh_fd = FiniteDiff.finite_difference_jacobian(x_opt_h, b)

        @test isapprox(Dh, Dh_fd, rtol=1e-4, atol=1e-4)
    end

    @testset "implicit_diff_q shape and finite-diff check" begin
        A = [1.0 0.0; 0.0 1.0]
        b = [2.0, 3.0]
        c = [1.0, 2.0]
        mu = 1.5

        cache = solve_barrier(A, b, c, mu)
        Dq = implicit_diff_q(cache)

        n = length(cache.x)
        @test size(Dq) == (n, n)

        # Finite-difference check
        function x_opt_c(c_perturb)
            c2 = solve_barrier(A, b, c_perturb, mu)
            return c2.x
        end
        Dq_fd = FiniteDiff.finite_difference_jacobian(x_opt_c, c)

        @test isapprox(Dq, Dq_fd, rtol=1e-4, atol=1e-4)
    end

    @testset "implicit_diff_h with 3-variable LP" begin
        A = [1.0 1.0 1.0; 0.0 1.0 1.0]
        b = [6.0, 4.0]
        c = [1.0, 2.0, 3.0]
        mu = 0.5

        cache = solve_barrier(A, b, c, mu)
        Dh = implicit_diff_h(cache)

        n, m = length(cache.x), length(cache.b)
        @test size(Dh) == (n, m)
        @test all(isfinite.(Dh))

        # Finite-diff check
        Dh_fd = FiniteDiff.finite_difference_jacobian(
            b_p -> solve_barrier(A, b_p, c, mu).x,
            b
        )
        @test isapprox(Dh, Dh_fd, rtol=1e-4, atol=1e-4)
    end

    @testset "recourse_multiplier returns dual variable" begin
        A = [1.0 0.0; 0.0 1.0]
        b = [2.0, 3.0]
        c = [1.0, 2.0]
        mu = 0.5

        cache = solve_barrier(A, b, c, mu)
        lam = recourse_multiplier(cache)

        @test length(lam) == length(b)
        @test all(isfinite.(lam))

        # Verify KKT: c - mu/x + A'lambda = 0
        residual = c - mu ./ cache.x + A' * lam
        @test norm(residual) < 1e-7
    end
end
