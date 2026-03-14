using Test
using ProblemBasedScenarioGeneration
using LinearAlgebra

@testset "Barrier Solver" begin
    @testset "Simple 2×2 LP" begin
        # min x1 + 2x2  s.t. x1 = 2, x2 = 3, x >= 0
        A = [1.0 0.0; 0.0 1.0]
        b = [2.0, 3.0]
        c = [1.0, 2.0]
        mu = 0.1

        cache = solve_barrier(A, b, c, mu)

        @test kkt_residual(cache) < 1e-8
        @test isapprox(cache.x, [2.0, 3.0], atol=1e-4)

        # Compare to LP solution (mu=0)
        x_lp, _ = solve_lp(A, b, c)
        @test isapprox(cache.x, x_lp, atol=1e-4)
    end

    @testset "Canonical LP correctness" begin
        # Simple problem: min x1+2*x2 s.t. x1+x2=4, x >= 0
        # Optimal: x1=4, x2=0 (but barrier forces x2 > 0)
        A = [1.0 1.0]
        b = [4.0]
        c = [1.0, 2.0]
        mu = 0.5

        cache = solve_barrier(A, b, c, mu)
        @test kkt_residual(cache) < 1e-8
        @test all(cache.x .> 0)
        @test isapprox(sum(cache.x), 4.0, atol=1e-6)  # Ax = b

        # As mu -> 0, solution should approach LP optimum
        cache_small_mu = solve_barrier(A, b, c, 1e-4)
        @test kkt_residual(cache_small_mu) < 1e-8
        @test cache_small_mu.x[1] > cache_small_mu.x[2]  # x1 should be larger (lower cost)
    end

    @testset "KKT conditions" begin
        A = [1.0 0.0 1.0; 0.0 1.0 1.0]
        b = [2.0, 3.0]
        c = [1.0, 1.0, 0.5]
        mu = 0.2

        cache = solve_barrier(A, b, c, mu)

        # Check KKT conditions: c - μ/x + A'λ = 0, Ax - b = 0
        x, lambda = cache.x, cache.lambda
        r_dual = c - mu ./ x + A' * lambda
        r_prim = A * x - b

        @test norm(r_dual) < 1e-7
        @test norm(r_prim) < 1e-7
        @test all(x .> 0)
    end

    @testset "Warm-starting convergence" begin
        A = [1.0 0.0; 0.0 1.0]
        b = [2.0, 3.0]
        c = [1.0, 2.0]
        mu = 0.1

        # Cold start
        cache1 = solve_barrier(A, b, c, mu)

        # Warm start with known solution
        cache2 = solve_barrier(A, b, c, mu; x0=cache1.x, lambda0=cache1.lambda)

        # Both should give the same answer
        @test isapprox(cache1.x, cache2.x, atol=1e-8)
        @test kkt_residual(cache2) < 1e-8
    end

    @testset "mu=0 fallback to LP" begin
        A = [1.0 0.0; 0.0 1.0]
        b = [2.0, 3.0]
        c = [1.0, 2.0]

        cache = solve_barrier(A, b, c, 0.0)
        x_lp, _ = solve_lp(A, b, c)

        @test isapprox(cache.x, x_lp, atol=1e-6)
    end

    @testset "3-variable problem vs HiGHS" begin
        # min x1 + 2x2 + 3x3  s.t. x1+x2+x3=6, x2+x3=4, x>=0
        A = [1.0 1.0 1.0; 0.0 1.0 1.0]
        b = [6.0, 4.0]
        c = [1.0, 2.0, 3.0]
        mu = 0.05

        cache = solve_barrier(A, b, c, mu)
        x_lp, _ = solve_lp(A, b, c)

        @test kkt_residual(cache) < 1e-7
        # Barrier solution is close to LP but not exact (μ > 0 pushes away from boundary)
        @test isapprox(cache.x, x_lp, atol=0.1)
    end
end
