using Test
using LinearAlgebra
using Random

include(joinpath(@__DIR__, "..", "..", "ProblemBasedScenarioGeneration.jl"))
include(joinpath(@__DIR__, "refactored_logbar_lp.jl"))
include(joinpath(@__DIR__, "refactored_lp_solver.jl"))
include(joinpath(@__DIR__, "refactored_logbar_lp_solver.jl"))

const PBSG = ProblemBasedScenarioGeneration

using .RefactoredLogbarLP
using .RefactoredLPSolvers
using .RefactoredLogbarLPSolvers

Random.seed!(42)

@testset "Refactored log-barrier matches canonical" begin
    A_eq = [1.0 1.0 1.0; 0.0 1.0 -1.0]
    b_eq = [1.0, 0.2]
    c = [1.5, 0.8, 0.4]
    mu = fill(0.05, length(c))
    A_ineq = -Matrix{Float64}(I, length(c), length(c))
    b_ineq = zeros(length(c))

    canonical_lp = PBSG.CanLP(A_eq, b_eq, c)
    canonical_log = PBSG.LogBarCanLP(canonical_lp, mu)
    x_old, λ_old = PBSG.LogBarCanLP_standard_solver(canonical_log)

    new_lp = InequalityEqualityLP(A_ineq, b_ineq, A_eq, b_eq, c)
    new_log = LogBarrierLP(new_lp, mu)
    x_new, λ_new = solve_log_barrier_lp(new_log; initial_point=x_old)

    @test x_new ≈ x_old atol=1e-6
    @test λ_new ≈ λ_old atol=1e-6

    K_old = PBSG.diff_KKT_Y(canonical_log, x_old)
    K_new = diff_KKT_Y(new_log, x_old, λ_old)
    @test K_new ≈ K_old atol=1e-8

    D_A_old, D_b_old, D_c_old = PBSG.diff_opt(canonical_log)
    solver_closure(instance) = solve_log_barrier_lp(instance; initial_point=x_old)
    derivs_new = diff_opt(new_log; solver=solver_closure)

    @test derivs_new.A_eq ≈ D_A_old atol=1e-6
    @test derivs_new.b_eq ≈ D_b_old atol=1e-6
    @test derivs_new.c ≈ D_c_old atol=1e-6
end

@testset "General LP solver" begin
    A_ineq = [-1.0 0.0; 0.0 -1.0; 1.0 0.0]
    b_ineq = [0.0, 0.0, 0.8]
    A_eq = [1.0 1.0]
    b_eq = [1.0]
    c = [1.0, 2.0]

    lp = InequalityEqualityLP(A_ineq, b_ineq, A_eq, b_eq, c)
    x_opt, (λ_ineq, λ_eq) = solve_general_lp(lp)

    @test x_opt ≈ [0.8, 0.2] atol=1e-6
    @test maximum(lp.A_ineq * x_opt .- lp.b_ineq) <= 1e-7
    @test maximum(abs.(lp.A_eq * x_opt .- lp.b_eq)) <= 1e-7
    @test length(λ_eq) == 1
    @test length(λ_ineq) == 3
end

@testset "Log-barrier solver behaviour" begin
    A_ineq = [-1.0 0.0; 0.0 -1.0; 1.0 1.0]
    b_ineq = [0.0, 0.0, 1.4]
    A_eq = [1.0 -1.0]
    b_eq = [0.2]
    c = [1.2, 0.9]

    lp = InequalityEqualityLP(A_ineq, b_ineq, A_eq, b_eq, c)
    mu = fill(0.1, size(A_ineq, 1))
    instance = LogBarrierLP(lp, mu)

    x0 = find_strictly_feasible_point(lp; margin=1e-8)
    x_opt, λ_opt = solve_log_barrier_lp(instance; initial_point=x0)

    @test is_strictly_feasible(instance, x_opt)
    @test length(λ_opt) == size(A_eq, 1)

    zero_mu_instance = LogBarrierLP(lp, zeros(size(A_ineq, 1)))
    x_zero, λ_zero = solve_log_barrier_lp(zero_mu_instance)
    x_lp, (_, λ_eq_lp) = solve_general_lp(lp)
    @test x_zero ≈ x_lp atol=1e-6
    @test λ_zero ≈ λ_eq_lp atol=1e-6
end

@testset "Derivative checks" begin
    A_ineq = [-1.0 0.0; 0.0 -1.0; 1.0 1.0]
    b_ineq = [0.0, 0.0, 1.6]
    A_eq = [1.0 -1.0]
    b_eq = [0.3]
    c = [0.7, 1.1]

    lp = InequalityEqualityLP(A_ineq, b_ineq, A_eq, b_eq, c)
    mu = fill(0.2, size(A_ineq, 1))
    instance = LogBarrierLP(lp, mu)

    x0 = find_strictly_feasible_point(lp; margin=1e-6)
    x_opt, λ_opt = solve_log_barrier_lp(instance; initial_point=x0)
    K = diff_KKT_Y(instance, x_opt, λ_opt)
    solver_closure(inst) = solve_log_barrier_lp(inst; initial_point=x_opt)

    D_c = diff_opt_c(instance, x_opt, λ_opt, K, solver_closure)

    eps = 1e-6
    fd = zeros(length(c), length(c))
    for j in 1:length(c)
        pert_c = copy(c)
        pert_c[j] += eps
        pert_lp = InequalityEqualityLP(A_ineq, b_ineq, A_eq, b_eq, pert_c)
        pert_instance = LogBarrierLP(pert_lp, mu)
        x_pert, _ = solve_log_barrier_lp(pert_instance; initial_point=x_opt)
        fd[:, j] = (x_pert - x_opt) / eps
    end

    @test fd ≈ D_c atol=5e-4
end
