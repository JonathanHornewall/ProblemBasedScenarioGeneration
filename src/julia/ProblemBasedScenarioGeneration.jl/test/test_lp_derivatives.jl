include("../src/problem_instances/resource_allocation/parameters.jl")
cz, qw, ρᵢ = vec(cz), vec(qw), vec(ρᵢ)

problem_data = ResourceAllocationProblemData(μᵢⱼ, cz, qw, ρᵢ)
problem_instance = ResourceAllocationProblem(problem_data)

regularization_parameter = 1.5

@testset "diff_opt_b matches finite differences" begin
    # Build a single-scenario TwoStageSLP from the resource allocation instance
    A1 = problem_instance.s1_constraint_matrix
    b1 = problem_instance.s1_constraint_vector
    c1 = problem_instance.s1_cost_vector
    W = problem_instance.s2_constraint_matrix
    T = problem_instance.s2_coupling_matrix
    q = problem_instance.s2_cost_vector
    
    # Function that returns optimal solution given scenario parameter ξ
    function opt_with_ξ(ξ)
        W, T, h, q = ProblemBasedScenarioGeneration.scenario_realization(problem_instance, ξ)
        two_slp = ProblemBasedScenarioGeneration.TwoStageSLP(A1, b1, c1, [W], [T], [h], [q], [1.0])
        logbar_two_slp = ProblemBasedScenarioGeneration.LogBarCanLP(two_slp, regularization_parameter)
        optimal_solution, optimal_dual = ProblemBasedScenarioGeneration.LogBarCanLP_standard_solver(logbar_two_slp)
        return optimal_solution
    end
    
    # ξ_test should match the scenario parameter dimension (30)
    ξ_test = float(ones(30))
    
    # Test derivative with respect to ξ_test
    g_fd = FiniteDiff.finite_difference_jacobian(opt_with_ξ, ξ_test)
    
    # Apply our derivative to the problem instance
    W, T, h, q = ProblemBasedScenarioGeneration.scenario_realization(problem_instance, ξ_test)
    two_slp = ProblemBasedScenarioGeneration.TwoStageSLP(A1, b1, c1, [W], [T], [h], [q], [1.0])
    logbar_two_slp = ProblemBasedScenarioGeneration.LogBarCanLP(two_slp, regularization_parameter)
    
    # Get the full derivative with respect to b
    g_ad_full = ProblemBasedScenarioGeneration.diff_opt_b(logbar_two_slp)
    
    g_ad = g_ad_full[:, size(T, 2)+1:end]
    
    # Both should be Jacobians: g_fd is (n, m) and g_ad is (n, m) where n=length(c1), m=length(ξ_test)
    @test size(g_ad) == size(g_fd)
    @test isapprox(g_ad, g_fd; rtol=1e-4, atol=1e-4)
end