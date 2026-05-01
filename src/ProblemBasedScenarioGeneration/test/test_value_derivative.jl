include("../src/problem_instances/resource_allocation/parameters.jl")
cz, qw, ρᵢ, = vec(cz), vec(qw), vec(ρᵢ)

problem_data = ResourceAllocationProblemData(μᵢⱼ, cz, qw, ρᵢ)
problem_instance = ResourceAllocationProblem(problem_data)

test_scenario = ones(30)
scenario_parameter = float(test_scenario)
regularization_parameter = 1.5

@testset "diff_s1_cost matches finite differences" begin
    # Build a single-scenario TwoStageSLP from the resource allocation instance
    A1 = problem_instance.s1_constraint_matrix
    b1 = problem_instance.s1_constraint_vector
    c1 = problem_instance.s1_cost_vector
    
    # Use scenario_collection_realization to get properly formatted arrays
    scenario_matrix = reshape(scenario_parameter, :, 1)
    Ws, Ts, hs, qs = ProblemBasedScenarioGeneration.scenario_collection_realization(problem_instance, scenario_matrix)
    
    two_slp = ProblemBasedScenarioGeneration.TwoStageSLP(A1, b1, c1, Ws, Ts, hs, qs, [1.0])

    # Pick a strictly positive feasible first-stage decision (A1 is rank-0 so any x>0 is feasible)
    x0 = ones(length(c1))
    μ = regularization_parameter

    f(x) = ProblemBasedScenarioGeneration.s1_cost(two_slp, x, μ)
    g_fd = FiniteDiff.finite_difference_gradient(f, x0)
    g_ad = ProblemBasedScenarioGeneration.diff_s1_cost(two_slp, x0, μ)

    @test isapprox(g_ad, g_fd; rtol=1e-4, atol=1e-4)
end