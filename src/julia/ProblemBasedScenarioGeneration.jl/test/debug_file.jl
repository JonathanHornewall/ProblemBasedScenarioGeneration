using FiniteDiff
using ProblemBasedScenarioGeneration
using ProblemBasedScenarioGeneration: convert_standard_to_canonical_form, CanLP, LogBarCanLP, LogBarCanLP_standard_solver, KKT
using ProblemBasedScenarioGeneration: convert_decision_standard_to_canonical, diff_opt_A, diff_opt_b, diff_opt_c
using ProblemBasedScenarioGeneration: diff_KKT_Y, diff_KKT_b, diff_cache_computation, diff_opt_b
using ProblemBasedScenarioGeneration: scenario_realization, ResourceAllocationProblem, ResourceAllocationProblemData, TwoStageSLP, cost_2s_LogBarCanLP, cost, isfeasible
using LinearAlgebra

# Include the necessary functions
include("../src/problem_instances/resource_allocation/parameters.jl")
cz, qw, ρᵢ, = vec(cz), vec(qw), vec(ρᵢ)

problem_data = ResourceAllocationProblemData(μᵢⱼ, cz, qw, ρᵢ)
problem_instance = ResourceAllocationProblem(problem_data)

# Test with a simple scenario
test_scenario = ones(30)
scenario_parameter = float(test_scenario)
reg_param_surr = 1.5
reg_param_ref = 1.5

println("=== Debugging Cost Discrepancy ===")

# Get the problem parameters
A, b, c = problem_instance.s1_constraint_matrix, problem_instance.s1_constraint_vector, problem_instance.s1_cost_vector
W, T, h, q = scenario_realization(problem_instance, scenario_parameter)

println("Problem dimensions:")
println("  First stage: $(length(c)) variables, $(length(b)) constraints")
println("  Second stage: $(length(q)) variables, $(length(h)) constraints")
println("  Number of scenarios: 1")

# Create the two-stage problem
twoslp = TwoStageSLP(A, b, c, [W], [T], [h], [q])
println("\nTwoStageSLP probabilities: ", twoslp.ps)

# Method 1: Evaluate cost at fixed first-stage decision (what evaluated_cost does)
println("\n=== Method 1: Fixed first-stage decision ===")
# Get a surrogate decision (just use a feasible point for testing)
surrogate_decision = ones(length(c))  # Simple feasible decision
println("Surrogate decision: ", surrogate_decision)

# Check first-stage cost
s1_lp = CanLP(A, b, c)
s1_reg_lp = LogBarCanLP(s1_lp, reg_param_surr)
s1_cost = cost(s1_reg_lp, surrogate_decision)
println("First-stage cost: ", s1_cost)

# Check second-stage cost
s2_constraint_matrix = W
s2_constraint_vector = h - T * surrogate_decision
s2_cost_vector = q * twoslp.ps[1]  # Scale by probability
s2_lp = CanLP(s2_constraint_matrix, s2_constraint_vector, s2_cost_vector)
s2_reg_lp = LogBarCanLP(s2_lp, reg_param_surr * twoslp.ps[1])

# Solve second-stage optimally
optimal_s2_decision, _ = LogBarCanLP_standard_solver(s2_reg_lp)
s2_cost = cost(s2_reg_lp, optimal_s2_decision)
println("Second-stage cost: ", s2_cost)
println("Total cost (Method 1): ", s1_cost + s2_cost)

# Method 2: Solve entire problem optimally (what opt_cost does)
println("\n=== Method 2: Optimal solution ===")
logbarlp = LogBarCanLP(twoslp, reg_param_surr)
println("LogBarCanLP regularization parameters:")
println("  First stage: ", logbarlp.regularization_parameters[1:length(c)])
println("  Second stage: ", logbarlp.regularization_parameters[length(c)+1:end])

# Solve optimally
optimal_solution, _ = LogBarCanLP_standard_solver(logbarlp)
opt_cost = cost(logbarlp, optimal_solution)
println("Optimal cost (Method 2): ", opt_cost)

# Compare the approaches
println("\n=== Comparison ===")
method1_cost = s1_cost + s2_cost
method2_cost = opt_cost
gap = method1_cost - method2_cost
println("Method 1 cost: ", method1_cost)
println("Method 2 cost: ", method2_cost)
println("Gap (Method 1 - Method 2): ", gap)
println("Is gap positive? ", gap > 0)

# Check if the surrogate decision is actually feasible
println("\n=== Feasibility Check ===")
println("First-stage feasibility: ", isfeasible(s1_reg_lp, surrogate_decision))
println("Second-stage feasibility: ", isfeasible(s2_reg_lp, optimal_s2_decision))

# Check the actual cost using cost_2s_LogBarCanLP
println("\n=== Using cost_2s_LogBarCanLP ===")
cost_2s_result = cost_2s_LogBarCanLP(twoslp, surrogate_decision, reg_param_surr)
println("cost_2s_LogBarCanLP result: ", cost_2s_result)
println("Matches Method 1? ", isapprox(cost_2s_result, method1_cost, rtol=1e-10))