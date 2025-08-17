# Debug script to investigate the cost discrepancy issue
using ProblemBasedScenarioGeneration
using Flux, LinearAlgebra, Statistics

# Load the training utilities
include("training.jl")

println("=== Detailed Cost Discrepancy Investigation ===")

# Load a saved experiment to investigate
model, training_data, testing_data, problem_instance, reg_params = load_experiment_state("experiment_state.jld2")

println("Loaded experiment with regularization parameters: $reg_params")

# Extract regularization parameters
reg_param_surr = reg_params["reg_param_surr"]
reg_param_ref = reg_params["reg_param_ref"]

println("\n=== Problem Setup ===")
println("Regularization parameters:")
println("  reg_param_surr: $reg_param_surr")
println("  reg_param_ref: $reg_param_ref")

# Get problem dimensions
A, b, c = problem_instance.s1_constraint_matrix, problem_instance.s1_constraint_vector, problem_instance.s1_cost_vector
println("Problem dimensions:")
println("  First stage: $(length(c)) variables, $(length(b)) constraints")

# Test with a specific sample
test_sample = first(testing_data)
x, ξ = test_sample
println("\n=== Testing with Sample ===")
println("Input x: $(x[1:5])... (length: $(length(x)))")
println("Scenario ξ: $(ξ[1:5])... (length: $(length(ξ)))")

# Get the surrogate decision
ξ_hat = model(x)
println("Predicted scenario ξ_hat: $(ξ_hat[1:5])... (length: $(length(ξ_hat)))")

# Method 1: Calculate evaluated_cost using the loss function
println("\n=== Method 1: Loss Function ===")
evaluated_cost = loss(problem_instance, reg_param_surr, reg_param_ref, ξ_hat, ξ)
println("evaluated_cost (from loss function): $evaluated_cost")

# Method 2: Manual calculation to verify
println("\n=== Method 2: Manual Calculation ===")

# Get surrogate decision
surrogate_decision = surrogate_solution(problem_instance, ξ_hat, reg_param_surr)
println("Surrogate decision: $(surrogate_decision[1:5])... (length: $(length(surrogate_decision)))")

# Calculate first-stage cost
s1_lp = CanLP(A, b, c)
s1_reg_lp = LogBarCanLP(s1_lp, reg_param_surr)
s1_cost = cost(s1_reg_lp, surrogate_decision)
println("First-stage cost: $s1_cost")

# Calculate second-stage cost
W, T, h, q = scenario_realization(problem_instance, ξ)
twoslp = TwoStageSLP(A, b, c, [W], [T], [h], [q])
println("TwoStageSLP probabilities: $(twoslp.ps)")

# Use cost_2s_LogBarCanLP to verify
cost_2s_result = cost_2s_LogBarCanLP(twoslp, surrogate_decision, reg_param_surr)
println("cost_2s_LogBarCanLP result: $cost_2s_result")

# Method 3: Calculate opt_cost
println("\n=== Method 3: Optimal Cost ===")
logbarlp = LogBarCanLP(twoslp, reg_param_surr)
opt_cost = optimal_value(logbarlp)
println("opt_cost: $opt_cost")

# Compare the approaches
println("\n=== Comparison ===")
gap = evaluated_cost - opt_cost
println("evaluated_cost: $evaluated_cost")
println("opt_cost: $opt_cost")
println("Gap (evaluated_cost - opt_cost): $gap")
println("Is gap positive? $(gap > 0)")

# Check if the surrogate decision is actually feasible
println("\n=== Feasibility Check ===")
println("First-stage feasibility: $(isfeasible(s1_reg_lp, surrogate_decision))")

# Check regularization parameters in both approaches
println("\n=== Regularization Parameter Check ===")
println("LogBarCanLP regularization parameters:")
println("  First stage: $(logbarlp.regularization_parameters[1:length(c)])")
println("  Second stage: $(logbarlp.regularization_parameters[length(c)+1:end])")

# Check if there's a mismatch in how regularization is applied
println("\n=== Detailed Second-Stage Analysis ===")
s2_constraint_matrix = W
s2_constraint_vector = h - T * surrogate_decision
s2_cost_vector = q * twoslp.ps[1]  # Scale by probability
s2_lp = CanLP(s2_constraint_matrix, s2_constraint_vector, s2_cost_vector)
s2_reg_lp = LogBarCanLP(s2_lp, reg_param_surr * twoslp.ps[1])

println("Second-stage regularization parameter: $(reg_param_surr * twoslp.ps[1])")
println("Second-stage cost vector (scaled by probability): $(s2_cost_vector[1:5])...")

# Solve second-stage optimally
optimal_s2_decision, _ = LogBarCanLP_standard_solver(s2_reg_lp)
s2_cost = cost(s2_reg_lp, optimal_s2_decision)
println("Second-stage cost: $s2_cost")

# Compare manual calculation vs cost_2s_LogBarCanLP
manual_total = s1_cost + s2_cost
println("\nManual calculation total: $manual_total")
println("cost_2s_LogBarCanLP total: $cost_2s_result")
println("Difference: $(manual_total - cost_2s_result)")

println("\n=== Summary ===")
println("The issue might be in how the regularization parameters are handled")
println("or in the probability scaling between the two approaches.")
