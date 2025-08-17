# Focused debug script to understand the cost discrepancy
using ProblemBasedScenarioGeneration: TwoStageSLP, LogBarCanLP, CanLP, extensive_form_canonical, cost_2s_LogBarCanLP, optimal_value, LogBarCanLP_standard_solver, cost
using Flux, LinearAlgebra

# Define missing constant
const feasibility_margin = 1e-8

println("=== Core Cost Discrepancy Investigation ===")

# Create a simple test case
println("Creating test problem...")

# Simple 2x2 problem for clarity
A = [1.0 0.0; 0.0 1.0]  # Identity matrix
b = [1.0, 1.0]           # Constraint vector
c = [1.0, 1.0]           # Cost vector

# Simple scenario
W = [1.0 0.0; 0.0 1.0]  # Second-stage constraint matrix
T = [0.5 0.0; 0.0 0.5]  # Coupling matrix
h = [0.5, 0.5]           # Second-stage constraint vector
q = [0.5, 0.5]           # Second-stage cost vector

# Create two-stage problem
twoslp = TwoStageSLP(A, b, c, [W], [T], [h], [q])
println("TwoStageSLP created with probabilities: $(twoslp.ps)")

# Test regularization parameter
reg_param = 0.1
println("Regularization parameter: $reg_param")

# Method 1: Calculate evaluated_cost (fixed first-stage, optimize second-stage)
println("\n=== Method 1: Fixed First-Stage Decision ===")
# Use a simple feasible first-stage decision
first_stage_decision = [0.5, 0.5]
println("First-stage decision: $first_stage_decision")

# Calculate cost using cost_2s_LogBarCanLP
evaluated_cost = cost_2s_LogBarCanLP(twoslp, first_stage_decision, reg_param)
println("evaluated_cost (cost_2s_LogBarCanLP): $evaluated_cost")

# Method 2: Calculate opt_cost (optimize both stages)
println("\n=== Method 2: Optimal Two-Stage Solution ===")
logbarlp = LogBarCanLP(twoslp, reg_param)
opt_cost = optimal_value(logbarlp)
println("opt_cost (optimal_value): $opt_cost")

# Compare
gap = evaluated_cost - opt_cost
println("\n=== Comparison ===")
println("evaluated_cost: $evaluated_cost")
println("opt_cost: $opt_cost")
println("Gap (evaluated_cost - opt_cost): $gap")
println("Is gap positive? $(gap > 0)")

# Let's break down what cost_2s_LogBarCanLP actually does
println("\n=== Detailed Breakdown of cost_2s_LogBarCanLP ===")

# First-stage cost
s1_lp = CanLP(A, b, c)
s1_reg_lp = LogBarCanLP(s1_lp, reg_param)
s1_cost = cost(s1_reg_lp, first_stage_decision)
println("First-stage cost: $s1_cost")

# Second-stage cost
s2_constraint_matrix = W
s2_constraint_vector = h - T * first_stage_decision
s2_cost_vector = q * twoslp.ps[1]  # Scale by probability
s2_lp = CanLP(s2_constraint_matrix, s2_constraint_vector, s2_cost_vector)
s2_reg_lp = LogBarCanLP(s2_lp, reg_param * twoslp.ps[1])

println("Second-stage constraint vector: $s2_constraint_vector")
println("Second-stage cost vector (scaled): $s2_cost_vector")
println("Second-stage regularization: $(reg_param * twoslp.ps[1])")

# Solve second-stage optimally
optimal_s2_decision, _ = LogBarCanLP_standard_solver(s2_reg_lp)
s2_cost = cost(s2_reg_lp, optimal_s2_decision)
println("Second-stage cost: $s2_cost")

# Manual total
manual_total = s1_cost + s2_cost
println("Manual calculation total: $manual_total")
println("cost_2s_LogBarCanLP result: $evaluated_cost")
println("Difference: $(manual_total - evaluated_cost)")

# Now let's see what LogBarCanLP does internally
println("\n=== LogBarCanLP Internal Structure ===")
println("LogBarCanLP regularization parameters:")
println("  First stage: $(logbarlp.regularization_parameters[1:length(c)])")
println("  Second stage: $(logbarlp.regularization_parameters[length(c)+1:end])")

# Check if there's a mismatch in regularization application
println("\n=== Regularization Mismatch Check ===")
println("In cost_2s_LogBarCanLP:")
println("  First stage: $reg_param")
println("  Second stage: $(reg_param * twoslp.ps[1])")

println("In LogBarCanLP:")
println("  First stage: $(logbarlp.regularization_parameters[1])")
println("  Second stage: $(logbarlp.regularization_parameters[length(c)+1])")

# The issue might be that LogBarCanLP applies regularization differently
# Let's check if the extensive form has different regularization
println("\n=== Extensive Form Analysis ===")
extensive_lp = extensive_form_canonical(twoslp)
println("Extensive form dimensions: $(size(extensive_lp.A))")
println("Extensive form cost vector length: $(length(extensive_lp.c))")

# Check if the extensive form regularization is different
extensive_reg_lp = LogBarCanLP(extensive_lp, reg_param)
println("Extensive form regularization parameters: $(extensive_reg_lp.regularization_parameters[1:5])...")

println("\n=== Summary ===")
println("The issue appears to be in how regularization is applied")
println("between the two approaches.")
