# Tests for the first experiments
# This script loads the pre-trained model and evaluates it on out-of-sample data
using ProblemBasedScenarioGeneration
using LinearAlgebra
using Flux, ChainRulesCore, ChainRulesTestUtils, FiniteDifferences
using DataLoaders: DataLoader
using SparseArrays
using Statistics
using Plots: plot
using Serialization
using ProblemBasedScenarioGeneration: LogBarCanLP, TwoStageSLP, LogBarCanLP_standard_solver, ResourceAllocationProblemData, 
ResourceAllocationProblem, scenario_realization, dataGeneration, cost, cost_2s_LogBarCanLP, optimal_value,
diff_cost_2s_LogBarCanLP, diff_opt_b, train_model!, CanLP
import ProblemBasedScenarioGeneration: primal_problem_cost

import Flux: params, gradient, Optimise, Adam

# Import data from local files
include("parameters.jl")
cz, qw, ρᵢ, = vec(cz), vec(qw), vec(ρᵢ)

include("neural_net.jl")
include("training.jl")
include("test_function.jl")

function main()

problem_data = ResourceAllocationProblemData(μᵢⱼ, cz, qw, ρᵢ)
problem_instance = ResourceAllocationProblem(problem_data)

# Load pre-trained model
println("Loading pre-trained model...")
if isfile("trained_model.jls")
    model = deserialize("trained_model.jls")
    println("Pre-trained model loaded successfully!")
else
    error("No trained model found at trained_model.jls")
end

# Generate out-of-sample data
println("Generating out-of-sample data...")
Nsamples = 10
Noutofsamples = 10
sigma = 5
p = 1
L = 3
Σ = 3

data_set_training, data_set_testing = dataGeneration(problem_instance, Nsamples, Noutofsamples, sigma, p, L, Σ)

println("Generated $(length(data_set_testing)) out-of-sample test scenarios")

# Set regularization parameters for testing
reg_param_surr_test = 1.0  # Same as used in training
reg_param_ref_test = 1.0   # Same as used in training

println("Testing model on out-of-sample data...")
println("Regularization parameters: surrogate=$reg_param_surr_test, reference=$reg_param_ref_test")

# Evaluate the model using the testing function
test_gap = testing(problem_instance, model, data_set_testing, reg_param_surr_test, reg_param_ref_test)

println("\n=== FINAL RESULTS ===")
println("Average optimality gap on out-of-sample data: $(round(test_gap * 100, digits=2))%")
println("Model evaluation completed!")

return test_gap

end

# Run the main function if this script is executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
