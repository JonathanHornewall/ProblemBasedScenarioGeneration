# Regression test for the trained neural network model
using Revise
using ProblemBasedScenarioGeneration
using LinearAlgebra
using Flux, ChainRulesCore, ChainRulesTestUtils, FiniteDifferences
using DataLoaders: DataLoader
using SparseArrays
using Statistics
using Plots: plot
using JLD2
using ProblemBasedScenarioGeneration: LogBarCanLP, TwoStageSLP, LogBarCanLP_standard_solver, ResourceAllocationProblemData, 
ResourceAllocationProblem, scenario_realization, dataGeneration, cost, s1_cost, optimal_value,
diff_s1_cost, diff_opt_b, train_model!, CanLP, extensive_form_canonical
import ProblemBasedScenarioGeneration: primal_problem_cost

import Flux: params, gradient, Optimise, Adam

# Import data
include("parameters.jl")
cz, qw, ρᵢ, = vec(cz), vec(qw), vec(ρᵢ)

include("neural_net.jl")
include("training.jl")
include("test_function.jl")
include("tests_SAA/test_function_SAA.jl")
include("load_experiment.jl")

function regression_test()

println("=== REGRESSION TEST ===")

# Load the trained model using the correct method
println("Loading trained model...")
model = load_trained_model("trained_model_mse.jls")
println("Model loaded successfully!")

# Load regression parameters
println("Loading regression parameters...")
@load "regression_params.jld2" reg_params
println("Parameters loaded: ", reg_params)

# Create problem instance
problem_data = ResourceAllocationProblemData(μᵢⱼ, cz, qw, ρᵢ)
problem_instance = ResourceAllocationProblem(problem_data)

# Generate test data
println("Generating test data...")
Ntraining_samples = 50
Ntesting_samples = 10
sigma = 5
p = 1
L = 3
Σ = 3
N_xi_per_x = 20

data_set_training, data_set_testing = dataGeneration(problem_instance, Ntraining_samples, Ntesting_samples, N_xi_per_x, sigma, p, L, Σ)

# Test the model on a few samples
println("\n=== TESTING MODEL PERFORMANCE ===")
println("Testing on first 5 training samples:")

total_mse = 0.0
total_samples = 0

for (i, (x, ξ_actual)) in enumerate(collect(data_set_training)[1:5])
    # Get model prediction
    ξ_predicted = model(x)
    
    # Calculate MSE for this sample
    mse_sample = mean((ξ_predicted .- ξ_actual).^2)
    total_mse += mse_sample
    total_samples += 1
    
    println("Sample $i:")
    println("  Input x: $x")
    println("  Actual ξ range: $(minimum(ξ_actual)) to $(maximum(ξ_actual))")
    println("  Predicted ξ range: $(minimum(ξ_predicted)) to $(maximum(ξ_predicted))")
    println("  Sample MSE: $(round(mse_sample, digits=2))")
    println()
end

avg_mse = total_mse / total_samples
println("Average MSE across $total_samples samples: $(round(avg_mse, digits=2))")

# Test on test data
println("\n=== TESTING ON UNSEEN DATA ===")
println("Testing on first 3 test samples:")

test_total_mse = 0.0
test_total_samples = 0

for (i, (x, ξ_actual)) in enumerate(collect(data_set_testing)[1:3])
    # Get model prediction
    ξ_predicted = model(x)
    
    # Calculate MSE for this sample
    mse_sample = mean((ξ_predicted .- ξ_actual).^2)
    test_total_mse += mse_sample
    test_total_samples += 1
    
    println("Test Sample $i:")
    println("  Input x: $x")
    println("  Actual ξ range: $(minimum(ξ_actual)) to $(maximum(ξ_actual))")
    println("  Predicted ξ range: $(minimum(ξ_predicted)) to $(maximum(ξ_predicted))")
    println("  Sample MSE: $(round(mse_sample, digits=2))")
    println()
end

test_avg_mse = test_total_mse / test_total_samples
println("Average MSE on test data: $(round(test_avg_mse, digits=2))")

# Performance summary
println("=== PERFORMANCE SUMMARY ===")
println("Training data average MSE: $(round(avg_mse, digits=2))")
println("Test data average MSE: $(round(test_avg_mse, digits=2))")
println("Generalization gap: $(round(test_avg_mse - avg_mse, digits=2))")

if test_avg_mse > avg_mse * 1.5
    println("⚠️  Warning: Model may be overfitting (test MSE significantly higher than training MSE)")
else
    println("✅ Model generalization looks good!")
end

println("\nRegression test completed!")
end

regression_test()
