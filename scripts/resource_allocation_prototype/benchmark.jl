# Benchmark script for evaluating trained neural network models
using Revise
using ProblemBasedScenarioGeneration
using LinearAlgebra
using Flux, ChainRulesCore, ChainRulesTestUtils, FiniteDifferences
using DataLoaders: DataLoader
using SparseArrays
using Statistics
using Plots: plot
using Serialization
using ProblemBasedScenarioGeneration: LogBarCanLP, TwoStageSLP, LogBarCanLP_standard_solver, ResourceAllocationProblemData, 
ResourceAllocationProblem, scenario_realization, dataGeneration, cost, s1_cost, optimal_value,
diff_s1_cost, diff_opt_b, train!, CanLP, extensive_form_canonical, loss, relative_loss, construct_neural_network,
load_trained_model, surrogate_solution

import Flux: params, gradient, Optimise, Adam
import ProblemBasedScenarioGeneration: loss, relative_loss

# Import the testing function
include("tests_SAA/test_function_SAA.jl")

function run_benchmark()
    println("=== Neural Network Model Benchmark ===")
    
    # Load the latest trained model
    println("Loading latest trained model...")
    model = load_trained_model("trained_model.jls")
    println("✓ Model loaded successfully")

    # Reconstruct problem instance from parameters
    include("parameters.jl")
    cz_v, qw_v, ρᵢ_v = vec(cz), vec(qw), vec(ρᵢ)
    problem_data = ResourceAllocationProblemData(μᵢⱼ, cz_v, qw_v, ρᵢ_v)
    problem_instance = ResourceAllocationProblem(problem_data)

    # Load testing data and regularization parameters from saved experiment
    println("Loading testing data and parameters from experiment_state.jls...")
    _, _, testing_data, _, reg_params = deserialize("experiment_state.jls")
    reg_param_surr = reg_params["reg_param_surr"]
    reg_param_ref = reg_params["reg_param_ref"]

    println("Regularization parameters:")
    println("  - Surrogate: $reg_param_surr")
    println("  - Reference: $reg_param_ref")
    
    # Set testing parameters
    N_xi_per_x = 100  # Same as in main.jl
    
    println("\n=== Running SAA Testing ===")
    println("Testing dataset size: $(length(testing_data))")
    println("N_xi_per_x: $N_xi_per_x")
    
    # Run the SAA testing
    test_result = testing_SAA(problem_instance, model, testing_data, reg_param_surr, reg_param_ref, N_xi_per_x)
    
    println("\n=== Benchmark Complete ===")
    println("Results have been saved to gap_boxplot.pdf")
    
    return test_result
end

# Run the benchmark if this file is executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    run_benchmark()
end
