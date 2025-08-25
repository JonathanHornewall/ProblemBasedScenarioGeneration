# Tests for the first experiments
using Revise
using ProblemBasedScenarioGeneration
using LinearAlgebra
using Flux, ChainRulesCore, ChainRulesTestUtils, FiniteDifferences
using DataLoaders: DataLoader
using SparseArrays
using Statistics
using Plots: plot
using ProblemBasedScenarioGeneration: LogBarCanLP, TwoStageSLP, LogBarCanLP_standard_solver, ResourceAllocationProblemData, 
ResourceAllocationProblem, scenario_realization, dataGeneration, cost, s1_cost, optimal_value,
diff_s1_cost, diff_opt_b, train!, CanLP, extensive_form_canonical, loss, relative_loss, construct_neural_network

import Flux: params, gradient, Optimise, Adam#, 
import ProblemBasedScenarioGeneration: loss, relative_loss   # error if any of these re-appear


# Import data
include("parameters.jl")
cz, qw, ρᵢ, = vec(cz), vec(qw), vec(ρᵢ)

#include("neural_net.jl")
#include("training.jl")
#include("test_function.jl")
#include("tests_SAA/test_function_SAA.jl")

function main()

problem_data = ResourceAllocationProblemData(μᵢⱼ, cz, qw, ρᵢ)
problem_instance = ResourceAllocationProblem(problem_data)

# Generate data
Ntraining_samples = 1
Ntesting_samples = 10
sigma = 5
p =1
L = 3
Σ = 3
N_xi_per_x = 1

data_set_training, data_set_testing =  dataGeneration(problem_instance, Ntraining_samples, Ntesting_samples, N_xi_per_x, sigma, p, L, Σ)

model = construct_neural_network(problem_instance)
# Train the neural network model
reg_param_surr = 1.0
reg_param_prim = 0.0
reg_param_ref = 0.0
batchsize = 1
epochs = 1
step_size = 1e-3
save_model = false

# Defining closure for loss function to run generic neural network training with custom functions
#input_loss(ξ_output, ξ_actual) = loss(problem_instance, reg_param_surr, reg_param_prim, reshape(ξ_output, :, 1), reshape(ξ_actual, :, 1))
#input_relative_loss(ξ_output, ξ_actual) = relative_loss(problem_instance, reg_param_surr, reg_param_prim, reshape(ξ_output, :, 1), reshape(ξ_actual, :, 1))

# Defining closure for loss function to run generic neural network training with loss function from ProblemBasedScenarioGeneration.jl
input_loss(ξ_output, ξ_actual) = loss(problem_instance, reg_param_surr, reg_param_prim, ξ_output, ξ_actual)
input_relative_loss(ξ_output, ξ_actual) = relative_loss(problem_instance, reg_param_surr, reg_param_prim, ξ_output, ξ_actual)

println("Starting training...")

# Train with original loss functions
train!(input_loss, input_relative_loss, model, data_set_training; 
        opt = Adam(step_size), epochs = epochs, batchsize = batchsize, display_iterations = true, 
        save_model = save_model, model_save_path = "trained_model.jls")

println("Training completed!")

#=
# Save the complete experiment state for decision-focused learning results

save_experiment_state(model, data_set_training, data_set_testing, problem_instance, 
                    Dict("reg_param_surr" => reg_param_surr, "reg_param_prim" => reg_param_prim, "reg_param_ref" => reg_param_ref), 
                    filepath = "experiment_state.jls")
                    

# Test the trained model
println("Testing the trained model...")
#test_result = testing(problem_instance, model, data_set_testing, reg_param_surr, reg_param_ref)
test_result = testing_SAA(problem_instance, model, data_set_testing, reg_param_surr, reg_param_ref, N_xi_per_x)
println("Test result: ", test_result)

println("Experiment completed and saved!")
end
=#
end
main()
