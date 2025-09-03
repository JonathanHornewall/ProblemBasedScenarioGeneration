using Revise
using ProblemBasedScenarioGeneration
using LinearAlgebra
using Flux, ChainRulesCore, ChainRulesTestUtils, FiniteDifferences
using DataLoaders: DataLoader
using SparseArrays
using Statistics
using Plots: plot
using ProblemBasedScenarioGeneration: LogBarCanLP, TwoStageSLP, LogBarCanLP_standard_solver, UnreliableNewsvendorProblemData, 
UnreliableNewsvendorProblem, scenario_realization, dataGeneration, cost, s1_cost, optimal_value,
diff_s1_cost, diff_opt_b, train!, CanLP, extensive_form_canonical, loss, relative_loss, construct_neural_network

import Flux: params, gradient, Optimise, Adam#, 
import ProblemBasedScenarioGeneration: loss, relative_loss   # error if any of these re-appear


include("optimality_test.jl")


function main()

problem_data = UnreliableNewsvendorProblemData(p,c,π,η)
problem_instance = UnreliableNewsvendorProblem(problem_data)

# Generate data
Ntraining_samples = 10000
Ntesting_samples = 1
N_xi_per_x = 1000

data_set_training, data_set_testing =  dataGeneration(problem_instance, Ntraining_samples, Ntesting_samples, N_xi_per_x)

model = construct_neural_network(problem_instance)
# Train the neural network model
reg_param_surr = 1.0 # to compute the first stage solution
reg_param_prim = 1.0 # to compute the cost of the first stage solution
reg_param_ref = 0.0 # I do not what it is used for 
batchsize = 1
epochs = 15
step_size = 1e-4
save_model = true

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
        save_model = save_model, model_save_path = "trained_model_with_prim_regulariz.jls")

println("Training completed!")

println(model([1.0])[1], " should be equal to ", z_star*model([1.0])[2])

z = surrogate_solution(problem_instance, reg_param_surr, model([1.0]))[1]
println("z equals ", z, " while z* equals ", z_star)

return model

end
model = main()


problem_data = UnreliableNewsvendorProblemData(p,c,π,η)
problem_instance = UnreliableNewsvendorProblem(problem_data)

# Generate data
Ntraining_samples = 10000
Ntesting_samples = 1
N_xi_per_x = 1000

data_set_training, data_set_testing =  dataGeneration(problem_instance, Ntraining_samples, Ntesting_samples, N_xi_per_x)

model = construct_neural_network(problem_instance)
# Train the neural network model
reg_param_surr = 1.0 # to compute the first stage solution
reg_param_prim = 1.0 # to compute the cost of the first stage solution
reg_param_ref = 0.0 # I do not what it is used for 
testing_SAA(problem_instance, model, data_set_testing, reg_param_surr, reg_param_ref, N_xi_per_x)
