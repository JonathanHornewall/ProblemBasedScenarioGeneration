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
include(joinpath(@__DIR__, "..", "resource_allocation_prototype", "tests_SAA", "test_function_SAA.jl"))

function main()

problem_data = UnreliableNewsvendorProblemData(p,c,π,η)
problem_instance = UnreliableNewsvendorProblem(problem_data)

# Generate data
Ntraining_samples = 1000
Ntesting_samples = 1
N_xi_per_x = 1000

save_model_training = true

data_set_training, data_set_testing =  dataGeneration(problem_instance, Ntraining_samples, Ntesting_samples, N_xi_per_x)

model = construct_neural_network(problem_instance)
# Train the neural network model

reg_param_ref = 0.0 # I do not what it is used for 
batchsize = 1
epochs = 10
step_size = 1e-4
save_model = true

# Defining closure for loss function to run generic neural network training with custom functions
#input_loss(ξ_output, ξ_actual) = loss(problem_instance, reg_param_surr, reg_param_prim, reshape(ξ_output, :, 1), reshape(ξ_actual, :, 1))
#input_relative_loss(ξ_output, ξ_actual) = relative_loss(problem_instance, reg_param_surr, reg_param_prim, reshape(ξ_output, :, 1), reshape(ξ_actual, :, 1))

# Defining closure for loss function to run generic neural network training with loss function from ProblemBasedScenarioGeneration.jl
input_loss(ξ_output, ξ_actual) = loss(problem_instance, reg_param_surr, reg_param_prim, ξ_output, ξ_actual)

println("Starting training...")

param_list = [1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.08, 0.06, 0.04, 0.02, 0.01]
#param_list = [0.01]
epoch_list = fill(epochs, length(param_list) + 1) # configurable epochs per stage
epoch_list[11] = 20
@assert length(epoch_list) == length(param_list) + 1 "epoch_list must be one longer than param_list"

# we keep only the model from the last annealing stage because of overwriting
filepath = "experiment_state_newsvendor_annealing.jls"

function run_training_stage(reg_param_surr_stage, reg_param_prim_stage, stage_epochs)
        input_loss(ξ_output, ξ_actual) = loss(problem_instance, reg_param_surr_stage, reg_param_prim_stage, ξ_output, ξ_actual)
        input_relative_loss(ξ_output, ξ_actual) = relative_loss(problem_instance, reg_param_surr_stage, reg_param_prim_stage, ξ_output, ξ_actual)

        train!(input_loss, input_relative_loss, model, data_set_training;
                opt = Adam(step_size), epochs = stage_epochs, batchsize = batchsize, display_iterations = true,
                save_model = save_model_training, model_save_path = "trained_model_annealing.jls")

        save_experiment_state(model, data_set_training, data_set_testing, problem_instance,
                Dict("reg_param_surr" => reg_param_surr_stage, "reg_param_prim" => reg_param_prim_stage, "reg_param_ref" => reg_param_ref),
                filepath = filepath)
end

for (idx, reg_param_surr) in enumerate(param_list)
        stage_epochs = epoch_list[idx]
        if idx == length(param_list)
                reg_param_prim_stage = 0.0
        else
                reg_param_prim_stage = reg_param_surr
        end
        println("Starting annealing stage $(idx) with reg_param_surr = $(reg_param_surr), reg_param_prim = $(reg_param_prim_stage), epochs = $(stage_epochs)")
        run_training_stage(reg_param_surr, reg_param_prim_stage, stage_epochs)
end

println("Training completed!")

println(model([1.0])[1], " should be equal to ", z_star*model([1.0])[2])

reg_param_surr = last(param_list)
z = surrogate_solution(problem_instance, reg_param_surr, model([1.0]))[1]
println("z equals ", z, " while z* equals ", z_star)

test_result = testing_SAA(problem_instance, model, data_set_testing, reg_param_surr, reg_param_ref, N_xi_per_x)

return problem_instance, model, data_set_testing

end

problem_instance, model, data_set_testing = main()
