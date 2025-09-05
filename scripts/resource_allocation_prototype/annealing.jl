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

import Flux: params, gradient, Optimise, Adam
import ProblemBasedScenarioGeneration: loss, relative_loss, surrogate_solution

include("/home/delannoypavysol/ProblemBasedScenarioGeneration/ProblemBasedScenarioGeneration/scripts/resource_allocation_prototype/outdated/neural_net.jl")


# Import data
include("parameters.jl")
cz, qw, ρᵢ, = vec(cz), vec(qw), vec(ρᵢ)

#include("../outdated/neural_net.jl")
#include("../outdated/training.jl")
#include("../outdated/test_function.jl")
include("tests_SAA/test_function_SAA.jl")

function main()
        problem_data = ResourceAllocationProblemData(μᵢⱼ, cz, qw, ρᵢ)
        problem_instance = ResourceAllocationProblem(problem_data)

        # Generate data
        Ntraining_samples = 100
        Ntesting_samples = 30
        sigma = 5 
        p = 2
        L = 3 
        N_xi_per_x = 1000

        data_set_training, data_set_testing, _, _=  dataGeneration(problem_instance, Ntraining_samples, Ntesting_samples, N_xi_per_x, sigma, p, L, nothing, nothing)

        model = construct_neural_network(problem_instance; nr_of_scenarios = 1)
        # Train the neural network model
        reg_param_prim = 0.0
        reg_param_ref = 0.0
        batchsize = 1
        epochs = 10 #nb of epoch per  reg parameter
        step_size = 1e-3
        save_model_training = true

        # Defining closure for loss function to run generic neural network training with custom functions
        #input_loss(ξ_output, ξ_actual) = loss(problem_instance, reg_param_surr, reg_param_prim, reshape(ξ_output, :, 1), reshape(ξ_actual, :, 1))
        #input_relative_loss(ξ_output, ξ_actual) = relative_loss(problem_instance, reg_param_surr, reg_param_prim, reshape(ξ_output, :, 1), reshape(ξ_actual, :, 1))

        println("Starting training with annealing...")

        # Train with original loss functions

        param_list = [1.0, 0.7, 0.4, 0.1]

        for reg_param_surr in param_list

            input_loss(ξ_output, ξ_actual) = loss(problem_instance, reg_param_surr, reg_param_prim, ξ_output, ξ_actual)
            input_relative_loss(ξ_output, ξ_actual) = relative_loss(problem_instance, reg_param_surr, reg_param_prim, ξ_output, ξ_actual)

            train!(input_loss, input_relative_loss, model, data_set_training; 
                    opt = Adam(step_size), epochs = epochs, batchsize = batchsize, display_iterations = true, 
                    save_model = save_model_training, model_save_path = "trained_model_annealing.jls")

            save_experiment_state(model, data_set_training, data_set_testing, problem_instance, 
                    Dict("reg_param_surr" => reg_param_surr, "reg_param_prim" => reg_param_prim, "reg_param_ref" => reg_param_ref), 
                    filepath = "experiment_state_annealing.jls")

        end 

        println("Training completed!")

        # Test the trained model
        println("Testing the trained model...")
        test_result = testing_SAA(problem_instance, model, data_set_testing, param_list[end], reg_param_ref, N_xi_per_x)
        println("Test result: ", test_result)

        println("Experiment completed and saved!")
end
main()
