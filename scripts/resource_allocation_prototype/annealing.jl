# Tests for the first experiments
import Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

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

include("custom_code/neural_net.jl")


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
        N_xi_per_x = 100
        collections_per_sample = 1

        data_set_training, data_set_testing, _, _ = dataGeneration(problem_instance, Ntraining_samples, Ntesting_samples, N_xi_per_x, sigma, p, L, collections_per_sample)
        collections_per_sample = size(first(values(data_set_testing)), 1)

        model = construct_neural_network(problem_instance; nr_of_scenarios = 1)
        # Train the neural network model
        reg_param_ref = 0.0
        base_batchsize = 1
        default_epochs = 10 # fallback epochs per stage
        base_step_size = 1e-3
        save_model_training = false

        # Defining closure for loss function to run generic neural network training with custom functions
        #input_loss(ξ_output, ξ_actual) = loss(problem_instance, reg_param_surr, reg_param_prim, reshape(ξ_output, :, 1), reshape(ξ_actual, :, 1))
        #input_relative_loss(ξ_output, ξ_actual) = relative_loss(problem_instance, reg_param_surr, reg_param_prim, reshape(ξ_output, :, 1), reshape(ξ_actual, :, 1))

        println("Starting training with annealing...")

        # Train with original loss functions

        #param_list = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01]
        param_list = [0.01]

        epoch_list = fill(default_epochs, length(param_list) + 1) # configurable epochs per stage
        epoch_list[1] = 3 * default_epochs

        # Step-size schedule fixed to keep learning rate consistent across stages
        step_size_schedule = fill(base_step_size, length(param_list)+1)
        step_size_schedule[2]=base_step_size/5

        # Batch-size schedule grows as μ shrinks to stabilise later stages (clamped to keep epochs practical)
        batchsize_schedule = [
            clamp(round(Int, base_batchsize * (param_list[1] / μ)), base_batchsize, 4 * base_batchsize)
            for μ in param_list
        ]
        push!(batchsize_schedule, batchsize_schedule[end]) # final stage

        @assert length(epoch_list) == length(step_size_schedule) == length(batchsize_schedule) == length(param_list) + 1 "Schedules must align with annealing stages"

        function run_training_stage(reg_param_surr_stage, reg_param_prim_stage, stage_epochs, stage_step_size, stage_batchsize)
            input_loss(ξ_output, ξ_actual) = loss(problem_instance, reg_param_surr_stage, reg_param_prim_stage, ξ_output, ξ_actual)
            input_relative_loss(ξ_output, ξ_actual) = relative_loss(problem_instance, reg_param_surr_stage, reg_param_prim_stage, ξ_output, ξ_actual)

            train!(input_loss, input_relative_loss, model, data_set_training;
                    opt = Adam(stage_step_size), epochs = stage_epochs, batchsize = stage_batchsize, display_iterations = true,
                    save_model = save_model_training, model_save_path = "trained_model_annealing.jls")

            save_experiment_state(model, data_set_training, data_set_testing, problem_instance,
                    Dict("reg_param_surr" => reg_param_surr_stage, "reg_param_prim" => reg_param_prim_stage, "reg_param_ref" => reg_param_ref),
                    filepath = "experiment_state_annealing.jls")
        end

        for (idx, reg_param_surr) in enumerate(param_list)
            stage_epochs = epoch_list[idx]
            reg_param_prim_stage = reg_param_surr
            stage_step_size = step_size_schedule[idx]
            stage_batchsize = batchsize_schedule[idx]
            println("Starting annealing stage $(idx) with reg_param_surr = $(reg_param_surr), reg_param_prim = $(reg_param_prim_stage), epochs = $(stage_epochs), step_size = $(stage_step_size), batchsize = $(stage_batchsize)")
            run_training_stage(reg_param_surr, reg_param_prim_stage, stage_epochs, stage_step_size, stage_batchsize)
        end

        final_reg_param_surr = param_list[end]
        final_stage_epochs = epoch_list[end]
        final_reg_param_prim = 0.0
        final_step_size = step_size_schedule[end]
        final_batchsize = batchsize_schedule[end]
        println("Starting final annealing stage with reg_param_surr = $(final_reg_param_surr), reg_param_prim = $(final_reg_param_prim), epochs = $(final_stage_epochs), step_size = $(final_step_size), batchsize = $(final_batchsize)")
        run_training_stage(final_reg_param_surr, final_reg_param_prim, final_stage_epochs, final_step_size, final_batchsize)

        println("Training completed!")

        # Test the trained model
        println("Testing the trained model...")
        test_result = testing_SAA(problem_instance, model, data_set_testing, final_reg_param_surr, reg_param_ref, N_xi_per_x)
        println("Test result: ", test_result)

        println("Testing the trained model, this time with reg_param_surr = 0")
        test_result_zero = testing_SAA(problem_instance, model, data_set_testing, 0, reg_param_ref, N_xi_per_x)
        println("Test result: ", test_result_zero)

        println("\n=== Baseline training with mean-square loss (no annealing) ===")
        mse_model = construct_neural_network(problem_instance; nr_of_scenarios = 1)
        mse_loss(ξ_output, ξ_actual) = norm(ξ_output .- ξ_actual)^2
        mse_relative_loss(ξ_output, ξ_actual) = relative_loss(problem_instance, final_reg_param_surr, reg_param_ref, ξ_output, ξ_actual)
        train!(mse_loss, mse_relative_loss, mse_model, data_set_training;
                opt = Adam(base_step_size), epochs = 60, batchsize = 10, display_iterations = true,
                save_model = false)
        println("Testing the baseline MSE-trained model...")
        mse_result = testing_SAA(problem_instance, mse_model, data_set_testing, final_reg_param_surr, reg_param_ref, N_xi_per_x)
        println("Baseline MSE test result: ", mse_result)

        println("Experiment completed and saved!")
end
main()
