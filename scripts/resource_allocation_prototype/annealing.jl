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
# Testing is skipped for now; this script only runs the annealing training flow.
# include("tests_SAA/test_function_SAA.jl")

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

        data_set_training, data_set_testing, _, _ = dataGeneration(problem_instance, Ntraining_samples, Ntesting_samples, N_xi_per_x, sigma, p, L)

        model = construct_neural_network(problem_instance; nr_of_scenarios = 1)
        # Train the neural network model
        reg_param_ref = 0.0
        batchsize = 1
        default_epochs = 10 # fallback epochs per stage
        step_size = 1e-3
        save_model_training = true

        # Defining closure for loss function to run generic neural network training with custom functions
        #input_loss(ξ_output, ξ_actual) = loss(problem_instance, reg_param_surr, reg_param_prim, reshape(ξ_output, :, 1), reshape(ξ_actual, :, 1))
        #input_relative_loss(ξ_output, ξ_actual) = relative_loss(problem_instance, reg_param_surr, reg_param_prim, reshape(ξ_output, :, 1), reshape(ξ_actual, :, 1))

        println("Starting training with annealing...")

        # Train with original loss functions

        param_list = [1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.08, 0.06, 0.04, 0.02, 0.01]
        epoch_list = fill(default_epochs, length(param_list) + 1) # configurable epochs per stage
        epoch_list[1] = 20
        @assert length(epoch_list) == length(param_list) + 1 "epoch_list must be one longer than param_list"

        function run_training_stage(reg_param_surr_stage, reg_param_prim_stage, stage_epochs)
            input_loss(ξ_output, ξ_actual) = loss(problem_instance, reg_param_surr_stage, reg_param_prim_stage, ξ_output, ξ_actual)
            input_relative_loss(ξ_output, ξ_actual) = relative_loss(problem_instance, reg_param_surr_stage, reg_param_prim_stage, ξ_output, ξ_actual)

            train!(input_loss, input_relative_loss, model, data_set_training;
                    opt = Adam(step_size), epochs = stage_epochs, batchsize = batchsize, display_iterations = true,
                    save_model = save_model_training, model_save_path = "trained_model_annealing.jls")

            save_experiment_state(model, data_set_training, data_set_testing, problem_instance,
                    Dict("reg_param_surr" => reg_param_surr_stage, "reg_param_prim" => reg_param_prim_stage, "reg_param_ref" => reg_param_ref),
                    filepath = "experiment_state_annealing.jls")
        end

        for (idx, reg_param_surr) in enumerate(param_list)
            stage_epochs = epoch_list[idx]
            reg_param_prim_stage = reg_param_surr
            println("Starting annealing stage $(idx) with reg_param_surr = $(reg_param_surr), reg_param_prim = $(reg_param_prim_stage), epochs = $(stage_epochs)")
            run_training_stage(reg_param_surr, reg_param_prim_stage, stage_epochs)
        end

        final_reg_param_surr = param_list[end]
        final_stage_epochs = epoch_list[end]
        final_reg_param_prim = 0.0
        println("Starting final annealing stage with reg_param_surr = $(final_reg_param_surr), reg_param_prim = $(final_reg_param_prim), epochs = $(final_stage_epochs)")
        run_training_stage(final_reg_param_surr, final_reg_param_prim, final_stage_epochs)

        println("Training completed!")

        println("Experiment completed and saved!")
end
main()
