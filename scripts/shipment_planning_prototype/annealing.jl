# Prototype annealing experiment for the shipment planning problem
import Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

using Revise
using ProblemBasedScenarioGeneration
using LinearAlgebra
using Flux, ChainRulesCore, ChainRulesTestUtils, FiniteDifferences
using Statistics
using Plots
using ProblemBasedScenarioGeneration: ShipmentPlanningProblem, ShipmentPlanningProblemData,
    shipment_planning_problem_data, dataGeneration, loss, relative_loss,
    construct_neural_network

include("custom_code/neural_net.jl")
include("custom_code/load_experiment.jl")
include("parameters.jl")

function main()
    problem_instance = ShipmentPlanningProblem(shipment_problem_data)

    cfg = shipment_training_config
    data_set_training, data_set_testing = dataGeneration(
        problem_instance,
        cfg[:Ntraining_samples],
        cfg[:Ntesting_samples],
        cfg[:N_xi_per_x],
        cfg[:sigma],
        cfg[:seasonal_scale],
        cfg[:trend_decay];
        collections_per_sample = cfg[:collections_per_sample]
    )

    model = construct_neural_network(problem_instance; nr_of_scenarios = 1)
    reg_param_ref = 0.0
    base_batchsize = 10
    default_epochs = 20
    base_step_size = 1e-3
    save_model_training = true
    experiment_path = "shipment_experiment_state.jls"
    model_save_path = "shipment_trained_model.jls"

    println("Starting shipment planning annealing loop...")

    param_list = [3.0, 1.0, 0.1, 0.01]
    epoch_list = fill(default_epochs, length(param_list) + 1)
    epoch_list[1] = Int(ceil(2.5 * default_epochs))

    step_size_schedule = fill(base_step_size, length(param_list) + 1)
    for i in eachindex(param_list)
        step_size_schedule[i] = sqrt(param_list[i] / param_list[1]) * base_step_size
    end
    step_size_schedule[end] = step_size_schedule[end - 1]

    batchsize_schedule = [
        clamp(round(Int, base_batchsize * (param_list[1] / μ)), base_batchsize, min(4 * base_batchsize, 32))
        for μ in param_list
    ]
    push!(batchsize_schedule, batchsize_schedule[end])

    function run_training_stage(reg_param_surr_stage, reg_param_prim_stage, stage_epochs, stage_step_size, stage_batchsize)
        input_loss(ξ_output, ξ_actual) = loss(problem_instance, reg_param_surr_stage, reg_param_prim_stage, ξ_output, ξ_actual)
        input_relative_loss(ξ_output, ξ_actual) = relative_loss(problem_instance, reg_param_surr_stage, reg_param_prim_stage, ξ_output, ξ_actual)

        train!(input_loss, input_relative_loss, model, data_set_training;
               opt = Adam(stage_step_size), epochs = stage_epochs, batchsize = stage_batchsize,
               display_iterations = true, save_model = save_model_training,
               model_save_path = model_save_path)

        reg_params = Dict("reg_param_surr" => reg_param_surr_stage,
                          "reg_param_prim" => reg_param_prim_stage,
                          "reg_param_ref" => reg_param_ref)
        save_experiment_state(model, data_set_training, data_set_testing, problem_instance, reg_params;
                              filepath = experiment_path)
    end

    for (idx, reg_param_surr) in enumerate(param_list)
        stage_epochs = epoch_list[idx]
        reg_param_prim_stage = reg_param_surr
        stage_step_size = step_size_schedule[idx]
        stage_batchsize = batchsize_schedule[idx]
        println("Stage $(idx): μ = $(reg_param_surr), epochs = $(stage_epochs), lr = $(stage_step_size), batch = $(stage_batchsize)")
        run_training_stage(reg_param_surr, reg_param_prim_stage, stage_epochs, stage_step_size, stage_batchsize)
    end

    final_reg_param_surr = param_list[end]
    final_stage_epochs = epoch_list[end]
    final_reg_param_prim = 0.0
    run_training_stage(final_reg_param_surr, final_reg_param_prim, final_stage_epochs, step_size_schedule[end], batchsize_schedule[end])

    println("Training completed. Evaluating on held-out contexts...")
    function reshape_scenarios(sample)
        if ndims(sample) == 1
            return reshape(sample, :, 1)
        elseif ndims(sample) == 2
            return sample
        elseif ndims(sample) == 3
            collections, N_xi, nloc = size(sample)
            return reshape(permutedims(sample, (3, 1, 2)), nloc, :)
        else
            error("Unsupported scenario tensor dimensions: $(ndims(sample))")
        end
    end

    function evaluate_gap(dataset)
        rel_losses = Float64[]
        for (x, ξ) in dataset
            ξ̂ = model(x)
            scenarios = reshape_scenarios(ξ)
            push!(rel_losses, relative_loss(problem_instance, final_reg_param_surr, reg_param_ref, ξ̂, scenarios))
        end
        mean(rel_losses)
    end

    println("Average relative loss on training set: ", evaluate_gap(data_set_training))
    println("Average relative loss on testing set: ", evaluate_gap(data_set_testing))
end

main()
