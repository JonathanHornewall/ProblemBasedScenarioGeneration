# Refactored variant of `scripts/resource_allocation_prototype/main.jl`
# that swaps in the new loss / gradient implementation while keeping the
# remainder of the experiment workflow identical.

using Revise
using ProblemBasedScenarioGeneration
using LinearAlgebra
using Flux, ChainRulesCore, FiniteDifferences
using DataLoaders: DataLoader
using SparseArrays
using Statistics
using Plots: plot
using ProblemBasedScenarioGeneration: LogBarCanLP, TwoStageSLP, LogBarCanLP_standard_solver,
    ResourceAllocationProblemData, ResourceAllocationProblem, scenario_realization, dataGeneration,
    cost, s1_cost, optimal_value, diff_s1_cost, diff_opt_b, train!, CanLP, extensive_form_canonical,
    construct_neural_network

import Flux: params, gradient, Optimise, Adam

# Bring in the refactored loss implementation.
include("RefactoredLoss.jl")
using .RefactoredLoss

# Locate the original prototype assets (parameters, custom NN utilities, tests).
const _prototype_dir = normpath(joinpath(@__DIR__, "..", "..", "..", "..", "..", "..",
    "scripts", "resource_allocation_prototype"))

include(joinpath(_prototype_dir, "parameters.jl"))
include(joinpath(_prototype_dir, "custom_code", "neural_net.jl"))
cz, qw, ρᵢ = vec(cz), vec(qw), vec(ρᵢ)

include(joinpath(_prototype_dir, "tests_SAA", "test_function_SAA.jl"))

function main(; epochs::Int = 30)
    problem_data = ResourceAllocationProblemData(μᵢⱼ, cz, qw, ρᵢ)
    problem_instance = ResourceAllocationProblem(problem_data)

    # Generate data
    Ntraining_samples = 100
    Ntesting_samples = 100
    sigma = 5
    p = 1
    L = 3
    Σ = 3
    N_xi_per_x = 100
    collections_per_sample = 1

    data_set_training, data_set_testing =
        dataGeneration(problem_instance, Ntraining_samples, Ntesting_samples,
                       N_xi_per_x, sigma, p, L, collections_per_sample)
    collections_per_sample = size(first(values(data_set_testing)), 1)

    model = construct_neural_network(problem_instance; nr_of_scenarios = 5)
    # Train the neural network model
    reg_param_surr = 0.01
    reg_param_prim = 0.01
    reg_param_ref = 0.0
    batchsize = 1
    step_size = 1e-3
    save_model_training = true

    state_dir = joinpath("experiment_states", "main")
    mkpath(state_dir)

    # Loss closures using the refactored implementation
    input_loss(ξ_output, ξ_actual) =
        refactored_loss(problem_instance, reg_param_surr, reg_param_prim,
                        ξ_output, ξ_actual)

    function input_relative_loss(ξ_output, ξ_actual)
        predicted = refactored_loss(problem_instance, reg_param_surr, reg_param_prim,
                                    ξ_output, ξ_actual)
        reference = refactored_loss(problem_instance, reg_param_prim, reg_param_prim,
                                    ξ_actual, ξ_actual)
        return (predicted - reference) / abs(reference)
    end

    println("Starting training...")

    model_save_path = joinpath(state_dir, "trained_model.jls")
    train!(input_loss, input_relative_loss, model, data_set_training;
           opt = Adam(step_size), epochs = epochs, batchsize = batchsize,
           display_iterations = true, save_model = save_model_training,
           model_save_path = model_save_path)

    println("Training completed!")

    save_experiment_state(model, data_set_training, data_set_testing, problem_instance,
                          Dict("reg_param_surr" => reg_param_surr,
                               "reg_param_prim" => reg_param_prim,
                               "reg_param_ref" => reg_param_ref),
                          filepath = joinpath(state_dir, "experiment_state.jls"))

    println("Testing the trained model...")
    test_result = testing_SAA(problem_instance, model, data_set_testing,
                              reg_param_surr, reg_param_ref, N_xi_per_x)
    println("Test result: ", test_result)

    println("Experiment completed and saved!")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
