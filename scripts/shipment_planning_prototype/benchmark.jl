# Benchmark script for shipment planning prototype
import Pkg
Pkg.activate(@__DIR__)

using ProblemBasedScenarioGeneration
using Serialization
using Statistics
using Plots

using ProblemBasedScenarioGeneration: ShipmentPlanningProblem, shipment_planning_problem_data,
    construct_neural_network

include("custom_code/neural_net.jl")
include("custom_code/load_experiment.jl")
include("parameters.jl")
include("tests_SAA/test_function_SAA.jl")

function run_benchmark()
    println("=== Shipment Planning Benchmark ===")
    model_path = "shipment_trained_model.jls"
    experiment_path = "shipment_experiment_state.jls"

    println("Loading trained model from $model_path ...")
    model = load_trained_model(model_path)
    println("✓ Model loaded")

    println("Loading experiment state from $experiment_path ...")
    _, _, testing_data, problem_instance, reg_params = load_experiment_state(experiment_path)
    reg_param_surr = reg_params["reg_param_surr"]
    reg_param_ref = reg_params["reg_param_ref"]
    println("Regularization parameters: surrogate=$(reg_param_surr), reference=$(reg_param_ref)")

    cfg = shipment_training_config
    N_xi_per_x = cfg[:N_xi_per_x]

    println("\nRunning SAA-style evaluation ...")
    test_result = testing_SAA(problem_instance, model, testing_data, reg_param_surr, reg_param_ref, N_xi_per_x)

    println("Benchmark completed. Results written to shipment_gap_boxplot.pdf")
    return test_result
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_benchmark()
end
