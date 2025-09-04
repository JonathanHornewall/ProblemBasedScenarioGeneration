"""
Experimental save/load utilities, copied from scripts.

Provides helpers to save and load trained models, datasets, and complete
experiment states using Serialization. Also includes convenience routines to
continue training and compare models.
"""

using ProblemBasedScenarioGeneration
using Flux
using LinearAlgebra
import Serialization

"""
    save_trained_model(model, filepath)

Save a trained Flux model to a file using Julia's built-in Serialization.
"""
function save_trained_model(model, filepath)
    Serialization.serialize(filepath, model)
end

"""
    load_trained_model(filepath)

Load a trained Flux model from a file using Julia's built-in Serialization.
"""
function load_trained_model(filepath)
    return Serialization.deserialize(filepath)
end

"""
    save_training_data(training_data, testing_data, filepath)

Save training and testing datasets to a file using Julia's built-in Serialization.
"""
function save_training_data(training_data, testing_data, filepath)
    Serialization.serialize(filepath, (training_data, testing_data))
end

"""
    load_training_data(filepath)

Load training and testing datasets from a file using Julia's built-in Serialization.
"""
function load_training_data(filepath)
    training_data, testing_data = Serialization.deserialize(filepath)
    return training_data, testing_data
end

"""
    save_experiment_state(model, training_data, testing_data, problem_instance,
                          reg_params; filepath = "experiment_state.jls")

Save the complete experiment state including model, data, and parameters.
"""
function save_experiment_state(model, training_data, testing_data, problem_instance,
                               reg_params; filepath = "experiment_state.jls")
    # Extract key parameters from problem instance
    problem_data = Dict(
        "s1_constraint_matrix" => problem_instance.s1_constraint_matrix,
        "s1_constraint_vector" => problem_instance.s1_constraint_vector,
        "s1_cost_vector" => problem_instance.s1_cost_vector,
    )

    # Save everything
    Serialization.serialize(filepath, (model, training_data, testing_data, problem_data, reg_params))
    println("Complete experiment state saved to: $filepath")
end

"""
    load_experiment_state(filepath)

Load the complete experiment state from a file.
"""
function load_experiment_state(filepath)
    model, training_data, testing_data, problem_data, reg_params = Serialization.deserialize(filepath)

    # Reconstruct problem instance
    problem_instance = ResourceAllocationProblem(ResourceAllocationProblemData(
        problem_data["s1_constraint_matrix"],
        problem_data["s1_constraint_vector"],
        problem_data["s1_cost_vector"],
    ))

    return model, training_data, testing_data, problem_instance, reg_params
end

"""
    load_and_continue_experiment(experiment_file = "experiment_state.jls")

Load a saved experiment and return all components for continued work.
"""
function load_and_continue_experiment(experiment_file = "experiment_state.jls")
    println("Loading experiment from: $experiment_file")

    # Load the complete experiment state
    model, training_data, testing_data, problem_instance, reg_params = load_experiment_state(experiment_file)

    println("✓ Model loaded successfully")
    println("✓ Training data loaded: $(length(training_data)) samples")
    println("✓ Testing data loaded: $(length(testing_data)) samples")
    println("✓ Problem instance reconstructed")
    println("✓ Regularization parameters: $reg_params")

    return model, training_data, testing_data, problem_instance, reg_params
end

"""
    continue_training(experiment_file = "experiment_state.jls",
                      additional_epochs = 10,
                      learning_rate = 1e-3)

Continue training a loaded model for additional epochs.
"""
function continue_training(experiment_file = "experiment_state.jls",
                           additional_epochs = 10,
                           learning_rate = 1e-3)
    model, training_data, testing_data, problem_instance, reg_params = load_and_continue_experiment(experiment_file)

    # Extract regularization parameters
    reg_param_surr = reg_params["reg_param_surr"]
    reg_param_ref = reg_params["reg_param_ref"]

    println("\n=== Continuing Training ===")
    println("Additional epochs: $additional_epochs")
    println("Learning rate: $learning_rate")

    # Continue training (this will call train! from the package training code)
    train!(problem_instance, reg_param_surr, reg_param_ref, model, training_data;
           opt = Adam(learning_rate), epochs = additional_epochs,
           display_iterations = true, save_model = true,
           model_save_path = "continued_training_model.jld2")

    # Save the updated experiment state
    save_experiment_state(model, training_data, testing_data, problem_instance, reg_params;
                          filepath = "continued_experiment_state.jld2")

    println("\n✓ Continued training completed and saved!")
    return model, training_data, testing_data, problem_instance, reg_params
end

"""
    compare_models(original_file = "experiment_state.jls",
                   continued_file = "continued_experiment_state.jls")

Compare the performance of original and continued training models.
"""
function compare_models(original_file = "experiment_state.jls",
                        continued_file = "continued_experiment_state.jls")
    println("=== Model Comparison ===")

    # Load original model
    println("Loading original model...")
    model_orig, _, testing_data, problem_instance, reg_params = load_experiment_state(original_file)

    # Load continued training model
    println("Loading continued training model...")
    model_cont, _, _, _, _ = load_experiment_state(continued_file)

    # Extract regularization parameters
    reg_param_surr = reg_params["reg_param_surr"]
    reg_param_ref = reg_params["reg_param_ref"]

    # Test both models (relies on package-provided testing function)
    println("\nTesting original model...")
    test_orig = testing(problem_instance, model_orig, testing_data, reg_param_surr, reg_param_ref)

    println("Testing continued training model...")
    test_cont = testing(problem_instance, model_cont, testing_data, reg_param_surr, reg_param_ref)

    println("\n=== Results ===")
    println("Original model test gap: $test_orig")
    println("Continued training test gap: $test_cont")
    println("Improvement: $(test_orig - test_cont)")

    return test_orig, test_cont
end

function example_usage()
    println("=== Example Usage ===")
    println("1. Load and test a saved model:")
    println("   model, data, test_data, problem, params = load_and_continue_experiment()")
    println()
    println("2. Continue training for 5 more epochs:")
    println("   continue_training(\"experiment_state.jls\", 5, 1e-3)")
    println()
    println("3. Compare original vs continued training:")
    println("   compare_models()")
    println()
    println("4. Load and work with model manually:")
    println("   model, data, test_data, problem, params = load_experiment_state(\"experiment_state.jls\")")
end
