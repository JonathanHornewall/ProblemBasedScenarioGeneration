# Utility script to load saved experiment states
# This allows you to continue working with trained models without retraining

using ProblemBasedScenarioGeneration
using Flux
using LinearAlgebra

# Load the training utilities
include("training.jl")

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
    quick_test_loaded_model(experiment_file = "experiment_state.jls")

Quick test to verify the loaded model works correctly.
"""
function quick_test_loaded_model(experiment_file = "experiment_state.jls")
    model, training_data, testing_data, problem_instance, reg_params = load_and_continue_experiment(experiment_file)
    
    # Extract regularization parameters
    reg_param_surr = reg_params["reg_param_surr"]
    reg_param_ref = reg_params["reg_param_ref"]
    
    println("\n=== Quick Model Test ===")
    
    # Test on a few training samples
    println("Testing on training data...")
    for (i, (x, ξ)) in enumerate(training_data[1:min(3, length(training_data))])
        ξ_hat = model(x)
        loss_val = loss(problem_instance, reg_param_surr, reg_param_ref, ξ_hat, ξ)
        println("  Sample $i - Loss: $loss_val")
    end
    
    # Test on testing data
    println("\nTesting on testing data...")
    test_result = testing(problem_instance, model, testing_data, reg_param_surr, reg_param_ref)
    println("  Test gap: $test_result")
    
    println("\n✓ Model test completed successfully!")
    
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
    
    # Continue training
    train!(problem_instance, reg_param_surr, reg_param_ref, model, training_data; 
           opt = Adam(learning_rate), epochs = additional_epochs, 
           display_iterations = true, save_model = true, 
           model_save_path = "continued_training_model.jld2")
    
    # Save the updated experiment state
    save_experiment_state(model, training_data, testing_data, problem_instance, reg_params, 
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
    
    # Test both models
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

# Example usage functions
function example_usage()
    println("=== Example Usage ===")
    println("1. Load and test a saved model:")
    println("   model, data, test_data, problem, params = quick_test_loaded_model()")
    println()
    println("2. Continue training for 5 more epochs:")
    println("   continue_training(\"experiment_state.jld2\", 5, 1e-3)")
    println()
    println("3. Compare original vs continued training:")
    println("   compare_models()")
    println()
    println("4. Load and work with model manually:")
    println("   model, data, test_data, problem, params = load_and_continue_experiment()")
end

# Show example usage when script is loaded
example_usage()
