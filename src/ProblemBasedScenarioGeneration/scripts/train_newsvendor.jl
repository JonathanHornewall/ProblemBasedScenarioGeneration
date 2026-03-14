"""
Training script for the unreliable newsvendor problem.

Usage:
    julia --project=.. scripts/train_newsvendor.jl
"""

using ProblemBasedScenarioGeneration
using Flux
using Statistics

println("Initializing Unreliable Newsvendor problem...")
prob = UnreliableNewsvendorProblem()

# Generate training dataset
println("Generating dataset...")
n_train = 50
dataset = generate_dataset(prob, n_train)

# Build scenario generator
# Input: 1D context (demand D), Output: 2D scenario [D, U]
generator = build_generator(1, 2; hidden_dim=32, n_layers=2)

println("Model: ", generator)
println("Parameters: ", sum(length(p) for p in Flux.params(generator)))

# Run continuation training
println("\nStarting μ-continuation training...")
loss_history = continuation_train!(
    generator, prob, dataset;
    mu_schedule = [1.0, 0.5, 0.2, 0.1, 0.05],
    epochs_per_stage = 10,
    first_stage_epochs = 20,
    finetune_epochs = 10,
    opt = Adam(1e-3),
    verbose = true
)

println("\nTraining complete!")
println("Final loss: ", loss_history[end])

# Evaluate on a test sample
test_x = [0.6]
raw = generator(test_x)
sc_pred = _params_to_scenarios(prob, raw)
println("\nTest prediction for D=0.6:")
println("  Predicted [D, U] = ", raw)
