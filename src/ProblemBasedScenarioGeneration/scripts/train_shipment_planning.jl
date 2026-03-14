"""
Training script for the shipment planning problem.

Usage:
    julia --project=.. scripts/train_shipment_planning.jl
"""

using ProblemBasedScenarioGeneration
using Flux
using Statistics

println("Initializing Shipment Planning problem...")
prob = ShipmentPlanningProblem()

# Generate training dataset
println("Generating dataset...")
n_train = 100
dataset = generate_dataset(prob, n_train; sigma=1.0)

# Build scenario generator
# Input: 3D context, Output: 4D demand scenario (4 locations)
generator = build_generator(prob; hidden_dim=128, n_layers=3)

println("Model: ", generator)
println("Parameters: ", sum(length(p) for p in Flux.params(generator)))

# Run continuation training
println("\nStarting μ-continuation training...")
loss_history = continuation_train!(
    generator, prob, dataset;
    mu_schedule = [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01],
    epochs_per_stage = 5,
    first_stage_epochs = 10,
    finetune_epochs = 5,
    opt = Adam(1e-3),
    verbose = true
)

println("\nTraining complete!")
println("Final loss: ", loss_history[end])

# Evaluate on a test sample
test_x = rand(3)
sc_pred = _params_to_scenarios(prob, generator(test_x))
println("\nTest prediction: $(length(sc_pred)) scenario(s)")
println("Predicted demands: ", sc_pred[1].h[1:4])
