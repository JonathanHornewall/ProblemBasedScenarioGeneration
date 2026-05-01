This guide explains how to save and load trained neural network models and experiment data to avoid retraining from scratch.

When you run training with saving enabled, the following are saved:

1. Trained Model Weights (`trained_model.jls`)
   - Complete Flux model with all trained parameters
2. Training and Testing Datasets (`experiment_state.jls`)
   - Trained model weights
   - Training dataset
   - Testing dataset
3. Full Experiment State (`experiment_state.jls`)
   - Everything needed to continue the experiment

How to Use

- Training with Auto-Save
  - The main training script can save the model and experiment state after training.

- Loading a Saved Model
  - Use the utilities in `load_experiment.jl`.

Example

```julia
include("load_experiment.jl")

# Quick test of loaded model
model, data, test_data, problem, params = load_and_continue_experiment()

# Continue training for 5 more epochs
continue_training("experiment_state.jls", 5, 1e-3)

# Compare models
compare_models()
```

Files

- `load_experiment.jl`: save/load helpers and convenience routines.
- `test_function_SAA.jl`: SAA testing and plotting utilities.

Notes

- These tools are experimental and may assume the main package is already loaded (`using ProblemBasedScenarioGeneration`).
- Some plotting or baseline comparison features may require optional CSV files; if not found, they will be skipped.

