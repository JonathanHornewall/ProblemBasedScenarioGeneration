# Model Saving and Loading Guide

This guide explains how to save and load trained neural network models and experiment data to avoid retraining from scratch.

## What Gets Saved

When you run training with saving enabled, the following are saved:

1. **Trained Model Weights** (`trained_model.jls`)
   - Complete Flux model with all trained parameters
   - Can be loaded and used directly for inference

2. **Complete Experiment State** (`experiment_state.jls`)
   - Trained model weights
   - Training dataset
   - Testing dataset  
   - Problem instance parameters
   - Regularization parameters
   - Everything needed to continue the experiment

## How to Use

### 1. Training with Auto-Save

The `main.jl` script now automatically saves the model and experiment state after training:

```julia
# This will automatically save:
# - trained_model.jld2
# - experiment_state.jld2
include("main.jl")
main()
```

### 2. Loading a Saved Model

Use the utility functions in `load_experiment.jl`:

```julia
# Load the training utilities
include("load_experiment.jl")

# Quick test of loaded model
model, data, test_data, problem, params = quick_test_loaded_model()

# Or load manually for custom work
model, data, test_data, problem, params = load_and_continue_experiment()
```

### 3. Continue Training

```julia
# Continue training for 5 more epochs
continue_training("experiment_state.jls", 5, 1e-3)

# This creates:
# - continued_training_model.jls  
# - continued_experiment_state.jls
```

### 4. Compare Models

```julia
# Compare original vs continued training performance
compare_models()
```

## File Structure

After running training, you'll have:

```
scripts/resource_allocation_prototype/
├── trained_model.jls               # Just the model weights
├── experiment_state.jls            # Complete experiment state
├── main.jl                         # Training script (modified)
├── training.jl                     # Training functions (modified)
├── load_experiment.jl              # Loading utilities
└── README_save_load.md             # This file
```

## Key Functions

### Saving Functions
- `save_trained_model(model, filepath)` - Save just the model
- `save_training_data(training_data, testing_data, filepath)` - Save datasets
- `save_experiment_state(...)` - Save everything

### Loading Functions  
- `load_trained_model(filepath)` - Load just the model
- `load_training_data(filepath)` - Load datasets
- `load_experiment_state(filepath)` - Load everything

### Utility Functions
- `quick_test_loaded_model()` - Quick verification
- `continue_training()` - Continue training from saved state
- `compare_models()` - Compare model performances

## Example Workflow

1. **First Run**: Train and save
   ```julia
   include("main.jl")
   main()  # This saves everything
   ```

2. **Subsequent Runs**: Load and continue
   ```julia
   include("load_experiment.jl")
   
   # Quick test
   model, data, test_data, problem, params = quick_test_loaded_model()
   
   # Continue training
   continue_training("experiment_state.jld2", 10, 1e-3)
   
   # Compare results
   compare_models()
   ```

## Benefits

- **No More Retraining**: Load saved models instantly
- **Experiment Continuity**: Continue training from where you left off
- **Model Comparison**: Easily compare different training stages
- **Reproducibility**: Save exact experiment conditions
- **Time Savings**: Skip hours of retraining

## Dependencies

The saving functionality uses Julia's built-in `Serialization` module, so no additional dependencies are required.

```julia
using Pkg
Pkg.instantiate()
```

## Troubleshooting

- **File not found**: Make sure you've run training at least once
- **Dependency errors**: Check that JLD2 is installed
- **Model loading issues**: Verify the saved files are complete

## Advanced Usage

You can also save intermediate checkpoints during training by modifying the training loop to save every N epochs, or save different versions of the model for comparison.
