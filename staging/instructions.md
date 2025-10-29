# Instructions for Extensive Numerical Tests

## Overview
- Goal: benchmark the decision-focused neural-net scenario generator against the baseline methods from Tito’s paper and recreate the paper’s plot with our method included.

All the code should be placed inside of the full_benchmark directory.

## References
- Reference: The problem we are benchmarking on is the resource allocation problem in scripts/resource_allocation_prototype.
- The decision focused neural net is the one implemented in scripts/resource_allocation_prototype/annealing.jl The training should follow its outline as closely as possible.
- Reuse the methods in `tito/resource_allocation` for all baseline methods. The legacy `method_compar` code is for visual/layout reference only—it must not be executed or duplicated. The baseline methods should resuse the code from Tito without changing it, unless absolutely necessary.

Generally, the code should try to re-use functionality as much as possible (e.g. reuse tito method implementation), and when that is not possible, it should copy existing design (e.g. use the DataGenerator function, but rewritten to take additional arguments).

## Step 1: Generate Testing Data
- Tasks:
  - Sample 30 covariates with parameter `p = 2` using the same data-generation routine as Tito’s experiments.
  - For each of 30 designated testing covariates, draw C independent batches of S demand scenarios (default C = 1, S = 100). Store with shape: `30 × C × S × J`.
- Artifacts (all written under `artifacts/testing/`):
  - `test_covariates.csv`: 30 rows with the covariate vectors used for evaluation.
  - `test_scenarios.jls`: serialized array storing the `30 × C × S × J` scenarios aligned with `test_covariates.csv`.
  - `full_context_pool.csv`: 100 rows capturing the complete covariate pool prior to splitting (so we can reproduce splits deterministically).

## Step 2: Compute SAA Baselines
- Tasks:
  - For each testing covariate, solve C independent SAA problems, each using one of that covariate’s S-scenario batches.
  - Average the C optimal costs to obtain the reference optimum per covariate (with default C = 1 this equals the single run).
- Artifacts (under `artifacts/testing/`):
  - `saa_runs.csv`: `30 × C` rows with columns `{covariate_id, run_id (1–C), objective_value, covariate_vector...}`.
  - `saa_optima.csv`: 30 rows with columns `{covariate_id, optimal_cost, covariate_vector...}` (the averaged optimums).
  - Optional: `saa_solver_metadata.json` describing solver settings (seed, tolerance) for reproducibility.

## Step 3: Generate Training Data
- Tasks:
  - Sample 100 training covariates. For each covariate, sample a scenario to build 100 context–scenario pairs. Use the same distributional parameters as for the testing data.
  - Ensure the random seeds and parameters are logged so the dataset can be regenerated.
- Artifacts (under `artifacts/training/`):
  - `training_pairs.jls`: serialized collection of 100 tuples `(x, ξ)` with ξ containing the scenario matrix per covariate.
  - `training_covariates.csv`: covariate vectors for all training pairs.
  - `data_generation_log.json`: parameters, RNG seeds, and any preprocessing notes.

## Step 4: Train Baseline (Non-Neural) Methods
- Tasks:
  - Load training artifacts.
  - Fit each baseline model (e.g., LS, ER-SAA, KNN, CART/M5, Nelder–Mead AD) without modifying the validated reference implementations.
  - Record hyperparameters and training diagnostics.
- Artifacts (under `artifacts/models/baselines/`):
  - One serialized model per method (e.g., `ls_model.jls`, `er_saa_model.jls`, `cart_model.jls`, `knn_model.jls`, `nm_model.jls`).
  - `baseline_training_report.json`: summary of training settings (hyperparameters, seeds, fit metrics).

## Step 5: Train Neural-Net Method
- Tasks:
  - Load training artifacts and train the neural network scenario generator.
  - Capture training curves, checkpoints, and final weights.
- Artifacts (under `artifacts/models/neural/`):
  - `neural_model_final.jls`: final trained model.
  - `neural_training_history.csv`: epoch-wise metrics.
  - Optional checkpoints (e.g., `checkpoints/epoch_XX.jls`) if early stopping or resumability is needed.

## Step 6: Compute Benchmark and Plot
- Tasks:
  - For each method (baselines + neural net), generate decisions on the testing covariates, evaluate cost against the precomputed SAA optima, and compute percentage gaps (`(evaluated / optimum − 1) × 100`).
  - Aggregate results across the 30 covariates and build the comparison figure mirroring `gap_boxplot.png`.
- Artifacts (under `artifacts/results/`):
  - `benchmark_gaps.csv`: tidy table with `{covariate_id, method, evaluated_cost, gap_percent}`.
  - `benchmark_summary.json`: aggregated statistics per method (mean gap, variance, percentile info).
  - `gap_boxplot.png`: reproduced plot including the neural-net method, matching the styling of the original figure.

## Main Experiment Driver
- Provide a script `scripts/run_experiment.jl` (or equivalent) that orchestrates Steps 1–6.
- Behavior:
  - When invoked with a destination directory `--output-dir=<path>`, create/synchronize the artifacts described above.
  - If `--input-dir=<path>` is supplied, inspect that directory for existing artifacts. For each step, skip recomputation when all expected artifacts are present and simply copy them into the new output directory.
  - Flags:
    - `--full_training`: regenerate training data, retrain all models, and re-run the full benchmark (reusing testing data and SAA artifacts from the input directory).
    - `--method_training`: retrain all baseline methods and re-run the benchmark (reusing training data, neural model, testing data, and SAA artifacts).
    - `--neural_training`: retrain only the neural network and update its benchmark results.
    - `--full_testing`: regenerate testing data and recompute SAA baselines (reusing existing training data and models from the input directory).
  - If multiple flags are passed, resolve them by the most comprehensive action (e.g., `--full_training` supersedes `--method_training`).

  In addition, the script should take arguments specifying:
  - size of the training data (default 100)
  - scenarios per context in the training data (default 1)
  - nr covariates in testing data (default 30)
  - nr of scenario collections per covariate in testing data (default 1)
  - size of scenario collection per covariate in testing data (default 100)
  - the data generation parameters of the training data and the testing data (default p =2)
  - Annealing schedule for neural net training (default 1.0, 0.1, 0.01)
  - Epoch schedule (default 100 for the first one, then 30 for the following 3). Used for training the neural net.

  - Step size schedule (default 10⁻3 for the first one, 10⁻4 for the following 3)
  - Batch size schedule (default 10 for first one, then 25 for the following 3)
  - surrogate parameter (default same as the last parameter in the annealing schedule)

  These all specify how the decision focused neural net should be trained. See annealing.jl for reference.
## Implementation Plan
# Here you should specify the implementation plan

# Testing and validation

Test 1:
Perform a small smoke run, with minimal values for training and testing data. The goal is just to ensure that the script runs to completion.

Test 2: Perform a test with the different flags to ensure that they behave correctly.

Iterate until the tests pass.
