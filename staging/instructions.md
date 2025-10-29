# Instructions for Extensive Numerical Tests

## Overview
- Goal: benchmark the decision-focused neural-net scenario generator against the baseline methods from Tito’s paper and recreate the paper’s plot with our method included.
- Reference: reuse the tested implementations in `tito/resource_allocation` for all baseline methods. The legacy `method_compar` code is for visual/layout reference only—it must not be executed or duplicated.

## Step 1: Generate Testing Data
- Tasks:
  - Sample 100 covariates with parameter `p = 2` using the same data-generation routine as Tito’s experiments.
  - For each of 30 designated testing covariates, draw 30 independent batches of 1,000 demand scenarios (shape: `30 × 30 × 1000 × J`).
- Artifacts (all written under `artifacts/testing/`):
  - `test_covariates.csv`: 30 rows with the covariate vectors used for evaluation.
  - `test_scenarios.jls`: serialized array storing the 30 × 30 × 1000 scenarios aligned with `test_covariates.csv`.
  - `full_context_pool.csv`: 100 rows capturing the complete covariate pool prior to splitting (so we can reproduce splits deterministically).

## Step 2: Compute SAA Baselines
- Tasks:
  - For each testing covariate, solve 30 independent SAA problems, each using one of the 1,000-scenario batches.
  - Average the 30 optimal costs to obtain the reference optimum per covariate.
- Artifacts (under `artifacts/testing/`):
  - `saa_runs.csv`: 900 rows with columns `{covariate_id, run_id (1–30), objective_value, covariate_vector...}`.
  - `saa_optima.csv`: 30 rows with columns `{covariate_id, optimal_cost, covariate_vector...}` (the averaged optimums).
  - Optional: `saa_solver_metadata.json` describing solver settings (seed, tolerance) for reproducibility.

## Step 3: Generate Training Data
- Tasks:
  - From the remaining 70 covariates (after reserving 30 for testing), build 100 context–scenario pairs using the same distributional parameters.
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

## Implementation Plan
| File | Purpose | Key Functions/Entrypoints |
| --- | --- | --- |
| `scripts/run_experiment.jl` | CLI entry point; parses flags, resolves artifact reuse logic, and sequences the step modules. | `run_experiment(args)`, `resolve_step_plan(config, artifacts_state)` |
| `scripts/steps/generate_testing_data.jl` | Implements Step 1 data generation and persistence. | `generate_testing_data(config)`, `save_testing_artifacts(data, output_dir)` |
| `scripts/steps/compute_saa_baselines.jl` | Implements Step 2 SAA solves and logging. | `compute_saa_baselines(test_data, config)`, `write_saa_artifacts(results, output_dir)` |
| `scripts/steps/generate_training_data.jl` | Implements Step 3 training dataset creation. | `generate_training_data(config)`, `save_training_artifacts(data, output_dir)` |
| `scripts/steps/train_baselines.jl` | Implements Step 4 for all non-neural methods. | `train_baseline_methods(training_data, config)`, `serialize_baseline_models(models, output_dir)` |
| `scripts/steps/train_neural.jl` | Implements Step 5 neural-net training workflow. | `train_neural_method(training_data, config)`, `save_neural_artifacts(model, history, output_dir)` |
| `scripts/steps/run_benchmark.jl` | Implements Step 6 evaluation and plotting. | `benchmark_methods(test_data, models, saa_optima, config)`, `save_benchmark_outputs(results, output_dir)` |
| `scripts/util/artifacts.jl` | Shared helpers for checking/copying artifacts. | `detect_artifacts(output_dir)`, `copy_artifacts(src, dest, manifest)` |
| `scripts/util/config.jl` | Centralizes configuration defaults and RNG seeding. | `load_config(args)`, `seed_rng(config)` |

- Each step module should expose a single `execute_<step>` function that `run_experiment.jl` can call after determining whether recomputation is required.
- Shared data structures (e.g., typed structs for covariates, scenarios, and model bundles) should live under `src/` so both scripts and tests can import them cleanly.
