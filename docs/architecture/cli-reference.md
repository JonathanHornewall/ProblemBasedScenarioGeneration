# Unit 5: CLI Reference & Usage Examples

## Overview

ProblemBasedScenarioGeneration provides a command-line interface (`pbsg`) for training, evaluating, and managing neural scenario generators for two-stage stochastic linear programming. The CLI is built on [ArgParse.jl](https://github.com/carlobaldassi/ArgParse.jl) and exposes five subcommands: **train**, **continue**, **test**, **evaluate**, and **info**.

---

## Installation & Setup

### Prerequisites

- Julia 1.9 or later
- The package and its dependencies installed (see `Project.toml`)

### Entry Point

The CLI entry point is `bin/pbsg.jl`, located at:

```
src/ProblemBasedScenarioGeneration/bin/pbsg.jl
```

All commands are invoked from the **package root** (`src/ProblemBasedScenarioGeneration/`):

```bash
julia --project=. bin/pbsg.jl <command> [options]
```

The entry point activates the project environment and calls `ProblemBasedScenarioGeneration.cli_main()`.

### Available Problems

The CLI ships with three built-in problem instances:

| Problem name           | Julia type                       | Context dim | Scenario dim | Noise pattern |
|------------------------|----------------------------------|:-----------:|:------------:|:-------------:|
| `resource_allocation`  | `ResourceAllocationProblem`      | 3           | J (clients)  | `H_ONLY`      |
| `shipment_planning`    | `ShipmentPlanningProblem`        | 3           | n_loc        | `H_ONLY`      |
| `newsvendor`           | `UnreliableNewsvendorProblem`    | 1           | 2 (D, U)     | `WH`          |

---

## Subcommand Reference

### `train` -- Train a New Model

Train a scenario generator from scratch. The model is a feed-forward neural network (`Flux.Chain`) that maps context vectors to scenario parameters. Training minimises decision regret: the cost gap between decisions made with predicted vs. actual scenarios.

```bash
julia --project=. bin/pbsg.jl train -p <problem> [options]
```

#### Flags and Arguments

**Problem**

| Flag              | Short | Type   | Default | Required | Description                                                                  |
|-------------------|-------|--------|---------|:--------:|------------------------------------------------------------------------------|
| `--problem`       | `-p`  | String | --      | Yes      | Problem type: `resource_allocation`, `shipment_planning`, or `newsvendor`    |
| `--nr-scenarios`  |       | Int    | `1`     | No       | Number of scenarios the model outputs per prediction                         |

**Data generation**

| Flag        | Short | Type    | Default | Description                                                              |
|-------------|-------|---------|---------|--------------------------------------------------------------------------|
| `--n-train` |       | Int     | `200`   | Number of training (context, scenario) pairs to generate                 |
| `--n-test`  |       | Int     | `50`    | Number of test samples for post-training evaluation (0 to skip)          |
| `--sigma`   |       | Float64 | `5.0`   | Noise standard deviation for data generation (ignored for newsvendor)    |

**Model architecture**

| Flag           | Short | Type   | Default | Description                                                           |
|----------------|-------|--------|---------|-----------------------------------------------------------------------|
| `--hidden-dim` |       | Int    | `128`   | Width of hidden layers                                                |
| `--n-layers`   |       | Int    | `3`     | Number of hidden layers (output layer is added automatically)         |
| `--activation` |       | String | `relu`  | Hidden-layer activation: `relu`, `tanh`, `sigmoid`, or `softplus`     |

**Optimization**

| Flag          | Short | Type    | Default | Description                                                          |
|---------------|-------|---------|---------|----------------------------------------------------------------------|
| `--epochs`    |       | Int     | `30`    | Number of training epochs (full passes over the dataset)             |
| `--batchsize` |       | Int     | `1`     | Mini-batch size (1 = pure SGD, higher = mini-batch SGD)              |
| `--lr`        |       | Float64 | `1e-3`  | Learning rate for the optimizer                                      |
| `--optimizer` |       | String  | `adam`  | Optimizer algorithm: `adam` (Adam) or `sgd` (gradient descent)       |

**Barrier parameters**

| Flag        | Short | Type    | Default | Description                                                            |
|-------------|-------|---------|---------|------------------------------------------------------------------------|
| `--mu-surr` |       | Float64 | `1.0`   | Log-barrier parameter for surrogate LP solve. Retained for API compatibility; standard training always uses the HiGHS LP solver (`solve_lp`) regardless of this value. Used by `continuation_train!` during mu-annealing. |
| `--mu-prim` |       | Float64 | `0.0`   | Log-barrier parameter for cost evaluation. Retained for API compatibility; standard training always uses the HiGHS LP solver (`solve_lp`) regardless of this value. Used by `continuation_train!` during mu-annealing. |

**Continuation schedule** (enabled with `--continuation`)

> **Note:** Standard (non-continuation) training always uses the HiGHS LP solver (`solve_lp`) for both the surrogate first-stage solve and recourse cost evaluation, regardless of `--mu-surr` and `--mu-prim` values. Continuation training (`continuation_train!`) uses the log-barrier solver (`solve_barrier`) with mu-annealing as described below.

| Flag                | Short | Type   | Default                                                    | Description                                                    |
|---------------------|-------|--------|------------------------------------------------------------|----------------------------------------------------------------|
| `--continuation`    |       | Bool   | `false`                                                    | Enable 3-phase mu-continuation: warmup -> anneal -> finetune   |
| `--mu-schedule`     |       | String | `1.0,0.8,0.6,0.4,0.2,0.1,0.08,0.06,0.04,0.02,0.01`      | Comma-separated decreasing mu values for the annealing phase   |
| `--epochs-per-stage`|       | Int    | `10`                                                       | Training epochs at each mu value during annealing              |
| `--warmup-epochs`   |       | Int    | `20`                                                       | Epochs for the warm-up phase (train at largest mu)             |
| `--finetune-epochs` |       | Int    | `10`                                                       | Epochs for fine-tuning phase (train at mu=0, exact LP)         |

**Output and logging**

| Flag              | Short | Type   | Default | Description                                                             |
|-------------------|-------|--------|---------|-------------------------------------------------------------------------|
| `--output`        | `-o`  | String | (auto)  | Path to save the final checkpoint (`.jls`). Auto-generated if omitted   |
| `--save-interval` |       | Int    | `0`     | Save an intermediate checkpoint every N epochs (0 = final only)         |
| `--loss-log`      |       | String | `""`    | Write per-epoch loss to this CSV file (columns: `epoch`, `loss`)        |
| `--verbose`       | `-v`  | Bool   | `false` | Print per-epoch loss and training phase transitions                     |
| `--seed`          |       | Int    | `-1`    | Random seed for reproducibility (-1 = no fixed seed)                    |

When `--output` is omitted, the checkpoint is saved as `pbsg_<problem>_<YYYYMMDD_HHMMSS>.jls`.

When `--save-interval` is set, intermediate checkpoints are written as `<output>_stageN.jls` in addition to the final checkpoint.

---

### `continue` -- Resume Training from a Checkpoint

Loads the model architecture and weights from a previously saved checkpoint, re-generates the training dataset (same size and sigma as the original run), and trains for additional epochs. Unspecified parameters default to values stored in the checkpoint.

```bash
julia --project=. bin/pbsg.jl continue -c <checkpoint.jls> [options]
```

#### Flags and Arguments

| Flag           | Short | Type    | Default | Required | Description                                                           |
|----------------|-------|---------|---------|:--------:|-----------------------------------------------------------------------|
| `--checkpoint` | `-c`  | String  | --      | Yes      | Path to the checkpoint (`.jls`) to resume from                        |
| `--epochs`     |       | Int     | `30`    | No       | Number of additional epochs to train                                  |
| `--lr`         |       | Float64 | `-1.0`  | No       | Learning rate override (-1 = use checkpoint value)                    |
| `--output`     | `-o`  | String  | (auto)  | No       | Path for the new checkpoint (default: `<original>_continued.jls`)     |
| `--mu-surr`    |       | Float64 | `-1.0`  | No       | Surrogate barrier parameter override (-1 = use checkpoint value)      |
| `--mu-prim`    |       | Float64 | `-1.0`  | No       | Evaluation barrier parameter override (-1 = use checkpoint value)     |
| `--batchsize`  |       | Int     | `-1`    | No       | Mini-batch size override (-1 = use checkpoint value)                  |
| `--verbose`    | `-v`  | Bool    | `false` | No       | Print per-epoch loss during training                                  |
| `--seed`       |       | Int     | `-1`    | No       | Random seed (-1 = no fixed seed)                                      |

**Override semantics:** A value of `-1` (for numeric flags) means "keep whatever was in the checkpoint." Any non-negative value overrides the checkpoint setting. The optimizer type (adam/sgd) is always inherited from the checkpoint and cannot be changed via `continue`.

---

### `test` -- Lightweight Test Evaluation

Loads the model from a checkpoint, generates fresh test (context, scenario) pairs, and computes decision regret and relative decision regret for each sample. Prints summary statistics.

```bash
julia --project=. bin/pbsg.jl test -c <checkpoint.jls> [options]
```

#### Flags and Arguments

| Flag           | Short | Type    | Default | Required | Description                                                                     |
|----------------|-------|---------|---------|:--------:|---------------------------------------------------------------------------------|
| `--checkpoint` | `-c`  | String  | --      | Yes      | Path to the trained model checkpoint (`.jls`)                                   |
| `--problem`    | `-p`  | String  | `""`    | No       | Problem type override (default: inferred from checkpoint metadata)               |
| `--n-test`     |       | Int     | `100`   | No       | Number of test (context, scenario) pairs to generate                            |
| `--mu-prim`    |       | Float64 | `0.0`   | No       | Barrier parameter for cost evaluation (0 = exact LP solution)                   |
| `--seed`       |       | Int     | `-1`    | No       | Random seed for test data generation (-1 = no fixed seed)                       |
| `--output`     | `-o`  | String  | `""`    | No       | Write per-sample results to this CSV file                                       |

---

### `evaluate` -- Full Evaluation with Plots and CSV

Generates detailed statistics (95% CI, threshold fractions, median) and publication-ready plots (loss curve, regret histogram, violin, CDF, scenario scatter). Requires `Plots.jl` and `StatsPlots.jl` (loaded on first use).

```bash
julia --project=. bin/pbsg.jl evaluate -c <checkpoint.jls> [options]
```

#### Flags and Arguments

| Flag           | Short | Type    | Default          | Required | Description                                              |
|----------------|-------|---------|------------------|:--------:|----------------------------------------------------------|
| `--checkpoint` | `-c`  | String  | --               | Yes      | Path to the trained model checkpoint (`.jls`)            |
| `--n-test`     |       | Int     | `100`            | No       | Number of test samples to generate                       |
| `--mu-prim`    |       | Float64 | `0.0`            | No       | Barrier parameter for cost evaluation (0 = exact LP)     |
| `--seed`       |       | Int     | `-1`             | No       | Random seed for test data generation (-1 = no fixed seed)|
| `--output-dir` | `-d`  | String  | `./eval_results` | No       | Directory to save plots and CSV output                   |
| `--format`     |       | String  | `png`            | No       | Plot file format: `png`, `pdf`, or `svg`                 |
| `--no-plots`   |       | Bool    | `false`          | No       | Skip plot generation (metrics only)                      |
| `--csv`        |       | Bool    | `false`          | No       | Save per-sample metrics to CSV                           |
| `--verbose`    | `-v`  | Bool    | `false`          | No       | Print detailed progress                                  |

#### Generated Plots

When plots are enabled (the default), the following files are created in `--output-dir`:

| File                       | Description                                                                       |
|----------------------------|-----------------------------------------------------------------------------------|
| `loss_curve.<fmt>`         | Training loss vs. epoch (line plot)                                               |
| `regret_histogram.<fmt>`   | Histogram of relative regret with mean line and 95% CI annotation                 |
| `regret_boxplot.<fmt>`     | Violin + boxplot of relative regret distribution                                  |
| `regret_cdf.<fmt>`         | Empirical CDF of absolute relative regret with threshold reference lines          |
| `scenario_scatter.<fmt>`   | Predicted vs. actual scenario parameters (subplots per dimension, skipped if dim > 8) |

#### CSV Output

When `--csv` is specified, a file `metrics.csv` is written to `--output-dir` with columns:

```
sample,decision_regret,relative_decision_regret,l2_distance
```

---

### `info` -- Inspect a Checkpoint

Display metadata stored in a checkpoint file: model architecture, training hyperparameters, problem type, loss history summary, and timestamp.

```bash
julia --project=. bin/pbsg.jl info -c <checkpoint.jls>
```

#### Flags and Arguments

| Flag           | Short | Type   | Default | Required | Description                                       |
|----------------|-------|--------|---------|:--------:|---------------------------------------------------|
| `--checkpoint` | `-c`  | String | --      | Yes      | Path to the checkpoint file (`.jls`) to inspect   |

---

## Expected Output Formats

### `train` Output

```
Problem: newsvendor (WH)
Generating 200 training samples...
Model: Chain(Dense(1 => 128, relu), Dense(128 => 128, relu), Dense(128 => 128, relu), Dense(128 => 2, softplus))
Parameters: 33666

Starting training...
[ Info: Epoch 1: loss = 14.532
[ Info: Epoch 2: loss = 12.871
[ Info: Epoch 3: loss = 11.204
...
[ Info: Epoch 30: loss = 0.342

Checkpoint saved: pbsg_newsvendor_20260315_143022.jls

Evaluating on 50 test samples...
  Mean regret:   0.2841
  Std regret:    0.1923

Training complete!
```

When `--loss-log` is specified, a CSV file is written:

```csv
epoch,loss
1,14.532
2,12.871
...
30,0.342
```

### `continue` Output

```
Loading checkpoint: model.jls
Model restored: Chain(Dense(1 => 128, relu), Dense(128 => 128, relu), Dense(128 => 128, relu), Dense(128 => 2, softplus))
Problem: newsvendor
Generating 200 training samples...

Continuing training for 10 epochs (from epoch 30)...
[ Info: Epoch 1: loss = 0.315
...
[ Info: Epoch 10: loss = 0.198

Checkpoint saved: model_continued.jls
Total epochs: 40
Final loss: 0.198
```

### `test` Output

```
Loading checkpoint: model.jls
Problem: newsvendor
Generating 100 test samples...
Evaluating...

=== Test Results (100 samples) ===
Decision regret:
  Mean:    0.2537
  Std:     0.1842
  Min:     0.0012
  Max:     0.9831
Relative decision regret:
  Mean:    0.0341
  Std:     0.0287
  Min:     0.0001
  Max:     0.1422
```

When `--output` is specified, a per-sample CSV is written:

```csv
sample,decision_regret,relative_decision_regret
1,0.253,0.034
2,0.118,0.015
...
```

### `evaluate` Output

```
Loading checkpoint: model.jls
Problem: newsvendor
Generating 100 test samples...
Computing evaluation metrics...

=== Evaluation Results (100 samples) ===

Decision regret:
  Mean:      0.2537
  Std:       0.1842
  95% CI:    [0.2176, 0.2898]
  Median:    0.2104
  Min:       0.0012
  Max:       0.9831

Relative decision regret:
  Mean:      0.0341
  Std:       0.0287
  95% CI:    [0.0285, 0.0397]
  Median:    0.0268
  Min:       0.0001
  Max:       0.1422

Threshold fractions (relative regret):
  < 1%:     23.0% of samples
  < 5%:     68.0% of samples
  < 10%:    89.0% of samples

Scenario L2 distance:
  Mean:      1.432
  Std:       0.891
  Median:    1.204

Generating plots in eval_out/ ...
  loss_curve.png
  regret_histogram.png
  regret_boxplot.png
  regret_cdf.png
  scenario_scatter.png
Plots saved to eval_out/
Metrics CSV saved: eval_out/metrics.csv
```

### `info` Output

```
File: model.jls
Size: 524288 bytes

=== Checkpoint Info ===
Timestamp:        2026-03-15T14:30:22
Problem:          newsvendor
Noise pattern:    WH
Nr of scenarios:  1

--- Model Architecture ---
Input dim:        1
Output dim:       2
Hidden dim:       128
Layers:           3
Activation:       relu
Total parameters: 33666

--- Training Config ---
  activation          relu
  batchsize           1
  continuation        false
  epochs              30
  epochs_per_stage    10
  finetune_epochs     10
  hidden_dim          128
  lr                  0.001
  mu_prim             0.0
  mu_schedule         1.0,0.8,0.6,0.4,0.2,0.1,0.08,0.06,0.04,0.02,0.01
  mu_surr             1.0
  n_layers            3
  n_train             200
  nr_scenarios        1
  seed                -1
  sigma               5.0
  total_epochs        30
  warmup_epochs       20

--- Training History ---
Total epochs:     30
Initial loss:     14.532
Final loss:       0.342
Best loss:        0.342
```

---

## Workflow Examples

### 1. Train a Model from Scratch

Basic training run for the newsvendor problem:

```bash
julia --project=. bin/pbsg.jl train \
    -p newsvendor \
    --epochs 30 \
    --n-train 200 \
    --n-test 50 \
    -v \
    -o newsvendor_model.jls
```

Train with a custom architecture and optimizer:

```bash
julia --project=. bin/pbsg.jl train \
    -p resource_allocation \
    --hidden-dim 256 \
    --n-layers 4 \
    --activation tanh \
    --lr 5e-4 \
    --optimizer adam \
    --epochs 50 \
    -v \
    -o ra_model.jls
```

Reproducible run with fixed seed and loss logging:

```bash
julia --project=. bin/pbsg.jl train \
    -p newsvendor \
    --seed 42 \
    --epochs 100 \
    --save-interval 10 \
    --loss-log losses.csv \
    -v \
    -o model.jls
```

### 2. Train with Mu-Continuation Schedule

The continuation schedule implements Algorithm 1 from the paper: warmup at high mu, anneal through a decreasing schedule, then fine-tune at mu=0.

```bash
julia --project=. bin/pbsg.jl train \
    -p shipment_planning \
    --continuation \
    --mu-schedule "1.0,0.5,0.1,0.01" \
    --warmup-epochs 20 \
    --epochs-per-stage 5 \
    --finetune-epochs 10 \
    -v \
    -o sp_continuation.jls
```

When `--continuation` is enabled, the `--epochs`, `--mu-surr`, and `--mu-prim` flags are ignored. Instead, the total number of training epochs is determined by:

```
total = warmup_epochs + len(mu_schedule) * epochs_per_stage + finetune_epochs
```

### 3. Inspect a Trained Model

```bash
julia --project=. bin/pbsg.jl info -c newsvendor_model.jls
```

This prints the model architecture, all training hyperparameters, and a loss history summary. Useful for comparing experiments or verifying a checkpoint before further training.

### 4. Test a Model on New Data

Lightweight evaluation that prints summary statistics:

```bash
julia --project=. bin/pbsg.jl test \
    -c newsvendor_model.jls \
    --n-test 200 \
    --seed 123
```

Save per-sample results to CSV for external analysis:

```bash
julia --project=. bin/pbsg.jl test \
    -c newsvendor_model.jls \
    --n-test 500 \
    -o test_results.csv
```

Override the problem type (e.g., to test a model on a different problem):

```bash
julia --project=. bin/pbsg.jl test \
    -c newsvendor_model.jls \
    -p newsvendor \
    --mu-prim 0.0
```

### 5. Run Full Evaluation with Plots

Generate plots and CSV output:

```bash
julia --project=. bin/pbsg.jl evaluate \
    -c newsvendor_model.jls \
    --n-test 100 \
    -d ./eval_results \
    --csv \
    -v
```

Generate PDF plots without CSV:

```bash
julia --project=. bin/pbsg.jl evaluate \
    -c newsvendor_model.jls \
    --format pdf \
    -d ./plots
```

Metrics only (no plot generation, faster):

```bash
julia --project=. bin/pbsg.jl evaluate \
    -c newsvendor_model.jls \
    --no-plots \
    --csv \
    -d ./metrics_only
```

### 6. Continue Training from a Checkpoint

Resume with default settings from the checkpoint:

```bash
julia --project=. bin/pbsg.jl continue \
    -c newsvendor_model.jls \
    --epochs 20 \
    -v
```

Resume with a lower learning rate and save to a new file:

```bash
julia --project=. bin/pbsg.jl continue \
    -c newsvendor_model.jls \
    --epochs 10 \
    --lr 1e-4 \
    -o newsvendor_model_v2.jls
```

The continued model appends to the original loss history, so the total epoch count accumulates across runs.

---

## Checkpoint File Format

Checkpoints are serialized Julia `Dict{String,Any}` objects (using `Serialization.serialize`), saved with the `.jls` extension. The dictionary contains:

| Key                | Type                  | Description                                              |
|--------------------|-----------------------|----------------------------------------------------------|
| `model_state`      | Flux model state      | Neural network weights (via `Flux.state`)                |
| `model_config`     | `Dict{String,Any}`    | Architecture: `input_dim`, `output_dim`, `hidden_dim`, `n_layers`, `activation` |
| `problem_type`     | `String`              | Problem name (e.g., `"newsvendor"`)                      |
| `noise_pattern`    | `String`              | Noise pattern (e.g., `"WH"`)                             |
| `nr_of_scenarios`  | `Int`                 | Number of output scenarios                               |
| `training_config`  | `Dict{String,Any}`    | All training hyperparameters                             |
| `loss_history`     | `Vector{Float64}`     | Per-epoch average loss values                            |
| `total_epochs`     | `Int`                 | Total number of training epochs                          |
| `timestamp`        | `String`              | ISO 8601 timestamp of checkpoint creation                |

Checkpoints can be loaded programmatically with:

```julia
using ProblemBasedScenarioGeneration
ckpt = load_checkpoint("model.jls")
model = restore_model(ckpt)
```

---

## Programmatic Access

All CLI functionality is also available as Julia functions. The CLI is a thin wrapper around:

| CLI subcommand | Julia function              | Source file              |
|----------------|-----------------------------|--------------------------|
| `train`        | `train!`, `continuation_train!` | `src/training/trainer.jl`, `src/training/continuation.jl` |
| `continue`     | `load_checkpoint`, `restore_model`, `train!` | `src/persistence.jl`, `src/training/trainer.jl` |
| `test`         | `decision_regret`, `relative_decision_regret` | `src/loss/decision_regret.jl` |
| `evaluate`     | `compute_evaluation_metrics`, `print_evaluation_summary`, `plot_*` | `src/evaluation/evaluate.jl` |
| `info`         | `print_checkpoint_info`     | `src/persistence.jl`     |

---

## Help

Each subcommand supports `--help`:

```bash
julia --project=. bin/pbsg.jl --help
julia --project=. bin/pbsg.jl train --help
julia --project=. bin/pbsg.jl evaluate --help
```
