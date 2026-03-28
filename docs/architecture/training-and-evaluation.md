# Unit 4: Training & Evaluation Pipeline

This document describes how ProblemBasedScenarioGeneration trains neural scenario generators end-to-end using decision-focused learning, how it evaluates trained models, and how it persists and restores training state.

---

## Table of Contents

1. [Full Pipeline Overview](#1-full-pipeline-overview)
2. [The `train!` Function — One Epoch Walkthrough](#2-the-train-function--one-epoch-walkthrough)
3. [Mu-Continuation Schedule](#3-mu-continuation-schedule)
4. [SGD with Flux and Zygote](#4-sgd-with-flux-and-zygote)
5. [Loss Function: Decision Regret](#5-loss-function-decision-regret)
6. [Evaluation Metrics](#6-evaluation-metrics)
7. [Plotting Functions](#7-plotting-functions)
8. [Checkpoint Save/Load/Restore Cycle](#8-checkpoint-saveloadrestore-cycle)

---

## 1. Full Pipeline Overview

The training pipeline implements decision-focused learning: a neural network is trained not to predict scenarios accurately in a statistical sense, but to produce scenarios that lead to good downstream optimization decisions. The full data flow is:

```
context x ──► Neural Net ──► raw params ──► scenario_realization() ──► Scenario(W,T,h,q)
                  │                                                          │
                  │                                                          ▼
                  │                                              TwoStageSLP (surrogate)
                  │                                                          │
                  │                                                          ▼
                  │                                              solve_lp() ──► x̂₁
                  │                                              (HiGHS LP)       │
                  │                                                                 ▼
                  │                                              evaluate_cost(x̂₁, actual scenario)
                  │                                                                 │
                  │                                                                 ▼
                  │                                                          loss (cost)
                  │                                                                 │
                  ◄─────────────── Zygote reverse-mode AD ──────────────────────────┘
                  │
                  ▼
           Flux.update!(opt_state, model, gradients)
```

The key insight is that gradients flow backward through the LP solver itself via the `rrule` defined on `solve_lp`, which uses a subgradient approximation based on the LP dual variables. This makes the entire pipeline differentiable.

**Source files involved:**

| File | Role |
|------|------|
| `src/training/trainer.jl` | `train!` loop, batching, gradient computation |
| `src/training/continuation.jl` | `continuation_train!` with 3-phase mu schedule |
| `src/loss/decision_regret.jl` | `decision_regret`, `surrogate_first_stage`, `evaluate_cost` |
| `src/evaluation/evaluate.jl` | Metrics, summary printing, CSV export, all 5 plot types |
| `src/persistence.jl` | Checkpoint serialization and model restoration |
| `src/models/scenario_generator.jl` | `build_generator` neural network factory |
| `src/models/output_heads.jl` | `build_output_head`, `build_full_model` |
| `src/diff/subgradient_rrule.jl` | ChainRules `rrule` for `solve_lp` (primary training path) |
| `src/diff/barrier_rrule.jl` | ChainRules `rrule` for `solve_barrier` (not on default training path) |
| `src/diff/implicit_diff.jl` | `implicit_diff_h`, `implicit_diff_q` (used by barrier rrule only) |

---

## 2. The `train!` Function — One Epoch Walkthrough

The `train!` function in `src/training/trainer.jl` is the core training loop. Here is a step-by-step walkthrough of what happens during a single epoch.

### Signature

```julia
function train!(
    model,
    prob::ProblemInstance,
    dataset::Vector{<:Tuple};
    mu_surr::Float64  = 1.0,
    mu_prim::Float64  = 0.0,
    opt               = Adam(1e-3),
    epochs::Int       = 30,
    batchsize::Int    = 1,
    verbose::Bool     = false,
    opt_state         = nothing
) -> Vector{Float64}
```

### Arguments

- **`model`**: A `Flux.Chain` mapping context vectors to raw scenario parameter vectors.
- **`prob`**: A `ProblemInstance` subtype (e.g., `ResourceAllocationProblem`) that defines the optimization problem structure.
- **`dataset`**: A `Vector` of `(context, Scenario)` tuples — each pair associates a context vector `x` with the actual scenario that was realized.
- **`mu_surr`**: Barrier parameter for the surrogate solve (the solve over predicted scenarios).
- **`mu_prim`**: Barrier parameter for cost evaluation on the actual scenario.
- **`opt`**: A Flux optimizer (default: `Adam(1e-3)`).
- **`batchsize`**: Mini-batch size. Default `1` means pure stochastic gradient descent.
- **`opt_state`**: If provided, reuses existing optimizer state (for resumed training). Otherwise initialized via `Flux.setup(opt, model)`.

### Step-by-Step: One Epoch

**Step 1 — Optimizer state initialization.** On the first call (or when `opt_state` is `nothing`), the function initializes the optimizer state tree:

```julia
opt_state = isnothing(opt_state) ? Flux.setup(opt, model) : opt_state
```

**Step 2 — Shuffle and partition.** The dataset indices are randomly permuted and partitioned into mini-batches:

```julia
indices = randperm(N)
for batch_idx in Iterators.partition(indices, batchsize)
    batch = dataset[batch_idx]
    ...
end
```

**Step 3 — Gradient computation.** For each mini-batch, `Flux.gradient` is called with a closure that:

1. Passes each context `x` through the model to get `raw_params`.
2. Converts `raw_params` to `Vector{Scenario}` via `_params_to_scenarios`.
3. Computes `decision_regret(prob, mu_surr, mu_prim, predicted_sc, actual_sc)`.
4. Averages the loss over the batch with `mean`.

```julia
gs = Flux.gradient(model) do m
    batch_loss = mean(batch) do (x, actual_sc)
        raw_params = m(x)
        predicted_sc = _params_to_scenarios(prob, raw_params)
        decision_regret(prob, mu_surr, mu_prim, predicted_sc, actual_sc)
    end
    batch_loss
end
```

Under the hood, Zygote traces through the entire forward pass — including the LP solver — and uses the custom `rrule` on `solve_lp` to compute gradients via a subgradient approximation (see [Section 4](#4-sgd-with-flux-and-zygote)).

**Step 4 — Parameter update.** The gradient is extracted and applied:

```julia
gmodel = gs isa Tuple ? gs[1] : gs
Flux.update!(opt_state, model, gmodel)
```

**Step 5 — Loss tracking.** After the update, the batch loss is recomputed without the gradient tape (for logging fidelity) and stored:

```julia
bl = mean(batch) do (x, actual_sc)
    raw_params = model(x)
    predicted_sc = _params_to_scenarios(prob, raw_params)
    decision_regret(prob, mu_surr, mu_prim, predicted_sc, actual_sc)
end
push!(epoch_losses, bl)
```

**Step 6 — Epoch aggregation and GC.** At the end of each epoch, the batch losses are averaged to produce the epoch loss. `GC.gc()` is called explicitly to free memory from intermediate AD computations:

```julia
avg_loss = mean(epoch_losses)
push!(loss_history, avg_loss)
GC.gc()
```

### The `_params_to_scenarios` Helper

This function bridges the neural network output and the optimization problem structure:

```julia
function _params_to_scenarios(prob::ProblemInstance, raw_params::AbstractVector)
    sc_dim = _scenario_param_dim(prob)
    n_sc   = length(raw_params) / sc_dim
    params_mat = reshape(raw_params, sc_dim, n_sc)
    return [scenario_realization(prob, params_mat[:, s]) for s in 1:n_sc]
end
```

It reshapes the flat output vector into columns (one per scenario), then calls `scenario_realization` on each column to construct the full `Scenario` struct with matrices `W`, `T`, `h`, `q`.

---

## 3. Mu-Continuation Schedule

The `continuation_train!` function in `src/training/continuation.jl` implements a 3-phase training schedule that gradually reduces the barrier parameter mu. This is critical because the log-barrier solver approximates the true LP solution, and the quality of the approximation depends on mu.

### Why Continuation Matters

The log-barrier solver solves:

```
min  c'x - mu * sum(log(x_i))   s.t. Ax = b, x > 0
```

- **Large mu**: The barrier term dominates, producing a smooth, well-conditioned loss landscape with gradients that are easy to follow. But the solution is far from the true LP optimum.
- **Small mu**: The solution approaches the true LP vertex, but the loss landscape becomes sharp and ill-conditioned, making optimization difficult.
- **mu = 0**: Falls back to an exact LP solver (HiGHS via JuMP), which is non-differentiable. This is used only for final evaluation.

The continuation strategy starts with a smoothed landscape (large mu) to find a good basin of attraction, then progressively sharpens the landscape to converge to a decision that is optimal under the true LP.

### The Three Phases

```julia
function continuation_train!(
    model, prob, dataset;
    mu_schedule = [1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.08, 0.06, 0.04, 0.02, 0.01],
    epochs_per_stage  = 10,
    first_stage_epochs = 20,
    finetune_epochs    = 10,
    opt = Adam(1e-3),
    verbose = false
) -> Vector{Float64}
```

#### Phase 1: Warm-up

```julia
train!(model, prob, dataset; mu_surr=mu0, mu_prim=mu0, epochs=first_stage_epochs, ...)
```

- Trains for `first_stage_epochs` (default: 20) at the largest mu value (`mu_schedule[1]`, default: 1.0).
- The smooth landscape helps the model escape poor initial regions and find a reasonable parameter basin.
- Both `mu_surr` and `mu_prim` are set to the same value.

#### Phase 2: Annealing

```julia
for (stage, mu) in enumerate(mu_schedule)
    train!(model, prob, dataset; mu_surr=mu, mu_prim=mu, epochs=epochs_per_stage, ...)
end
```

- Steps through each value in `mu_schedule`, training for `epochs_per_stage` (default: 10) at each level.
- The default schedule has 11 stages: `[1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.08, 0.06, 0.04, 0.02, 0.01]`.
- As mu decreases, the barrier solver solution moves closer to the true LP optimum.
- The model continuously adapts to the sharpening landscape.

#### Phase 3: Fine-tuning

```julia
train!(model, prob, dataset; mu_surr=0.0, mu_prim=0.0, epochs=finetune_epochs, ...)
```

- Trains for `finetune_epochs` (default: 10) at `mu = 0`.
- At `mu = 0`, `surrogate_first_stage` falls back to the exact LP solver (`solve_lp` via HiGHS).
- This phase polishes the model under the true (non-smoothed) decision loss.

### Total Epoch Count

With default settings:
- Warm-up: 20 epochs
- Annealing: 11 stages x 10 epochs = 110 epochs
- Fine-tuning: 10 epochs
- **Total: 140 epochs**

### Visual Timeline

```
Epochs:  1         20        30   40   ...  120  130       140
         |-- Warm-up --|-- Anneal (11 stages) --|-- Finetune --|
mu:      1.0           1.0  0.8  0.6  ...  0.02  0.01    0.0
```

---

## 4. SGD with Flux and Zygote

### How Automatic Differentiation Works Through the Full Pipeline

The training pipeline differentiates through the entire forward pass: neural network, scenario construction, LP solve, and cost evaluation. This is possible because of three interlocking mechanisms:

#### 1. Flux.gradient + Zygote Reverse-Mode AD

`Flux.gradient(model) do m ... end` invokes Zygote's reverse-mode AD. Zygote traces the forward computation and builds a "pullback" tape. On the backward pass, it propagates the gradient of the scalar loss back through every operation.

#### 2. The ChainRules rrule on `solve_lp` (Subgradient)

The HiGHS LP solver is not natively differentiable. The custom `rrule` in `src/diff/subgradient_rrule.jl` provides a subgradient approximation that makes it differentiable on the default training path:

```julia
function ChainRulesCore.rrule(::typeof(solve_lp), A, b, c; tol=1e-9)
    (x_opt, lambda_opt) = solve_lp(A, b, c; tol)

    function solve_lp_pullback(dx_bar)
        db_bar = -lambda_opt * sum(dx_bar)
        dc_bar =  x_opt * sum(dx_bar)
        return (NoTangent(), NoTangent(), db_bar, dc_bar)
    end

    return (x_opt, lambda_opt), solve_lp_pullback
end
```

Key design choices:
- **Subgradient via LP duality**: The pullback uses the optimal dual variable `lambda_opt` as a subgradient for `b`, and the optimal primal `x_opt` as a subgradient for `c`. This follows from LP sensitivity analysis: `dV/db = -lambda*` where `V(b) = min c'x s.t. Ax=b, x>=0`.
- **No tangent for A**: Only gradients w.r.t. `b` (the RHS) and `c` (the cost vector) are propagated. This is sufficient because scenario parameters affect the LP through `b` and `c`.
- **Lightweight pullback**: The gradients are computed directly from the cached primal and dual solutions, without needing to solve any additional linear systems.

> **Note:** A barrier-based `rrule` for `solve_barrier` also exists in `src/diff/barrier_rrule.jl`, which uses implicit differentiation of the KKT conditions (see Section 4.3). However, it is **not on the default training path** since `surrogate_first_stage` and `recourse_cost` now use `solve_lp` exclusively.

#### 3. Implicit Differentiation of KKT Conditions (Barrier Path Only)

The functions `implicit_diff_h` and `implicit_diff_q` in `src/diff/implicit_diff.jl` compute the sensitivity of the barrier LP optimum `x*` with respect to problem data by applying the implicit function theorem to the KKT conditions. These are used by the barrier `rrule` (not on the default training path), but are documented here for completeness:

```
KKT conditions:
    g₁(x,λ) = c - μ/x + A'λ = 0   (stationarity)
    g₂(x,λ) = Ax - b          = 0   (feasibility)

KKT Jacobian:
    K = [D   A']   where D = diag(μ/x²)
        [A   0 ]

Sensitivity w.r.t. b:  K [∂x/∂b; ∂λ/∂b] = [0; I_m]   →  ∂x*/∂b = (K⁻¹)₁:ₙ,ₙ₊₁:ₙ₊ₘ
Sensitivity w.r.t. c:  K [∂x/∂c; ∂λ/∂c] = [-I_n; 0]  →  ∂x*/∂c = -(K⁻¹)₁:ₙ,₁:ₙ
```

This amounts to solving linear systems with the KKT matrix, which is well-conditioned when mu > 0 (the diagonal block D ensures positive definiteness of the (1,1) block).

#### Gradient Flow Through the Full Pipeline

Here is the complete chain of differentiation, from loss back to model parameters:

```
d(loss)/d(model_params)
  = d(loss)/d(cost) * d(cost)/d(x₁) * d(x₁)/d(b,c) * d(b,c)/d(scenario_params) * d(scenario_params)/d(model_params)
    \_____________/   \____________/   \_____________/   \________________________/   \___________________________/
     evaluate_cost     recourse_cost    subgradient       scenario_realization          Flux.Chain (Dense layers)
                                        via solve_lp
                                        rrule
```

#### Zygote Compatibility Notes

The codebase takes care to avoid Zygote-incompatible patterns:
- `build_mu_vector` uses `vcat` instead of `append!` (no mutation).
- Scenario construction uses comprehensions rather than in-place array modification.
- `_kkt_matrix` in `implicit_diff.jl` and barrier solver KKT assembly use `vcat`/`hcat` to build the KKT matrix without in-place mutation (e.g., `K = vcat(hcat(Diagonal(D_diag), A'), hcat(A, zeros(T, m, m)))`).
- `scenario_realization` for `ShipmentPlanningProblem` uses `promote_type(eltype(param), eltype(prob.q))` to handle Float32 model outputs mixing with Float64 problem data, ensuring type stability through the AD pipeline.
- The LP solver runs in the forward pass only; its pullback uses the cached primal and dual solutions.

---

## 5. Loss Function: Decision Regret

The loss function is defined in `src/loss/decision_regret.jl`. It measures the suboptimality of decisions made using predicted scenarios versus the actual realization.

### `decision_regret`

```julia
function decision_regret(
    prob::ProblemInstance,
    mu_surr::Real,
    mu_prim::Real,
    predicted_scenarios::Vector{<:Scenario},
    actual_scenario::Scenario
) -> Float64
```

**Step 1 — Surrogate decision.** Build a `TwoStageSLP` from the predicted scenarios and solve it with barrier parameter `mu_surr` to obtain the first-stage decision `x̂₁`:

```julia
slp_surr = TwoStageSLP(A1, b1, c1, predicted_scenarios)
x1_hat = surrogate_first_stage(slp_surr, mu_surr)
```

**Step 2 — Cost evaluation.** Evaluate `x̂₁` on the actual scenario with barrier parameter `mu_prim`:

```julia
slp_prim = TwoStageSLP(A1, b1, c1, [actual_scenario])
cost_val = evaluate_cost(x1_hat, slp_prim, mu_prim)
```

The returned cost is `c₁'x̂₁ + Q(x̂₁, actual_scenario)`, where `Q` is the recourse (second-stage) value function.

### `surrogate_first_stage`

Solves the extensive-form two-stage LP:

1. Calls `extensive_form(slp)` to build the block-structured constraint matrix.
2. Solves with `solve_lp` (HiGHS), which is made differentiable via the subgradient `rrule`.
3. Returns the first `n₁` components of the solution (first-stage decision variables).

### `evaluate_cost`

Computes the total cost of a first-stage decision:

```
cost = c₁'x₁ + sum_s p_s * Q(x₁, s)
```

where `Q(x₁, s)` is the recourse cost: solve the LP `min q'y s.t. Wy = h - Tx₁, y >= 0` via `solve_lp`.

### `relative_decision_regret`

Normalizes the regret relative to the optimal cost:

```
relative_regret = (surrogate_cost - optimal_cost) / |optimal_cost|
```

where `optimal_cost` is the cost when the actual scenario is used for both the decision and the evaluation. The denominator includes a small epsilon (`1e-10`) for numerical stability.

---

## 6. Evaluation Metrics

The evaluation system is implemented in `src/evaluation/evaluate.jl`.

### `compute_evaluation_metrics`

```julia
function compute_evaluation_metrics(
    generator, prob, test_data, mu_surr, mu_prim
) -> Dict
```

For each `(context, actual_scenario)` pair in `test_data`, this function:

1. Runs the generator: `raw = generator(x)`.
2. Converts to scenarios: `pred_sc = _params_to_scenarios(prob, raw)`.
3. Computes **absolute decision regret** via `decision_regret(...)`.
4. Computes **relative decision regret** via `relative_decision_regret(...)`.
5. Extracts predicted and actual scenario parameter vectors.
6. Computes the **L2 distance** between predicted and actual parameters.

Returns a `Dict` with keys:

| Key | Type | Description |
|-----|------|-------------|
| `"regrets"` | `Vector{Float64}` | Per-sample absolute decision regret |
| `"rel_regrets"` | `Vector{Float64}` | Per-sample relative decision regret |
| `"l2_distances"` | `Vector{Float64}` | Per-sample L2 distance in scenario parameter space |
| `"predicted_params"` | `Vector{Vector{Float64}}` | Per-sample predicted scenario parameters |
| `"actual_params"` | `Vector{Vector{Float64}}` | Per-sample actual scenario parameters |
| `"n"` | `Int` | Number of test samples |

### `print_evaluation_summary`

Prints a formatted summary to an `IO` stream with four sections:

1. **Decision regret**: mean, std, 95% confidence interval (`1.96 * std / sqrt(n)`), median, min, max.
2. **Relative decision regret**: same statistics.
3. **Threshold fractions**: percentage of samples with absolute relative regret below 1%, 5%, and 10%.
4. **Scenario L2 distance**: mean, std, median.

The 95% confidence interval assumes approximate normality of the sample mean (Central Limit Theorem).

### `save_evaluation_csv`

Writes per-sample metrics as a CSV file with columns: `sample`, `decision_regret`, `relative_decision_regret`, `l2_distance`.

### Scenario Parameter Extraction

The `_extract_scenario_params` function is specialized per problem type to invert `scenario_realization`:

- **`UnreliableNewsvendorProblem`**: Extracts `[D, U]` from `sc.h[1]` and `sc.T[2,1]`.
- **`ShipmentPlanningProblem`**: Extracts `sc.h[1:n_loc]`.
- **`ResourceAllocationProblem`**: Extracts `sc.h[(I+1):end]` (the demand components).

---

## 7. Plotting Functions

All five plotting functions are defined in `src/evaluation/evaluate.jl`. They use the `Plots.jl` and `StatsPlots.jl` backends. The headless GR backend is forced via `ENV["GKSwstype"] = "100"` to enable plot generation without a display server.

Each function takes the data to visualize and a file path, and calls `Plots.savefig(path)` to write the output. The file format is determined by the path extension (typically `.png`, `.pdf`, or `.svg`).

### 1. `plot_loss_curve(loss_history, path)`

**What it visualizes:** Training loss (decision regret) as a function of epoch number.

**Implementation:** A simple line plot via `Plots.plot`. The x-axis is `1:length(loss_history)`, the y-axis is the per-epoch average loss. Uses `linewidth=2`, no legend, and an 800x500 canvas.

**When to use:** To diagnose convergence — look for a decreasing trend. With continuation training, you will see distinct segments corresponding to each mu phase.

### 2. `plot_regret_histogram(rel_regrets, ci, path)`

**What it visualizes:** Distribution of relative decision regrets across test samples.

**Implementation:** A histogram via `Plots.histogram` with `alpha=0.7` for semi-transparency. Overlays:
- A vertical dashed red line at the mean via `Plots.vline!`.
- A text annotation showing `Mean +/- CI` near the top of the plot via `annotate!`.

**Arguments:** `ci` is the half-width of the 95% confidence interval on the mean, pre-computed by the caller.

### 3. `plot_regret_boxplot(rel_regrets, path)`

**What it visualizes:** The shape of the relative regret distribution, including quartiles and outliers.

**Implementation:** A composite violin + boxplot using `StatsPlots.violin` and `StatsPlots.boxplot!`. The violin shows the full density estimate; the overlaid boxplot shows the quartiles. Both use semi-transparency (`alpha=0.7` and `alpha=0.5`). Canvas size is 600x500.

### 4. `plot_regret_cdf(rel_regrets, path)`

**What it visualizes:** The empirical cumulative distribution function of the absolute relative regret.

**Implementation:**
1. Sorts `abs.(rel_regrets)` in ascending order.
2. Plots the sorted values against `(1:n) ./ n` (fraction of samples).
3. Adds horizontal and vertical reference lines (gray dotted) at thresholds 1%, 5%, and 10%, showing what fraction of samples fall below each threshold.

**When to use:** To answer questions like "what fraction of test samples have relative regret below 5%?" — read the intersection of the curve with the vertical line at 0.05.

### 5. `plot_scenario_scatter(predicted, actual, path; max_dims=8)`

**What it visualizes:** Predicted vs. actual scenario parameters, one subplot per parameter dimension. Points on the 45-degree line indicate perfect prediction.

**Implementation:**
- Skipped entirely if the scenario parameter dimension exceeds `max_dims` (default: 8).
- Creates a grid of scatter plots with `ceil(dim / 4)` rows and `min(dim, 4)` columns.
- Each subplot shows `actual` on the x-axis, `predicted` on the y-axis, with a red dashed 45-degree reference line.
- Marker size is 3 with `alpha=0.6` for density visibility.
- Uses `Plots.plot(plts...; layout=(...))` to combine subplots.

**When to use:** To assess whether the model is systematically over- or under-predicting specific scenario parameters.

---

## 8. Checkpoint Save/Load/Restore Cycle

The persistence system in `src/persistence.jl` uses Julia's built-in `Serialization` module to save and restore training state.

### What Gets Serialized

A checkpoint is a `Dict{String, Any}` with the following entries:

| Key | Type | Description |
|-----|------|-------------|
| `"model_state"` | `Flux.state(model)` | All model weights and biases (nested NamedTuples) |
| `"model_config"` | `Dict{String,Any}` | Architecture: `input_dim`, `output_dim`, `hidden_dim`, `n_layers`, `activation` |
| `"problem_type"` | `String` | One of `"resource_allocation"`, `"shipment_planning"`, `"newsvendor"` |
| `"noise_pattern"` | `String` | Stringified `NoisePattern` enum value |
| `"nr_of_scenarios"` | `Int` | Number of scenarios the model outputs |
| `"training_config"` | `Dict{String,Any}` | All training hyperparameters (lr, mu values, batchsize, epochs, etc.) |
| `"loss_history"` | `Vector{Float64}` | Per-epoch average loss for the entire training run |
| `"total_epochs"` | `Int` | Total number of completed training epochs |
| `"timestamp"` | `String` | ISO 8601 timestamp of when the checkpoint was saved |

### Save: `save_checkpoint`

```julia
save_checkpoint(path, model, prob, config, loss_history; nr_of_scenarios=1)
```

The function infers architecture metadata directly from the `Flux.Chain`:
- `input_dim` from the first layer's weight matrix columns.
- `output_dim` from the last layer's weight matrix rows.
- `hidden_dim` from the first layer's weight matrix rows.
- `n_layers` = `length(model.layers) - 1` (hidden layers only; the output layer is implicit).
- `activation` is resolved to a string name via `ACTIVATION_MAP` (supports `"relu"`, `"tanh"`, `"sigmoid"`, `"softplus"`, `"identity"`).

Model weights are captured via `Flux.state(model)`, which returns a nested structure of arrays without any Flux type dependencies — making it safe for serialization.

### Load: `load_checkpoint`

```julia
load_checkpoint(path) -> Dict{String, Any}
```

A thin wrapper around `Serialization.deserialize`. Validates that the file exists.

### Restore: `restore_model`

```julia
restore_model(checkpoint) -> Flux.Chain
```

Rebuilds the model from scratch and loads the saved weights:

1. Reads `model_config` from the checkpoint.
2. Maps the activation string back to a function via `ACTIVATION_MAP`.
3. Calls `build_generator(input_dim, output_dim; hidden_dim, n_layers, activation)` to reconstruct the architecture.
4. Calls `Flux.loadmodel!(model, checkpoint["model_state"])` to inject the saved weights.

This two-step process (rebuild architecture, then load weights) is more robust than serializing the model object directly, because it avoids issues with Julia's type system across versions.

### Resume Training Flow

The CLI `continue` command demonstrates the full resume cycle:

```
load_checkpoint(path)
    │
    ▼
restore_model(checkpoint)     ──► Flux.Chain with saved weights
    │
    ▼
resolve_problem(ckpt["problem_type"])  ──► ProblemInstance
    │
    ▼
generate fresh training data (same n_train and sigma from checkpoint config)
    │
    ▼
train!(model, prob, data; epochs=N, ...)   ──► new losses
    │
    ▼
all_losses = vcat(old_loss_history, new_losses)
    │
    ▼
save_checkpoint(new_path, model, prob, updated_config, all_losses)
```

Note that optimizer state is **not** persisted across checkpoints. When training is resumed, a fresh optimizer is created. The learning rate and other optimizer parameters can be overridden from the CLI or fall back to checkpoint values.

### `print_checkpoint_info`

Provides a human-readable summary of a checkpoint, including:
- Timestamp and problem metadata.
- Model architecture details and total parameter count (computed analytically by `_count_params`).
- All training configuration key-value pairs (sorted alphabetically).
- Training history summary: total epochs, initial/final/best loss.

### Intermediate Checkpoints

The CLI `train` command supports periodic checkpoint saving via `--save-interval N`. When enabled, training is split into chunks of `N` epochs, and after each chunk an intermediate checkpoint is saved with a `_stageK` suffix (e.g., `model_stage1.jls`, `model_stage2.jls`). The `opt_state` is shared across chunks within the same run to maintain optimizer momentum.

---

## Appendix: Key Type Signatures

```julia
# Training
train!(model, prob, dataset; mu_surr, mu_prim, opt, epochs, batchsize, verbose, opt_state) -> Vector{Float64}
continuation_train!(model, prob, dataset; mu_schedule, epochs_per_stage, first_stage_epochs, finetune_epochs, opt, verbose) -> Vector{Float64}

# Loss
decision_regret(prob, mu_surr, mu_prim, predicted_scenarios, actual_scenario) -> Float64
relative_decision_regret(prob, mu_surr, mu_prim, predicted_scenarios, actual_scenario) -> Float64
surrogate_first_stage(slp, mu) -> Vector{Float64}
evaluate_cost(x1, slp, mu) -> Float64
recourse_cost(x1, sc, mu) -> Float64

# Evaluation
compute_evaluation_metrics(generator, prob, test_data, mu_surr, mu_prim) -> Dict
print_evaluation_summary(io, metrics) -> nothing
save_evaluation_csv(metrics, path) -> nothing

# Plots
plot_loss_curve(loss_history, path) -> nothing
plot_regret_histogram(rel_regrets, ci, path) -> nothing
plot_regret_boxplot(rel_regrets, path) -> nothing
plot_regret_cdf(rel_regrets, path) -> nothing
plot_scenario_scatter(predicted, actual, path; max_dims=8) -> nothing

# Persistence
save_checkpoint(path, model, prob, config, loss_history; nr_of_scenarios) -> String
load_checkpoint(path) -> Dict{String, Any}
restore_model(checkpoint) -> Flux.Chain
print_checkpoint_info(io, checkpoint) -> nothing
resolve_problem(name) -> ProblemInstance
```
