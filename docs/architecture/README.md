# Unit 1: Overview & Architecture

## 1. Project Overview

**ProblemBasedScenarioGeneration** is a Julia package for *decision-focused learning* in two-stage stochastic linear programming (2SLP). Rather than training a neural network to predict scenarios that match historical data (predict-then-optimize), this package trains a scenario generator that directly minimizes the downstream *decision regret* -- the cost gap between decisions made with predicted scenarios versus the true scenario.

The core idea: a neural network observes contextual features and outputs scenario parameters. Those scenarios are fed into a differentiable LP solver, which produces a first-stage decision. The decision is evaluated against the actual scenario, and the resulting cost is backpropagated through the entire pipeline to update the network weights.

Key capabilities:

- **End-to-end differentiable pipeline** from features through LP solving to decision cost
- **HiGHS LP solver** (primary, for training) with subgradient rrule, and **log-barrier Newton solver** (for implicit differentiation research) with implicit differentiation rrule
- **mu-continuation training schedule** that anneals the barrier parameter from smooth to exact
- **Problem-agnostic framework** with a `ProblemInstance` interface for plugging in new 2SLPs
- **Three built-in problems**: resource allocation, shipment planning, unreliable newsvendor
- **CLI tool** (`pbsg`) for training, evaluation, checkpointing, and plotting

## 2. Architecture Diagram

```
                    ProblemBasedScenarioGeneration Pipeline
  ============================================================================

  TRAINING DATA                    NEURAL NETWORK              SCENARIO CONSTRUCTION
  +------------------+        +---------------------+       +---------------------+
  | (context, actual |        | build_generator()   |       | scenario_realization|
  |  scenario) pairs | -----> | Dense -> Dense ->   | ----> | (prob, params)      |
  | from             |  x_i   | ... -> Dense+softplus|  raw | -> Vector{Scenario} |
  | generate_dataset |        | (Flux.Chain)        | params|                     |
  +------------------+        +---------------------+       +---------------------+
                                       |                             |
                                       | (weights updated            | predicted
                                       |  via backprop)              | scenarios
                                       |                             v
  LOSS BACKPROPAGATION         +---------------------+     +---------------------+
  +------------------+         | decision_regret()   |     | TwoStageSLP         |
  | Flux.gradient    | <------ | cost = c'x1 +       |     | (A, b, c, scenarios)|
  | through entire   |  loss   |   sum p_s * Q(x1,s) |     +---------------------+
  | pipeline via     |         +---------------------+              |
  | ChainRules rrule |                  ^                           v
  +------------------+                  |                  +---------------------+
         |                              |                  | extensive_form()    |
         v                      actual scenario            | -> (A_ext, b_ext,  |
  +------------------+                  |                  |    c_ext)           |
  | Flux.update!     |                  |                  +---------------------+
  | (Adam optimizer) |                  |                           |
  +------------------+                  |                           v
                                        |                  +---------------------+
                                        |                  | solve_lp()          |
                                        +----------------- | (HiGHS LP solver)   |
                                                           | s.t. Ax = b, x >= 0|
                                          x1* (first-      +---------------------+
                                          stage decision)           |
                                                           +---------------------+
                                                           | rrule (pullback)    |
                                                           | subgradient_rrule   |
                                                           | dx/db, dx/dc via    |
                                                           | LP dual subgradients|
                                                           +---------------------+
```

### Data Flow Summary

```
features x_i --> neural_net(x_i) --> raw_params --> scenario_realization() --> Scenario[]
                                                                                  |
                                                                                  v
            loss <-- evaluate_cost(x1*, actual) <-- surrogate_first_stage(TwoStageSLP)
              |                                          |
              v                                    solve_lp() + subgradient rrule
         backprop via Flux.gradient()                    |
              |                                    LP dual-based subgradients
              v
         Flux.update!(opt_state, model, grads)
```

## 3. Module Dependency Graph

```
ProblemBasedScenarioGeneration.jl (entry point)
|
+-- core/
|   +-- scenario.jl            Scenario{T} struct
|   +-- two_stage_lp.jl        TwoStageSLP{T} struct (depends on: Scenario)
|   +-- canonical_form.jl      to_canonical, extensive_form (depends on: TwoStageSLP, Scenario)
|
+-- solvers/
|   +-- lp_solver.jl           solve_lp, solve_lp_primal (depends on: JuMP, HiGHS)
|   +-- barrier_solver.jl      BarrierCache, solve_barrier (depends on: lp_solver)
|
+-- diff/
|   +-- implicit_diff.jl       implicit_diff_h, implicit_diff_q (depends on: BarrierCache)
|   +-- barrier_rrule.jl       rrule(solve_barrier) (depends on: implicit_diff, solve_barrier)
|   +-- subgradient_rrule.jl   rrule(solve_lp) (depends on: solve_lp)
|
+-- problems/
|   +-- interface.jl           ProblemInstance, NoisePattern (no internal deps)
|   +-- resource_allocation.jl ResourceAllocationProblem (depends on: interface, Scenario)
|   +-- shipment_planning.jl   ShipmentPlanningProblem (depends on: interface, Scenario)
|   +-- newsvendor.jl          UnreliableNewsvendorProblem (depends on: interface, Scenario)
|
+-- loss/
|   +-- decision_regret.jl     decision_regret, evaluate_cost, surrogate_first_stage
|                              (depends on: TwoStageSLP, extensive_form, solve_lp,
|                               ProblemInstance)
|
+-- models/
|   +-- scenario_generator.jl  build_generator (depends on: Flux, ProblemInstance)
|   +-- output_heads.jl        build_output_head, build_full_model
|                              (depends on: scenario_generator, scenario_realization)
|
+-- training/
|   +-- trainer.jl             train! (depends on: Flux, decision_regret, ProblemInstance)
|   +-- continuation.jl        continuation_train! (depends on: train!)
|
+-- evaluation/
|   +-- evaluate.jl            compute_evaluation_metrics, plotting functions
|                              (depends on: decision_regret, Plots, StatsPlots)
|
+-- persistence.jl             save_checkpoint, load_checkpoint, restore_model
|                              (depends on: Serialization, Flux, build_generator)
|
+-- cli.jl                     cli_main, cmd_train, cmd_test, cmd_evaluate, cmd_info, cmd_continue
                               (depends on: ArgParse, all modules above)
```

## 4. Design Philosophy

### 4.1 Differentiable Optimization

The entire pipeline from neural network output to decision cost must be differentiable so that Flux/Zygote can compute gradients for training. Two solver backends are available, each with its own ChainRules rrule:

- **`solve_lp` (HiGHS)** with **subgradient rrule** (`subgradient_rrule.jl`): the **primary solver used during training**. Uses LP duals as subgradients to provide approximate gradients. Fast, robust, and fully Zygote-compatible.
- **`solve_barrier` (Newton log-barrier)** with **implicit differentiation rrule** (`barrier_rrule.jl`): solves `min c'x - mu * sum(log(x_i))` subject to `Ax = b, x > 0`, then applies the implicit function theorem to the KKT conditions to compute `dx*/db` and `dx*/dc`. **Retained for research and alternative use cases** but no longer on the default training path.

Both `surrogate_first_stage` and `recourse_cost` now call `solve_lp` directly (via HiGHS), so standard training does not require the barrier solver.

### 4.2 No-Mutation for AD Compatibility

Zygote (the AD engine used by Flux) does not support in-place mutation. The codebase follows a strict no-mutation discipline:

- `extensive_form` builds the block-structured constraint matrix using `hcat`/`vcat` and comprehensions instead of pre-allocating and filling.
- `build_mu_vector` uses `vcat` instead of `append!`.
- `_kkt_matrix` in `implicit_diff.jl` constructs the KKT system using `vcat`/`hcat` (e.g., `vcat(hcat(Diagonal(D_diag), A'), hcat(A, zeros(T, m, m)))`) for Zygote safety.
- Barrier solver KKT assembly in `barrier_solver.jl` similarly uses mutation-free `vcat`/`hcat` construction.
- `scenario_realization` in problem implementations uses `promote_type` for Float32/Float64 compatibility -- Flux models typically output Float32 while problem data is Float64, so type promotion ensures consistent element types throughout the pipeline.
- All scenario construction uses functional patterns (creating new arrays rather than modifying existing ones).

### 4.3 Problem-Agnostic Framework

The package separates the optimization framework from specific problem instances via the `ProblemInstance` abstract type. Any new two-stage stochastic LP can be added by implementing four methods:

| Method | Signature | Purpose |
|--------|-----------|---------|
| `first_stage_data` | `(prob) -> (A, b, c)` | First-stage constraint data |
| `scenario_realization` | `(prob, param) -> Scenario` | Map parameter vector to `Scenario` |
| `generate_dataset` | `(prob, n) -> Vector{Tuple}` | Generate (context, scenario) training pairs |
| `noise_pattern` | `(prob) -> NoisePattern` | Which second-stage matrices vary (e.g., `H_ONLY`, `WH`) |

The `NoisePattern` enum (`H_ONLY`, `Q_ONLY`, `W_ONLY`, `WH`, `WQ`, `WHQ`) determines which output head is used to map raw network output to `Scenario` structs.

### 4.4 mu-Continuation for the Log-Barrier Solver

Standard training now uses `solve_lp` directly -- the `mu` parameter accepted by `surrogate_first_stage` and `recourse_cost` does not affect solver choice, as both always call `solve_lp` (HiGHS).

Continuation training with the barrier solver remains available via `continuation_train!`, which implements a three-phase schedule:

1. **Warm-up**: train at the largest mu (e.g., 1.0) for `first_stage_epochs` to find a good region.
2. **Annealing**: step through a decreasing `mu_schedule` (e.g., `[1.0, 0.8, ..., 0.01]`), training `epochs_per_stage` epochs at each level.
3. **Fine-tuning**: train at mu = 0 using the `solve_lp` subgradient rrule for `finetune_epochs`.

When mu = 0, `solve_barrier` falls back to `solve_lp` (HiGHS), and the subgradient rrule in `subgradient_rrule.jl` provides approximate gradients using LP duals.

## 5. File Manifest

### Source Files (`src/`)

| File | Description |
|------|-------------|
| `ProblemBasedScenarioGeneration.jl` | Module entry point; imports, includes, and exports |
| `core/scenario.jl` | `Scenario{T}` struct holding W, T, h, q for one second-stage scenario |
| `core/two_stage_lp.jl` | `TwoStageSLP{T}` struct combining first-stage data with scenario vector |
| `core/canonical_form.jl` | `to_canonical`, `to_canonical_decision`, `extensive_form` for LP form conversion |
| `solvers/lp_solver.jl` | `solve_lp`, `solve_lp_primal` via JuMP/HiGHS for exact LP solving |
| `solvers/barrier_solver.jl` | `BarrierCache`, `solve_barrier`, `kkt_residual` -- Newton log-barrier LP solver |
| `diff/implicit_diff.jl` | `implicit_diff_h`, `implicit_diff_q`, `recourse_multiplier` via KKT implicit function theorem |
| `diff/barrier_rrule.jl` | ChainRules `rrule` for `solve_barrier` enabling AD through the barrier solver |
| `diff/subgradient_rrule.jl` | ChainRules `rrule` for `solve_lp` using LP duals as subgradients (mu=0 case) |
| `problems/interface.jl` | `ProblemInstance` abstract type, `NoisePattern` enum, interface method stubs |
| `problems/resource_allocation.jl` | `ResourceAllocationProblem` (20 clients, 30 resources, H_ONLY noise) |
| `problems/shipment_planning.jl` | `ShipmentPlanningProblem` (12 warehouses, 4 locations, H_ONLY noise) |
| `problems/newsvendor.jl` | `UnreliableNewsvendorProblem` (1D order quantity, WH noise with demand + reliability) |
| `loss/decision_regret.jl` | `decision_regret`, `relative_decision_regret`, `surrogate_first_stage`, `evaluate_cost`, `recourse_cost` |
| `models/scenario_generator.jl` | `build_generator` neural network factory, `_context_dim`, `_scenario_param_dim` helpers |
| `models/output_heads.jl` | `build_output_head`, `build_full_model` -- maps raw network output to `Vector{Scenario}` |
| `training/trainer.jl` | `train!` training loop with Flux.gradient and Adam optimizer |
| `training/continuation.jl` | `continuation_train!` implementing the three-phase mu-annealing schedule |
| `evaluation/evaluate.jl` | `compute_evaluation_metrics`, `print_evaluation_summary`, `save_evaluation_csv`, plotting functions |
| `persistence.jl` | `save_checkpoint`, `load_checkpoint`, `restore_model`, `print_checkpoint_info`, `resolve_problem` |
| `cli.jl` | `cli_main`, ArgParse settings, subcommand handlers (`cmd_train`, `cmd_test`, `cmd_evaluate`, `cmd_continue`, `cmd_info`) |

### Other Files

| File | Description |
|------|-------------|
| `bin/pbsg.jl` | CLI entry point script; activates project and calls `cli_main()` |
| `Project.toml` | Package metadata, dependencies, and compatibility bounds |
| `test/runtests.jl` | Test suite entry point |
| `test/test_barrier_solver.jl` | Tests for the Newton log-barrier solver |
| `test/test_implicit_diff.jl` | Tests for implicit differentiation correctness |
| `test/test_decision_regret.jl` | Tests for the decision regret loss computation |
| `test/test_problems.jl` | Tests for problem instance construction and interface |

## 6. Dependencies

From `Project.toml` (v0.2.0, requires Julia >= 1.9):

| Package | Purpose | Compat |
|---------|---------|--------|
| **Flux** | Neural network framework (layers, optimizers, training) | 0.14 -- 0.16 |
| **ChainRulesCore** | Custom reverse-mode AD rules (rrules for solvers) | 1.x |
| **Zygote** | Automatic differentiation engine (used by Flux) | 0.6 |
| **JuMP** | Algebraic modeling for LP formulation | 1.x |
| **HiGHS** | LP solver backend for exact (mu=0) solves | 1.x |
| **Distributions** | Sampling for data generation (Normal, Uniform, Beta, MvNormal) | 0.25 |
| **ArgParse** | Command-line argument parsing for the `pbsg` CLI | 1.x |
| **Plots** | Visualization (loss curves, histograms, CDFs) | 1.x |
| **StatsPlots** | Violin/boxplot extensions for Plots | 0.15 |
| **Serialization** | Model checkpoint save/load (Julia stdlib) | stdlib |
| **LinearAlgebra** | Matrix operations, `Diagonal`, `pinv`, `dot` (Julia stdlib) | stdlib |
| **SparseArrays** | Sparse matrix support (Julia stdlib) | stdlib |
| **Statistics** | `mean`, `std`, `median` (Julia stdlib) | stdlib |
| **Random** | `randperm`, `seed!` (Julia stdlib) | stdlib |
| **Dates** | Timestamps for checkpoints (Julia stdlib) | stdlib |

**Test-only dependencies**: `Test`, `FiniteDiff` (2.x), `FiniteDifferences` (0.12).

## 7. Key Types and Functions Reference

### Core Types

- **`Scenario{T<:Real}`** -- holds `W` (m2 x n2), `T` (m2 x n1), `h` (m2,), `q` (n2,) for one second-stage scenario
- **`TwoStageSLP{T<:Real}`** -- holds `A`, `b`, `c` (first stage) plus `scenarios::Vector{Scenario{T}}` and `p` (probabilities)
- **`BarrierCache{T<:Real}`** -- stores barrier solve results (`x`, `lambda`, `mu`, `A`, `b`, `c`) for downstream implicit differentiation
- **`ProblemInstance`** -- abstract type; subtypes: `ResourceAllocationProblem`, `ShipmentPlanningProblem`, `UnreliableNewsvendorProblem`
- **`NoisePattern`** -- enum: `H_ONLY`, `Q_ONLY`, `W_ONLY`, `WH`, `WQ`, `WHQ`

### Core Functions

- **`solve_barrier(A, b, c, mu) -> BarrierCache`** -- Newton log-barrier solver
- **`solve_lp(A, b, c) -> (x, lambda)`** -- exact LP via HiGHS
- **`implicit_diff_h(cache) -> Matrix`** -- dx*/db sensitivity
- **`implicit_diff_q(cache) -> Matrix`** -- dx*/dc sensitivity
- **`decision_regret(prob, mu_surr, mu_prim, predicted, actual) -> Float64`** -- the training loss
- **`surrogate_first_stage(slp, mu) -> Vector`** -- solve extensive form, return x1*
- **`build_generator(prob; nr_of_scenarios, ...) -> Flux.Chain`** -- neural network factory
- **`train!(model, prob, dataset; mu_surr, mu_prim, ...) -> Vector{Float64}`** -- training loop
- **`continuation_train!(model, prob, dataset; mu_schedule, ...) -> Vector{Float64}`** -- mu-annealing training

## 8. Further Reading

- [Mathematical Foundations](../mathematical-foundations.md) -- barrier method derivation, implicit differentiation theory, KKT conditions
- [Problems and Models](../problems-and-models.md) -- detailed description of each problem instance and neural network architecture
- [Training and Evaluation](../training-and-evaluation.md) -- training loop internals, continuation schedule, evaluation metrics
- [CLI Reference](../cli-reference.md) -- complete command-line interface documentation for `pbsg`
