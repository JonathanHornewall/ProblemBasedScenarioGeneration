# Refactored loss prototype

This directory hosts an experimental reimplementation of `loss.jl` with a
bespoke reverse-mode rule that avoids differentiating through the full solver
stack.  The main entry point is `RefactoredLoss.refactored_loss`, which mirrors
the behaviour of `ProblemBasedScenarioGeneration.loss` but supplies its own
`rrule`.

## Why this should be faster

The custom `rrule` factors the derivative into pieces that are already
available from analytic primitives instead of letting AD re-run the entire
solver stack:

* The surrogate solve is reused for both the primal value and reverse sweep, so
  ChainRules never re-invokes the expensive interior-point solver in the
  pullback.  Gradients only consume cached tensors produced by
  `derivative_surrogate_solution` rather than differentiating through
  `LogBarCanLP_standard_solver_primal` again.
* The pullback contracts those cached sensitivities with the first-stage cost
  gradient via `_contract_tensor_list!` / `_contract_matrix_list!` without
  building new tapes, which removes a large nested AD loop.
* Only the inexpensive `scenario_collection_realization` maps are left to
  Zygote, greatly reducing the amount of differentiation through linear-algebra
  solves compared to the baseline.

Together these changes cut the amount of work done during reverse-mode: the
pullback boils down to tensor contractions and a single gradient through the
scenario mapping instead of re-solving and re-factorising the surrogate LP.

## Running the experiment

To verify correctness and performance, run:

```bash
julia --project=src/julia/ProblemBasedScenarioGeneration.jl \
    src/julia/ProblemBasedScenarioGeneration.jl/src/dev/refactored_loss/run_refactored_loss_tests.jl
```

The script compares the original and refactored losses, checks gradients
against finite differences, and benchmarks reverse-mode differentiation.
