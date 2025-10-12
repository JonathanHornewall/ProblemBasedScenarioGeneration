# Refactored Log-Barrier LP Utilities

This directory contains an experimental reimplementation of the linear-program differentiators used in `differentials_logbar_lp.jl`. The refactored code supports linear programs that separate equality and inequality constraints and does not rely on implicit non-negativity of the decision variables. Inequality feasibility is enforced through explicit log-barrier terms that operate on arbitrary linear inequalities.

The files provide

- a new problem description (`refactored_logbar_lp.jl`) with full derivative support,
- deterministic solvers for both the raw and log-barrier formulations (`refactored_lp_solver.jl` and `refactored_logbar_lp_solver.jl`), and
- regression and sensitivity tests (`tests.jl`).

## Tests

The test suite serves three complementary goals:

1. **Regression against the canonical implementation** – the tests confirm that, when inequalities encode non-negativity, the new KKT system and derivatives match the existing canonical formulation.
2. **Solver validation** – dedicated cases ensure that the general-purpose LP solver and the log-barrier solver return feasible, optimal solutions and correctly reduce to the non-barrier solver when the barrier weights vanish.
3. **Derivative verification** – finite-difference checks compare analytical sensitivities against numerical approximations, ensuring that the reported derivatives remain trustworthy.

Run the tests from the project root with:

```julia
julia --project -e 'include("src/dev/refactored_logbar_lp/tests.jl")'
```
