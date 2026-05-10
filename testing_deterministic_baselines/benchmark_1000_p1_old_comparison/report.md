# 1000-Scenario Deterministic Baseline Validation

Generated at: 2026-05-05T18:09:52.039Z

## Sandbox

- All generated files are under `testing_deterministic_baselines/benchmark_1000_p1_old_comparison`.
- Source packages and source experiment artifacts were read-only inputs.

## Data

- Training: 100 context-scenario pairs, demand power 1.0, seed 202615051.
- Test: 30 cached contexts with 1000 scenarios per context.
- Optimality results: 30 cached rows, each with 20 evaluation batches.
- Local training digest: sha1:0e2e5445c38af36f9dcc9a559c8a18afc6760c75.

## ContextualDFL Ranking

| Rank | Method | Gap % | Mean Relative Regret | Mean Regret | Eval Seconds |
|---:|---|---:|---:|---:|---:|
| 1 | Residual SAA | 3.758 | 0.03758 | 32.242 | 144.742 |
| 2 | Least Squares | 5.942 | 0.05942 | 50.886 | 131.193 |
| 3 | kNN-SAA | 6.477 | 0.06477 | 53.54 | 138.793 |
| 4 | SAA | 15.872 | 0.15872 | 129.216 | 131.442 |

## Old Implementation Agreement

| Method | New Gap % | Old Decision Gap % | Old Native Gap % | Native-vs-NewEval Delta pp | Roughly Agrees |
|---|---:|---:|---:|---:|:---:|
| SAA | 15.872 | 15.872 | 15.872 | 0.0 | true |
| Least Squares | 5.942 | 5.942 | 5.942 | 0.0 | true |
| Residual SAA | 3.758 | 3.758 | 3.758 | 0.0 | true |
| kNN-SAA | 6.477 | 6.477 | 6.477 | 0.0 | true |

Overall agreement check: true.

## Artifacts

- `artifacts/source_cache/seed_1/test_dataset.jls` copied from `/Users/jonathanhornewall/Projects/Ph.D_Projects/ProblemBasedScenarioGeneration/test_value_stochastic_solution_resource_allocation/p_sweep/p_1p0/seed_1/test_dataset.jls`.
- `artifacts/source_cache/seed_1/optimal_results.jls` copied from `/Users/jonathanhornewall/Projects/Ph.D_Projects/ProblemBasedScenarioGeneration/test_value_stochastic_solution_resource_allocation/p_sweep/p_1p0/seed_1/optimal_results.jls`.
- `artifacts/source_cache/seed_1/summary.csv` copied from `/Users/jonathanhornewall/Projects/Ph.D_Projects/ProblemBasedScenarioGeneration/test_value_stochastic_solution_resource_allocation/p_sweep/p_1p0/seed_1/summary.csv`.
- `artifacts/source_cache/seed_1/per_context.csv` copied from `/Users/jonathanhornewall/Projects/Ph.D_Projects/ProblemBasedScenarioGeneration/test_value_stochastic_solution_resource_allocation/p_sweep/p_1p0/seed_1/per_context.csv`.
- `artifacts/source_cache/seed_2/test_dataset.jls` copied from `/Users/jonathanhornewall/Projects/Ph.D_Projects/ProblemBasedScenarioGeneration/test_value_stochastic_solution_resource_allocation/p_sweep/p_1p0/seed_2/test_dataset.jls`.
- `artifacts/source_cache/seed_2/optimal_results.jls` copied from `/Users/jonathanhornewall/Projects/Ph.D_Projects/ProblemBasedScenarioGeneration/test_value_stochastic_solution_resource_allocation/p_sweep/p_1p0/seed_2/optimal_results.jls`.
- `artifacts/source_cache/seed_2/summary.csv` copied from `/Users/jonathanhornewall/Projects/Ph.D_Projects/ProblemBasedScenarioGeneration/test_value_stochastic_solution_resource_allocation/p_sweep/p_1p0/seed_2/summary.csv`.
- `artifacts/source_cache/seed_2/per_context.csv` copied from `/Users/jonathanhornewall/Projects/Ph.D_Projects/ProblemBasedScenarioGeneration/test_value_stochastic_solution_resource_allocation/p_sweep/p_1p0/seed_2/per_context.csv`.
- `artifacts/source_cache/seed_3/test_dataset.jls` copied from `/Users/jonathanhornewall/Projects/Ph.D_Projects/ProblemBasedScenarioGeneration/test_value_stochastic_solution_resource_allocation/p_sweep/p_1p0/seed_3/test_dataset.jls`.
- `artifacts/source_cache/seed_3/optimal_results.jls` copied from `/Users/jonathanhornewall/Projects/Ph.D_Projects/ProblemBasedScenarioGeneration/test_value_stochastic_solution_resource_allocation/p_sweep/p_1p0/seed_3/optimal_results.jls`.
- `artifacts/source_cache/seed_3/summary.csv` copied from `/Users/jonathanhornewall/Projects/Ph.D_Projects/ProblemBasedScenarioGeneration/test_value_stochastic_solution_resource_allocation/p_sweep/p_1p0/seed_3/summary.csv`.
- `artifacts/source_cache/seed_3/per_context.csv` copied from `/Users/jonathanhornewall/Projects/Ph.D_Projects/ProblemBasedScenarioGeneration/test_value_stochastic_solution_resource_allocation/p_sweep/p_1p0/seed_3/per_context.csv`.
- `results/summary.csv`
- `results/per_sample.csv`
- `results/decisions.csv`
- `results/comparison.csv`
