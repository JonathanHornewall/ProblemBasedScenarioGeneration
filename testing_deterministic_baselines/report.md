# Deterministic Baseline Validation

Generated at: 2026-05-05T15:05:08.301Z

## Data

- Training: 100 context-scenario pairs, seed 202615051.
- Test: 30 cached contexts with 100 scenarios per context.
- Optimality results: 30 cached rows reused from `temp_experiments/dfl_suite/artifacts/test_cache`.
- Local artifact digest: training dataset sha1:c60f99cdc98c01bc7b688f73703abde1c9897daf.

## Ranking

| Rank | Method | Gap % | Mean Relative Regret | Mean Regret | Policy Eval Seconds |
|---:|---|---:|---:|---:|---:|
| 1 | kNN-SAA | 11.159 | 0.11159 | 96.281 | 19.8 |
| 2 | Residual SAA | 13.295 | 0.13295 | 110.195 | 36.09 |
| 3 | Least Squares | 17.038 | 0.17038 | 130.802 | 13.408 |
| 4 | SAA | 41.748 | 0.41748 | 319.217 | 14.885 |

## Validation

- Training dataset length and per-context scenario count validated.
- Test dataset length, per-context scenario count, and optimal-result count validated.
- All baseline decisions are finite 20-dimensional first-stage vectors.
- SAA context-independence check: true.

## Artifacts

- `artifacts/test_dataset.jls` copied from `/Users/jonathanhornewall/Projects/Ph.D_Projects/ProblemBasedScenarioGeneration/temp_experiments/dfl_suite/artifacts/test_cache/test_dataset.jls`.
- `artifacts/test_optimal_results.jls` copied from `/Users/jonathanhornewall/Projects/Ph.D_Projects/ProblemBasedScenarioGeneration/temp_experiments/dfl_suite/artifacts/test_cache/test_optimal_results.jls`.
- `artifacts/source_test_cache_metadata.csv` copied from `/Users/jonathanhornewall/Projects/Ph.D_Projects/ProblemBasedScenarioGeneration/temp_experiments/dfl_suite/artifacts/test_cache/metadata.csv`.
- `results/summary.csv`
- `results/per_sample.csv`
- `results/decisions.csv`
