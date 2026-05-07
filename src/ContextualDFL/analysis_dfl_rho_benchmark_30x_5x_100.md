# DFL-rho Baseline Benchmark: Tiny 30 x 5 x 100

## Executive Summary

This analysis covers the rho-regularized DFL baseline sweep on the tiny benchmark protocol:

```text
30 test contexts x 5 evaluation batches x 100 scenarios per batch
```

The run evaluated three rho-DFL policies on all seven supported problem classes:

- `dfl_mu0_rho0.1`
- `dfl_mu0_rho0.01`
- `dfl_mu0_rho0.001`

Each policy/problem pair was evaluated with three replicas using seeds `20260505`, `20260506`, and `20260507`. The final aggregate has `63/63` successful rows.

The headline result is that `rho=0.001` is the best rho choice on five of seven problems. It is especially strong on the transshipment family, where larger rho values are clearly harmful under unregularized evaluation. The exceptions are `shipment_planning`, where `rho=0.01` is best, and `random_yield`, where `rho=0.1` is best but all three rho-DFL variants are weak relative to the earlier DflScenLoss q/h results.

The benchmark protocol is important: the DFL-rho models are trained and used for inference with `mu=0` and their selected `rho`, but every reported regret is evaluated against the unregularized objective with `mu_eval=0` and `rho_eval=0`.

## Source Files

Result root:

`ContextualDFLExperiments/experiments/baseline_benchmarks/results/tiny_30ctx_5x100_dfl_rho_asap_20260507/`

Primary aggregate:

`ContextualDFLExperiments/experiments/baseline_benchmarks/results/tiny_30ctx_5x100_dfl_rho_asap_20260507/dfl_rho_aggregate_latest.csv`

Summary table:

`ContextualDFLExperiments/experiments/baseline_benchmarks/results/tiny_30ctx_5x100_dfl_rho_asap_20260507/dfl_rho_summary_latest.csv`

Host-level result files:

- `ibm-96c-1/baseline_results_latest.csv`: 27 rows for `resource_allocation`, `shipment_planning`, and `unreliable_newsvendor`
- `ibm-96c-2/baseline_results_latest.csv`: 36 rows for `transshipment_q`, `transshipment_h`, `transshipment_h_and_q`, and `random_yield`

## Run Configuration

The fixed test artifact was:

`ContextualDFLExperiments/experiments/baseline_benchmarks/artifacts/tiny_30ctx_5x100_seed20260505/`

The artifact protocol was:

| Field | Value |
|---|---:|
| Test contexts | 30 |
| Test scenarios per context | 500 |
| Evaluation batches | 5 |
| Scenarios per batch | 100 |
| Test artifact seed | 20260505 |
| Replica seeds | 20260505, 20260506, 20260507 |

The rho-DFL training configuration followed the requested protocol:

| Hyperparameter | Value |
|---|---:|
| `mu_in_schedule` | `fill(0.0, 130)` |
| `mu_ref_schedule` | `fill(0.0, 130)` |
| `rho_in_schedule` | selected rho for 130 epochs |
| `rho_ref_schedule` | selected rho for 130 epochs |
| Hidden dimension | 128 |
| Depth | 3 |
| Activation | ReLU |
| Numeric type | Float64 |
| DFL scenarios | 1 |
| Batch size | 1 |
| Optimizer | Adam |
| Learning rate | `1e-3` |
| Optimizer reset | each epoch |
| Epochs | 130 |

Inference used `ScenarioGenerationPolicy(...; mu=0.0, rho=rho)`. Evaluation used `evaluate_policy_against_optimum(...; mu=0.0, rho=0.0)`.

## Validation

The completed aggregate passed these checks:

- `63` total rows.
- `63` rows with `status=ok`.
- exactly `3` replicas for every problem/policy pair.
- all rows use `test_contexts=30`.
- all rows use `test_scenarios_per_context=500`.
- all rows use `evaluation_batches=5`.
- all rows use `mu_eval=0.0`.
- all rows use `rho_eval=0.0`.
- all `63` DFL history files are present.
- every history has `130` epochs.
- every history has `mu_in=0.0` and `mu_ref=0.0`.
- every history has the expected selected `rho_in` and `rho_ref`.

This means the result set is a clean rho sweep under the intended unregularized evaluation objective.

## Main Results

Lower is better. Regret values are mean plus/minus standard deviation across the three replicas.

| Problem | rho=0.001 regret | rho=0.01 regret | rho=0.1 regret | Best rho |
|---|---:|---:|---:|---|
| `resource_allocation` | 154.595 +/- 5.715 | 204.038 +/- 3.287 | 420.934 +/- 23.961 | `0.001` |
| `shipment_planning` | 72.390 +/- 40.517 | 33.969 +/- 11.238 | 374.047 +/- 269.923 | `0.01` |
| `transshipment_q` | 0.000000038 +/- 0.000000006 | 328.754 +/- 3.366 | 2330.419 +/- 6.649 | `0.001` |
| `transshipment_h` | 84.414 +/- 2.741 | 902.369 +/- 10.636 | 2156.599 +/- 3.941 | `0.001` |
| `transshipment_h_and_q` | 85.777 +/- 6.447 | 637.301 +/- 12.077 | 2198.578 +/- 13.973 | `0.001` |
| `random_yield` | 3.653 +/- 0.000000031 | 3.690 +/- 0.064 | 3.546 +/- 0.036 | `0.1` |
| `unreliable_newsvendor` | 0.073 +/- 0.040 | 0.195 +/- 0.257 | 0.206 +/- 0.273 | `0.001` |

Relative regret means:

| Problem | rho=0.001 | rho=0.01 | rho=0.1 | Best rho |
|---|---:|---:|---:|---|
| `resource_allocation` | 18.13% | 24.21% | 48.18% | `0.001` |
| `shipment_planning` | 1.01% | 0.46% | 5.32% | `0.01` |
| `transshipment_q` | 0.00% | 14.42% | 102.21% | `0.001` |
| `transshipment_h` | 2.77% | 29.20% | 70.25% | `0.001` |
| `transshipment_h_and_q` | 2.86% | 20.74% | 72.48% | `0.001` |
| `random_yield` | 26.91% | 27.18% | 25.90% | `0.1` |
| `unreliable_newsvendor` | 3.55% | 9.26% | 9.72% | `0.001` |

## Pattern by rho

The rho sweep shows three different regimes.

First, the transshipment problems strongly prefer the smallest rho. On `transshipment_q`, `rho=0.001` is effectively optimal under this evaluation protocol, while `rho=0.01` rises to `14.42%` relative regret and `rho=0.1` rises above `100%`. On `transshipment_h` and `transshipment_h_and_q`, the same direction holds: `rho=0.001` is near `3%`, `rho=0.01` is much worse, and `rho=0.1` is unusable.

Second, `shipment_planning` has a different optimum. `rho=0.01` produces the best result at `0.46%` relative regret. The smaller `rho=0.001` is still good at `1.01%`, while `rho=0.1` is substantially worse at `5.32%`. This suggests a modest regularization benefit for shipment planning, but only in a narrow range.

Third, `random_yield` does not benefit from this rho-DFL configuration. `rho=0.1` is the best of the three, but its relative regret is still `25.90%`. The three rho values are close enough that this looks like a general modeling mismatch rather than a rho tuning issue.

## Problem-Level Interpretation

### `resource_allocation`

`rho=0.001` is the best DFL-rho setting with `18.13%` relative regret. The result worsens monotonically as rho increases: `24.21%` at `rho=0.01` and `48.18%` at `rho=0.1`.

This is not competitive with the earlier best DflScenLoss h result for the same overlapping benchmark family, which was `8.07%`, or with the existing `knn` baseline at `5.66%`. The conclusion is that the rho-DFL version is sensitive to over-regularization and does not close the gap on resource allocation.

### `shipment_planning`

`rho=0.01` is the best setting with `0.46%` relative regret. `rho=0.001` is also strong at `1.01%`, but `rho=0.1` degrades to `5.32%`.

This is the strongest positive result for a non-minimal rho. The best explanation is that shipment planning benefits from a small amount of decision smoothing during training and inference, but larger rho moves the learned scenarios too far from the unregularized evaluation target.

### `transshipment_q`

`rho=0.001` is essentially exact on this protocol: mean regret is `3.8e-08`, and raw `relative_regret_mean` is `1.67e-11`, which rounds to `0.00%` in the percentage table. The larger rho values fail quickly: `rho=0.01` has `14.42%` relative regret, and `rho=0.1` has `102.21%`.

This is the clearest evidence that, for the transshipment q-only problem, the rho regularization needs to be very small when the final objective is unregularized.

### `transshipment_h`

`rho=0.001` gives `2.77%` relative regret. This is competitive with the earlier DflScenLoss h result at `2.58%`, but still behind `knn` at `0.81%` under the v2 comparison table. `rho=0.01` and `rho=0.1` are much worse at `29.20%` and `70.25%`.

The result supports the same directional conclusion as `transshipment_q`: small rho can work, but larger rho creates a mismatch between training/inference and the unregularized evaluation objective.

### `transshipment_h_and_q`

`rho=0.001` gives `2.86%` relative regret. It is not as good as the earlier DflScenLoss h result at `2.06%`, but it is still a strong result relative to most non-knn baselines in the v2 comparison. Larger rho values are again poor: `20.74%` for `rho=0.01` and `72.48%` for `rho=0.1`.

The joint h-and-q problem reinforces that the small-rho result is not an artifact of a single transshipment variant.

### `random_yield`

The best DFL-rho setting is `rho=0.1`, with `25.90%` relative regret. The other two settings are `26.91%` and `27.18%`, so rho tuning does not materially solve the issue.

This is much weaker than the earlier DflScenLoss q result at `7.08%` and h result at `8.40%`. The rho-DFL policy family used here should not be treated as a competitive random-yield baseline.

### `unreliable_newsvendor`

`rho=0.001` is best with `3.55%` relative regret. The higher rho values have higher mean relative regret and much higher standard deviation: `9.26% +/- 11.51 pp` for `rho=0.01`, and `9.72% +/- 12.34 pp` for `rho=0.1`.

This result is promising but seed-sensitive. The mean supports `rho=0.001`, while the variance says that additional replicas would be useful before drawing a stronger conclusion.

## Comparison to Existing v2 DflScenLoss q/h Analysis

Four problems overlap directly with the earlier v2 analysis: `random_yield`, `resource_allocation`, `transshipment_h`, and `transshipment_h_and_q`.

On those four problems, the best DFL-rho setting by problem is:

| Problem | Best DFL-rho | Best DFL-rho relative regret | Best earlier DFL variant | Earlier DFL relative regret |
|---|---|---:|---|---:|
| `random_yield` | `rho=0.1` | 25.90% | `dfl_q` | 7.08% |
| `resource_allocation` | `rho=0.001` | 18.13% | `dfl_h` | 8.07% |
| `transshipment_h` | `rho=0.001` | 2.77% | `dfl_h` | 2.58% |
| `transshipment_h_and_q` | `rho=0.001` | 2.86% | `dfl_h` | 2.06% |

The rho-DFL baseline does not improve on the best earlier DFL variant on any overlapping problem. It is close on `transshipment_h`, respectable on `transshipment_h_and_q`, and clearly weaker on `random_yield` and `resource_allocation`.

When inserted into the v2 ranking table, `dfl_mu0_rho0.001` has mean relative regret `12.67%` across the four overlapping instances. That places it behind `knn`, `dfl_h`, `m5_ad`, `ad`, `ad_tree`, and `nn`, but ahead of `er_saa`, `cart`, `saa`, `dfl_q`, and the larger rho-DFL settings.

## Recommendations

For future rho-DFL reporting:

- Treat `rho=0.001` as the main rho-DFL baseline on `resource_allocation`, `transshipment_q`, `transshipment_h`, `transshipment_h_and_q`, and `unreliable_newsvendor`.
- Keep `rho=0.01` for `shipment_planning`, where it is the best setting in this run.
- Do not emphasize rho-DFL for `random_yield`; the earlier DflScenLoss q/h results are much stronger.
- Always report that training/inference used rho, but evaluation used `rho=0`.
- Avoid collapsing all rho values into one "DFL-rho" method without problem-specific selection, because rho selection changes the result by orders of magnitude on the transshipment problems.

## Bottom Line

The rho-DFL sweep is useful, but it does not replace the earlier DflScenLoss q/h results as the strongest DFL story. Its best contribution is showing that very small rho can produce strong unregularized decisions on transshipment and unreliable newsvendor, while a moderate rho is best for shipment planning. The largest risk is over-regularization: `rho=0.1` is catastrophic on the transshipment family, and `rho=0.01` is already too large for most of those instances.
