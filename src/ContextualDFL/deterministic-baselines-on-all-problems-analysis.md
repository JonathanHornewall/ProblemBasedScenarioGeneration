# Deterministic Baselines on All Problems: Run Analysis

Date analyzed: 2026-05-06

This note summarizes the completed deterministic baseline benchmark run across all
problem instances currently wired into `ContextualDFLExperiments`.

The proper run was executed on `gcp-16c-4` from the synced project checkout, with
no Julia execution on the local machine. The aggregate result file is:

`~/ProblemBasedScenarioGeneration/src/ContextualDFL/ContextualDFLExperiments/experiments/baseline_benchmarks/results/proper_direct_20260506_145010/baseline_results_all.csv`

The smoke run also completed with 28 rows and zero non-OK statuses.

## Executive Summary

All 28 proper benchmark rows completed with status `ok`: 7 problem variants times
4 deterministic policies.

ER-SAA is the strongest baseline on this run. Excluding the degenerate
`transshipment_q` case, ER-SAA has the best absolute regret on 5 of 6
discriminating problems and the best relative regret on 4 of 6. KNN wins
`random_yield`, and SAA is effectively tied with ER-SAA on
`unreliable_newsvendor` depending on whether absolute or relative regret is used.

`transshipment_q` should not be used to rank the policies in this configuration:
all four policies obtain the same objective value as the optimum to numerical
tolerance.

The resource-allocation run used the precomputed tiny experiment artifacts
rather than recomputing optima. Those artifacts have 30 contexts, 1000 scenarios
per context, and 1 evaluation batch. This matches the scenario count per context
used by the proper baseline setup, but it intentionally differs from the 20-batch
protocol used by the generated-optimum problems.

## Run Protocol

| Benchmark | Train contexts | Train scenarios/context | Test contexts | Test scenarios/context | Eval batches | Optimum source | Optimum/cache seconds |
|---|---:|---:|---:|---:|---:|---|---:|
| `resource_allocation` | 100 | 1 | 30 | 1000 | 1 | precomputed artifacts | 1.32 |
| `shipment_planning` | 100 | 1 | 30 | 1000 | 20 | generated during run | 29.44 |
| `transshipment_q` | 100 | 1 | 30 | 1000 | 20 | generated during run | 38.74 |
| `transshipment_h` | 100 | 1 | 30 | 1000 | 20 | generated during run | 43.73 |
| `transshipment_h_and_q` | 100 | 1 | 30 | 1000 | 20 | generated during run | 43.39 |
| `random_yield` | 100 | 1 | 30 | 1000 | 20 | generated during run | 17.19 |
| `unreliable_newsvendor` | 100 | 1 | 30 | 1000 | 20 | generated during run | 9.66 |

Resource-allocation artifact load was confirmed in the run log:

`resource_allocation artifacts loaded: 30 contexts, 1000 scenarios/context, 1 evaluation batch(es)`

## Relative Regret Results

Lower is better. Values are percentages.

| Benchmark | SAA | KNN | Least squares | ER-SAA | Best by relative regret |
|---|---:|---:|---:|---:|---|
| `resource_allocation` | 32.513 | 10.739 | 11.110 | 8.284 | ER-SAA |
| `shipment_planning` | 10.734 | 2.046 | 7.664 | 0.296 | ER-SAA |
| `transshipment_q` | ~0.000 | ~0.000 | ~0.000 | ~0.000 | tied / non-discriminating |
| `transshipment_h` | 5.050 | 2.014 | 1.640 | 0.709 | ER-SAA |
| `transshipment_h_and_q` | 5.136 | 1.987 | 2.516 | 0.832 | ER-SAA |
| `random_yield` | 0.700 | 0.448 | 2.257 | 0.898 | KNN |
| `unreliable_newsvendor` | 8.336 | 37.915 | 67.476 | 8.356 | SAA, narrowly |

## Absolute Regret Results

Lower is better. These values are objective-scale dependent and should not be
averaged across problems.

| Benchmark | SAA | KNN | Least squares | ER-SAA | Best by absolute regret |
|---|---:|---:|---:|---:|---|
| `resource_allocation` | 322.866 | 122.493 | 112.771 | 85.725 | ER-SAA |
| `shipment_planning` | 736.790 | 144.174 | 539.222 | 21.127 | ER-SAA |
| `transshipment_q` | ~0.000 | ~0.000 | ~0.000 | ~0.000 | tied / non-discriminating |
| `transshipment_h` | 138.115 | 55.032 | 45.946 | 20.244 | ER-SAA |
| `transshipment_h_and_q` | 140.435 | 55.094 | 69.913 | 23.027 | ER-SAA |
| `random_yield` | 0.0546 | 0.0349 | 0.1766 | 0.0700 | KNN |
| `unreliable_newsvendor` | 0.0405 | 0.1863 | 0.3286 | 0.0404 | ER-SAA, narrowly |

## Aggregate Policy Statistics

Including `transshipment_q`:

| Policy | Mean relative regret % | Median relative regret % | Max relative regret % | Mean fit seconds | Mean eval seconds | Absolute-regret wins | Relative-regret wins |
|---|---:|---:|---:|---:|---:|---:|---:|
| SAA | 8.924 | 5.136 | 32.513 | 1.610 | 75.641 | 1 | 2 |
| KNN | 7.878 | 2.014 | 37.915 | 0.030 | 73.715 | 1 | 1 |
| Least squares | 13.238 | 2.516 | 67.476 | 1.429 | 73.034 | 0 | 0 |
| ER-SAA | 2.768 | 0.832 | 8.356 | 0.015 | 80.396 | 5 | 4 |

Excluding `transshipment_q`:

| Policy | Mean relative regret % | Median relative regret % | Max relative regret % | Mean fit seconds | Mean eval seconds | Absolute-regret wins | Relative-regret wins |
|---|---:|---:|---:|---:|---:|---:|---:|
| SAA | 10.411 | 6.736 | 32.513 | 1.781 | 77.378 | 0 | 1 |
| KNN | 9.191 | 2.030 | 37.915 | 0.030 | 75.409 | 1 | 1 |
| Least squares | 15.444 | 5.090 | 67.476 | 1.432 | 74.700 | 0 | 0 |
| ER-SAA | 3.229 | 0.865 | 8.356 | 0.015 | 82.808 | 5 | 4 |

## Detailed Per-Policy Metrics

`gap stderr` is the mean reported standard error column from the evaluation
output. For resource allocation it is zero because the artifact protocol has one
evaluation batch.

| Benchmark | Policy | Policy value mean | Optimum value mean | Regret mean | Relative regret % | Gap stderr | Fit seconds | Eval seconds |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `resource_allocation` | SAA | 1218.237 | 895.371 | 322.866 | 32.513 | 0.000 | 8.320 | 206.690 |
| `resource_allocation` | KNN | 1017.863 | 895.371 | 122.493 | 10.739 | 0.000 | 0.030 | 206.139 |
| `resource_allocation` | Least squares | 1008.141 | 895.371 | 112.771 | 11.110 | 0.000 | 1.368 | 202.088 |
| `resource_allocation` | ER-SAA | 981.095 | 895.371 | 85.725 | 8.284 | 0.000 | 0.012 | 240.076 |
| `shipment_planning` | SAA | 7764.231 | 7027.441 | 736.790 | 10.734 | 7.570 | 0.391 | 48.686 |
| `shipment_planning` | KNN | 7171.615 | 7027.441 | 144.174 | 2.046 | 5.962 | 0.030 | 46.646 |
| `shipment_planning` | Least squares | 7566.663 | 7027.441 | 539.222 | 7.664 | 27.628 | 1.393 | 47.042 |
| `shipment_planning` | ER-SAA | 7048.568 | 7027.441 | 21.127 | 0.296 | 4.161 | 0.014 | 48.969 |
| `transshipment_q` | SAA | 2280.000 | 2280.000 | ~0.000 | ~0.000 | ~0.000 | 0.586 | 65.222 |
| `transshipment_q` | KNN | 2280.000 | 2280.000 | ~0.000 | ~0.000 | ~0.000 | 0.031 | 63.554 |
| `transshipment_q` | Least squares | 2280.000 | 2280.000 | ~0.000 | ~0.000 | ~0.000 | 1.406 | 63.038 |
| `transshipment_q` | ER-SAA | 2280.000 | 2280.000 | ~0.000 | ~0.000 | ~0.000 | 0.013 | 65.924 |
| `transshipment_h` | SAA | 2873.389 | 2735.273 | 138.115 | 5.050 | 3.376 | 0.580 | 65.196 |
| `transshipment_h` | KNN | 2790.305 | 2735.273 | 55.032 | 2.014 | 2.538 | 0.031 | 62.981 |
| `transshipment_h` | Least squares | 2781.219 | 2735.273 | 45.946 | 1.640 | 2.509 | 1.408 | 62.849 |
| `transshipment_h` | ER-SAA | 2755.517 | 2735.273 | 20.244 | 0.709 | 1.568 | 0.013 | 66.856 |
| `transshipment_h_and_q` | SAA | 2877.980 | 2737.545 | 140.435 | 5.136 | 3.941 | 0.576 | 65.131 |
| `transshipment_h_and_q` | KNN | 2792.639 | 2737.545 | 55.094 | 1.987 | 2.780 | 0.030 | 62.425 |
| `transshipment_h_and_q` | Least squares | 2807.459 | 2737.545 | 69.913 | 2.516 | 3.552 | 1.581 | 62.228 |
| `transshipment_h_and_q` | ER-SAA | 2760.572 | 2737.545 | 23.027 | 0.832 | 1.857 | 0.022 | 66.629 |
| `random_yield` | SAA | 7.833 | 7.778 | 0.0546 | 0.700 | 0.0045 | 0.373 | 40.740 |
| `random_yield` | KNN | 7.813 | 7.778 | 0.0349 | 0.448 | 0.0040 | 0.032 | 38.764 |
| `random_yield` | Least squares | 7.954 | 7.778 | 0.1766 | 2.257 | 0.0038 | 1.454 | 38.946 |
| `random_yield` | ER-SAA | 7.848 | 7.778 | 0.0700 | 0.898 | 0.0037 | 0.017 | 39.268 |
| `unreliable_newsvendor` | SAA | -0.4501 | -0.4906 | 0.0405 | 8.336 | 0.0115 | 0.445 | 37.826 |
| `unreliable_newsvendor` | KNN | -0.3043 | -0.4906 | 0.1863 | 37.915 | 0.0224 | 0.029 | 35.499 |
| `unreliable_newsvendor` | Least squares | -0.1620 | -0.4906 | 0.3286 | 67.476 | 0.0335 | 1.390 | 35.047 |
| `unreliable_newsvendor` | ER-SAA | -0.4502 | -0.4906 | 0.0404 | 8.356 | 0.0105 | 0.013 | 35.048 |

## Critical Interpretation

ER-SAA is the best default deterministic baseline in this run. It substantially
improves over SAA on resource allocation, shipment planning, and both
transshipment variants involving `h`. The shipment-planning result is especially
large: ER-SAA's relative regret is 0.296%, compared with 2.046% for KNN and
10.734% for SAA.

KNN is still important. It wins `random_yield` and is the runner-up on several
other problems. Its fit cost is negligible, and its evaluation time is generally
slightly lower than ER-SAA. If a cheap, robust baseline is needed, KNN is a
reasonable second baseline to keep.

Least squares is inconsistent. It is competitive on `transshipment_h`, but it is
weak on `shipment_planning`, `random_yield`, and especially
`unreliable_newsvendor`. This is consistent with a linear certainty-equivalent
model being brittle when the scenario-to-decision mapping is nonlinear,
piecewise, or sensitive to distributional tails rather than only conditional
means.

SAA is a useful lower bar but not a strong benchmark for most of these
contextual problems. Its poor resource-allocation result is expected because the
training protocol uses 100 contexts with only one scenario per context; ignoring
context loses information. The one exception is `unreliable_newsvendor`, where
SAA and ER-SAA are effectively tied. ER-SAA has slightly lower absolute regret,
while SAA has slightly lower mean relative regret.

`transshipment_q` is not informative under this setup. All policies return
objective value 2280.0 against optimum 2280.0, with numerical-regret artifacts
around `-8.6e-13`. This likely means the tested `q` variation does not affect the
first-stage decision in a way these contexts expose, or the deterministic
program has a trivial optimum for this instance. It should be reported for
coverage, but excluded from aggregate conclusions about policy quality.

Resource-allocation comparability needs care. The benchmark successfully uses
the precomputed artifacts, which avoids expensive optimum recomputation. However,
those artifacts represent one evaluation batch, while the other proper
benchmarks use 20 batches. This makes the resource-allocation `gap_stderr_mean`
column structurally zero and makes its uncertainty estimate not comparable with
the other problems. The artifact payload also does not carry explicit
`optimality_mu` and `optimality_rho` metadata, so compatibility is inferred from
the deterministic baseline setup and the current zero-risk/default solver
assumptions rather than proven from serialized metadata alone.

Runtime is dominated by evaluation solves, not policy fitting. Across all seven
problems, total evaluation time was roughly 511 to 563 seconds per policy family.
Fit time is small except for least squares, which is around 1.4 seconds per
problem, and SAA on resource allocation, which took 8.32 seconds. ER-SAA's
stronger statistical performance costs some evaluation time: its mean evaluation
time excluding `transshipment_q` is 82.8 seconds versus 75.4 seconds for KNN.

Finally, this is a single-seed result. The rankings are strong enough to justify
using ER-SAA as the main deterministic baseline in subsequent comparisons, but
not enough to make claims about variance across random training/test draws.
Repeating the proper profile across several seeds would be the next step if the
goal is publishable performance claims rather than a project benchmark sweep.
