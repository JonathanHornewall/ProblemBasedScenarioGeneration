# Tabular Experiment Evidence

Regenerated from `analysis_dfl_tiny_v3.md` on 2026-05-07. The main metric is relative regret, reported as `100 * relative_regret_mean`; lower is better.

## Source Inventory

| Source id | Source file or directory | Context used in this report |
|---|---|---|
| V3 analysis | `analysis_dfl_tiny_v3.md` | Primary qualitative and tabular source for the q/h DflScenLoss comparison, accepted baseline comparison, rho-DFL sweep interpretation, and seed-level q/h rows. |
| Accepted baselines | `ContextualDFLExperiments/experiments/baseline_benchmarks/results/evalbatch1_seed20260505_20260507_all_baselines/performance_analysis_30x_x1_1000.md` | Accepted 7-benchmark baseline analysis with 63/63 successful rows, 7 benchmarks x 9 policies. |
| Accepted baseline CSV | `ContextualDFLExperiments/experiments/baseline_benchmarks/results/evalbatch1_seed20260505_20260507_all_baselines/baseline_results_latest.csv` | Raw policy values, regrets, relative regrets, fit seconds, evaluation seconds, and metadata for the accepted baseline aggregate. |
| q/h DflScenLoss run | `/home/rwl/ProblemBasedScenarioGeneration/experiments/temp/2026-05-07_dflscenloss_tiny_qh_gcp16c4/results_final_eval_once` | Cited by the V3 analysis as the source for q/h DflScenLoss rows; V3 provides the aggregate and seed-level rows used here. |
| rho-DFL analysis | `analysis_dfl_rho_benchmark_30x_5x_100.md` | Backing analysis for the rho-regularized DFL sweep and its validation checks. |
| rho-DFL aggregate | `ContextualDFLExperiments/experiments/baseline_benchmarks/results/tiny_30ctx_5x100_dfl_rho_asap_20260507/dfl_rho_aggregate_latest.csv` | Raw 63-row rho-DFL aggregate over 7 problems x 3 rho settings x 3 replicas. |
| rho-DFL summary | `ContextualDFLExperiments/experiments/baseline_benchmarks/results/tiny_30ctx_5x100_dfl_rho_asap_20260507/dfl_rho_summary_latest.csv` | Problem/policy means, standard deviations, fit seconds, and evaluation seconds for rho-DFL. |
| SPO+ transshipment-q | `ContextualDFLExperiments/experiments/baseline_benchmarks/results/transshipment_q_spoplus_vs_dfl_nn_20260507/spoplus_transshipment_q_nn_results.csv` | Focused one-seed comparison of `spo_plus` and `dfl_scen` on `transshipment_q`, with history files in the same directory. |
| Large-eval logs | `ContextualDFLExperiments/experiments/baseline_benchmarks/results/realworld_30ctx_20x1000_20260507_logs/**/baseline_results_latest.csv` | Partial single-seed baseline sanity checks with 30 test contexts, 20 evaluation batches, and 20,000 test scenarios per context. |
| Problem implementations | `ContextualDFLExperiments/src/implementations/**` and `ContextualDFL/src/implementations/TransShipmentProblem/TransShipmentProblem.jl` | Problem dimensions, random component descriptions, and decoder support. |

## Protocols

Source: V3 analysis, accepted baseline analysis, rho-DFL analysis, and corresponding CSV metadata.

| Result family | Problems | Seeds or replicas | Train contexts | Train scenarios/context | Test contexts | Test scenarios/context | Eval batches | Scenarios/batch | Notes |
|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| Accepted baselines | 7 | seed `20260505` | 100 | 1 | 30 | 1000 | 1 | 1000 | 63/63 rows `status=ok`; used as accepted classical baseline aggregate. |
| q/h DflScenLoss | 4 | `20260505`, `20260506`, `20260507` | 100 | 1 | 30 | 500 | 5 | 100 | V3 reports this under the common `30 x 5 x 100` convention. |
| rho-DFL | 7 | replicas `20260505`, `20260506`, `20260507` | 100 | 1 | 30 | 500 | 5 | 100 | 63/63 rows `status=ok`; training/inference used `mu=0` and selected `rho`; evaluation used `mu_eval=0`, `rho_eval=0`. |
| SPO+ transshipment-q | 1 | seed `20260505` | not summarized in V3 | 1 generated scenario | 30 | from result artifact | not separately reported | not separately reported | Focused sanity check; not part of V3 aggregate tables. |
| Large-eval logs | 4 | seed `20260505` | 100 | 1 | 30 | 20000 | 20 | 1000 | Partial baseline-only logs; useful for sanity checking the tiny accepted baseline ordering. |

## Benchmark Setup

Sources: problem implementation files listed in the source inventory; dimensions are also summarized in the earlier experiment instructions and validated by tests where noted.

| Benchmark | n1 | n2 | m2 | Random component | Learned component(s) reported here | Train contexts | Test contexts | Test scenarios/context | Notes |
|---|---:|---:|---:|---|---|---:|---:|---:|---|
| `resource_allocation` | 20 | 680 | 50 | `h` | `dfl_h`, `dfl_q`, rho-DFL | 100 | 30 | 500 or 1000 | Service-rate matrix is 20 x 30; recourse variables are `30 + 20*30 + 20 + 30 = 680`; recourse rows are `20 + 30 = 50`. |
| `shipment_planning` | 4 | 68 | 16 | `h` | rho-DFL | 100 | 30 | 500 or 1000 | Default shipment cost matrix has 4 warehouses and 12 demand nodes; recourse variables are `4 + 4*12 + 12 + 4 = 68`; equality rows are `12 + 4 = 16`. |
| `transshipment_q` | 7 | 77 | 35 | `q` | rho-DFL; SPO+ focused run | 100 | 30 | 500 or 1000 | Fixed feasible-set objective-vector setting; appropriate for SPO+. |
| `transshipment_h` | 7 | 77 | 35 | `h` | `dfl_h`, `dfl_q`, rho-DFL | 100 | 30 | 500 or 1000 | Transshipment wrapper supports `:h_only`, `:q_only`, and `:h_and_q`. |
| `transshipment_h_and_q` | 7 | 77 | 35 | `h,q` | `dfl_h`, `dfl_q`, rho-DFL | 100 | 30 | 500 or 1000 | Mixed component stress test for learned-component choice. |
| `random_yield` | 5 | 20 | 5 | `W` | `dfl_q`, `dfl_h`, rho-DFL | 100 | 30 | 500 or 1000 | Synthetic random-recourse-matrix stress test; small instance uses `r=5`, `a=10`, `K_support=5`, so `n2 = a + 2r = 20`. |
| `unreliable_newsvendor` | 1 | 3 | 2 | reliability/demand | rho-DFL | 100 | 30 | 500 or 1000 | Auxiliary/smoke-style benchmark; not recommended as a main benchmark centerpiece. |

## Accepted Classical Baseline Panel

Source: `performance_analysis_30x_x1_1000.md` and `baseline_results_latest.csv` in `evalbatch1_seed20260505_20260507_all_baselines`. Protocol: 30 test contexts, 1000 test scenarios/context, one evaluation batch, seed `20260505`.

| Benchmark | SAA | ER-SAA | kNN-SAA | OLS-CE | CART-CE | NN | AD | AD-tree | M5-AD | Best |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `random_yield` | 23.62% | 46.81% | 3.77% | 29.14% | 30.64% | 26.07% | 26.72% | 17.48% | 26.71% | `knn` |
| `resource_allocation` | 23.54% | 6.27% | 5.66% | 8.52% | 13.36% | 11.73% | 6.17% | 11.38% | 5.92% | `knn` |
| `shipment_planning` | 10.61% | 0.19% | 1.94% | 7.55% | 30.42% | 6.90% | 0.32% | 7.40% | 1.40% | `er_saa` |
| `transshipment_h` | 15.48% | 4.34% | 0.81% | 6.44% | 12.57% | 2.61% | 4.44% | 7.78% | 3.21% | `knn` |
| `transshipment_h_and_q` | 13.22% | 4.08% | 0.56% | 6.02% | 10.77% | 6.99% | 3.53% | 7.96% | 3.38% | `knn` |
| `transshipment_q` | 0.00% | 0.03% | 0.00% | 0.32% | 0.00% | 0.00% | 0.03% | 0.00% | 0.00% | tied near-zero |
| `unreliable_newsvendor` | 21.41% | 3.25% | 1.59% | 17.63% | 30.69% | 8.84% | 1.18% | 6.65% | 2.08% | `ad` |

## Accepted Baseline Winner Detail

Source: `ContextualDFLExperiments/experiments/baseline_benchmarks/results/evalbatch1_seed20260505_20260507_all_baselines/baseline_results_latest.csv`. This table keeps the requested cost, regret, uncertainty, fit-time, and evaluation-time fields for the best accepted baseline on each benchmark.

| Benchmark | Best policy | Test cost | Oracle value | Regret | Rel. regret | Gap stderr | Fit sec | Eval sec |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `random_yield` | `knn` | 14.6090 | 14.0812 | 0.5277 | 3.77% | 0.0000 | 0.95 | 360.69 |
| `resource_allocation` | `knn` | 894.2299 | 841.4969 | 52.7329 | 5.66% | 0.0000 | 0.77 | 656.79 |
| `shipment_planning` | `er_saa` | 7048.5685 | 7034.9015 | 13.6670 | 0.19% | 0.0000 | 2.02 | 352.99 |
| `transshipment_h` | `knn` | 3083.9507 | 3059.7964 | 24.1543 | 0.81% | 0.0000 | 1.03 | 378.01 |
| `transshipment_h_and_q` | `knn` | 3085.2351 | 3068.0130 | 17.2221 | 0.56% | 0.0000 | 1.14 | 379.45 |
| `transshipment_q` | `saa` | 2280.0000 | 2280.0000 | 0.0000 | 0.00% | 0.0000 | 17.67 | 365.22 |
| `unreliable_newsvendor` | `ad` | -1.6639 | -1.6830 | 0.0191 | 1.18% | 0.0000 | 596.78 | 166.73 |

## Main Overlap Table With DFL

Sources: V3 analysis, accepted baseline aggregate, q/h DflScenLoss run cited by V3, and `dfl_rho_aggregate_latest.csv`. Baseline columns come from the accepted `30 x 1 x 1000` aggregate; V3 treats the comparison under the common `30 x 5 x 100` reporting convention. DFL rows use 30 test contexts x 5 evaluation batches x 100 scenarios per batch and three seeds/replicas. This table is limited to the four problems where q/h DflScenLoss, rho-DFL, and accepted baselines overlap.

| Benchmark | SAA | ER-SAA | kNN-SAA | OLS-CE | CART-CE | NN | AD | AD-tree | M5-AD | DFL-q | DFL-h | rho=0.001 | rho=0.01 | rho=0.1 | Best |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `random_yield` | 23.62% | 46.81% | 3.77% | 29.14% | 30.64% | 26.07% | 26.72% | 17.48% | 26.71% | 7.08% | 8.40% | 26.91% | 27.18% | 25.90% | `knn` |
| `resource_allocation` | 23.54% | 6.27% | 5.66% | 8.52% | 13.36% | 11.73% | 6.17% | 11.38% | 5.92% | 24.31% | 8.07% | 18.13% | 24.21% | 48.18% | `knn` |
| `transshipment_h` | 15.48% | 4.34% | 0.81% | 6.44% | 12.57% | 2.61% | 4.44% | 7.78% | 3.21% | 30.58% | 2.58% | 2.77% | 29.20% | 70.25% | `knn` |
| `transshipment_h_and_q` | 13.22% | 4.08% | 0.56% | 6.02% | 10.77% | 6.99% | 3.53% | 7.96% | 3.38% | 26.95% | 2.06% | 2.86% | 20.74% | 72.48% | `knn` |

## Overall Ranking on Four Overlap Problems

Source: V3 analysis. Ranking uses `random_yield`, `resource_allocation`, `transshipment_h`, and `transshipment_h_and_q`.

| Policy | Mean relative regret | Median relative regret | Max relative regret | Average rank |
|---|---:|---:|---:|---:|
| `knn` | 2.70% | 2.29% | 5.66% | 1.00 |
| `dfl_h` | 5.28% | 5.33% | 8.40% | 3.00 |
| `m5_ad` | 9.81% | 4.65% | 26.71% | 4.75 |
| `ad` | 10.22% | 5.30% | 26.72% | 6.00 |
| `ad_tree` | 11.15% | 9.67% | 17.48% | 7.25 |
| `nn` | 11.85% | 9.36% | 26.07% | 6.50 |
| `least_squares` | 12.53% | 7.48% | 29.14% | 8.25 |
| `dfl_mu0_rho0.001` | 12.67% | 10.50% | 26.91% | 6.75 |
| `er_saa` | 15.38% | 5.31% | 46.81% | 7.50 |
| `cart` | 16.84% | 12.97% | 30.64% | 10.50 |
| `saa` | 18.97% | 19.51% | 23.62% | 9.50 |
| `dfl_q` | 22.23% | 25.63% | 30.58% | 10.25 |
| `dfl_mu0_rho0.01` | 25.33% | 25.69% | 29.20% | 11.75 |
| `dfl_mu0_rho0.1` | 54.20% | 59.22% | 72.48% | 12.00 |

## Best DFL Configuration by Problem

Source: V3 analysis. This compares the DFL family only: q/h DflScenLoss and rho-DFL on the overlapping four problems.

| Benchmark | Best DFL family member | Relative regret | Second-best DFL family member | Relative regret | Interpretation |
|---|---|---:|---|---:|---|
| `random_yield` | `dfl_q` | 7.08% | `dfl_h` | 8.40% | q/h DflScenLoss dominates rho-DFL. |
| `resource_allocation` | `dfl_h` | 8.07% | `dfl_mu0_rho0.001` | 18.13% | rho-DFL is not close. |
| `transshipment_h` | `dfl_h` | 2.58% | `dfl_mu0_rho0.001` | 2.77% | rho-DFL is close but still second. |
| `transshipment_h_and_q` | `dfl_h` | 2.06% | `dfl_mu0_rho0.001` | 2.86% | rho-DFL is useful but weaker. |

## DflScenLoss Decoder Ablation

Source: V3 analysis; q/h DflScenLoss rows are means over seeds `20260505`, `20260506`, and `20260507`.

| Benchmark | Better DFL decoder | DFL-q relative regret | DFL-h relative regret | Difference |
|---|---|---:|---:|---|
| `random_yield` | `q` | 7.08% | 8.40% | `q` better by 1.32 pp |
| `resource_allocation` | `h` | 24.31% | 8.07% | `h` better by 16.24 pp |
| `transshipment_h` | `h` | 30.58% | 2.58% | `h` better by 27.99 pp |
| `transshipment_h_and_q` | `h` | 26.95% | 2.06% | `h` better by 24.88 pp |

## rho-DFL Ablation

Sources: V3 analysis, `analysis_dfl_rho_benchmark_30x_5x_100.md`, `dfl_rho_aggregate_latest.csv`, and `dfl_rho_summary_latest.csv`. Each cell gives relative regret, with raw regret mean +/- standard deviation in parentheses.

| Problem | rho=0.001 | rho=0.01 | rho=0.1 | Best rho |
|---|---:|---:|---:|---|
| `resource_allocation` | 18.13% (154.595 +/- 5.715) | 24.21% (204.038 +/- 3.287) | 48.18% (420.934 +/- 23.961) | `0.001` |
| `shipment_planning` | 1.01% (72.390 +/- 40.517) | 0.46% (33.969 +/- 11.238) | 5.32% (374.047 +/- 269.923) | `0.01` |
| `transshipment_q` | 0.00% (0.000000038 +/- 0.000000006) | 14.42% (328.754 +/- 3.366) | 102.21% (2330.419 +/- 6.649) | `0.001` |
| `transshipment_h` | 2.77% (84.414 +/- 2.741) | 29.20% (902.369 +/- 10.636) | 70.25% (2156.599 +/- 3.941) | `0.001` |
| `transshipment_h_and_q` | 2.86% (85.777 +/- 6.447) | 20.74% (637.301 +/- 12.077) | 72.48% (2198.578 +/- 13.973) | `0.001` |
| `random_yield` | 26.91% (3.653 +/- 0.000000031) | 27.18% (3.690 +/- 0.064) | 25.90% (3.546 +/- 0.036) | `0.1` |
| `unreliable_newsvendor` | 3.55% (0.073 +/- 0.040) | 9.26% (0.195 +/- 0.257) | 9.72% (0.206 +/- 0.273) | `0.001` |

## Best rho-DFL Detail

Source: `ContextualDFLExperiments/experiments/baseline_benchmarks/results/tiny_30ctx_5x100_dfl_rho_asap_20260507/dfl_rho_aggregate_latest.csv`. Means are over three replicas. This table keeps the cost, regret, uncertainty, fit-time, and evaluation-time fields for the best rho-DFL setting on each benchmark.

| Benchmark | Best rho-DFL policy | Test cost | Oracle value | Regret | Rel. regret | Gap stderr | Fit sec | Eval sec |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `random_yield` | `dfl_mu0_rho0.1` | 17.6480 | 14.1020 | 3.5460 | 25.90% | 0.0708 | 1.14 | 267.42 |
| `resource_allocation` | `dfl_mu0_rho0.001` | 995.8342 | 841.2391 | 154.5951 | 18.13% | 0.6411 | 1.56 | 609.73 |
| `shipment_planning` | `dfl_mu0_rho0.01` | 7066.5644 | 7032.5957 | 33.9686 | 0.46% | 5.5938 | 1.50 | 342.47 |
| `transshipment_h` | `dfl_mu0_rho0.001` | 3146.1655 | 3061.7510 | 84.4144 | 2.77% | 5.3328 | 1.37 | 286.07 |
| `transshipment_h_and_q` | `dfl_mu0_rho0.001` | 3149.0674 | 3063.2905 | 85.7770 | 2.86% | 5.4831 | 1.47 | 284.92 |
| `transshipment_q` | `dfl_mu0_rho0.001` | 2280.0000 | 2280.0000 | 0.0000 | 0.00% | 0.0000 | 1.42 | 286.20 |
| `unreliable_newsvendor` | `dfl_mu0_rho0.001` | -1.6156 | -1.6885 | 0.0729 | 3.55% | 0.0090 | 1.56 | 315.94 |

## SPO+ Focused Transshipment-q Check

Source: `ContextualDFLExperiments/experiments/baseline_benchmarks/results/transshipment_q_spoplus_vs_dfl_nn_20260507/spoplus_transshipment_q_nn_results.csv`. This is a focused one-seed run, not part of the V3 aggregate; it is useful because `transshipment_q` has a fixed feasible set and objective-vector uncertainty.

| Benchmark | Method | Seed | Epochs | Scenarios | Hidden dim | Train sec | Eval sec | Final train loss | Regret | Relative regret |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `transshipment_q` | `dfl_scen` | 20260505 | 20 | 1 | 128 | 114.35 | 202.14 | 2280.0001 | 0.0000667 | 0.00000293% |
| `transshipment_q` | `spo_plus` | 20260505 | 20 | 1 | 128 | 48.14 | 221.20 | 13.4813 | 0.000000000126 | 0.00000000000551% |

## Running Learning Updates

Source: user-provided inline updates on 2026-05-07 while learning runs were still in progress. These rows are provisional and should be replaced or extended as new run outputs arrive. Relative regret is recorded exactly as provided, as a fraction rather than a percentage.

Latest compact running update:

| Problem | Policy | Epochs | Eval | Regret | Rel. regret |
|---|---|---:|---|---:|---:|
| `random_yield` | `spoplus_qconv` | 3 | 20 ctx x 20 scen | 4.2795 | 0.3180 |
| `random_yield` | `dfl_qconv` | 3 | 20 ctx x 20 scen | 10.4089 | 0.7513 |
| `resource_allocation` | `dfl_ra_physical_cost` | 1 | 5 ctx x 5 scen | 3311.7277 | 3.9465 |
| `resource_allocation` | `dfl_ra_full_cost` | 1 | 5 ctx x 5 scen | 3312.1010 | 3.9469 |
| `resource_allocation` | `dfl_ra_original_cost` | 1 | 5 ctx x 5 scen | 3308.0583 | 3.9419 |
| `resource_allocation` | `dfl_ra_economic_cost` | 1 | 5 ctx x 5 scen | 3311.7277 | 3.9465 |

Detailed rows from the prior inline update, retained where policy values, optima, and timings were supplied:

| Problem | Policy | Status | Epochs | Eval | Policy value | Optimal | Regret | Rel. regret | Fit s | Eval s |
|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| `random_yield` | `spoplus_qconv` | `ok` | 3 | 20 x 20 | 18.2679 | 13.9885 | 4.2795 | 0.3180 | 47.8 | 4.1 |
| `random_yield` | `dfl_qconv` | `ok` | 3 | 20 x 20 | 24.3974 | 13.9885 | 10.4089 | 0.7513 | 53.2 | 3.2 |
| `resource_allocation` | `dfl_ra_physical_cost` | `ok` | 1 | 5 x 5 | 4149.4941 | 837.7664 | 3311.7277 | 3.9465 | 59.8 | 3.2 |
| `resource_allocation` | `dfl_ra_full_cost` | `ok` | 1 | 5 x 5 | 4149.8674 | 837.7664 | 3312.1010 | 3.9469 | 59.5 | 3.3 |

Earlier partial qconv rows from the first inline update:

| Run | Policy | Regret | Relative regret | Fit sec | Eval sec |
|---|---|---:|---:|---:|---:|
| earlier full-ish 50 epoch seed 20260505 | `spoplus_qconv` | 4.612 | 0.334 | 135.7 | 88.2 |
| earlier full-ish 50 epoch seed 20260505 | `dfl_qconv` | 9.772 | 0.706 | 179.8 | 92.2 |

## Timing Summary

Sources: accepted baseline CSV and rho-DFL aggregate CSV. These timings are not directly comparable across all methods because the accepted baselines use a `30 x 1 x 1000` evaluation protocol while rho-DFL uses `30 x 5 x 100`. q/h DflScenLoss timing was not available in the V3 analysis table.

| Family | Policy | Rows | Mean fit sec | Mean eval sec | Mean relative regret |
|---|---|---:|---:|---:|---:|
| Accepted baselines | `knn` | 7 | 0.9 | 400.0 | 2.05% |
| Accepted baselines | `m5_ad` | 7 | 2651.9 | 236.6 | 6.10% |
| Accepted baselines | `ad` | 7 | 1823.1 | 215.7 | 6.06% |
| Accepted baselines | `er_saa` | 7 | 2.7 | 409.1 | 9.28% |
| Accepted baselines | `nn` | 7 | 678.5 | 274.3 | 9.02% |
| Accepted baselines | `ad_tree` | 7 | 1890.7 | 274.6 | 8.38% |
| Accepted baselines | `least_squares` | 7 | 2.9 | 390.7 | 10.80% |
| Accepted baselines | `saa` | 7 | 14.4 | 378.6 | 15.41% |
| Accepted baselines | `cart` | 7 | 4.2 | 287.5 | 18.35% |
| rho-DFL | `dfl_mu0_rho0.001` | 21 | 1.43 | 342.09 | 7.89% |
| rho-DFL | `dfl_mu0_rho0.01` | 21 | 1.42 | 351.71 | 17.92% |
| rho-DFL | `dfl_mu0_rho0.1` | 21 | 1.40 | 353.06 | 47.72% |

## Large-Evaluation Baseline Sanity Check

Source: `ContextualDFLExperiments/experiments/baseline_benchmarks/results/realworld_30ctx_20x1000_20260507_logs/**/baseline_results_latest.csv`. These are partial single-seed baseline logs with 30 contexts, 20,000 test scenarios/context, and 20 evaluation batches. They should not be merged with the V3 aggregate, but they are useful because the rankings broadly match the accepted tiny baseline results.

| Benchmark | SAA | ER-SAA | kNN-SAA | OLS-CE | CART-CE | NN | AD | Best available |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `random_yield` | 23.54% | 46.63% | 3.83% | 29.07% | 30.42% | 25.94% | 26.64% | `knn` |
| `transshipment_h` | 15.45% | 4.33% | 0.82% | 6.40% | 12.54% | 2.58% | 4.43% | `knn` |
| `transshipment_h_and_q` | 13.20% | 4.07% | 0.57% | 5.99% | 10.76% | 6.95% | -- | `knn` |
| `transshipment_q` | 0.00% | 0.03% | 0.00% | 0.33% | 0.00% | 0.00% | 0.03% | tied near-zero |

## q/h DflScenLoss Seed-Level Rows

Source: V3 analysis, derived from the q/h DflScenLoss run directory cited above. Lower is better.

| Problem | Output | Seed | Regret | Relative regret |
|---|---|---:|---:|---:|
| `random_yield` | `h` | 20260505 | 1.4971 | 10.30% |
| `random_yield` | `h` | 20260506 | 0.9781 | 6.82% |
| `random_yield` | `h` | 20260507 | 1.1522 | 8.08% |
| `random_yield` | `q` | 20260505 | 1.0289 | 7.43% |
| `random_yield` | `q` | 20260506 | 0.9077 | 6.68% |
| `random_yield` | `q` | 20260507 | 0.9841 | 7.15% |
| `resource_allocation` | `h` | 20260505 | 77.8981 | 9.18% |
| `resource_allocation` | `h` | 20260506 | 69.7575 | 8.25% |
| `resource_allocation` | `h` | 20260507 | 56.6957 | 6.78% |
| `resource_allocation` | `q` | 20260505 | 64.9643 | 7.49% |
| `resource_allocation` | `q` | 20260506 | 173.0531 | 20.16% |
| `resource_allocation` | `q` | 20260507 | 388.7423 | 45.28% |
| `transshipment_h` | `h` | 20260505 | 100.1445 | 3.28% |
| `transshipment_h` | `h` | 20260506 | 67.3695 | 2.23% |
| `transshipment_h` | `h` | 20260507 | 68.0943 | 2.23% |
| `transshipment_h` | `q` | 20260505 | 965.9477 | 31.26% |
| `transshipment_h` | `q` | 20260506 | 903.6260 | 29.21% |
| `transshipment_h` | `q` | 20260507 | 965.9490 | 31.26% |
| `transshipment_h_and_q` | `h` | 20260505 | 54.5839 | 1.80% |
| `transshipment_h_and_q` | `h` | 20260506 | 59.3342 | 1.96% |
| `transshipment_h_and_q` | `h` | 20260507 | 71.8003 | 2.43% |
| `transshipment_h_and_q` | `q` | 20260505 | 1055.3685 | 33.83% |
| `transshipment_h_and_q` | `q` | 20260506 | 734.1741 | 24.09% |
| `transshipment_h_and_q` | `q` | 20260507 | 698.9256 | 22.92% |

## Reporting Notes

| Point | Consequence for the paper |
|---|---|
| `knn` remains the best overall method on the four q/h DflScenLoss overlap problems. | The main table should not claim that DFL beats all classical contextual baselines; the cleaner claim is that `dfl_h` is the strongest non-kNN method on this overlap set and is especially competitive on transshipment. |
| Decoder choice is decisive. | Report `dfl_h` for resource allocation and transshipment; report `dfl_q` for random yield. Avoid presenting `dfl_q` as the default for `h`-random benchmarks. |
| rho is highly sensitive. | Use `rho=0.001` for resource allocation, transshipment, and unreliable newsvendor; use `rho=0.01` for shipment planning; do not emphasize rho-DFL on random yield. |
| q/h DflScenLoss rows are absent in V3 for `shipment_planning`, `transshipment_q`, and `unreliable_newsvendor`. | For those problems, the currently tabulated DFL evidence comes from rho-DFL; `transshipment_q` also has the focused SPO+ sanity check. |
| The large-evaluation logs are partial and single-seed. | Use them only as supporting sanity checks, not as the main performance table. |
