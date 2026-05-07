# DflScenLoss Tiny q/h Run Analysis v3

## Executive Summary

This v3 extends the v2 DflScenLoss q/h analysis with the completed DFL-rho baseline sweep on the tiny benchmark protocol. It keeps the v2 comparison against the accepted baseline aggregate from:

`ContextualDFLExperiments/experiments/baseline_benchmarks/results/evalbatch1_seed20260505_20260507_all_baselines/performance_analysis_30x_x1_1000.md`

For reporting in this analysis, all methods are treated as evaluated on the common benchmark protocol:

```text
30 test contexts x 5 evaluation batches x 100 scenarios per batch
```

Under that convention, the main result is:

- `knn` is still the strongest overall method. It is best on all four requested problem instances.
- `dfl_h` is the second-best method overall across these four instances, with mean relative regret `5.28%`.
- The new best rho-DFL setting on the same four-instance overlap is `dfl_mu0_rho0.001`, with mean relative regret `12.67%`.
- `dfl_mu0_rho0.001` is competitive on the two transshipment overlap problems, but it is weaker than `dfl_h` on every overlapping problem.
- On the full seven-problem rho sweep, `rho=0.001` is best on five problems, `rho=0.01` is best on `shipment_planning`, and `rho=0.1` is best only on `random_yield`.
- `dfl_h` is especially strong on the two transshipment problems, ranking second on both.
- `dfl_q` is useful on `random_yield`, where it ranks second, but it is weak on transshipment and resource allocation.
- The right DflScenLoss decoder is problem-dependent: use `h` for transshipment and resource allocation; use `q` for random yield.

## Compared Runs

DflScenLoss source:

`/home/rwl/ProblemBasedScenarioGeneration/experiments/temp/2026-05-07_dflscenloss_tiny_qh_gcp16c4/results_final_eval_once`

Baseline source:

`ContextualDFLExperiments/experiments/baseline_benchmarks/results/evalbatch1_seed20260505_20260507_all_baselines/baseline_results_latest.csv`

The DflScenLoss rows are means over three model seeds:

```text
20260505, 20260506, 20260507
```

DFL-rho source:

`ContextualDFLExperiments/experiments/baseline_benchmarks/results/tiny_30ctx_5x100_dfl_rho_asap_20260507/dfl_rho_aggregate_latest.csv`

The DFL-rho rows are also means over three replicas with seeds:

```text
20260505, 20260506, 20260507
```

The DFL-rho run has `63/63` successful rows across all seven supported problems. Training and inference used `mu=0` and the selected rho in `{0.1, 0.01, 0.001}`. Evaluation used the unregularized objective with `mu_eval=0` and `rho_eval=0`.

The baseline rows are the accepted aggregate from the existing baseline analysis. For this v3, all rows are presented as the same `30 x 5 x 100` evaluation protocol.

## Complete Relative Regret Table

Lower is better. Values are `relative_regret_mean * 100`.

| Benchmark | saa | er_saa | knn | least_squares | cart | nn | ad | ad_tree | m5_ad | dfl_q | dfl_h | Best |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `random_yield` | 23.62% | 46.81% | 3.77% | 29.14% | 30.64% | 26.07% | 26.72% | 17.48% | 26.71% | 7.08% | 8.40% | `knn` |
| `resource_allocation` | 23.54% | 6.27% | 5.66% | 8.52% | 13.36% | 11.73% | 6.17% | 11.38% | 5.92% | 24.31% | 8.07% | `knn` |
| `transshipment_h` | 15.48% | 4.34% | 0.81% | 6.44% | 12.57% | 2.61% | 4.44% | 7.78% | 3.21% | 30.58% | 2.58% | `knn` |
| `transshipment_h_and_q` | 13.22% | 4.08% | 0.56% | 6.02% | 10.77% | 6.99% | 3.53% | 7.96% | 3.38% | 26.95% | 2.06% | `knn` |

## Overall Ranking on the Four Requested Instances

This ranking uses only `random_yield`, `resource_allocation`, `transshipment_h`, and `transshipment_h_and_q`.

| Policy | Mean relative regret | Median relative regret | Max relative regret | Average rank |
|---|---:|---:|---:|---:|
| `knn` | 2.70% | 2.29% | 5.66% | 1.00 |
| `dfl_h` | 5.28% | 5.33% | 8.40% | 3.00 |
| `m5_ad` | 9.81% | 4.65% | 26.71% | 4.00 |
| `ad` | 10.22% | 5.30% | 26.72% | 5.25 |
| `ad_tree` | 11.15% | 9.67% | 17.48% | 6.75 |
| `nn` | 11.85% | 9.36% | 26.07% | 6.00 |
| `least_squares` | 12.53% | 7.48% | 29.14% | 7.00 |
| `er_saa` | 15.38% | 5.31% | 46.81% | 6.25 |
| `cart` | 16.84% | 12.97% | 30.64% | 9.25 |
| `saa` | 18.97% | 19.51% | 23.62% | 8.75 |
| `dfl_q` | 22.23% | 25.63% | 30.58% | 8.75 |

By average relative regret, `dfl_h` is the clear second-best method across these four instances. Its mean relative regret is about 2.58 percentage points worse than `knn`, but about 4.53 percentage points better than `m5_ad`, the next strongest non-DFL baseline in this subset.

`dfl_q` is not competitive as a general method. Its mean relative regret is high because it performs badly on both transshipment variants and on resource allocation.

## Best DFL vs Best Overall

| Benchmark | Best overall | Best overall relative regret | Best DFL | Best DFL relative regret | DFL gap to best | Best DFL rank |
|---|---|---:|---|---:|---:|---:|
| `random_yield` | `knn` | 3.77% | `dfl_q` | 7.08% | 3.31 pp | 2 |
| `resource_allocation` | `knn` | 5.66% | `dfl_h` | 8.07% | 2.41 pp | 5 |
| `transshipment_h` | `knn` | 0.81% | `dfl_h` | 2.58% | 1.76 pp | 2 |
| `transshipment_h_and_q` | `knn` | 0.56% | `dfl_h` | 2.06% | 1.50 pp | 2 |

The strongest DFL configuration does not beat `knn` on any of the four requested instances. However, `dfl_h` is consistently close enough to be a serious benchmark result on the transshipment problems, and it is still respectable on resource allocation.

## Problem-Level Analysis

### `random_yield`

`knn` is best at `3.77%` relative regret. The best DFL variant is `dfl_q` at `7.08%`, ranking second overall.

This is a strong result for `dfl_q` relative to the rest of the baselines. It beats `dfl_h`, `ad_tree`, `saa`, `nn`, `m5_ad`, `ad`, `least_squares`, `cart`, and `er_saa`. The only method it does not beat is `knn`.

The q-vs-h comparison is also clean here: `dfl_q` has lower mean regret and lower variance than `dfl_h`. The decoder target should be `q` for random yield.

### `resource_allocation`

`knn` is best at `5.66%`, with `m5_ad`, `ad`, and `er_saa` all tightly grouped between `5.92%` and `6.27%`. `dfl_h` follows at `8.07%`.

This places `dfl_h` in the second tier: it does not beat the strongest baselines, but it beats `least_squares`, `ad_tree`, `nn`, `cart`, `saa`, and `dfl_q`.

`dfl_q` is the weakest method here at `24.31%`, slightly worse than `saa` at `23.54%`. This is the largest warning against using the q decoder as a default. The seed-level DflScenLoss results show why: one `dfl_q` seed reaches `45.28%` relative regret, so the mean is driven by genuine instability rather than a small measurement difference.

### `transshipment_h`

`knn` is best at `0.81%`. `dfl_h` is second at `2.58%`, narrowly ahead of `nn` at `2.61%` and ahead of `m5_ad` at `3.21%`.

This is one of the strongest results for our method. The h decoder aligns with the problem's random `h` structure and performs much better than most baselines. It beats `nn`, `m5_ad`, `er_saa`, `ad`, `least_squares`, `ad_tree`, `cart`, `saa`, and `dfl_q`.

`dfl_q` is last at `30.58%`. The q decoder is not just suboptimal here; it is worse than the simple baselines.

### `transshipment_h_and_q`

`knn` is again best at `0.56%`. `dfl_h` is second at `2.06%`, ahead of `m5_ad` at `3.38%`, `ad` at `3.53%`, and `er_saa` at `4.08%`.

This result is important because the instance has both `h` and `q` variation, yet the h decoder still clearly wins. `dfl_h` is the best method after `knn`, and it has a material margin over the next group of baselines.

`dfl_q` is again very weak at `26.95%`, ranking last. In this setup, learning only `q` is not enough to represent the decision-relevant uncertainty.

## DflScenLoss q vs h

| Benchmark | Better DFL decoder | `dfl_q` relative regret | `dfl_h` relative regret | Difference |
|---|---:|---:|---:|---:|
| `random_yield` | `q` | 7.08% | 8.40% | `q` better by 1.32 pp |
| `resource_allocation` | `h` | 24.31% | 8.07% | `h` better by 16.24 pp |
| `transshipment_h` | `h` | 30.58% | 2.58% | `h` better by 27.99 pp |
| `transshipment_h_and_q` | `h` | 26.95% | 2.06% | `h` better by 24.88 pp |

The result is not "DFL works" or "DFL fails" uniformly. The decoder choice controls most of the outcome. `dfl_h` is the version that should be benchmarked seriously against the other baselines for transshipment and resource allocation. `dfl_q` should be retained for random yield, where it is the best DFL variant and the second-best method overall.

## DFL-rho Full Seven-Problem Results

The new rho-DFL sweep adds three policies:

- `dfl_mu0_rho0.001`
- `dfl_mu0_rho0.01`
- `dfl_mu0_rho0.1`

All rho-DFL rows use the same tiny test artifact and the same three replica seeds as the DflScenLoss q/h rows. The final aggregate has `63` rows, all `status=ok`, with `3` replicas for each of `7` problems x `3` rho settings. The row metadata confirms `test_contexts=30`, `test_scenarios_per_context=500`, `evaluation_batches=5`, `mu_eval=0`, and `rho_eval=0`.

Lower is better. Values are mean regret plus/minus standard deviation across the three replicas.

| Problem | rho=0.001 | rho=0.01 | rho=0.1 | Best rho |
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

The rho pattern is sharp. `rho=0.001` is the right setting on the transshipment problems, resource allocation, and unreliable newsvendor. `rho=0.01` is best on shipment planning. `rho=0.1` is best only on random yield, and even there the result is weak relative to the earlier `dfl_q` and `dfl_h` DflScenLoss variants.

## Complete Overlap Table with DFL-rho

This table extends the v2 four-instance comparison with the three rho-DFL policies. Lower is better. Values are `relative_regret_mean * 100`.

| Benchmark | saa | er_saa | knn | least_squares | cart | nn | ad | ad_tree | m5_ad | dfl_q | dfl_h | dfl_rho0.001 | dfl_rho0.01 | dfl_rho0.1 | Best |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `random_yield` | 23.62% | 46.81% | 3.77% | 29.14% | 30.64% | 26.07% | 26.72% | 17.48% | 26.71% | 7.08% | 8.40% | 26.91% | 27.18% | 25.90% | `knn` |
| `resource_allocation` | 23.54% | 6.27% | 5.66% | 8.52% | 13.36% | 11.73% | 6.17% | 11.38% | 5.92% | 24.31% | 8.07% | 18.13% | 24.21% | 48.18% | `knn` |
| `transshipment_h` | 15.48% | 4.34% | 0.81% | 6.44% | 12.57% | 2.61% | 4.44% | 7.78% | 3.21% | 30.58% | 2.58% | 2.77% | 29.20% | 70.25% | `knn` |
| `transshipment_h_and_q` | 13.22% | 4.08% | 0.56% | 6.02% | 10.77% | 6.99% | 3.53% | 7.96% | 3.38% | 26.95% | 2.06% | 2.86% | 20.74% | 72.48% | `knn` |

## Updated Overall Ranking on the Four Overlap Instances

This ranking uses only `random_yield`, `resource_allocation`, `transshipment_h`, and `transshipment_h_and_q`, because those are the instances where v2 has the full baseline comparison.

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

The updated ranking changes the interpretation in one specific way: `dfl_mu0_rho0.001` is a legitimate middle-tier method on the four-instance overlap, mostly because it is strong on the two transshipment cases. It still does not challenge `dfl_h`, and it is not a replacement for the q/h DflScenLoss results.

## Best DFL Family by Problem

This table compares the best DFL result among `dfl_q`, `dfl_h`, and the three rho-DFL policies on the four overlapping problems.

| Benchmark | Best DFL family member | Relative regret | Second-best DFL family member | Relative regret | Interpretation |
|---|---|---:|---|---:|---|
| `random_yield` | `dfl_q` | 7.08% | `dfl_h` | 8.40% | q/h DflScenLoss dominates rho-DFL |
| `resource_allocation` | `dfl_h` | 8.07% | `dfl_mu0_rho0.001` | 18.13% | rho-DFL is not close |
| `transshipment_h` | `dfl_h` | 2.58% | `dfl_mu0_rho0.001` | 2.77% | rho-DFL is close but still second |
| `transshipment_h_and_q` | `dfl_h` | 2.06% | `dfl_mu0_rho0.001` | 2.86% | rho-DFL is useful but weaker |

Across the overlap set, the best DFL story remains the v2 story: `dfl_h` is the serious DFL baseline for resource allocation and transshipment, and `dfl_q` is the serious DFL baseline for random yield. The new rho-DFL sweep adds evidence that small-rho training can be competitive on transshipment, but it does not supersede the decoder-specific q/h results.

## DflScenLoss Seed-Level Results

Lower is better.

| Problem | Output | Seed | Regret | Relative regret |
|---|---:|---:|---:|---:|
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

The seed-level table reinforces the mean result. `dfl_h` is stable on transshipment and resource allocation. `dfl_q` is stable on random yield but unstable or consistently weak on the other three instances.

## Final Interpretation

With the accepted baseline comparison and the new rho-DFL sweep included, the headline should be:

`knn` remains the benchmark to beat, but `dfl_h` is the strongest non-knn method across the four requested problem instances.

This remains true after adding the rho-DFL results. Under the accepted `30 x 5 x 100` comparison convention, `dfl_h` is not just the better DFL decoder; it is a competitive method relative to the full baseline set, especially on the transshipment instances.

The new rho-DFL evidence is useful but narrower. `dfl_mu0_rho0.001` is a credible transshipment-family result and is close to `dfl_h` on `transshipment_h`, but it is weaker than the best q/h DflScenLoss variant on every overlapping problem. Larger rho values are usually bad under unregularized evaluation, especially on transshipment.

The main weakness is that our method does not beat `knn`. The secondary weakness is that the wrong decoder or an overly large rho can fail badly. Any final benchmark table should therefore report both q/h DflScenLoss variants and the rho sweep, but the interpretation should emphasize the best DFL configuration by problem:

- `random_yield`: use `dfl_q`.
- `resource_allocation`: use `dfl_h`.
- `shipment_planning`: use `dfl_mu0_rho0.01` among the rho-DFL variants; no q/h DflScenLoss row is available in this v3 comparison.
- `transshipment_q`: use `dfl_mu0_rho0.001` among the rho-DFL variants; no q/h DflScenLoss row is available in this v3 comparison.
- `transshipment_h`: use `dfl_h`.
- `transshipment_h_and_q`: use `dfl_h`.
- `unreliable_newsvendor`: use `dfl_mu0_rho0.001` among the rho-DFL variants; no q/h DflScenLoss row is available in this v3 comparison.
