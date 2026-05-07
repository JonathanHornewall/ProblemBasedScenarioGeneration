# Baseline Benchmark Performance Analysis

Codex session: 019e01d7-87ad-72b0-84c9-b3a5b7c7b0c3

Source: `baseline_results_latest.csv` in `evalbatch1_seed20260505_20260507_all_baselines`

## Run Coverage

The batch contains 63 completed runs: 7 benchmarks evaluated with 9 policies each. Every row has status `ok`.

Common run configuration:

| Field | Value |
|---|---:|
| Profile | `proper` |
| Seed | `20260505` |
| Train contexts | 100 |
| Train scenarios per context | 1 |
| Test contexts | 30 |
| Test scenarios per context | 1000 |
| Evaluation batches | 1 |
| Sample count | 30 |

The primary comparison metric in this report is `relative_regret_mean`, reported as a percentage. Lower is better. This is the most useful headline metric because raw objective values and raw regrets are on very different scales across benchmarks.

## Executive Summary

`knn` is the strongest overall baseline in this batch. It has the lowest mean relative regret at `2.05%`, the lowest median relative regret at `1.59%`, the best tie-aware average rank at `1.57`, and it is best or tied-best on 5 of the 7 benchmarks.

The next most competitive policies are `ad` and `m5_ad`. They have similar mean relative regret, `6.06%` and `6.10%` respectively, but differ in cost: `ad` averages about `1823` fit seconds per benchmark, while `m5_ad` averages about `2652` fit seconds. `m5_ad` has a better average rank than `ad` because it is consistently near the front on several benchmarks, even though it does not win outright except in the near-tied `transshipment_q` case.

`cart`, `saa`, and `least_squares` are the weakest overall policies by average relative regret. `cart` is especially poor on `shipment_planning` and `unreliable_newsvendor`; `saa` is weak on the resource and transshipment benchmarks except for the degenerate-looking `transshipment_q`; `least_squares` is never the best policy in this batch.

`transshipment_q` is not very discriminative in this run. Most policies achieve effectively zero regret, and even the worst policy, `least_squares`, is only at `0.32%` relative regret. Interpret aggregate win counts with that tie-heavy benchmark in mind.

## Complete Relative Regret Table

Lower is better. Values are `relative_regret_mean * 100`.

| Benchmark | saa | er_saa | knn | least_squares | cart | nn | ad | ad_tree | m5_ad | Best |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| random_yield | 23.62% | 46.81% | 3.77% | 29.14% | 30.64% | 26.07% | 26.72% | 17.48% | 26.71% | knn |
| resource_allocation | 23.54% | 6.27% | 5.66% | 8.52% | 13.36% | 11.73% | 6.17% | 11.38% | 5.92% | knn |
| shipment_planning | 10.61% | 0.19% | 1.94% | 7.55% | 30.42% | 6.90% | 0.32% | 7.40% | 1.40% | er_saa |
| transshipment_h | 15.48% | 4.34% | 0.81% | 6.44% | 12.57% | 2.61% | 4.44% | 7.78% | 3.21% | knn |
| transshipment_h_and_q | 13.22% | 4.08% | 0.56% | 6.02% | 10.77% | 6.99% | 3.53% | 7.96% | 3.38% | knn |
| transshipment_q | 0.00% | 0.03% | 0.00% | 0.32% | 0.00% | 0.00% | 0.03% | 0.00% | 0.00% | tied near-zero |
| unreliable_newsvendor | 21.41% | 3.25% | 1.59% | 17.63% | 30.69% | 8.84% | 1.18% | 6.65% | 2.08% | ad |

## Best Result by Benchmark

This table shows the best policy for each benchmark using relative regret, along with the corresponding objective and raw regret fields from the CSV.

| Benchmark | Optimal value mean | Best policy | Policy value mean | Regret mean | Relative regret | Worst policy |
|---|---:|---|---:|---:|---:|---|
| random_yield | 14.0812 | knn | 14.6090 | 0.5277 | 3.77% | er_saa (46.81%) |
| resource_allocation | 841.4969 | knn | 894.2299 | 52.7329 | 5.66% | saa (23.54%) |
| shipment_planning | 7034.9015 | er_saa | 7048.5685 | 13.6670 | 0.19% | cart (30.42%) |
| transshipment_h | 3059.7964 | knn | 3083.9507 | 24.1543 | 0.81% | saa (15.48%) |
| transshipment_h_and_q | 3068.0130 | knn | 3085.2351 | 17.2221 | 0.56% | saa (13.22%) |
| transshipment_q | 2280.0000 | tied near-zero | 2280.0000 | 0.0000 | 0.00% | least_squares (0.32%) |
| unreliable_newsvendor | -1.6830 | ad | -1.6639 | 0.0191 | 1.18% | cart (30.69%) |

## Overall Policy Ranking

Average rank is tie-aware: policies with the same relative regret after numerical rounding receive the same rank. This mainly affects `transshipment_q`, where many policies are effectively tied at zero regret.

| Policy | Mean relative regret | Median relative regret | Max relative regret | Tie-aware average rank | Wins or near-ties | Mean fit seconds | Mean eval seconds |
|---|---:|---:|---:|---:|---:|---:|---:|
| knn | 2.05% | 1.59% | 5.66% | 1.57 | 5 | 0.9 | 400.0 |
| m5_ad | 6.10% | 3.21% | 26.71% | 2.71 | 1 | 2651.9 | 236.6 |
| ad | 6.06% | 3.53% | 26.72% | 3.14 | 1 | 1823.1 | 215.7 |
| er_saa | 9.28% | 4.08% | 46.81% | 4.00 | 1 | 2.7 | 409.1 |
| nn | 9.02% | 6.99% | 26.07% | 4.43 | 1 | 678.5 | 274.3 |
| ad_tree | 8.38% | 7.78% | 17.48% | 4.86 | 1 | 1890.7 | 274.6 |
| least_squares | 10.80% | 7.55% | 29.14% | 5.71 | 0 | 2.9 | 390.7 |
| saa | 15.41% | 15.48% | 23.62% | 6.71 | 1 | 14.4 | 378.6 |
| cart | 18.35% | 13.36% | 30.69% | 7.29 | 1 | 4.2 | 287.5 |

The ranking above is sorted by tie-aware average rank rather than mean relative regret. That ordering favors policies that are consistently near the top across benchmarks. By mean relative regret alone, the top policies are `knn`, `ad`, `m5_ad`, `ad_tree`, and `nn`.

## Benchmark-Level Interpretation

### random_yield

`knn` is the clear winner at `3.77%` relative regret. The second-best policy, `ad_tree`, is much worse at `17.48%`, so this is not a close result. Most other policies cluster between roughly `23%` and `31%`, while `er_saa` is the outlier worst at `46.81%`.

This benchmark is the main weakness for several otherwise competitive policies. `ad`, `m5_ad`, and `nn` all exceed `26%` relative regret here, which substantially raises their overall mean regret.

### resource_allocation

`knn` is again best, but the top group is tight. `knn` reaches `5.66%`, followed by `m5_ad` at `5.92%`, `ad` at `6.17%`, and `er_saa` at `6.27%`. These four policies are effectively the competitive set for this benchmark.

`saa` is the worst policy here at `23.54%`, suggesting that direct sample-average decisions are poorly matched to this resource allocation setting under the current training regime.

### shipment_planning

`er_saa` is best at `0.19%`, with `ad` very close at `0.32%`. `m5_ad` is also strong at `1.40%`, and `knn` remains competitive at `1.94%`.

The gap between the leading group and the rest is large. `cart` is the worst at `30.42%`, while `saa`, `least_squares`, `nn`, and `ad_tree` are materially behind the top four.

### transshipment_h

`knn` is best at `0.81%`. `nn` is second at `2.61%`, followed by `m5_ad` at `3.21%`. `er_saa` and `ad` are close to each other around `4.3%` to `4.4%`.

`saa` is worst at `15.48%`. This reinforces the pattern that the transshipment benchmarks reward context-sensitive policies more than plain SAA.

### transshipment_h_and_q

This benchmark is another strong result for `knn`, which achieves `0.56%` relative regret. `m5_ad`, `ad`, and `er_saa` form the next group, ranging from `3.38%` to `4.08%`.

`saa` is worst at `13.22%`, and `cart` is also weak at `10.77%`. The ordering is broadly consistent with `transshipment_h`, which suggests the advantage of `knn` is not limited to a single transshipment parameterization.

### transshipment_q

This benchmark is nearly solved by most policies in this batch. `saa`, `knn`, `cart`, `nn`, `ad_tree`, and `m5_ad` all have effectively zero relative regret after rounding. `er_saa` and `ad` are also very close at `0.03%`.

Because the regret range is tiny, `transshipment_q` should not be over-weighted when deciding which policy is strongest overall. It provides little separation among policies compared with the other benchmarks.

### unreliable_newsvendor

`ad` is best at `1.18%`, with `knn` close behind at `1.59%`. `m5_ad` is third at `2.08%`, and `er_saa` follows at `3.25%`.

`cart` is the worst at `30.69%`, and `saa` is also weak at `21.41%`. The best result here is one of the few cases where `ad` clearly improves on `knn`.

## Runtime Considerations

The cheapest fit-time policies are `knn`, `er_saa`, `least_squares`, and `cart`, all averaging under 5 fit seconds per benchmark. Among these, `knn` is the only one that is also consistently high-performing, making it the best speed-quality tradeoff in this batch.

The optimization-heavy policies are much more expensive:

| Policy | Mean fit seconds | Mean relative regret |
|---|---:|---:|
| nn | 678.5 | 9.02% |
| ad | 1823.1 | 6.06% |
| ad_tree | 1890.7 | 8.38% |
| m5_ad | 2651.9 | 6.10% |

`ad` and `m5_ad` are competitive in accuracy, but their fit-time cost is very high relative to `knn`. In this batch, that extra cost does not buy a better overall result. `ad` is still valuable for `unreliable_newsvendor` and is nearly best on `shipment_planning`, but it is not the strongest general-purpose baseline here.

Evaluation time is less decisive than fit time. The spread in mean evaluation time is smaller than the spread in fit time, and the best policy by accuracy, `knn`, has one of the slower evaluation averages. If repeated evaluation is the bottleneck, this should be considered separately, but for model selection the fit-time penalty of `ad`, `ad_tree`, and `m5_ad` is the larger practical concern.

## Conclusions

For this batch, `knn` should be treated as the primary baseline to beat. It is the best or tied-best on most benchmarks, has the best average rank, and achieves this with negligible fit time.

`ad` and `m5_ad` are the strongest challengers. `ad` is particularly good on `unreliable_newsvendor` and `shipment_planning`; `m5_ad` is consistently competitive on `resource_allocation`, `shipment_planning`, and the transshipment variants. However, both are much more expensive to fit than `knn`.

`er_saa` has isolated strong performance, especially on `shipment_planning`, but it is unstable because of its very poor `random_yield` result. `nn` is moderately competitive but does not justify its fit-time cost in this run. `cart`, `saa`, and `least_squares` are weaker overall and should mainly be retained as simple reference baselines rather than serious contenders.
