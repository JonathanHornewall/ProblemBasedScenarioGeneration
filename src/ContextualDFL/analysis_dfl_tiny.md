# DflScenLoss Tiny q/h Run Analysis

## Executive Summary

The DflScenLoss tiny benchmark completed successfully on `gcp-16c-4`. The final run produced 24 successful jobs: 4 problem instances, 2 decoder outputs (`q` and `h`), and 3 model seeds (`20260505`, `20260506`, `20260507`). Every job trained for 130 epochs, saved resumable state and final model weights, and was evaluated once at the end against the precomputed test optima stored in the tiny artifact bundle.

The main result is that the choice of learned output matters sharply. Learning `h` is much better for the two transshipment settings and for resource allocation. Learning `q` is better for random yield, but the gap there is modest compared with the transshipment and resource-allocation gaps.

The run should not yet be described as a complete benchmark against all other baselines for this exact tiny protocol. I checked the local and remote baseline result trees and found no successful baseline rows with `test_contexts=30`, `test_scenarios_per_context=500`, and `evaluation_batches=5`. The existing complete baseline aggregate under `evalbatch1_seed20260505_20260507_all_baselines` uses `test_scenarios_per_context=1000` and `evaluation_batches=1`, so it is not directly comparable to this DflScenLoss run.

## Run Configuration

Runner:

`/home/rwl/ProblemBasedScenarioGeneration/experiments/temp/2026-05-07_dflscenloss_tiny_qh_gcp16c4/benchmark_dflscenloss_tiny_qh.jl`

Remote output directory:

`/home/rwl/ProblemBasedScenarioGeneration/experiments/temp/2026-05-07_dflscenloss_tiny_qh_gcp16c4/results_final_eval_once`

Test artifact bundle:

`/home/rwl/ProblemBasedScenarioGeneration/src/ContextualDFL/ContextualDFLExperiments/experiments/baseline_benchmarks/artifacts/tiny_30ctx_5x100_seed20260505`

Invocation recorded in `manifest.txt`:

```text
--artifact-dir /home/rwl/ProblemBasedScenarioGeneration/src/ContextualDFL/ContextualDFLExperiments/experiments/baseline_benchmarks/artifacts/tiny_30ctx_5x100_seed20260505
--output-dir /home/rwl/ProblemBasedScenarioGeneration/experiments/temp/2026-05-07_dflscenloss_tiny_qh_gcp16c4/results_final_eval_once
--local-workers 12
--job-batch-size 12
--problems transshipment_h,transshipment_h_and_q,random_yield,resource_allocation
--outputs q,h
--seeds 20260505,20260506,20260507
--max-epochs 130
```

Model/training settings:

- Network: generic vector decoder network with `hidden_dim=128`, `depth=3`.
- Optimizer: Adam, learning rate `1e-3`.
- Batch size: `1`.
- Training examples: the training data inside the tiny artifact bundle.
- Evaluation data: 30 test contexts, 500 scenarios per context, 5 evaluation batches.
- Smoothing schedule: geometric `mu` from `1e-1` to `1e-4`, followed by a final stage with `mu_in=1e-4` and `mu_ref=0`.
- Epoch schedule: 130 epochs total. Stage 1 has 20 epochs, then ten 10-epoch geometric stages, then a final 10-epoch reference-unsmoothed stage.

The schedule in the manifest starts at `mu_in=mu_ref=0.1`, passes through roughly `0.0501`, `0.0251`, `0.0126`, `0.00631`, `0.00316`, `0.00158`, `0.000794`, `0.000398`, `0.000199`, and ends at `0.0001`. Epochs 121-130 use `mu_in=0.0001` and `mu_ref=0.0`.

## Evaluation Semantics

Training does not evaluate against optimal solutions at each epoch.

During training, `ContextualDFL.train!` calls `DflScenLoss`. That loss:

1. Decodes the model output as a generated scenario collection.
2. Solves the stochastic program under the generated scenario using the current `mu_in`.
3. Decodes the reference scenario from the training data.
4. Evaluates the generated decision under the reference scenario with `mu_ref`.

That is a decision-focused training loss, but it is not train regret and it is not a comparison to precomputed optimal solutions. The `epoch_results.csv` file therefore records training loss only.

The comparison against optimal solutions happens once after the final epoch. The runner wraps the trained model in a `ScenarioGenerationPolicy` and calls `evaluate_policy_against_optimum` with the artifact's `optimal_results`. That final evaluation uses `mu=0` and `rho=0` for objective comparison and computes policy value, optimal value, regret, relative regret, and per-sample outputs.

This is the corrected behavior after debugging the long run. The earlier version evaluated the test set at every epoch. Since each test pass solves or evaluates across 30 contexts and 5 stored evaluation batches, doing that 130 times per job dominated wall time. The final version evaluates once at the end and keeps only the latest checkpoint plus final model weights.

## Run Integrity

The final result set is complete:

- 24/24 jobs completed with status `ok`.
- 24 final metrics files.
- 24 final result objects.
- 24 epoch-history CSVs.
- 24 per-sample final CSVs.
- 24 final model files.
- 3,120 epoch rows in `epoch_results.csv`, matching 24 jobs x 130 epochs.
- 24 rows in `individual_results.csv`.
- 8 rows in `summary_by_config.csv`, matching 4 problems x 2 decoder outputs.
- Remote deserialization check: `ok=24 bad=0`.

Resume behavior is in place. Each job writes `state.jls`, `epoch_history.csv`, `model_checkpoint_latest.jls`, `model_final.jls`, `test_metrics_final.csv`, `comparison_final.jls`, `per_sample_final.csv`, and `final_result.jls`. If the experiment crashes, completed jobs are skipped and partial jobs resume from `state.jls`.

## Mean Final Results

Lower regret and lower relative regret are better.

| Problem | Learned output | Seeds | Regret mean | Regret std | Relative regret mean | Relative regret std |
|---|---:|---:|---:|---:|---:|---:|
| `random_yield` | `h` | 3 | 1.2091 | 0.2642 | 8.40% | 1.76% |
| `random_yield` | `q` | 3 | 0.9736 | 0.0613 | 7.08% | 0.38% |
| `resource_allocation` | `h` | 3 | 68.1171 | 10.6960 | 8.07% | 1.21% |
| `resource_allocation` | `q` | 3 | 208.9199 | 164.8419 | 24.31% | 19.24% |
| `transshipment_h` | `h` | 3 | 78.5361 | 18.7173 | 2.58% | 0.61% |
| `transshipment_h` | `q` | 3 | 945.1743 | 35.9823 | 30.58% | 1.19% |
| `transshipment_h_and_q` | `h` | 3 | 61.9062 | 8.8917 | 2.06% | 0.33% |
| `transshipment_h_and_q` | `q` | 3 | 829.4894 | 196.4137 | 26.95% | 5.99% |

## Individual Final Results

| Problem | Output | Seed | Regret | Relative regret | Policy value | Optimal value |
|---|---:|---:|---:|---:|---:|---:|
| `random_yield` | `h` | 20260505 | 1.4971 | 10.30% | 15.5991 | 14.1020 |
| `random_yield` | `h` | 20260506 | 0.9781 | 6.82% | 15.0801 | 14.1020 |
| `random_yield` | `h` | 20260507 | 1.1522 | 8.08% | 15.2543 | 14.1020 |
| `random_yield` | `q` | 20260505 | 1.0289 | 7.43% | 15.1310 | 14.1020 |
| `random_yield` | `q` | 20260506 | 0.9077 | 6.68% | 15.0097 | 14.1020 |
| `random_yield` | `q` | 20260507 | 0.9841 | 7.15% | 15.0862 | 14.1020 |
| `resource_allocation` | `h` | 20260505 | 77.8981 | 9.18% | 919.1372 | 841.2391 |
| `resource_allocation` | `h` | 20260506 | 69.7575 | 8.25% | 910.9966 | 841.2391 |
| `resource_allocation` | `h` | 20260507 | 56.6957 | 6.78% | 897.9347 | 841.2391 |
| `resource_allocation` | `q` | 20260505 | 64.9643 | 7.49% | 906.2034 | 841.2391 |
| `resource_allocation` | `q` | 20260506 | 173.0531 | 20.16% | 1014.2922 | 841.2391 |
| `resource_allocation` | `q` | 20260507 | 388.7423 | 45.28% | 1229.9814 | 841.2391 |
| `transshipment_h` | `h` | 20260505 | 100.1445 | 3.28% | 3161.8956 | 3061.7510 |
| `transshipment_h` | `h` | 20260506 | 67.3695 | 2.23% | 3129.1205 | 3061.7510 |
| `transshipment_h` | `h` | 20260507 | 68.0943 | 2.23% | 3129.8454 | 3061.7510 |
| `transshipment_h` | `q` | 20260505 | 965.9477 | 31.26% | 4027.6987 | 3061.7510 |
| `transshipment_h` | `q` | 20260506 | 903.6260 | 29.21% | 3965.3770 | 3061.7510 |
| `transshipment_h` | `q` | 20260507 | 965.9490 | 31.26% | 4027.7000 | 3061.7510 |
| `transshipment_h_and_q` | `h` | 20260505 | 54.5839 | 1.80% | 3117.8744 | 3063.2905 |
| `transshipment_h_and_q` | `h` | 20260506 | 59.3342 | 1.96% | 3122.6247 | 3063.2905 |
| `transshipment_h_and_q` | `h` | 20260507 | 71.8003 | 2.43% | 3135.0908 | 3063.2905 |
| `transshipment_h_and_q` | `q` | 20260505 | 1055.3685 | 33.83% | 4118.6590 | 3063.2905 |
| `transshipment_h_and_q` | `q` | 20260506 | 734.1741 | 24.09% | 3797.4646 | 3063.2905 |
| `transshipment_h_and_q` | `q` | 20260507 | 698.9256 | 22.92% | 3762.2161 | 3063.2905 |

## Training-Loss Summary

The training loss is the DflScenLoss objective under the current annealing parameters. It should be read as a training signal, not as regret. In particular, final train loss can move differently from final test regret because the final reported regret is a post-training comparison against exact test optima.

| Problem | Output | Epoch 1 loss mean | Epoch 130 loss mean | Min loss mean | Mean training seconds/job | Mean epoch seconds |
|---|---:|---:|---:|---:|---:|---:|
| `random_yield` | `h` | 23.77 | 13.36 | 13.33 | 231.8 | 1.78 |
| `random_yield` | `q` | 24.36 | 14.26 | 14.00 | 247.6 | 1.90 |
| `resource_allocation` | `h` | 2758 | 860.5 | 852.5 | 2204.7 | 17.0 |
| `resource_allocation` | `q` | 2985 | 1021 | 975.8 | 2021.7 | 15.6 |
| `transshipment_h` | `h` | 3376 | 2932 | 2931 | 444.7 | 3.42 |
| `transshipment_h` | `q` | 3786 | 3893 | 3779 | 529.7 | 4.07 |
| `transshipment_h_and_q` | `h` | 3424 | 3026 | 3009 | 591.0 | 4.55 |
| `transshipment_h_and_q` | `q` | 3838 | 3882 | 3778 | 519.2 | 3.99 |

The training curves are consistent with the final test results in the broad cases where `h` wins: `transshipment_h`, `transshipment_h_and_q`, and `resource_allocation` all show substantially better final training losses for `h` than for `q`. Random yield is the exception: `h` has a lower final training loss, but `q` has better final test regret. That suggests the random-yield `h` training objective is not selecting policies that generalize as well as the `q` decoder on this test bundle.

## Problem-by-Problem Interpretation

### `transshipment_h`

Learning `h` is clearly the right decoder choice for this instance. Mean relative regret is 2.58% for `h` versus 30.58% for `q`. The gap is roughly 28 percentage points, and it is stable across all three seeds.

The `q` decoder produces very poor policies here. All three `q` seeds land near 29-31% relative regret, while all three `h` seeds stay near 2-3%. This fits the data-generating structure: this instance varies `h`, so a decoder that can output `h` is aligned with the uncertainty the policy needs to represent.

### `transshipment_h_and_q`

Learning `h` again dominates. Mean relative regret is 2.06% for `h` versus 26.95% for `q`.

The mixed `h+q` setting is important because learning `q` has access to one varying component but still performs badly. The result suggests that, at least for this tiny setup and this one-scenario DflScenLoss policy, representing the `h` side of the uncertainty is more important than representing `q`. The `q` results are also less stable than `h`; the seed-level relative regrets range from 22.92% to 33.83%.

### `random_yield`

Learning `q` is best. Mean relative regret is 7.08% for `q` versus 8.40% for `h`.

The improvement is smaller than in transshipment: about 1.32 percentage points in relative regret. The `q` decoder is also more stable, with relative-regret standard deviation of 0.38% compared with 1.76% for `h`. This is the one case where the `q` implementation is the better final choice.

### `resource_allocation`

Learning `h` is much better and much more stable. Mean relative regret is 8.07% for `h` versus 24.31% for `q`.

The `q` result is seed-sensitive. Seed `20260505` is competitive at 7.49% relative regret, but seed `20260506` rises to 20.16% and seed `20260507` fails badly at 45.28%. The `h` decoder is much more consistent, ranging from 6.78% to 9.18%.

This is the strongest stability warning in the run. Even if the best `q` seed is acceptable, the mean and variance indicate that `q` is not reliable on this resource-allocation tiny benchmark.

## q vs h Decoder Takeaway

Best decoder by problem:

| Problem | Better decoder | Relative regret difference |
|---|---:|---:|
| `transshipment_h` | `h` | `h` better by 27.99 percentage points |
| `transshipment_h_and_q` | `h` | `h` better by 24.88 percentage points |
| `random_yield` | `q` | `q` better by 1.32 percentage points |
| `resource_allocation` | `h` | `h` better by 16.24 percentage points |

The overall pattern is not that one decoder universally dominates. The useful rule from this run is that the decoder should match the uncertainty channel that matters for the induced decision. For the two transshipment variants and resource allocation, `h` carries the decision-relevant variation better. For random yield, `q` is the better target.

## Runtime and Debugging Notes

The run was initially too long for two reasons.

First, the initial version evaluated test performance at every epoch. That was unnecessary and expensive. Testing requires evaluating policy decisions against precomputed optimal values over the full test bundle. With 130 epochs and 24 jobs, per-epoch testing would require 3,120 test passes. The corrected version performs 24 test passes, one per finished job.

Second, the first version wrote one model file per epoch. That created unnecessary disk pressure on the remote machine. The corrected version writes a resumable `state.jls`, overwrites `model_checkpoint_latest.jls`, and saves `model_final.jls` at completion.

Observed final timing:

- Random yield jobs finished in roughly 5.7-7.4 minutes each including final evaluation.
- Transshipment jobs finished in roughly 1.6-4.1 minutes each depending on evaluation timing and worker contention.
- Resource allocation jobs were the expensive cases, around 36.8-40.9 minutes each.
- Final evaluation alone took about 35-60 seconds for random yield, 64-102 seconds for transshipment, and 141-165 seconds for resource allocation.

The resource-allocation training loop is the dominant runtime cost, averaging around 16-17 seconds per epoch.

## Baseline Comparison Status

This run currently provides the DflScenLoss q/h benchmark, not a completed comparison against all other baselines on the same tiny protocol.

I checked both local and remote result trees for baseline CSV rows with:

```text
test_contexts=30
test_scenarios_per_context=500
evaluation_batches=5
```

No matching rows were found.

The complete existing aggregate:

`ContextualDFLExperiments/experiments/baseline_benchmarks/results/evalbatch1_seed20260505_20260507_all_baselines/baseline_results_latest.csv`

contains successful rows, but it uses:

```text
test_contexts=30
test_scenarios_per_context=1000
evaluation_batches=1
```

That is a different evaluation protocol. It should not be mixed into this tiny DflScenLoss table as if it were directly comparable.

There is already infrastructure suggesting the intended tiny full-baseline protocol:

- `run_baselines.jl` has `TINY_CONFIG` with `test_contexts=30`, `test_scenarios_per_context=500`, `evaluation_batches=5`.
- `run_baselines.jl` supports `--tiny-full-baselines`, which sets profile `tiny`, enables the full baseline grid, and uses tiny data artifacts.
- `aggregate_tiny_full_baselines.jl` expects 196 successful rows across deterministic, replicated, and DFL baseline policies and explicitly rejects rows from the `30x1000` or wrong-batch protocols.

The missing next step for a true "benchmark against all other baselines" is to run the tiny full-baseline grid against the same artifact bundle and then aggregate those rows together with this DflScenLoss q/h result table.

## Artifacts to Keep

Primary result files:

- `summary.md`
- `summary_by_config.csv`
- `individual_results.csv`
- `epoch_results.csv`

Per-job files:

- `epoch_history.csv`
- `state.jls`
- `model_checkpoint_latest.jls`
- `model_final.jls`
- `test_metrics_final.csv`
- `comparison_final.jls`
- `per_sample_final.csv`
- `final_result.jls`

The important analysis inputs are `summary_by_config.csv` for mean performance, `individual_results.csv` for seed-level performance, and `epoch_results.csv` for training-loss history.

## Caveats

The results use only three model seeds. That is enough to expose the large q-vs-h separations in transshipment and resource allocation, but it is still a small sample for estimating variance.

The test evaluation is final-only by design. This avoids the original runtime issue, but it means the run cannot answer which epoch had the best test regret.

Training loss is not regret. It is useful for monitoring optimization, but the final decision quality is measured only by the post-training `evaluate_policy_against_optimum` call.

The current result set is not yet an all-baseline benchmark for the tiny protocol because the matching tiny full-baseline rows are absent.
