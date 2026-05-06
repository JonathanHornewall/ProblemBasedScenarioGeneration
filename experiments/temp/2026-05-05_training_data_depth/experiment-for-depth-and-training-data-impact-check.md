# Experiment Report: Impact of Training Data and Network Depth

## Executive Summary

The suite completed successfully with 50 intended successful runs:

- Baseline standard configuration: 10/10 runs.
- Training-data experiment: 5/5 runs for 500 training contexts and 5/5 runs for 1000 training contexts.
- Depth experiment: 6/6 runs for each depth 5, 6, 10, 20, and 40.

The main outcome metric is average test relative regret; lower is better.

The strongest conclusions are:

1. Increasing training data from 100 to 500 or 1000 contexts under the normalized gradient-computation budget did not produce a clear improvement over the baseline. The 1000-context condition had the best mean regret, but the difference from baseline is small relative to run-to-run variability.
2. Network depth has a much stronger effect than training-data size in this suite. Depth 6 was the best depth setting among the tested alternatives, but the original depth-4 baseline remained slightly better on mean regret.
3. Very deep models were unstable or poorly matched to this training setup. Depth 10 and 20 were substantially worse than depth 5/6, and depth 40 failed badly in half of its runs.
4. Target-prediction MSE deteriorated as depth increased, especially for depths 10, 20, and 40. This suggests the deeper networks were not simply overfitting the optimization objective while preserving demand predictions; they were also producing much poorer demand-level approximations.

## Experiment Design

All runs used the resource-allocation annealing setup from `annealing.jl` as the baseline:

- GeLU hidden activation.
- Four hidden layers for the standard baseline.
- Annealing schedule from the baseline script.
- Softplus output activation for nonnegative demand predictions.
- Batch size 1.
- Baseline training set size: 100 contexts.
- Baseline epoch count: 130.

For the training-data experiment, the number of epochs was scaled to keep the total number of per-context gradient computations constant:

| Training contexts | Epochs | Context-epoch budget |
|---:|---:|---:|
| 100 | 130 | 13,000 |
| 500 | 26 | 13,000 |
| 1000 | 13 | 13,000 |

For the depth experiment, the training-data size remained at 100 contexts and the epoch count remained at 130. The tested depths were 5, 6, 10, 20, and 40.

All reported summaries below use the cleaned result files:

- `results/runs.csv`
- `results/epochs.csv`
- `results/test_per_sample.csv`
- `results/baseline_summary.csv`
- `results/data_amount_summary.csv`
- `results/depth_summary.csv`

The final result set contains 50 successful rows in `runs.csv`, with no failed rows.

## Primary Outcome: Test Relative Regret

| Phase | Config | Runs | Epochs | Train contexts | Mean regret | SD | 95% CI | Median | Min | Max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | standard_n100_depth4_gelu | 10 | 130 | 100 | 0.0367 | 0.0128 | [0.0275, 0.0459] | 0.0349 | 0.0180 | 0.0567 |
| data_amount | n500 | 5 | 26 | 500 | 0.0409 | 0.0237 | [0.0114, 0.0703] | 0.0436 | 0.0145 | 0.0713 |
| data_amount | n1000 | 5 | 13 | 1000 | 0.0344 | 0.0153 | [0.0155, 0.0534] | 0.0351 | 0.0116 | 0.0513 |
| depth | depth5 | 6 | 130 | 100 | 0.0593 | 0.0269 | [0.0311, 0.0875] | 0.0608 | 0.0160 | 0.0964 |
| depth | depth6 | 6 | 130 | 100 | 0.0471 | 0.0179 | [0.0283, 0.0658] | 0.0414 | 0.0282 | 0.0702 |
| depth | depth10 | 6 | 130 | 100 | 0.1438 | 0.1798 | [-0.0450, 0.3325] | 0.0799 | 0.0389 | 0.5082 |
| depth | depth20 | 6 | 130 | 100 | 0.1214 | 0.0794 | [0.0381, 0.2047] | 0.0888 | 0.0519 | 0.2412 |
| depth | depth40 | 6 | 130 | 100 | 2.3047 | 1.8077 | [0.4074, 4.2021] | 2.4276 | 0.2471 | 3.9413 |

Note: the negative lower CI bound for depth 10 is just a small-sample normality artifact around a nonnegative metric; it should not be interpreted literally.

## Experiment 1: Impact of Training Data

The training-data experiment compared the standard 100-context baseline against 500 and 1000 contexts while keeping the context-epoch budget fixed at 13,000.

| Config | Mean regret | Difference vs baseline | Relative change vs baseline |
|---|---:|---:|---:|
| baseline n=100 | 0.0367 | 0.0000 | 0.0% |
| n500 | 0.0409 | +0.0042 | +11.4% worse |
| n1000 | 0.0344 | -0.0022 | -6.1% better |

The headline result is that more data did not create a decisive improvement when the total gradient-computation budget was held constant. The 1000-context condition was slightly better than baseline on the mean, but the difference is much smaller than the standard deviations of either condition. The 500-context condition was slightly worse than baseline and had the largest variance among the three data-size conditions.

Per-run test relative regrets:

| Config | Per-run regrets |
|---|---|
| baseline n=100 | r01=0.0324, r02=0.0374, r03=0.0523, r04=0.0180, r05=0.0242, r06=0.0567, r07=0.0272, r08=0.0440, r09=0.0279, r10=0.0464 |
| n500 | r01=0.0201, r02=0.0436, r03=0.0548, r04=0.0145, r05=0.0713 |
| n1000 | r01=0.0513, r02=0.0351, r03=0.0116, r04=0.0445, r05=0.0297 |

Interpretation:

- The n1000 condition had the best single run among these three groups, with relative regret 0.0116.
- The n500 condition also had a strong best run, 0.0145, but a weak worst run, 0.0713.
- The baseline was more stable than n500 and comparable in stability to n1000.
- If there is a benefit to adding more contexts, this suite suggests it is modest under the fixed gradient-computation budget. More contexts also mean fewer passes through each context, which may be limiting optimization.

## Experiment 2: Impact of Network Depth

The depth experiment is much more decisive. The tested deeper networks generally worsened performance relative to the depth-4 baseline.

| Config | Mean regret | Difference vs baseline | Relative change vs baseline |
|---|---:|---:|---:|
| baseline depth4 | 0.0367 | 0.0000 | 0.0% |
| depth5 | 0.0593 | +0.0226 | +61.7% worse |
| depth6 | 0.0471 | +0.0104 | +28.4% worse |
| depth10 | 0.1438 | +0.1071 | +292.2% worse |
| depth20 | 0.1214 | +0.0847 | +231.1% worse |
| depth40 | 2.3047 | +2.2680 | +6185.0% worse |

Depth 6 was the best of the tested non-baseline depths, with mean regret 0.0471. It was still worse than the depth-4 baseline on the mean. Depth 5 was slightly worse than depth 6. Depths 10 and 20 were worse by a large margin, and depth 40 was qualitatively different: three of six runs landed at relative regret 3.9413.

Per-run test relative regrets:

| Config | Per-run regrets |
|---|---|
| depth5 | r01=0.0964, r02=0.0553, r03=0.0663, r04=0.0160, r05=0.0728, r06=0.0489 |
| depth6 | r01=0.0371, r02=0.0282, r03=0.0456, r04=0.0336, r05=0.0677, r06=0.0702 |
| depth10 | r01=0.0792, r02=0.0548, r03=0.1011, r04=0.0805, r05=0.0389, r06=0.5082 |
| depth20 | r01=0.0519, r02=0.2412, r03=0.1949, r04=0.0629, r05=0.1147, r06=0.0627 |
| depth40 | r01=0.2471, r02=0.9139, r03=0.8435, r04=3.9413, r05=3.9413, r06=3.9413 |

Interpretation:

- Depth 6 is the only deeper configuration that is close to baseline. Its worst run, 0.0702, is not catastrophic, and its best run, 0.0282, is competitive with baseline.
- Depth 5 has one very good run, 0.0160, but its mean is pulled up by several weaker runs.
- Depth 10 is unstable. Five of six runs are in a plausible range, but one run at 0.5082 makes the mean much worse.
- Depth 20 is consistently worse than depth 5/6 and has two large-regret runs.
- Depth 40 is not viable under this setup. Half the runs hit the same very high regret value, suggesting a recurring failure mode rather than ordinary sampling variation.

## Training Curves and Target Prediction Metrics

The primary metric is regret, not demand target MSE. Still, the target MSE curves are useful diagnostics because they show whether the learned scenario generator is producing plausible demand-level outputs.

The table below reports mean final epoch metrics. For checkpoint-recovered baseline runs, the canonical epoch history was reconstructed by run ID so the final training curve remains represented.

| Config | Final train target MSE | Final test target MSE | Final DFL training loss |
|---|---:|---:|---:|
| standard_n100_depth4_gelu | 560.9 | 474.8 | 910.3 |
| n500 | 753.9 | 653.7 | 908.4 |
| n1000 | 892.1 | 791.4 | 906.1 |
| depth5 | 1337.3 | 1166.5 | 924.7 |
| depth6 | 3407.9 | 3061.9 | 936.7 |
| depth10 | 8123.7 | 7160.5 | 1002.1 |
| depth20 | 7383.6 | 6520.9 | 1068.8 |
| depth40 | 26967.6 | 26163.9 | 3143.0 |

Several points stand out:

- The DFL training loss decreases substantially from early training to final training for most configurations, but lower DFL loss is not enough to rescue the very deep networks.
- Target MSE gets worse as depth increases, especially at depths 10, 20, and 40.
- Depth 40 has both extremely poor target MSE and extremely poor regret, so its failure is visible under both diagnostics.
- The larger-data conditions have higher final target MSE than the baseline, despite similar or slightly better DFL loss. This may reflect the smaller number of epochs: each context is seen fewer times, and the model does not settle into as strong a demand-prediction fit.

Representative mean training trajectory:

| Config | Epoch | Mean train target MSE | Mean test target MSE | Mean DFL loss | Runs represented |
|---|---:|---:|---:|---:|---:|
| standard_n100_depth4_gelu | 1 | 770.8 | 649.4 | 1779.7 | 10 |
| standard_n100_depth4_gelu | 65 | 595.9 | 516.6 | 1033.5 | 10 |
| standard_n100_depth4_gelu | 130 | 560.9 | 474.8 | 910.3 | 10 |
| n500 | 1 | 677.5 | 563.5 | 1137.5 | 5 |
| n500 | 13 | 809.2 | 713.6 | 1038.0 | 5 |
| n500 | 26 | 753.9 | 653.7 | 908.4 | 5 |
| n1000 | 1 | 924.6 | 759.6 | 1052.5 | 5 |
| n1000 | 6 | 875.0 | 781.0 | 1084.8 | 5 |
| n1000 | 13 | 892.1 | 791.4 | 906.1 | 5 |
| depth5 | 1 | 1051.5 | 939.4 | 1665.2 | 6 |
| depth5 | 65 | 1234.0 | 1089.0 | 1037.7 | 6 |
| depth5 | 130 | 1337.3 | 1166.5 | 924.7 | 6 |
| depth6 | 1 | 1295.2 | 1133.2 | 1592.9 | 6 |
| depth6 | 65 | 1801.9 | 1504.8 | 1048.7 | 6 |
| depth6 | 130 | 3407.9 | 3061.9 | 936.7 | 6 |
| depth10 | 1 | 2143.6 | 1914.2 | 1441.7 | 6 |
| depth10 | 65 | 7495.2 | 6765.3 | 1078.7 | 6 |
| depth10 | 130 | 8123.7 | 7160.5 | 1002.1 | 6 |
| depth20 | 1 | 2976.5 | 2717.1 | 1443.5 | 6 |
| depth20 | 65 | 7664.5 | 7199.5 | 1182.6 | 6 |
| depth20 | 130 | 7383.6 | 6520.9 | 1068.8 | 6 |
| depth40 | 1 | 4661.1 | 4394.2 | 1521.0 | 6 |
| depth40 | 65 | 7064.1 | 6570.5 | 2933.4 | 6 |
| depth40 | 130 | 26967.6 | 26163.9 | 3143.0 | 6 |

The depth trend in target MSE is much stronger than the data-size trend. This suggests that the current architecture/optimization recipe is sensitive to depth, and that simply making the MLP deeper is not a good direction without additional stabilization.

## Stability and Variance

Run-to-run variance is important in this suite.

For training-data size:

- Baseline SD: 0.0128.
- n500 SD: 0.0237.
- n1000 SD: 0.0153.

The n500 condition is visibly noisier than baseline and n1000. The n1000 condition is slightly better than baseline on the mean, but its confidence interval overlaps heavily with the baseline interval.

For depth:

- Depth 5 and 6 are moderate-variance conditions.
- Depth 10 has one large outlier.
- Depth 20 has several elevated-regret runs.
- Depth 40 has repeated collapse-level outcomes.

This makes depth the clearer experimental factor. Data amount produces subtle changes; depth can change the result by orders of magnitude.

## Conclusions

### Training Data

The normalized-computation data experiment does not support the claim that more contexts automatically improve performance. With the same total gradient-computation budget, 1000 contexts is marginally better than the 100-context baseline on mean regret, but not decisively. The 500-context condition is worse on the mean and more variable.

The practical read is:

- More data may help slightly if moved from 100 to 1000 contexts.
- The benefit is small under the current annealing/training budget.
- The fixed compute budget means larger datasets receive fewer epochs, which may be the limiting factor.

### Network Depth

The depth experiment strongly argues against increasing depth aggressively under the current setup.

Depth ranking by mean regret:

1. Baseline depth 4: 0.0367.
2. Depth 6: 0.0471.
3. Depth 5: 0.0593.
4. Depth 20: 0.1214.
5. Depth 10: 0.1438.
6. Depth 40: 2.3047.

The original depth-4 baseline remains the best choice among all depth settings tested. If a deeper model is needed, depth 6 is the least damaging candidate, but it does not improve the baseline in this suite.

## Recommended Next Checks

1. Keep depth 4 as the default architecture for this annealing setup.
2. If testing additional data sizes, include a non-normalized-compute condition where larger datasets get more total context updates. The current experiment answers the fixed-budget question, not the question of whether more data helps with more compute.
3. If testing deeper networks again, add stabilization before depth 10+: residual connections, normalization, smaller initialization, smaller learning rate, gradient clipping, or a different annealing/optimizer schedule.
4. Re-run depth 6 against depth 4 with more replicates if the goal is to decide whether depth 6 is meaningfully competitive. The current evidence says depth 6 is close but still worse on the mean.
5. Treat depth 40 as failed under the current recipe. It should not be used without architectural changes.

## Artifact Pointers

All artifacts for this suite are contained in:

`/home/rwl/ProblemBasedScenarioGeneration/experiments/temp/2026-05-05_training_data_depth`

Important files:

- `results/runs.csv`: one row per successful run.
- `results/epochs.csv`: per-epoch training and test target metrics.
- `results/test_per_sample.csv`: per-context test metrics.
- `results/checkpoints.csv`: model checkpoint manifest.
- `results/checkpoints/`: serialized model checkpoints.
- `results/baseline_summary.csv`: baseline aggregate.
- `results/data_amount_summary.csv`: training-data aggregate.
- `results/depth_summary.csv`: depth aggregate.
