# MLflow Grid Search Analysis: `gridsearch_1777943178038`

Run ID: `17a04f1251414442a64da311c8e2f8c5`  
Experiment ID: `4`  
Experiment name: `ContextualDFLTraining/resource_allocation_experiment_1`

## Summary

The MLflow hierarchy for this grid search has three layers:

- 1 grid parent run: `17a04f1251414442a64da311c8e2f8c5`
- 48 config-parent runs, one for each hyperparameter candidate
- 240 repeat/training runs, five repeats per candidate

The parent and config-parent runs are still marked `RUNNING`. That appears to be a logging/coordinator issue rather than evidence that useful training is still ongoing: the coordinator was terminated before it closed those parent runs and logged final aggregate metrics.

For performance analysis, only repeat runs with final evaluation metrics should be counted.

```text
FINISHED repeat runs with final eval metrics: 139
FAILED repeat runs without final eval:        98
stale RUNNING repeat runs:                    3
total repeat runs:                          240
```

## Experiment Context

This is the resource allocation experiment. The learned model maps 3 context features to 30 demand outputs. Training uses `DflScenLoss`: predicted demand scenarios are decoded, a stochastic program is solved under those predicted scenarios, and the resulting first-stage decision is scored against reference scenarios.

Final reporting includes both prediction metrics and decision-quality metrics. The important evaluation path is:

- train and report metrics in `ContextualDFLTraining/src/train_run.jl`
- compare policy decisions against precomputed optima in `ContextualDFLExperiments/src/testing/evaluation/evaluation.jl`
- compute regret as policy value minus optimal value

The grid searched:

- activation: `relu`, `silu`, `geelu`
- depth: `2`, `3`, `4`, `5`
- mu schedule: `piecewise`, `linear`, `geometric`, `exponential`
- repeats: `5` per candidate

Important implementation detail: `geometric` and `exponential` are equivalent in the code, because both resolve to the same log-spaced schedule.

## Best Runs

Best single repeat by test regret:

```text
candidate:            13
activation:           relu
depth:                5
schedule:             piecewise
repeat:               2
training data seed:   450313993
test_regret_mean:     13.94
relative_regret_mean: 1.53%
test_mse:             500.93
```

This is the best single run, but it is not the best basis for choosing a configuration because repeat variability is large.

Best candidate mean over successful repeats:

```text
candidate:            11
activation:           relu
depth:                4
schedule:             geometric
successful repeats:   2 / 5
mean test regret:     24.09
relative regret:      2.39%
```

This is the lowest mean regret, but it only has 2 successful repeats, so it is statistically thin.

Best fully observed candidate, with all 5 repeats successful:

```text
candidate:            43
activation:           geelu
depth:                4
schedule:             geometric
successful repeats:   5 / 5
mean test regret:     36.86
relative regret:      3.73%
test_mse:             337.23
validation_mse:       277.72
```

Best candidate with at least 4 successful repeats:

```text
candidate:            8
activation:           relu
depth:                3
schedule:             exponential/log
successful repeats:   4 / 5
mean test regret:     33.24
relative regret:      3.40%
test_mse:             290.06
validation_mse:       202.91
```

## Main Effects

Depth had the largest impact on regret.

Candidate-level main-effect ranges:

```text
depth:      regret range 16.38, eta2 0.348
activation: regret range 13.44, eta2 0.225
schedule:   regret range  8.02, eta2 0.078
```

Candidate-mean regret by depth:

```text
depth 4: 38.63
depth 3: 44.69
depth 5: 52.81
depth 2: 55.01
```

Candidate-mean regret by activation:

```text
relu:  40.62
geelu: 46.39
silu:  54.06
```

Candidate-mean regret by effective schedule:

```text
piecewise:                 44.85
log/geometric/exponential: 45.84
linear:                    51.86
```

The practical conclusion is that depth `3` to `4` is the strongest region, with depth `4` best for regret. `relu` and `geelu` are more promising than `silu`. Linear mu annealing looks weaker for regret than piecewise or log schedules.

## Metric Alignment

Regret and relative regret agree almost perfectly, as expected.

```text
test_relative_regret_mean vs test_regret_mean:
Pearson  0.996
Spearman 0.994
```

MSE does not explain regret in this grid:

```text
test_mse vs test_regret_mean:
Pearson  -0.066
Spearman -0.081

validation_mse vs test_regret_mean:
Pearson  -0.080
Spearman -0.128
```

This is one of the most important findings. Better demand prediction does not reliably imply better decisions in this setup.

Examples:

- Candidate 34 has excellent validation MSE, `100.55`, but poor mean regret, `61.38`.
- Candidate 43 has worse MSE, `337.23`, but much better regret, `36.86`.

Training loss has a weak positive association with regret:

```text
loss vs test_regret_mean:
Pearson  0.253
Spearman 0.332
```

So loss is more directionally useful than MSE here, but still not strong enough to replace direct regret evaluation.

## Prediction Metrics Tradeoff

Depth `2` is best for MSE and tolerance accuracy but worst for regret:

```text
depth 2 candidate-mean test_mse: 173.13
depth 2 candidate-mean tolerance accuracy: 0.462
depth 2 candidate-mean regret: 55.01
```

Depth `4` is best for regret but not for MSE:

```text
depth 4 candidate-mean test_mse: 392.53
depth 4 candidate-mean tolerance accuracy: 0.301
depth 4 candidate-mean regret: 38.63
```

This reinforces that regret should be treated as the primary model-selection criterion for this experiment.

## Failure And Completeness Notes

The following candidates had zero successful final evaluations:

```text
candidate 1:  relu, depth 2, piecewise
candidate 4:  relu, depth 2, exponential
candidate 7:  relu, depth 3, geometric
candidate 16: relu, depth 5, exponential
candidate 20: silu, depth 2, exponential
candidate 24: silu, depth 3, exponential
```

Three stale `RUNNING` repeat runs were all candidate `48`:

```text
candidate 48: geelu, depth 5, exponential
repeat 3: last logged step 36
repeat 4: last logged step 33
repeat 5: last logged step 41
```

Many failed runs logged all 120 epochs but never logged final evaluation metrics. Those should not be included in regret or MSE rankings.

## Evaluation Caveat

Successful runs used either 10 or 20 test evaluation batches:

```text
test_evaluation_batches = 10: 51 runs
test_evaluation_batches = 20: 88 runs
```

The corresponding optimal test means differ slightly:

```text
10 batches: 934.894
20 batches: 934.631
```

The offset is small, about `0.263`, compared with the main regret differences, but it should be fixed before using this as a publication-quality comparison.

## Recommendation

Use regret, not validation MSE, as the primary selection metric.

The strongest practical candidates are:

```text
candidate 8:  relu,  depth 3, exponential/log, 4 successful repeats, mean regret 33.24
candidate 43: geelu, depth 4, geometric/log,    5 successful repeats, mean regret 36.86
candidate 39: geelu, depth 3, geometric/log,    5 successful repeats, mean regret 40.08
candidate 5:  relu,  depth 3, piecewise,        5 successful repeats, mean regret 40.03
```

For a follow-up sweep, focus on:

- depth `3` and `4`
- activations `relu` and `geelu`
- piecewise and log-spaced mu schedules
- consistent final evaluation settings across every repeat

Avoid choosing based on validation MSE alone; it selects shallow models that predict demand well but make worse resource-allocation decisions.
