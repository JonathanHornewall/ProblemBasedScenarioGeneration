# Impacts of Data Cardinality, Network Depth, Width, and Annealing Schedules

This report summarizes the completed resource-allocation annealing sweep in:

`/home/rwl/ProblemBasedScenarioGeneration/sandbox/resource_allocation_annealing_method_sweep`

Primary CSV outputs:

- `results/decisions.csv`
- `results/final_selection.csv`
- `results/runs.csv`
- `results/epochs.csv`
- `results/test_per_sample.csv`
- `phases/full/*/summary.csv`
- `phases/full/*/screening_summary.csv`

## Executive Summary

The best configuration written by the staged suite was:

| Parameter | Value |
|---|---:|
| Training contexts | 100 |
| Epochs | 130 |
| Final fine-tuning epochs | 10 |
| Depth | 3 |
| Hidden width | 32 |
| Activation | relu |
| Schedule kind | piecewise_linear |
| starting_mu | 1.0 |
| ending_mu | 0.05 |
| piece_length | 10 |
| nr_pieces | 11 |
| starting_phase_length | 20 |
| fine_tuning_phase_length | 10 |

This is the best **piecewise-linear** configuration selected by the final phase.
However, the strongest overall evidence in the suite is for the **geometric**
annealing schedule found in phase 4:

| Configuration | Mean average test loss |
|---|---:|
| Geometric schedule, `1.0 -> 0.01` | 0.038311 |
| Best piecewise-linear, `ending_mu=0.05` | 0.049345 |

Therefore:

- If the implementation can use any schedule shape, use the geometric schedule.
- If the implementation must remain piecewise-linear, use the `ending_mu=0.05`
  piecewise schedule.

## Methodological Notes

The suite originally started as a full 6-replicate sweep, then was updated to a
staged design:

1. Run short screening trials.
2. Keep the top candidates.
3. Run 6 full-budget replicates only for finalists.

The full finalist results are the trustworthy comparisons. Screening results are
useful for direction but were noisy enough to mis-rank several candidates.

Important bookkeeping details:

- The test cache was precomputed once and reused throughout.
- The full test cache contains 100 test data points.
- All completed rows in `results/runs.csv` have status `ok`.
- `results/runs.csv` contains 122 successful rows total.
- 114 rows correspond to the current staged design.
- 8 successful depth rows are legacy rows from before the staged redesign and
  should be ignored for current phase conclusions.
- No early stopping was actually active in the completed run. There is no
  `results/early_checks.csv`, and screening configs have
  `early_stop_enabled=false`.

## Overall Phase Decisions

From `results/decisions.csv`:

| Phase | Selected candidate | Mean average test loss | Successful full reps |
|---|---:|---:|---:|
| data_amount | n100 | 0.057696 | 6 |
| depth | depth3 | 0.058084 | 6 |
| width | width32 | 0.040422 | 6 |
| schedule_shape | geometric | 0.038311 | 6 |
| piecewise_linear | higher_end_mu | 0.049345 | 6 |

The final row in `results/final_selection.csv` uses the phase-5
piecewise-linear result. That is a valid answer for the piecewise-linear family,
but it should not be read as beating the geometric schedule.

## Phase 1: Impact of Data Cardinality

The data-cardinality experiment compared 100, 500, and 1000 training contexts
under normalized compute. More training contexts received fewer epochs:

| Candidate | Contexts | Epochs | Mean loss | Std | Min | Max |
|---|---:|---:|---:|---:|---:|---:|
| n100 | 100 | 130 | 0.057696 | 0.030920 | 0.028413 | 0.110564 |
| n500 | 500 | 26 | 0.080285 | 0.049586 | 0.028548 | 0.135453 |
| n1000 | 1000 | 13 | 0.132048 | 0.136754 | 0.010845 | 0.314806 |

Per-replicate losses:

| Candidate | Replicate losses |
|---|---|
| n100 | 0.058932, 0.110564, 0.029335, 0.046902, 0.028413, 0.072031 |
| n500 | 0.135453, 0.031827, 0.120483, 0.046658, 0.028548, 0.118739 |
| n1000 | 0.058627, 0.294095, 0.314806, 0.092244, 0.021671, 0.010845 |

Interpretation:

The 100-context setting performed best under the normalized compute budget. This
does not necessarily mean less data is intrinsically better. It means that, with
this optimizer, schedule, batch setup, and fixed compute normalization, the
larger-data settings appear undertrained. The 1000-context setting is especially
unstable because it only receives 13 epochs.

Training-loss behavior supports this interpretation:

| Candidate | First train loss mean | Last train loss mean |
|---|---:|---:|
| n100 | 1888.536 | 887.873 |
| n500 | 1158.685 | 901.117 |
| n1000 | 1056.758 | 986.350 |

The 1000-context case starts lower but improves the least, consistent with
insufficient epochs rather than a clearly inferior data regime.

## Phase 2: Impact of Network Depth

Depth screening tested 3, 4, 5, 10, and 20 layers. The two finalists were
`depth10` and `depth3`.

Screening results:

| Candidate | Screening mean | Screening std |
|---|---:|---:|
| depth10 | 0.045083 | 0.012651 |
| depth3 | 0.052808 | 0.037035 |
| depth5 | 0.110139 | 0.043718 |
| depth20 | 0.117339 | 0.063240 |
| depth4 | 0.136492 | 0.042503 |

Full finalist results:

| Candidate | Mean loss | Std | Min | Max |
|---|---:|---:|---:|---:|
| depth3 | 0.058084 | 0.022244 | 0.034688 | 0.096187 |
| depth10 | 0.079801 | 0.035044 | 0.033361 | 0.116775 |

Per-replicate losses:

| Candidate | Replicate losses |
|---|---|
| depth3 | 0.096187, 0.034688, 0.061149, 0.067433, 0.042263, 0.046781 |
| depth10 | 0.033361, 0.116775, 0.109949, 0.083046, 0.041275, 0.094398 |

Interpretation:

The screening stage slightly favored depth 10, but the full 6-replicate run
favored depth 3. The deeper model has a better best replicate, but its typical
behavior is worse and more variable. The 3-layer model is the better default for
this problem and compute budget.

## Phase 3: Impact of Hidden Width

Width screening tested 32, 64, 128, 256, and 512 hidden units. The two
finalists were `width512` and `width32`.

Screening results:

| Candidate | Screening mean | Screening std |
|---|---:|---:|
| width512 | 0.034089 | 0.010802 |
| width32 | 0.049926 | 0.012207 |
| width128 | 0.068899 | 0.051351 |
| width256 | 0.098782 | 0.052246 |
| width64 | 0.109991 | 0.100235 |

Full finalist results:

| Candidate | Mean loss | Std | Min | Max |
|---|---:|---:|---:|---:|
| width32 | 0.040422 | 0.015134 | 0.023478 | 0.065623 |
| width512 | 0.055088 | 0.013133 | 0.043437 | 0.073737 |

Per-replicate losses:

| Candidate | Replicate losses |
|---|---|
| width32 | 0.065623, 0.034008, 0.023478, 0.033431, 0.035264, 0.050731 |
| width512 | 0.048371, 0.045869, 0.073737, 0.043437, 0.049341, 0.069775 |

Interpretation:

The full runs reverse the screening ranking. Width 512 looked best in the short
screening stage, but width 32 won after full-budget training. This is a strong
warning that short screening is useful for pruning but not reliable enough to
make final decisions. The smaller model is also much cheaper:

| Candidate | Mean training seconds per full run | Mean evaluation seconds |
|---|---:|---:|
| width32 | 2025.650 | 219.036 |
| width512 | 3574.626 | 256.531 |

Width 32 is both better and faster in this suite.

## Phase 4: Impact of Annealing Schedule Shape

Schedule-shape screening tested:

- `piecewise_linear`
- `linear`
- `geometric`
- `cosine`
- `delayed_quadratic`
- `early_quadratic`

The two finalists were `cosine` and `geometric`.

Screening results:

| Candidate | Screening mean | Screening std |
|---|---:|---:|
| cosine | 0.043375 | 0.007863 |
| geometric | 0.058092 | 0.010780 |
| delayed_quadratic | 0.065974 | 0.054188 |
| piecewise_linear | 0.066936 | 0.012321 |
| early_quadratic | 0.239676 | 0.173831 |
| linear | 0.312194 | 0.324346 |

Full finalist results:

| Candidate | Mean loss | Std | Min | Max |
|---|---:|---:|---:|---:|
| geometric | 0.038311 | 0.013028 | 0.026509 | 0.058334 |
| cosine | 0.123345 | 0.137430 | 0.025369 | 0.379518 |

Per-replicate losses:

| Candidate | Replicate losses |
|---|---|
| geometric | 0.058334, 0.043932, 0.028529, 0.026983, 0.026509, 0.045580 |
| cosine | 0.379518, 0.035083, 0.025369, 0.027447, 0.105600, 0.167053 |

Per-sample test-regret distribution:

| Candidate | Mean | Median | P90 | P95 | Max |
|---|---:|---:|---:|---:|---:|
| geometric | 0.038311 | 0.021658 | 0.080968 | 0.123422 | 0.509116 |
| cosine | 0.123345 | 0.084416 | 0.364231 | 0.410499 | 0.652794 |

Interpretation:

The geometric schedule is the strongest result in the entire sweep. It is not
only better on mean loss, but also more stable across replicates and has much
better tail behavior on the test set. Cosine can produce excellent runs, but it
also produced severe failures.

The geometric schedule starts at `mu=1.0` and decreases multiplicatively toward
`mu=0.01`. Its preview begins:

`1.0, 0.902725, 0.814913, 0.735642, 0.664083, ...`

This means it moves away from the high-mu regime much earlier than the default
piecewise-linear schedule.

## Phase 5: Impact of Piecewise-Linear Schedule Parameters

Piecewise-linear screening tested:

- `default`
- `lower_end_mu`
- `higher_end_mu`
- `short_start`
- `long_start`
- `more_pieces`
- `fewer_pieces`
- `long_finetune`

The two finalists were `higher_end_mu` and `lower_end_mu`.

Screening results:

| Candidate | Screening mean | Screening std |
|---|---:|---:|
| higher_end_mu | 0.060575 | 0.017497 |
| lower_end_mu | 0.095057 | 0.047885 |
| long_start | 0.099518 | 0.084205 |
| default | 0.119352 | 0.067239 |
| fewer_pieces | 0.142962 | 0.032356 |
| long_finetune | 0.150045 | 0.159927 |
| short_start | 0.223892 | 0.122538 |
| more_pieces | 0.241033 | 0.195871 |

Full finalist results:

| Candidate | ending_mu | Mean loss | Std | Min | Max |
|---|---:|---:|---:|---:|---:|
| higher_end_mu | 0.05 | 0.049345 | 0.018560 | 0.023637 | 0.076773 |
| lower_end_mu | 0.001 | 0.071240 | 0.064025 | 0.025517 | 0.165765 |

Per-replicate losses:

| Candidate | Replicate losses |
|---|---|
| higher_end_mu | 0.057494, 0.034604, 0.023637, 0.049179, 0.054384, 0.076773 |
| lower_end_mu | 0.032134, 0.025517, 0.035175, 0.140516, 0.028331, 0.165765 |

Per-sample test-regret distribution:

| Candidate | Mean | Median | P90 | P95 | Max |
|---|---:|---:|---:|---:|---:|
| higher_end_mu | 0.049345 | 0.031026 | 0.099547 | 0.141649 | 0.663395 |
| lower_end_mu | 0.071240 | 0.035305 | 0.174524 | 0.219223 | 0.683943 |

Interpretation:

Among piecewise-linear schedules, a higher terminal `mu` is better. Driving
`mu` down to `0.001` produced some strong runs, but it also produced unstable
replicates with high regret. The best piecewise setting keeps `ending_mu=0.05`,
which appears to avoid the worst tail behavior.

The result also implies that the default `ending_mu=0.01` may be too aggressive
for the current training procedure.

## Cross-Phase Interpretation

The most important nuance is that phase 4 selected `geometric`, but phase 5 then
searched piecewise-linear schedules and wrote a final piecewise-linear config.
This makes the final suite output a bit ambiguous:

- `results/final_selection.csv` gives the best configuration after forcing the
  final phase back to `piecewise_linear`.
- The best overall measured schedule remains `geometric`.

Direct comparison:

| Candidate | Mean loss | Std | Median per-sample regret | P95 per-sample regret |
|---|---:|---:|---:|---:|
| geometric | 0.038311 | 0.013028 | 0.021658 | 0.123422 |
| piecewise_linear, ending_mu=0.05 | 0.049345 | 0.018560 | 0.031026 | 0.141649 |

Paired replicate comparison, using matching replicate seeds, gives a mean
difference of about `0.0110` in favor of geometric. With only 6 replicates this
is not definitive in a strict statistical sense, but it is consistent with the
mean, variance, and per-sample tail metrics.

## Training Dynamics

For successful full finalist attempts, the average final training losses were:

| Phase/Candidate | First train loss mean | Last train loss mean |
|---|---:|---:|
| data n100 | 1888.536 | 887.873 |
| data n500 | 1158.685 | 901.117 |
| data n1000 | 1056.758 | 986.350 |
| depth3 | 1905.385 | 903.336 |
| depth10 | 1377.993 | 920.691 |
| width32 | 2460.231 | 877.717 |
| width512 | 1420.206 | 895.271 |
| geometric | 2460.952 | 878.941 |
| cosine | 2460.673 | 945.091 |
| higher_end_mu | 2458.238 | 888.308 |
| lower_end_mu | 2460.861 | 908.078 |

The best test configurations generally also achieve lower final training losses,
but the relationship is not perfect. For example, the data-cardinality phase
shows that lower initial training loss in larger datasets does not compensate
for insufficient training epochs under the normalized budget.

## Reliability and Limitations

The conclusions are useful but should be read with these limitations:

1. The staged design did not give every candidate 6 full-budget replicates.
   Only finalists received full 6-replicate evaluation.

2. Screening was noisy. It mis-ranked width and schedule shape:
   `width512` screened better than `width32`, but lost in full runs;
   `cosine` screened better than `geometric`, but lost badly in full runs.

3. The data-cardinality conclusion is conditional on normalized compute. It is
   not evidence that 100 data points would beat 1000 data points if both were
   trained to convergence.

4. Early stopping was requested but not active in the completed run. The current
   evidence does not evaluate the effect of early stopping.

5. The final piecewise-linear selection should not be treated as the best global
   schedule unless the implementation is required to remain piecewise-linear.

## Recommendations

Recommended best overall configuration:

```text
training_contexts = 100
epochs = 130
final_epochs = 10
depth = 3
hidden_size = 32
activation = relu
schedule_kind = geometric
starting_mu = 1.0
ending_mu = 0.01
```

Recommended best piecewise-linear configuration:

```text
training_contexts = 100
epochs = 130
final_epochs = 10
depth = 3
hidden_size = 32
activation = relu
schedule_kind = piecewise_linear
starting_mu = 1.0
ending_mu = 0.05
piece_length = 10
nr_pieces = 11
starting_phase_length = 20
fine_tuning_phase_length = 10
```

Recommended follow-up experiments:

1. Re-run the geometric schedule against the best piecewise-linear schedule with
   more than 6 replicates.
2. Revisit data cardinality without strict normalized epochs, especially 500 and
   1000 contexts with enough epochs to converge.
3. Test geometric schedules with different `ending_mu` values, especially
   `0.05`, `0.02`, `0.01`, and `0.005`.
4. Repeat width comparison around the winning small model: 16, 24, 32, 48, 64.
5. Fix or re-enable early stopping only after deciding whether it should apply
   to screening, full finalist runs, or both.

