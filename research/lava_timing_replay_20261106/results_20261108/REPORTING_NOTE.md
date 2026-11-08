# Completed LAVA timing replays

All 19 original winning configurations completed 300 optimizer epochs on CERMICS on 8 November 2026. They cover 36 cells across the linear and validation-selected tables. Each timing is one replay using the first original confirmation seed bundle: data151101, initialization161101, shuffle171101, perturbation181101 and structure101. These are not ten-seed timing averages.

The resident loader loads each distinct original vertex and binding-normal array once. It preserves the public collation, padding, LAVA loss and Adam loop. SP25 N500 passed exact comparisons of all collated tensors, loss, prediction gradients and model-parameter gradients on four original-order minibatches (32,32,32,20). Its complete timing fit passed before the remaining18 were admitted. The source, seeds, parameters and runtime are retained in the compact evidence.

Training wall time covers minibatch construction, forward and backward propagation and optimizer updates for all300 epochs. Preprocessing, one-time array loading, parity checking, validation, testing and checkpoint persistence are outside that clock and recorded separately. Regret tables continue to use the original confirmation runs. Do not attach their ten-seed uncertainty estimates to these one-seed timings.

`lava_training_replay_19.csv` contains the new measurements and the recorded historical loader times. The old/new ratios compare separate runs, not a controlled loader-only microbenchmark. `training_times_300_epochs_corrected.csv` replaces exactly36 LAVA timing cells in the preceding full table; other rows and the historical source table are unchanged. Updated cells have measured_fits=1 and no between-seed standard deviation.

Shortest-path training is much faster in these replays: SP25 N500 is18.44min versus134.48min previously recorded. Its fresh geometry construction is7.89h separately. SP25 N1000 training is36.93min. KP1000 N1000 still needs3.95h of training and the worker phase peaks at87.73GiB. All runs used single bound Intel Xeon6730P cores.

See `../ops/cermics_20261107/closeout/REPORT.md` for all19 rows, shared-preprocessing clocks and resource measurements. All owned jobs and temporary remote paths are released; the finite monitor is deleted. No new regret evaluation or tuning was performed.
