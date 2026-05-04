# Quadratic Smoothing Annealing Sandbox

This sandbox runs the resource-allocation `annealing.jl` workflow with quadratic
smoothing without editing the package source or the canonical experiment
template.

Default sweep:

- baseline: `rho = 0.0`, one replicate
- smoothing choices: `rho = 0.01`, `0.05`, `0.1`, three replicates each
- testing objective uses `CDFL_SANDBOX_OPTIMALITY_RHO=0.0` by default, so policy
  performance is compared on the original objective while policy inference uses
  the training `rho`.

Artifacts are written under:

```text
src/ContextualDFL/ContextualDFLExperiments/experiments/resource_allocation_annealing/results/quadratic_smoothing_sandbox/<sweep-id>/
```

That `results/` directory is already excluded by `~/sync-julia-code.sh`, so the
remote artifacts stay isolated from normal code syncs.

Run from the local repo:

```bash
sandbox/quadratic_smoothing_annealing/launch_remote_sweep.sh
```

Useful overrides:

```bash
CDFL_SANDBOX_MAX_PARALLEL=3 \
CDFL_SANDBOX_SWEEP_ID=my-rho-sweep \
sandbox/quadratic_smoothing_annealing/launch_remote_sweep.sh
```

For a short remote smoke run:

```bash
CDFL_SANDBOX_TRAINING_SAMPLES=2 \
CDFL_SANDBOX_TESTING_SAMPLES=1 \
CDFL_SANDBOX_TEST_CONTEXTS=1 \
CDFL_SANDBOX_TESTING_SPLITS=1 \
CDFL_SANDBOX_XI_PER_X=1 \
CDFL_SANDBOX_PARAM_LIST=0.01 \
CDFL_SANDBOX_FIRST_STAGE_EPOCHS=1 \
CDFL_SANDBOX_DEFAULT_EPOCHS=1 \
CDFL_SANDBOX_RHOS=0.01 \
CDFL_SANDBOX_BASELINE_RHO= \
sandbox/quadratic_smoothing_annealing/launch_remote_sweep.sh
```
