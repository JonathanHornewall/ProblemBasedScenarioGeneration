# Fixing Slow Training Experiment

Scope: isolated benchmark scripts and Julia project are in this directory only.

Command used for the final run:

```bash
FIX_SLOW_N_TRAIN=10 bash experiments/temp/fixing_slow_training/run_all.sh
```

Settings:

- `N_TRAIN=10`
- `EPOCHS=1`
- `BATCHSIZE=1`
- `MU_SURR=1.0`
- `MU_REF=1.0`
- model architecture matches the resource-allocation annealing scripts
- old prototype includes `scripts/resource_allocation_prototype/custom_code/neural_net.jl`

Results after each script's internal one-sample warmup:

| implementation | display mode | training seconds | iterations | seconds / iteration |
| --- | --- | ---: | ---: | ---: |
| old prototype | display off | 0.631640625 | 10 | 0.0631640625 |
| old prototype | prototype `display_iterations=true` | 1.262174416 | 10 | 0.1262174416 |
| current ContextualDFL | display off | 0.330561583 | 10 | 0.0330561583 |
| current ContextualDFL | annealing-style `display_smooth=true` | 0.720921167 | 10 | 0.0720921167 |

Impression:

- The current core training loop is not slower in this small run; it is about 1.9x faster per iteration than the old prototype core path.
- The old prototype's `display_iterations=true` path is slower than both current measured paths because it computes relative loss per mini-batch and plots.
- The current annealing-style display path roughly doubles the measured training time compared with current core training, because it precomputes reference losses and reports smooth display losses.
- The current `resource_allocation_annealing/annealing.jl` also runs SAA testing by default after training. The old prototype `annealing.jl` stops after training and saving. For end-to-end script runtime, that default testing block is likely the largest difference.
