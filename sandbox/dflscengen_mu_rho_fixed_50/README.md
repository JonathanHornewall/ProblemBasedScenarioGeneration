# DFLScenGen Fixed Mu/Rho Sandbox

This sandbox runs the resource-allocation DFLScenGen annealing experiment with
the data/model settings from
`src/ContextualDFL/ContextualDFLExperiments/experiments/resource_allocation_annealing/annealing.jl`.

Requested jobs:

- `mu = 0.001`, `rho = 0.0`, 50 epochs
- `mu = 0.0001`, `rho = 0.0`, 50 epochs
- `mu = 0.0`, `rho = 0.001`, 50 epochs
- `mu = 0.0`, `rho = 0.0001`, 50 epochs

All run artifacts are written under `results/<sweep-id>/`. The shared generated
training/test data is serialized under `ground_truth/` so all four jobs use the
same train/test set for the same seed.

Launch on `ibm-96c-2`:

```bash
sandbox/dflscengen_mu_rho_fixed_50/launch_remote_fixed.sh
```
