# Converted-Q and RA Cost-Decoder Add-On

## Results

| Problem | Method | Target / decoder | Reps | Regret mean | Regret var | Rel. regret mean | Rel. regret var | Fit s mean |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| random_yield | SPO+ | converted q | 2 | 4.5940 | 0.000680 | 0.3340 | 2.56e-7 | 138.38 |
| random_yield | DFLScen | converted q | 2 | 9.3747 | 0.3164 | 0.6674 | 0.00291 | 177.62 |
| resource_allocation | DFLScen | original cost | 2 | 3291.4761 | 3.67e-18 | 3.9078 | 4.90e-24 | 1728.82 |
| resource_allocation | DFLScen | full cost | 2 | 3294.6857 | 0.0993 | 3.9117 | 1.48e-7 | 1416.66 |
| resource_allocation | DFLScen | physical cost | 2 | 3294.9175 | 0.0797 | 3.9120 | 1.18e-7 | 1427.47 |
| resource_allocation | DFLScen | economic cost | 2 | 3294.9175 | 0.0797 | 3.9120 | 1.18e-7 | 1437.35 |

## Minimal Configuration

- Code entry points: `run_baselines.jl`, `aggregate_tiny_full_baselines.jl`, `run_tiny_full_baselines.sh`.
- Data protocol: tiny artifacts, `train_contexts=100`, `train_scenarios_per_context=1`, `test_contexts=30`, `test_scenarios_per_context=500`, `evaluation_batches=5`, artifact seed `20260505`.
- Replications for this add-on: replica seeds `20260505` and `20260506`; table reports sample variance across replicas.
- Solver stack: `ContextualDFL.Solver(IpoptSolver(), HiGHSSolver())`; test optima from generated artifacts and `solve_dataset_to_optimality`; q-conv runs used `ipopt_max_iter=10000` and `constraint_tolerance=1e-8`.
- Neural training: ReLU MLP, depth `3`, hidden width `128`, Adam, learning rate `1e-3`, batch size `1`, 50 epochs.
- Random-yield instance: `RandomYieldProblem(r=5, a=10, K_support=5)`, context dimension `3`, q dimension `20`.
- Random-yield conversion: labels from converted-q data with full base scenario, `conversion_mu=1e-4`, `conversion_rho=0`, lower-bound margin `1e-4`, decoded by `LowerBoundedQDecoder`; `spoplus_qconv` uses `SPOPlusLoss`, `dfl_qconv` uses `DflScenLoss` with the standard schedule.
- Resource-allocation instance: default `ResourceAllocationProblem`, `20` resources, `30` demands, q dimension `680`; all cost-decoder runs use fixed demand `ones(30)`, one decoded scenario, `rho=1e-3`, and DFLScen.
- Resource-allocation decoder widths: original cost `30`, physical cost `630`, economic cost `630`, full cost `680`.
