# CERMICS LAVA timing replay

All 19 original-seed configurations completed 300 optimizer epochs and supply 36 entries across the linear and validation-selected timing tables. Execution finished November 8 at 02:55:07 Santiago, 16 h 37 min 58 s after the repaired Slurm job started. Regret estimates come from the original confirmations. [Completion receipt](evidence/COMPLETE.json), [frozen manifest and panel mapping](evidence/manifest.json)

SP25 N500 matched the reference and resident input tensors, loss, prediction gradients and all parameter gradients exactly on four original-order minibatches of 32, 32, 32, 20 observations. The first fit then took 1106.69 s of optimizer wall time and 1101.10 s of process CPU time. The initial parity check took 7.61 s. The gate released all five follow-on routes and retained the SP25 prefix on node07. [Parity](evidence/parity.json), [first fit](evidence/results/lava_complete__3612427e518ddbe8fdb8aa37929a8c00.json)

## Fit timings

Each row is one measured replay. L denotes the linear panel and S the validation-selected panel. Loader time is the one-time materialization of the resident arrays. Training covers the 300-epoch optimizer loop, with validation and test access disabled. The receipts also retain full setup clocks and the archived original fit identities. [Fit receipts](evidence/results)

| Route | N | Depth | Panels | Loader wall s | Loader CPU s | Training wall s | Training CPU s |
|---|---:|---:|---|---:|---:|---:|---:|
| SP10 | 100 | 0 | L + S | 0.15 | 0.15 | 7.10 | 7.07 |
| SP10 | 300 | 0 | L + S | 0.40 | 0.40 | 21.03 | 20.94 |
| SP10 | 500 | 0 | L + S | 0.63 | 0.63 | 35.23 | 35.07 |
| SP10 | 1000 | 0 | L + S | 1.18 | 1.17 | 71.72 | 71.40 |
| SP15 | 100 | 0 | L + S | 0.63 | 0.63 | 29.44 | 29.32 |
| SP15 | 300 | 0 | L + S | 1.99 | 1.98 | 88.69 | 88.34 |
| SP15 | 500 | 0 | L + S | 3.43 | 3.41 | 144.05 | 143.49 |
| SP15 | 1000 | 0 | L + S | 6.83 | 6.80 | 291.55 | 290.42 |
| SP20 | 100 | 0 | L + S | 1.94 | 1.93 | 98.69 | 98.33 |
| SP20 | 300 | 0 | L + S | 6.04 | 6.02 | 277.30 | 276.28 |
| SP20 | 500 | 0 | L + S | 10.20 | 10.17 | 464.81 | 463.07 |
| SP20 | 1000 | 0 | L + S | 20.71 | 20.63 | 910.67 | 907.27 |
| SP25 | 100 | 0 | L + S | 4.81 | 4.79 | 215.01 | 213.93 |
| SP25 | 300 | 0 | L + S | 14.69 | 14.60 | 653.95 | 650.67 |
| SP25 | 500 | 0 | L + S | 26.91 | 26.70 | 1106.69 | 1101.10 |
| SP25 | 1000 | 0 | L + S | 49.07 | 48.85 | 2215.79 | 2204.50 |
| CFL50 | 100 | 0 | L | 0.55 | 0.54 | 1132.75 | 1129.25 |
| CFL50 | 100 | 1 | S | 0.46 | 0.46 | 1132.66 | 1129.10 |
| KP1000 | 1000 | 0 | L + S | 18.72 | 18.63 | 14236.40 | 14173.64 |

The sum of the 19 optimizer clocks is 6.43 wall hours and 6.40 CPU hours across the parallel routes.

## Shared preparation

Seven measured stages cover six exact banks. SP25 first built N500 and then extended the saved prefix to N1000 on the same node and allocation. Each stage is charged once. The stage receipts record the wall and CPU clocks for input reconstruction, geometry and identity checks. The SP25 continuation includes the solver-order replay needed to restore the original construction. [Preparation receipts](evidence/preprocessing), [prefix and builder records](evidence/seeded_banks_v2)

| Bank | Prefix or extension | Input wall s | Geometry wall s | Geometry CPU s | Identity checks wall s |
|---|---|---:|---:|---:|---:|
| SP10 | N1000 | 1.38 | 2214.13 | 2207.27 | 0.00 |
| SP15 | N1000 | 1.40 | 8658.37 | 8614.60 | 0.30 |
| SP20 | N1000 | 1.66 | 21877.47 | 21785.06 | 0.65 |
| SP25 | N500 | 2.40 | 28410.97 | 28206.47 | 0.52 |
| SP25 | N500 to N1000 | 0.46 | 27146.55 | 26996.77 | 0.97 |
| CFL50 | N100 | 0.54 | 479.14 | 476.40 | 1.03 |
| KP1000 | N1000 | 1.73 | 285.42 | 283.72 | 0.01 |

The construction stages sum to 24.74 wall hours and 24.60 CPU hours across the routes. SP10 and KP1000 have original prefix metadata as their archived geometry identity evidence.

## Resources and closeout

Both nodes used Intel Xeon 6730P CPUs. Each numerical route used one bound core. The worker-phase RSS peaks cover setup, preparation and fits in the phase. The reported peak is the larger of the sampled process-tree peak and the child high-water mark. [Slurm and process snapshot](../snapshots/2026-11-08T112755.194887+0000.json), [runtime receipts](evidence/runtime)

| Route | Slurm job | Node | Phase | Process wall s | Actual CPU s | Peak RSS GiB | Guard GiB |
|---|---|---|---|---:|---:|---:|---:|
| SP10 | 19979 | node09 | remaining | 2365.44 | 2349.53 | 0.95 | 8 |
| SP15 | 19977 | node07 | remaining | 9236.61 | 9185.42 | 3.61 | 12 |
| SP20 | 19975 | node07 | remaining | 23679.37 | 23575.89 | 10.44 | 24 |
| SP25 | 19678 | node07 | first | 29565.73 | 29348.44 | 13.19 | 32 |
| SP25 | 19678 | node07 | remaining | 30310.55 | 30140.27 | 24.41 | 40 |
| CFL50 | 19978 | node09 | remaining | 2755.52 | 2741.11 | 4.53 | 20 |
| KP1000 | 19976 | node09 | remaining | 14552.81 | 14481.69 | 87.73 | 112 |

The six successful Slurm allocations total 112,469 allocated CPU-seconds (31.24 h). The seven finished worker phases account for 111,822.35 measured CPU-seconds (31.06 h). The node03 attempt 19671 stopped at scratch admission after two allocated CPU-seconds. Six one-minute-cap scratch probes informed the placement repair. [Failed admission](../overlays/01-node03-19671/ADMISSION_FAILURE.json), [scratch probe receipts](../SCRATCH_PROBE_READBACK.json)

The compact copy contains 195 checksummed files, including pinned source, the frozen configurations and seeds, all 19 timing receipts, parity, preparation and operational measurements. All six numerical jobs and both cleanup jobs are terminal. The shared root and the node07 and node09 scratch paths were removed after verification. The two cleanup jobs had a generic accounting name and were verified by exact submitted ID, node and exit status. [Retrieval receipt](RETRIEVAL_VERIFIED.json), [scratch release](SCRATCH_CLEANUP_VERIFIED.json), [shared root release](ROOT_CLEANUP_VERIFIED.json)
