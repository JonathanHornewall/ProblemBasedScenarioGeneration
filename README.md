# LAVA completion campaign, 2 November 2026

The campaign is active under [CONTRACT.json](CONTRACT.json). The absolute deadline remains **9 November 2026 04:27:29 UTC** (00:27 La Paz), with scientific cutoff 30 minutes earlier and at most 100 running plus pending campaign CPUs. Only CERMICS is authorized. Main owns scientific decisions; this Monitor owns remote mutations.

The latest completed impact reconciliation is [ops/FINAL_REPAIR_RECONCILIATION_20261102.json](ops/FINAL_REPAIR_RECONCILIATION_20261102.json). It retains **281 completed SP fits** after exact random-transition branch replay:160 development fits, 91 SP10 confirmations and30 SP15 confirmations. Four original seeded KP1000 recoveries completed 300 epochs. Seven completed CFL50 screens used affected unseeded geometry and are excluded from selection; their immutable receipts remain, with seven explicit replacement IDs in [ops/COORDINATOR_FINAL_READBACK.json](ops/COORDINATOR_FINAL_READBACK.json). No CFL selection had frozen. These are dated counts, not final campaign totals.

The shared-builder RNG fault is a correctness repair. A fit can use a legacy prefix only after the original LP/basis sequence proves the public random-transition branch unreachable, or use the original initialization seed and seeded-v2 cache. See [ops/SEED_REPAIR_DISPOSITION.json](ops/SEED_REPAIR_DISPOSITION.json), [ops/geometry_provenance.py](ops/geometry_provenance.py), [ops/safe_geometry.py](ops/safe_geometry.py), and [ops/bank_geometry.py](ops/bank_geometry.py). Seeded caches preserve NumPy state before/after each observation, replay the original LP objective sequence, and restore the exact saved RNG position. Proven random-free legacy anchors can be reused; unknown rows cannot. The focused interrupted/uninterrupted fixture passed with exact RNG states and rows; its caches never enter science. A final entry-point check also verifies original initialization seeds 201,8201 and161101.

Largest-CFL qualification used the original seed17 training observations 0/1, native physical LP relaxation, HiGHS, cap 250 and exact duplicates. The qualifier did not initialize NumPy's random transition RNG. The following remain factual measured computational attempts, with that reproducibility limitation; no expensive rerun is authorized merely to certify them.

| Route/sample | Wall min | Process CPU min | Peak RSS GiB | Raw vertex GiB | Outcome |
|---|---:|---:|---:|---:|---|
| CFL200/0 |24.04|18.71|22.40|1.34|Completed |
| CFL200/1 |19.13|18.73|23.98|1.16|Completed |
| CFL250/0 |69.90|36.49|28.57|1.78|Completed |
| CFL250/1 |32.08|31.54|28.63|1.12|Completed |
| CFL500/0 |46.64|45.89|247.96|—|SIGKILL; cause unproven |
| CFL500/1 |57.07|56.74|220.24|—|Explicit memory-headroom censor |

Evidence is [monitor_snapshots/1793647367.json](monitor_snapshots/1793647367.json) and [ops/HEARTBEAT_20261102_DETAIL.json](ops/HEARTBEAT_20261102_DETAIL.json). Extrapolating 1000 distinct anchors from these two CFL200/250 rows gives over 1 TiB of **uncompressed** vertex storage per bank. This is not a lower bound or disk-infeasibility proof: repeated anchors share files, normals use lossless compression, and vertices are lazily mapped. No broad compression project is part of this campaign. CFL200/250/500 remain pending admission; independent work continues.

The 19:58 UTC resource reconciliation had 51 allocated CPUs and 22 numerical children at roughly 98–99% CPU each. Allocations and actual work are reported separately. SP15's ten N100 banks took944–990 s and at most 44.3 MB of vertex files; original larger banks are admitted with existing time/RSS/storage guards. SP20 also advances from its ten measured N100 banks; SP25/50 remain at bounded N100 admission. CFL100 has a seeded first-paired N100 bank probe; it is not a completed fit. [ADMISSION.json](ADMISSION.json) records the current boundary.

Operational instructions, exact jobs, recovery rules and terminal cleanup are in [ops/HANDOFF.md](ops/HANDOFF.md). The single heartbeat is every four hours with a finite count before the unchanged deadline. No IntOpt, electricity, cloud, new methods, extra tuning seeds, or test-driven selection.

The live serial KP bank is proved random-free through all 1000 supplied training observations, so its in-flight fit is retained; see [ops/KP_CONSUMED_PREFIX_PROVENANCE.json](ops/KP_CONSUMED_PREFIX_PROVENANCE.json). Current job/admission ownership and the passing final CLI/resume check are saved in [ops/CURRENT_OPERATIONAL_STATE.json](ops/CURRENT_OPERATIONAL_STATE.json).

The final request is52 CPUs running plus pending (51 running,1 pending in the saved operational receipt), within the100-CPU envelope. The additional node09 queue lets the diagnosed CFL50 repair progress independently of the active CFL100 probe.
