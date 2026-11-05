# LAVA completion: results, timing and resource limits

Execution closed at **2026-11-04T19:51:27.847678+00:00** with **670 valid successful fits +40 accepted CFL100 resource censors =710 required outcomes covered**. All active fits drained. This is not710 successful fits. The terminal check found zero numerical processes, zero actual numerical CPU use and no reservations. Both remaining campaign allocations16726/16740 were released once and verified absent; no scientific CPU remains allocated. The heartbeat was paused before this compilation.

| Required route | Valid successes | Accepted resource censors |
|---|---:|---:|
| cfl_100x10 | 0 | 40 |
| cfl_50x5 | 104 | 0 |
| knapsack_1000x2 | 6 | 0 |
| shortest_path_10x10 | 260 | 0 |
| shortest_path_15x15 | 100 | 0 |
| shortest_path_20x20 | 100 | 0 |
| shortest_path_25x25 | 100 | 0 |

SP successes include160 original development screens plus400 confirmations; CFL50 includes64 screens plus40 confirmations. The6 KP rows are the six authorized recovery fits, not a fresh ten-seed KP comparison. The original paired seeds, complete16-point grids,14 frozen selections, native physical relaxation, HiGHS1.12.0, cap250/exact duplicates and original loss/batch/300epoch settings were preserved. Validation alone selected configurations and checkpoints. No CFL100-specific tuning occurred. SP50, CFL200/250/500 and larger-N CFL are scope exclusions; completed excluded measurements remain supplementary and cannot extend execution.

[All710 outcomes and per-fit seeds/results/times](artifacts/required_outcomes.csv), [frozen configurations](artifacts/selected_configurations.csv), [existing224 validation-table entries](artifacts/existing_validation_tables.csv), [confirmation summaries](artifacts/confirmation_summary.csv), [machine-readable summary](artifacts/final_summary.json). Confirmation means/medians/ranges summarize existing per-seed LAVA test results and do not change selection. Ten paired confirmations populate each retained complete SP/CFL50 configuration group. CFL100 has no measured test regret; a missing value is never zero, a method win or a CP-SPO loss. The six KP recoveries alone do not estimate the full original KP benchmark.

## Measured timing and resource accounting

| SP25 training N | Successful fits | Whole wall minutes | Whole process CPU minutes | Whole process peak RSS GiB |
|---|---:|---:|---:|---:|
| 100 | 40 | 26.77–31.39 | 26.65–31.29 | 1.58–1.64 |
| 300 | 20 | 78.41–88.20 | 78.13–87.89 | 1.62–1.69 |
| 500 | 20 | 131.00–146.62 | 130.40–146.14 | 1.64–1.71 |
| 1000 | 20 | 261.83–277.29 | 260.88–276.40 | 1.70–1.74 |

The table uses fit-reported whole-process clocks. Historical dispatcher attempt clocks additionally include the guard wrapper; these are overlapping measurements, not separate costs. The ten SP25 N1000 continuation attempts use their original exact seeds/RNG checkpoints. Initial N100 builds and replay/continuation are distinct attempts; completed nested prefixes are views of a shared build and are not charged again. [Every recorded attempt](artifacts/attempt_costs.csv) and [external shared-build attempts](artifacts/shared_build_attempt_costs.csv) retain failures, holds and replay costs. In-process construction, metadata cache loads, optimizer, validation and physical test clocks are separate columns in the per-fit file. Lazy row I/O remains inside optimizer clocks.

Valid fit processes consumed **379.58 measured core-hours** including process overhead and any in-process construction; their optimizer portion was **373.69 core-hours**. These two figures overlap and must not be added. External build attempts are separate; whole-attempt and cumulative prefix clocks overlap and must not be added. The11 worker-step-loss attempts add a measured CPU **lower bound3.13 core-hours**; their terminal rusage is unavailable and two starts were unsampled. A complete campaign CPU total is therefore not claimed. Actual CPU consumption and Slurm allocation are distinct; the original100 running-plus-pending campaign CPU envelope was never increased.

All ten original CFL100 N100 banks completed; final preparation attempts peaked56.7–65.0GiB, with earlier failed/resumed costs retained. All40 original fits were accepted resource-censored before a completed epoch under the unchanged12GiB fit guard. Batch32 padded vertex tensors alone measured12.09–13.51GiB in two banks before inputs/autograd. Their aggregate attempted CPU was565.16s;5-second guard sampling explains recorded overshoot. No further CFL100 attempt, cap/batch/loss change or resource escalation was made.

SP25 admission was corrected to16GiB per builder /3GiB per fit after measured peaks, while child termination guards remained24/6GiB and node/pool budgets230/96GiB. Nine continuation banks peaked8.87–9.00GiB; the final bank had8.99GiB peak and826.13wall/821.07CPU minutes. Healthy work was preserved. The later metadata-step lifetime fault interrupted11 attempts over9 fit IDs; its repair restored only absent workers, and successful fits were not rerun. The23 explicit startup-failure replacements and7 seed-RNG screen replacements remain separate records; the7 superseded CFL successes are excluded from all valid counts/tables here.

The closed, separately authorized loader diagnostic measured a6.48× pipeline ratio on three fixed SP25 N500 batches, not whole-fit acceleration: normal reading/decompression accounted for82.09percent and public LAVA forward/backward3.57percent of sampled CPU. Production loading and scientific settings were unchanged. [Diagnostic scope and measurements](ops/loader_profile_20261104/summary.md). The original larger-CFL qualifications had an unseeded transition-RNG limitation and remain supplementary resource observations only. [Current observed resource table](CFL_RESOURCE_COSTS.md) preserves these limits and dated projections separately from measurements.

## Provenance and retention

Scientific source commit`ed80f8b45fc291d27f9bf67b95b58fc9ee4642a9`; source bundle SHA256`1375e613d7c97c15402f31d5a65d16af91cb5be8748fbabd1ece31ced119f8f0`. The contract's original scientific cutoff9Nov03:57UTC and absolute deadline9Nov04:27UTC were not extended. This closeout adds no scientific audit, fit, qualification, solve or enumeration.

The verified22MB remote bundle contains9,035 compact files, plus the nested10,973 node07 and21,589 node09 seed/RNG records. Transfer size and each compact file's SHA256 were checked once. It contains source, original seeds, selected configurations, existing validation tables, per-seed measurements, clocks, failures, provenance and logs; no neural weights, optimizer states, prediction arrays or dense geometry caches. [Retrieval evidence](ops/FINAL_RETRIEVAL_VERIFIED.json), [compact bundle](retained_compact/remote_compact_final.tar.gz). GitHub/Drive backup and exact cleanup receipts are recorded separately in`ops/FINAL_BACKUP_VERIFIED.json` and`ops/MONITOR_CLOSED.json`; remote deletion follows verified backup. The readable report and scalar tables are durable alongside the bundle.
