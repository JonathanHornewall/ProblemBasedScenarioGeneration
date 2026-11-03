# Observed CFL resource costs

The active CFL scope is CFL50/100 at N100 only; SP50 has also been excluded from new work by the subsequent human amendment. Larger cases and data sizes are excluded by the human amendment; their completed/failed resource attempts remain evidence and are never ranked as missing-method losses. No new qualification or enumeration was run to compile this table.

Wall/CPU columns are minutes. RSS is RAM; geometry/pickle columns are disk GiB. Prefix clocks are constructor clocks at a completed prefix; whole-attempt clocks include imports/replay and must not be added to those cumulative prefix clocks. Shared construction is charged once per exact bank, never once per fit. All interrupted/replay attempts remain separately recorded.

| Phase, input/seed and host | Wall min | CPU min | Peak RSS GiB | Geometry/cache disk GiB | Observed outcome |
|---|---:|---:|---:|---|---|
| Qualification cfl_200x10; data17, obs0/1; node04 | 19.13–24.04 | 18.71–18.73 | 22.40–23.98 | 1.16–1.34 raw / 1.20–1.37 pickle | 2 completed resource attempts; RNG limitation |
| Qualification cfl_250x10; data17, obs0/1; node05 | 32.08–69.90 | 31.54–36.49 | 28.57–28.63 | 1.12–1.78 raw / 1.18–1.84 pickle | 2 completed resource attempts; RNG limitation |
| Qualification CFL500; data17, obs0; node07 | 46.64 | 45.89 | 247.96 | Not completed | Failed SIGKILL, cause unproven |
| Qualification CFL500; data17, obs1; node07 | 57.07 | 56.74 | 220.24 | Not completed | Memory-censored by220GiB guard |
| Admitted CFL50 N100 prefixes (2 banks); data17/151101; nodes07/09 | 5.25–8.06 | 5.22–8.00 | Not sampled at prefix | 1.38–1.45 raw | N100 inputs complete; shared across fits |
| Legacy CFL50 N1000 preprocessing (9 banks); paired data151103–151119; nodes07/09 | 75.79–87.64 | 75.23–87.14 | 12.90–16.43 | 11.87–15.30 raw | Completed supplementary resource evidence; legacy RNG provenance |
| Other interrupted CFL50 preprocessing (4 attempts); dev/paired; nodes07/09 | 6.26–41.44 | 6.19–41.11 | 2.00–9.21 | Retained prefixes; not added again | Allocation stops or seed-correctness stop; costs charged |
| Seeded CFL50 development N300; init8201; node07 | 24.43 | 24.24 | 5.22 | 4.22–4.25 committed raw | Completed supplementary preparation; N100 retained |
| Unseeded CFL50 paired attempt; data151101; node09 | 887.50 | 882.79 | 2.25 | 1.38 committed raw | Stopped for seed-provenance repair |
| Seeded CFL50 larger suffix; init161101; node09 | 133.35 | 132.78 | 2.26 | 1.38 committed raw | Scope-stopped; N100 input durable |
| Seeded CFL100 N100; init161101; node09 | 121.12 | 120.67 | 48.45 | Not sampled at censor | Memory-censored at48GiB;96GiB recovery provisioned |
| Scope-stopped SP50 N100 preparation; data151107/init161107; node07 | 10.34 | 10.22 | 78.88 | Completed/partial files preserved | Human scope stop; no N100 prefix or dependent healthy fit |
| CFL50 N100 screen fitting (64 valid fits); original seeds; nodes07/09 | 23.02–36.13 | 22.95–36.02 | 3.52–3.58 | Shared geometry above | 300epochs complete; RSS is whole-fit peak |
| CFL50 N100 confirmation fitting (8 valid fits); original seeds; nodes07/09 | 21.75–25.23 | 21.68–25.15 | 2.99–3.96 | Shared geometry above | 300epochs complete; RSS is whole-fit peak |
| Superseded CFL50 screen fitting (7 attempts); node07 | 24.04–24.59 | 23.96–24.51 | 3.85–3.89 | Shared geometry | Measured costs retained; scientific results superseded |

At3Nov00:12UTC, all64 valid CFL50 N100 screens were complete: validation took 1.80–2.36 seconds per fit and no screen test was performed. The first8 valid confirmations took 2.06–2.65 seconds validation and 0.66–0.76 seconds physical test evaluation per fit. Optimizer wall/CPU and whole-process RSS appear separately in the table. The original43-fit cache-load sample measured0.00033–0.00045 seconds wall and0.00033–0.00044 seconds CPU for metadata/view initialization; lazy row reads occur during fitting. The later64/8 fitting rows come from [the bounded snapshot](monitor_snapshots/1793664735.json); preparation/qualification rows retain their original measured receipt dates.

The qualification used original data seed17, original observations0/1, structural seed101, HiGHS1.12.0, physical LP relaxation, cap250 and exact duplicates. Its NumPy transition RNG was unseeded, so these are factual resource measurements rather than exactly reproducible original-initialization runs. Legacy full-bank construction is likewise labelled separately; only a proved random-free prefix or the original-seeded v2 cache can enter a retained fit. The failed/unseeded attempts remain charged, not relabelled as successful LAVA fits.

Nodes07/08/09 were identified as Intel Xeon6730P at the scope snapshot. Every numerical child uses one bound CPU and one numerical thread; nodes04/05 qualification used one bound CPU. Sources are the unchanged pinned public enumeration (`9dba06da...`), HiGHS adapter (`1933a96d...`), the original input hashes and the current run source receipts. Original initialization is8201 for CFL50 family development,161101/161103/.../161119 for paired confirmations; shuffling/perturbation/structural seeds remain in the original manifests. CFL100 uses the same family validation-selected configuration, with no per-size tuning.

The96GiB CFL100 preparation reservation is a **new resource allowance**, not an observed peak. It repairs the measured48GiB censor and resumes saved original-seed RNG checkpoints. The retained scope requires all ten original paired banks. Unmeasured censor-time cache bytes and prefix-specific RSS are shown as unavailable; no raw archive reconstruction was attempted.

The earlier >1TiB/1000-distinct-anchor CFL200/250 storage calculation was an **uncompressed extrapolation**, not a lower bound or proof of disk infeasibility. Repeated anchors share files, normals are compressed and vertices are lazy-mapped. It is not mixed into the measurement columns, and those cells are now excluded by instruction.

Detailed measured quantities: [preprocessing/qualification/fit times](ops/CFL_RESOURCE_COSTS_20261102.json), [whole-fit RSS/attempt receipts](ops/CFL_FIT_ATTEMPT_COSTS_20261102.json), [scope-change identities/hardware](ops/SCOPE_CHANGE_SNAPSHOT.json), [scope stop clocks and durable prefix](ops/CFL_SCOPE_SUFFIX_STOP.txt), [required CFL100 recovery](ops/CFL100_N100_RECOVERY_READBACK.json). These snapshots precede later completions; refresh the same table from agreed compact receipts at campaign closeout. No weights or dense caches belong in the final retained artifact set.
