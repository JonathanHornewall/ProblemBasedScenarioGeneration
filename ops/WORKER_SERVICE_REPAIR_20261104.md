# Worker-step lifetime recovery —4November2026

At the05:23UTC bounded heartbeat check,643 valid successful fits and40 accepted CFL100 censors covered683/710 outcomes;27 SP25 fits remained. Actual process telemetry found20 numerical children (1 builder,19 fits) consuming19.74 cores against32 allocated CPUs, while11 old worker records were stale and ready work lacked workers. No new failed-attempt receipt had been emitted.

The fault was in the earlier metadata handoff service. `refresh_idle_once.py` kept its Slurm step alive only while one of its originally launched worker PIDs remained. An unstarted child replaced its original metadata parent at the admission boundary, retaining the old step/cgroup. After the last original parent exited, the launcher returned and Slurm cleaned up the replacement descendants. Accounting confirms steps16740.5 and16726.19 completed; all stale child/parent PIDs are absent, and their final heartbeat times coincide with those endings. A completed launcher step was not a successful fit.

The launcher now remains alive until the shared campaign terminal marker or original scientific cutoff, independently of the original PID list. One focused metadata/process regression verified that an exited original PID does not end the service while a successor remains, then verified terminal-marker exit and a finite cutoff. Zero scientific fits, solves or enumerations were used by that regression. The protected production loader, loss, batching, data, seeds, grids, selections and fit guards were unchanged.

Eleven exact dead slots were restored on their existing indices/cores:7 in16726 and4 in16740, with no process signalled or allocation added. The restoration steps are16726.33 and16740.8 and use the corrected finite service. Old worker snapshots and interrupted attempt records were preserved before their heartbeat files were replaced. Active fits were preserved; successful receipts were neither removed nor rerun. Some interrupted missing fits had already restarted automatically in the healthy fresh pool; those current attempts were retained.

Applied readback at **04 Nov 01:36:03 La Paz /04 Nov 05:36:03 UTC** verified31 live numerical worker slots,1 bank builder and20 training fits consuming20.81 effective cores over the measured interval. No dispatchable fit remained unstarted; four original seed151119 N500/N1000 fits await durable prefixes from its healthy builder. All20 numerical processes seen in the diagnosis either retained their exact PID/birth and advanced CPU (19) or naturally completed300epochs (1). The singleton coordinator retained PID2281033/core46. Current node admission76/230GiB, resource-pool28/96GiB, sampled current RSS27.11GiB, host RAM available188.14GiB. The original16/3GiB admission and24/6GiB child termination guards are unchanged; allocation remains32CPU,0 pending.

| Quantity | Disposition |
|---|---|
|Valid successes|646, including76 SP25 fits|
|Accepted required-cell resource censors|40 CFL100; no further attempt|
|Remaining required fits|24 SP25:20 training,4 awaiting prefixes|
|Restored dead metadata slots|11;31 numerical slots now live|
|Interrupted attempts retained|11 attempts affecting9 distinct original fit IDs|
|Measured interrupted CPU lower bound|11,268.47seconds =3.13core-hours|
|Interrupted sampled RSS high-water maximum|1.66GiB; final peaks unavailable|
|Unknown quantities|Terminal return codes/final rusage and two attempts' unsampled initial CPU/RSS; zero samples are lower bounds, not proof of zero use|

These infrastructure-interrupted attempts are neither extra required outcomes nor accepted scientific resource censors. Their last observed CPU samples must be charged separately from valid completed-fit training; their lost final rusage must remain explicitly unavailable. Two fit IDs suffered two distinct interruptions and have separate records. The measured lower bounds are not complete attempt totals. No canonical failure file was written over a current healthy replacement, so continuation is not blocked by bookkeeping.

Nine original N1000 banks are complete; the last seed151119 was330/1000 in the initial05:23 snapshot and remains healthy. Its N300 prefix is durable and the two corresponding fits are active. Completed9-bank continuation attempts measured814.64–822.31minutes wall /811.78–819.41minutes CPU and8.87–9.00GiB whole-process peak RSS. They include imports/replay and must not be added again to their nested cumulative prefixes. The current compact [resource table](../CFL_RESOURCE_COSTS.md) also retains all36 measured larger-N completed-fit timings and the separately authorized, already closed loader diagnostic.

The campaign remains active under the same four-hour heartbeat and unchanged9Nov03:57:29UTC scientific cutoff /04:27:29UTC absolute deadline. Required coverage stays710:670 valid successes plus40 accepted censors if SP25 completes, followed by drain. No new allocation, memory increase, method, tuning, loader optimization, CFL100 attempt, qualification or scientific audit was introduced. The loader profile from the prior user request is closed and was not rerun.

Evidence: [initial snapshot](../monitor_snapshots/1793769780.json), [exact process/step diagnosis](WORKER_STEP_LOSS_DIAGNOSIS_20261104.json), [dead-slot plan and source records](WORKER_SLOT_RESTORE_PLAN_20261104.json), [source deployment and focused regression](WORKER_SERVICE_REPAIR_DEPLOYMENT_20261104.json), [applied continuity/dispatch verification](WORKER_SERVICE_REPAIR_APPLIED_20261104.json), [current structured state](HEARTBEAT_20261104_0522_SUMMARY.json), [all interrupted attempt records](../recovery/worker_step_loss).
