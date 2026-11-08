# CERMICS LAVA timing replay, 7 November 2026

The authorized scope is the 19-fit timing replay in `research/lava_timing_replay_20261106`. CERMICS is the authorized site. The historical completion campaign and its archive remain closed.

Main is `01a14a8b-3d7c-7d23-a8bb-2baa8f90c86c`. The ordinary Sol Max Monitor `01a19a7d-6b4c-7e41-981f-ea544f90aed0` owns remote mutations. Reuse Analyzer `01a167fc-0385-7502-b7aa-744fc489c456` for read-only compilation.

## Closeout

Execution finished on November 8 at 02:55 Santiago. The original-seed SP25 N500 parity check passed and all 19 timing fits completed 300 epochs. The 195-file compact copy was verified before the shared root and both scratch directories were removed. All owned jobs are terminal. The heartbeat was paused before compilation and deleted after closeout. [Timing and resource report](closeout/REPORT.md), [terminal marker](MONITOR_CLOSED.json)

## Verified launch

At 13:17:56 UTC on November 7, Slurm job `19678` was building the original SP25 N500 bank on node07, core 0, in `defq-gpu`. Slurm reported one allocated CPU, zero GPUs and a 40 GiB memory request. The first process sample recorded 27.42 CPU-seconds, 0.914 cores over the sample interval and 3.32 GiB sampled RSS. Startup reported 402.42 GiB free in local scratch and passed the pinned numerical runtime checks. Parity is pending, there are zero valid timing receipts and the other five routes await the gate. [Launch receipt](LAUNCH_VERIFIED.json)

The initial job `19671` stopped at node03 scratch admission with 15.12 GiB free, before numerical work. It and the six scratch probes `19672` through `19677` are terminal. The versioned site repair preserved the failed attempt, original submission and source hashes. Node07 hosts SP25, SP20 and SP15. Node09 hosts KP1000, CFL50 and SP10. Each route uses one core. [Admission failure](overlays/01-node03-19671/ADMISSION_FAILURE.json), [scratch measurements](SCRATCH_PROBE_READBACK.json), [repair deployment](REPAIR_DEPLOYMENT.json)

The exact remote root is `/home/jonathan.hornewall/runs/lava_timing_replay_20261107`. The exact node-local scratch directory is `/tmp/lava_timing_replay_20261107`. The pinned package and manifest hashes are in the run plan. The absolute cutoff is November 9 at 22:30 UTC, 19:30 Santiago. [Run plan](RUN_PLAN.json)

## Frozen requested order

Use the original paired seed bundle `151101/161101/171101/181101/101` and the 19 frozen configurations. The first route performs SP25 N500 exact input, loss, prediction-gradient and parameter-gradient parity, then the full 300-epoch timing fit. Both receipts must validate before the worker submits the other five routes once and continues the other three SP25 fits in the same allocation. Keep exact geometry prefixes on node07. Time preprocessing once per exact bank and record cache loads, training and attempts separately. Validation, test, retuning and paid resources are outside this replay.

## Monitoring procedure used for this replay

Reuse `monitor-lava-timing-replay` in this chat. The schedule state records activation and early checks. After the two hourly checks, switch this heartbeat to a finite four-hour cadence. Use one invocation of `python3 -B check_once.py` per wake with the exact fresh SSH transport in `remote.py`. Record owned queue rows, measured process CPU and RSS, and receipts. Stay quiet when healthy and send Main material findings only. Reconcile uncertain mutations against exact job names and submission receipts. [Schedule state](SCHEDULE_STATE.json)

On completion, stop or cutoff, pause the heartbeat before compilation. Verify release of exact owned jobs once. Retrieve and verify the compact source, seeds, configurations, parity, preparation and timing evidence and report. Clean the exact new root and node-local scratch after retrieval. Write the current replay's terminal marker and report once. At exhausted read-only transport, pause and alert Main once.
