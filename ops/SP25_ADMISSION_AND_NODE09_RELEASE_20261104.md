# SP25 admission correction and node09 release

Completed under the direct human instruction to measure memory, admit more SP25 fits and release idle node09 without interrupting healthy fits or changing science. Final observation: **3Nov2026 21:46:08 La Paz /4Nov01:46:08 UTC**.

**LAVA allocation fell48→32CPUs while active numerical children rose11→31:9→10 original bank builders and2→21 fits.** The final five-second sample measured30.60 numerical cores. All11 originally healthy children kept the same PID and birth time and advanced CPU; none was signalled or restarted. Original data/seeds, batch, loss, solver/cap250,300epochs, frozen configurations and validation/test selection remain unchanged. The40 accepted CFL100 censors remain closed to new attempts.

| Allocation | Before | Applied after release |
|---|---|---|
|16726,node07,24CPU|5 builders,2 training fits,17 idle workers|5 builders,18 training fits,1 lightweight coordinator on reserved core46|
|16740,node07,8CPU|4 builders,4 idle workers|5 builders,3 training fits|
|16223,node09,16CPU|16 idle numerical workers plus lightweight coordinator|Released once; verified absent|
|Total LAVA|48 allocated,11 numerical children|32 allocated,31 numerical children and1 coordinator;0 pending|

The final process sample measured22.69 numerical cores in16726 and7.90 in16740. All21 fits reported training; no active validation/test was observed. Numeric processes plus coordinator have32 distinct single-core affinities. Shared bank claims prevent duplicate construction and fit claims prevent duplicate training. Seed151119/init161119 received one proven-idle worker with explicit prerequisite priority; its original N100 prefix is reused and its N1000 builder is active. No successful fit was rerun.

| SP25 process | Whole-process sizing evidence | New admission claim | Retained child kill limit | Measured headroom |
|---|---|---:|---:|---:|
|Bank construction|9 original builders' Linux VmHWM8.69–8.88GiB near90percent complete, including temporaries so far|16GiB|24GiB|1.80× largest observed peak|
|Training fit|52 completed N100/N300/N500 fits; largest wait4 ru_maxrss1.69GiB|3GiB|6GiB|1.78× largest observed peak|

Admission and termination limits are now separate. Live ledger claims were repriced under the reservation lock while each active parent retained its imported original guard. Twenty proven-idle metadata workers were replaced; one further idle worker was retired for the coordinator core. An old active worker adopts current code only at its next unstarted fit boundary. Newly launched children capture their unchanged termination limit once before starting. The five-second final sample had87.70GiB combined current RSS and158.01GiB host RAM available. Admission totals223/230GiB node-wide and89/96GiB in16740. Observations are measurements, not bounds on unseen future peaks.

Exactly one metadata-only admission regression passed on one existing allocated CPU:3GiB claims remain unchanged, an obsolete6GiB unstarted parent hands off on its same core/cgroup, and16/3GiB admission remains separate from24/6GiB termination. It ran zero fits, solves or enumerations. Applied verification then established ten builders,21 fits, fresh singleton READY,40 preserved censors, distinct CPU affinities and no new failed attempt. The source diff changes operational admission/control only.

Old coordinator163065 exited on node09 before fresh coordinator2281033 acquired the singleton lock on node07/16726/core46. Fresh READY updates were verified before release. A separate immediate pre-release node09 check proved zero numerical children and zero coordinator. Exact owned job16223 received one scancel; after the finite35-second release grace, one scheduler read verified it absent. Jobs16741/16742/16744/16745/16746 had already been released; KP16727 had finished. Remaining owned allocations are only16726/16740. Unrelated Jonathan jobs16900 and17282 were preserved.

Before node09 release,21,589 small seed/RNG/JSON records were retained on shared home and copied locally. The4,303,294-byte compressed archive contains116,877,109 uncompressed bytes; every file's size/SHA256 and archive identity were verified. It includes no neural weights or dense geometry/cache arrays. Archive SHA256: `ef976b1afbd8b45a9a4c64b9544386e067fe9212951e32c08c4e8ad4e43fd832`. Node09 dense scratch remains for the authorized terminal cleanup; compact records are durable. Final GitHub/Drive campaign backup remains part of closeout.

Science status stays622 valid successes +40 accepted censors =662/710 outcomes covered;48 SP25 fits are unresolved,21 currently training. Execution remains active. Eventual coverage670 valid successes +40 censors closes execution after drain; missing CFL100 regret is not a method win. Scientific cutoff9Nov03:57:29UTC and absolute deadline04:27:29UTC are unchanged. The same finite four-hour heartbeat now points to the measured policy and remaining pools; no early checks, duplicate monitor, new allocation or deadline extension.

Evidence: [memory/ownership preflight](SP25_MEMORY_PREFLIGHT_20261104.json), [policy and authorization](SP25_ADMISSION_POLICY_20261104.json), [focused test](SP25_ADMISSION_SOURCE_DEPLOYMENT_20261104.json), [locked claim reconciliation](SP25_ADMISSION_REBALANCE_NODE07.json), [applied continuity/CPU verification](SP25_ADMISSION_APPLIED_20261104.json), [post-release process evidence](SP25_NODE07_AFTER_RELEASE_20261104.json), [node09 release](NODE09_RELEASE_20261104.json), [compact-copy verification](NODE09_COMPACT_LOCAL_VERIFIED_20261104.json), [compact seed archive](../retained_compact/node09_seed_records.tar.gz), [current structured state](SP25_ADMISSION_FINAL_STATE_20261104.json).
