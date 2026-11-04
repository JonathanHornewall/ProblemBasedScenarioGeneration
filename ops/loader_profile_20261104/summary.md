# SP25 loader versus public LAVA: one bounded diagnostic

**The current lazy normal-loading path dominates these sampled SP25 N500 batches.** Re-reading/decompressing normal matrices cost1.383CPU seconds per batch, **82.09%** of the measured current pipeline. Public LAVA forward+prediction-gradient backward cost0.06014CPU seconds, **3.57%**. Including normal dtype conversion and the unchanged public normal padding, handling matrices unused by LAVA cost **94.50%** of this measured pipeline. The reading/decompression is caused by our lazy cache representation; conversion and normal padding are also present in the unchanged public dataset/collator and remain in the resident comparison.

Mean current-path A: **1.68525CPU /1.69374wall seconds per full batch32**. Same sampled raw data resident in RAM B: **0.26023CPU /0.26169wall seconds**. The measured pipeline ratio is **6.48× CPU /6.47× wall**, an84.56% CPU reduction for this isolated data-assembly+loss pipeline. This is not an observed end-to-end training speedup. The existing8041.07CPU-second fit must not be attributed entirely to inherent LAVA loss complexity.

The question was explicitly authorized for one isolated diagnostic. Only completed SP25 N500 geometry from original data151101/init161101 was used. Three full batches at positions0,7,14 came from the original epoch1 order: `default_rng(derive_seed(171101,'epoch',1))`, derived seed450263392. Each A/B path ran four times per batch, in AB/BA/AB/BA order:24 measured pipelines total,96 distinct observed training examples. No model checkpoint, generator, validation/test bank, fit, optimizer update, solve, enumeration, installation or live-process profiler was used.

The identical maintained `HighsBindingDataset.__getitem__`, unchanged public `collate_fn_2` and public `LAVA.forward` were called. Standalone wrappers timed raw sequence access, the same `torch.as_tensor(...,dtype=float32)` calls and the same two public `pad_sequence` calls. Raw normal and vertex arrays remained float64; converted/padded fields and predictions remained float32. B preloaded only the selected batch's raw normal matrices and vertices, then performed the same conversion, padding and loss. All source/input identities and exact indices are retained in the raw JSON.

| Component | A CPU ms/batch | Share of A CPU | B CPU ms/batch |
|---|---:|---:|---:|
| Lazy normals: read + ZIP decompression | 1383.437 | 82.09% | 0.000 |
| Lazy vertices: file open / mmap creation | 7.628 | 0.45% | 0.000 |
| Normal float64→float32 conversion | 107.825 | 6.40% | 85.539 |
| Vertex conversion / materialization | 11.106 | 0.66% | 2.905 |
| Other dataset dtype conversion | 1.245 | 0.07% | 0.830 |
| Public collator: normal padding | 101.268 | 6.01% | 100.108 |
| Public collator: vertex padding | 9.702 | 0.58% | 9.626 |
| Public collator: other | 0.166 | 0.01% | 0.249 |
| Public LAVA forward | 49.776 | 2.95% | 49.939 |
| Public LAVA prediction-gradient backward | 10.364 | 0.61% | 10.201 |
| Dataset bookkeeping | 2.735 | 0.16% | 0.830 |
| Other timed pipeline bookkeeping | 0.000 | 0.00% | 0.000 |
| **Total** | **1685.251** | **100%** | **260.229** |

CPU medians/maxima across the12 measurements per path: A1.6932/1.7435s; B0.2736/0.2833s. Mean wall times by component are retained alongside CPU times in [summary metrics](summary_metrics.json); the raw JSON retains every repeat.

| Epoch1 full-batch position | A mean CPU seconds | B mean CPU seconds | Separate one-time preload CPU seconds |
|---|---:|---:|---:|
| 0 | 1.7055 | 0.2747 | 1.5770 |
| 7 | 1.6986 | 0.2756 | 1.4886 |
| 14 | 1.6517 | 0.2303 | 1.4399 |

One-time preloading was1.440–1.577CPU seconds per sampled batch (1.447–1.586wall), separately charged and excluded from repeated B visits. Each selected batch held737,894,400bytes (703.71MiB) of raw normal matrices and roughly36–38MiB of raw vertices. Public normal padding produced the same32×2402×1200 float32 tensor (351.86MiB) for A/B. Vertex padded shapes were32×402×1200,32×284×1200 and32×231×1200 for the three batches. The N500 bank's original neighbor counts remain31–442, mean135.832. Preserved padding is part of the public loss's mean denominator.

On the first paired batch, scalar loss and all prediction-gradient entries were **bitwise identical**, with maximum gradient difference0. Nonzero losses/gradient norms were measured; this was not a zero-gradient shortcut. Predictions were fixed deterministic sin/cos cost tensors, not outputs of a trained network. Thus this measures data handling and public loss/backward with representative shapes, while omitting model forward/backward, optimizer work, evaluations, epoch/persistence overhead and one-time shared geometry construction. It is not an exact authors' end-to-end training replication and does not establish timings for another route or seed.

No page cache was flushed. Normal timing combines ZIP file reading and decompression; it does not distinguish their individual costs. Vertex file timing covers opening/mmap creation; page faults/materialization may appear in dtype conversion. The CPU was isolated while other campaign workers used different physical cores on the same node, so shared memory bandwidth and OS caches remain part of the environment.

A conditional calculation only:300epochs×15 full batches gives4500 full-batch visits. If the sampled A−B difference1.42502CPU seconds applied to every such visit, it would total6412.60CPU seconds of difference. The300 shorter20-example batches, model/optimizer work and other unmeasured costs prevent treating that number or6.48× as a measured whole-fit speedup. Likewise a whole N500 raw-normal resident bank would extrapolate to10.74GiB, beyond the unchanged6GiB fit guard before other memory; whole-bank residency was not implemented or tested.

Diagnostic measurements finished **03 Nov 2026 23:32:05 La Paz /04 Nov 03:32:05 UTC**. Exact diagnostic job17476 used one physical CPU on node07: OS CPU1, physical socket1/core2. Its placement check verified no overlap with any campaign numerical child or coordinator. One diagnostic Torch-proxy startup error (missing dtype attribute) consumed4.181wall/3.177CPU seconds; it is preserved in [failed attempt](failed_attempt.json). The same job was requeued once after the wrapper correction; no replacement job was needed. The completed profile consumed31.710wall/30.613CPU seconds. Across both attempts: **35.891process-wall /33.790actual CPU seconds**,40allocated CPU-wall seconds, peakRSS1.94GiB, within the original five-minute execution budget. Slurm's requeue reset its limit to the original5minutes; standalone260s checks/270s alarm/275s CPU guards remained, and actual aggregate use stayed below300s. Slurm TotalCPU returned unusable zeros; actual CPU above is measured process time.

The diagnostic is **complete and released**: job17476 is COMPLETED0:0 and absent from the queue, verified once at terminal retrieval. No cancellation was needed after natural completion. No campaign source, cache file, dispatcher, allocation setting, scientific result, active process or heartbeat was changed. No further profiling or optimization is scheduled.

Inspect [the one diagnostic source](profile.py), [raw timings and source/sample identities](raw_timing.json), [derived component metrics](summary_metrics.json) and [exact terminal/release record](terminal.json). The profile question is closed; production optimization is outside this diagnostic.
