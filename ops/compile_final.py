"""Compile existing compact receipts; no fits, solves, enumeration or retuning."""
import collections, csv, datetime, json, statistics
from pathlib import Path

root = Path(__file__).resolve().parents[1]
remote = root / 'retained_compact/remote_snapshot'
out = root / 'artifacts'
out.mkdir(exist_ok=True)
read = lambda p: json.loads(p.read_text())
terminal = read(remote / 'EXECUTION_OUTCOMES_COMPLETE.json')
superseded = set(read(remote / 'ops/AFFECTED_JOB_IDS.json'))
censors = read(remote / 'ops/RESOURCE_CENSORED_CELLS_20261103.json')
allowed = {'dys_shortest_path_' + n + 'x' + n for n in ('10', '15', '20', '25')} | {'dys_cfl_50x5', 'dys_cfl_100x10', 'dys_knapsack_1000x2'}

def write_csv(name, rows):
    assert rows
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with (out / name).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys, lineterminator='\n'); writer.writeheader(); writer.writerows(rows)

receipts = {}
for p in sorted((remote / 'results/receipts').glob('*.json')):
    v = read(p); j = v.get('job', {})
    if v['status'] != 'complete' or p.stem in superseded or j.get('route') not in allowed:
        continue
    if j['route'].startswith('dys_cfl_') and j['config']['sizes']['train'] != 100:
        continue
    assert v['summary']['epochs_completed'] == 300
    receipts[p.stem] = v
assert len(receipts) == terminal['successful_fits'] == 670
assert len(censors['attempts']) == terminal['resource_censored_fits'] == 40
rows = []
for jid, v in sorted(receipts.items()):
    j, s = v['job'], v['summary']; c = j['config']; g = s.get('geometry') or {}; t = s.get('test') or {}
    rows.append({'id': jid, 'outcome': 'valid_success', 'route': j['route'], 'stage': j['stage'],
                 'phase': j.get('campaign_phase'), 'train_N': c['sizes']['train'], 'validation_N': c['sizes']['validation'],
                 'depth': c['model']['depth'], **{k + '_seed': val for k, val in c['seeds'].items()},
                 'epochs_completed': s['epochs_completed'], 'best_epoch': s.get('best_epoch'),
                 'validation_mean_regret': s.get('best_validation', {}).get('mean_regret'),
                 'test_mean_regret': t.get('mean_regret'), 'test_normalized_mean_regret': t.get('normalized_mean_regret'),
                 'test_standard_error_regret': t.get('standard_error_regret'), 'test_samples': t.get('sample_count'),
                 'whole_wall_seconds': v.get('total_wall_seconds', v.get('elapsed_seconds')),
                 'whole_process_CPU_seconds': v.get('resource_usage', {}).get('process_cpu_seconds'),
                 'whole_process_peak_RSS_bytes': v.get('resource_usage', {}).get('peak_process_resident_bytes'),
                 'cache_load_wall_seconds': g.get('cache_load_wall_seconds'), 'cache_load_CPU_seconds': g.get('cache_load_cpu_seconds'),
                 'in_process_build_wall_seconds': g.get('cache_build_wall_seconds_this_call'),
                 'in_process_build_CPU_seconds': g.get('cache_build_cpu_seconds_this_call'),
                 'optimization_wall_seconds': s.get('optimization_wall_seconds'), 'optimization_CPU_seconds': s.get('optimization_cpu_seconds'),
                 'validation_wall_seconds': s.get('validation_wall_seconds'), 'test_wall_seconds': s.get('test_evaluation_wall_seconds'),
                 'source_receipt': 'retained_compact/remote_snapshot/results/receipts/' + jid + '.json'})
for a in censors['attempts']:
    jid = a['id']; j = read(remote / 'jobs' / (jid + '.json'))['jobs'][0]; c = j['config']
    assert jid not in receipts
    rows.append({'id': jid, 'outcome': 'accepted_resource_censor', 'route': j['route'], 'stage': j['stage'],
                 'phase': j.get('campaign_phase'), 'train_N': c['sizes']['train'], 'validation_N': c['sizes']['validation'],
                 'depth': c['model']['depth'], **{k + '_seed': val for k, val in c['seeds'].items()},
                 'epochs_completed': 0, 'whole_wall_seconds': a['elapsed_wall_seconds'],
                 'whole_process_CPU_seconds': a['process_cpu_seconds'], 'sampled_peak_RSS_bytes': a['peak_rss_bytes'],
                 'censor_reason': a['reason'], 'test_mean_regret': None, 'test_normalized_mean_regret': None,
                 'source_receipt': 'ops/RESOURCE_CENSORED_CELLS_20261103.json'})
assert len(rows) == len({v['id'] for v in rows}) == 710
write_csv('required_outcomes.csv', rows)

selections, tables = [], []
for p in sorted((remote / 'selections').rglob('*.json')):
    v = read(p); j = v['selected_template']; c = j['config']
    assert v['test_scores_read'] is False and len(v['validation_table']) == 16
    selections.append({'family': v['family'], 'phase': v['phase'], 'depth': c['model']['depth'],
                       'train_N': c['sizes']['train'], 'selected_screen_id': j['id'],
                       'choices': json.dumps(j['choices'], sort_keys=True), 'validation_score': v['score'],
                       'test_scores_read': False, 'source': str(p.relative_to(root))})
    for index, item in enumerate(v['validation_table']):
        tables.append({'family': v['family'], 'phase': v['phase'], 'original_grid_index': index,
                       **{k: json.dumps(val, sort_keys=True) if isinstance(val, dict) else val for k, val in item.items()}})
assert len(selections) == 14 and len(tables) == 224
write_csv('selected_configurations.csv', selections); write_csv('existing_validation_tables.csv', tables)

groups = collections.defaultdict(list)
for row in rows:
    if row['outcome'] == 'valid_success' and row['stage'] == 'confirmation':
        groups[(row['route'], row['train_N'], row['depth'])].append(row)
aggregate = []
for (route, n, depth), values in sorted(groups.items()):
    scores = [v['test_normalized_mean_regret'] for v in values]
    aggregate.append({'route': route, 'train_N': n, 'depth': depth, 'successful_confirmation_fits': len(values),
                      'mean_test_normalized_regret': statistics.mean(scores), 'median_test_normalized_regret': statistics.median(scores),
                      'minimum_test_normalized_regret': min(scores), 'maximum_test_normalized_regret': max(scores),
                      'sample_SD_between_fit_means': statistics.stdev(scores) if len(scores) > 1 else None,
                      'mean_optimization_CPU_seconds': statistics.mean(v['optimization_CPU_seconds'] for v in values)})
write_csv('confirmation_summary.csv', aggregate)

attempt_rows = []
for p in sorted((remote / 'attempts').glob('*.json')):
    a = read(p); attempt_rows.append({'attempt_record': str(p.relative_to(root)), **a})
write_csv('attempt_costs.csv', attempt_rows)
builds = [a for a in attempt_rows if a.get('kind') == 'prepare']
write_csv('shared_build_attempt_costs.csv', builds)
scope_counts = collections.Counter((v['route'], v['outcome']) for v in rows)
fit_stats = {}
for size in (100, 300, 500, 1000):
    values = [v for v in rows if v['outcome'] == 'valid_success' and v['route'] == 'dys_shortest_path_25x25' and v['train_N'] == size]
    fit_stats[str(size)] = {'fits': len(values), **{k: [min(v[k] for v in values), max(v[k] for v in values)]
                                                 for k in ('whole_wall_seconds', 'whole_process_CPU_seconds', 'whole_process_peak_RSS_bytes')}}
interrupted = [read(p) for p in (root / 'recovery/worker_step_loss').glob('*.json')]
finished = datetime.datetime.fromtimestamp(terminal['finished_unix'], datetime.timezone.utc).isoformat()
summary = {'terminal': terminal, 'execution_finished_utc': finished,
           'scope_counts': {'|'.join(k): v for k, v in scope_counts.items()}, 'frozen_selection_groups': len(selections),
           'original_grid_entries': len(tables), 'SP25_complete_fit_ranges': fit_stats,
           'measured_valid_whole_fit_CPU_seconds': sum(v['whole_process_CPU_seconds'] for v in rows if v['outcome'] == 'valid_success'),
           'measured_optimization_CPU_seconds_valid_fits': sum(v['optimization_CPU_seconds'] for v in rows if v['outcome'] == 'valid_success'),
           'censor_CPU_seconds': censors['total_attempt_CPU_seconds'],
           'recorded_external_build_attempts': len(builds),
           'measured_external_build_attempt_CPU_seconds': sum(a.get('process_cpu_seconds') or 0 for a in builds),
           'missing_external_build_CPU_records': sum(a.get('process_cpu_seconds') is None for a in builds),
           'interrupted_worker_attempts': len(interrupted), 'interrupted_worker_CPU_seconds_lower_bound': 11268.47,
           'is_complete_campaign_CPU_total': False,
           'CPU_accounting_note': 'Whole fit process CPU includes optimization, cache/import/evaluation and any in-process construction. External build attempts are separately measured. Nested cumulative prefix clocks are not added. Interrupted final rusage is unavailable; no complete total is claimed.',
           'superseded_successful_CFL_screen_results_excluded': len(superseded), 'explicit_startup_failure_replacements': 23,
           'actual_numerical_processes_at_terminal_check': 0, 'actual_numerical_CPU_cores_at_terminal_check': 0,
           'allocated_campaign_CPUs_after_release': 0, 'retention': 'source, seeds/RNG, configurations, validation tables, scalar per-seed measurements, cost/failure records and logs; no weights, predictions arrays or dense caches'}
(out / 'final_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
(root / 'EXECUTION_OUTCOMES_COMPLETE.json').write_text(json.dumps(terminal, indent=2) + '\n')

table = '\n'.join('| ' + route.replace('dys_', '') + ' | ' + str(scope_counts[(route, 'valid_success')]) + ' | ' + str(scope_counts[(route, 'accepted_resource_censor')]) + ' |' for route in sorted(allowed))
sp25 = '\n'.join('| ' + str(n) + ' | ' + str(v['fits']) + ' | ' + ' | '.join(f'{v[k][0]/60:.2f}–{v[k][1]/60:.2f}' for k in ('whole_wall_seconds', 'whole_process_CPU_seconds')) + ' | ' + f"{v['whole_process_peak_RSS_bytes'][0]/2**30:.2f}–{v['whole_process_peak_RSS_bytes'][1]/2**30:.2f}" + ' |' for n, v in fit_stats.items())
report = f'''# LAVA completion: results, timing and resource limits

Execution closed at **{finished}** with **670 valid successful fits +40 accepted CFL100 resource censors =710 required outcomes covered**. All active fits drained. This is not710 successful fits. The terminal check found zero numerical processes, zero actual numerical CPU use and no reservations. Both remaining campaign allocations16726/16740 were released once and verified absent; no scientific CPU remains allocated. The heartbeat was paused before this compilation.

| Required route | Valid successes | Accepted resource censors |
|---|---:|---:|
{table}

SP successes include160 original development screens plus400 confirmations; CFL50 includes64 screens plus40 confirmations. The6 KP rows are the six authorized recovery fits, not a fresh ten-seed KP comparison. The original paired seeds, complete16-point grids,14 frozen selections, native physical relaxation, HiGHS1.12.0, cap250/exact duplicates and original loss/batch/300epoch settings were preserved. Validation alone selected configurations and checkpoints. No CFL100-specific tuning occurred. SP50, CFL200/250/500 and larger-N CFL are scope exclusions; completed excluded measurements remain supplementary and cannot extend execution.

[All710 outcomes and per-fit seeds/results/times](artifacts/required_outcomes.csv), [frozen configurations](artifacts/selected_configurations.csv), [existing224 validation-table entries](artifacts/existing_validation_tables.csv), [confirmation summaries](artifacts/confirmation_summary.csv), [machine-readable summary](artifacts/final_summary.json). Confirmation means/medians/ranges summarize existing per-seed LAVA test results and do not change selection. Ten paired confirmations populate each retained complete SP/CFL50 configuration group. CFL100 has no measured test regret; a missing value is never zero, a method win or a CP-SPO loss. The six KP recoveries alone do not estimate the full original KP benchmark.

## Measured timing and resource accounting

| SP25 training N | Successful fits | Whole wall minutes | Whole process CPU minutes | Whole process peak RSS GiB |
|---|---:|---:|---:|---:|
{sp25}

The ten SP25 N1000 continuation attempts use their original exact seeds/RNG checkpoints. Initial N100 builds and replay/continuation are distinct attempts; completed nested prefixes are views of a shared build and are not charged again. [Every recorded attempt](artifacts/attempt_costs.csv) and [external shared-build attempts](artifacts/shared_build_attempt_costs.csv) retain failures, holds and replay costs. In-process construction, metadata cache loads, optimizer, validation and physical test clocks are separate columns in the per-fit file. Lazy row I/O remains inside optimizer clocks.

Valid fit processes consumed **{summary['measured_valid_whole_fit_CPU_seconds']/3600:.2f} measured core-hours** including process overhead and any in-process construction; their optimizer portion was **{summary['measured_optimization_CPU_seconds_valid_fits']/3600:.2f} core-hours**. These two figures overlap and must not be added. External build attempts are separate; whole-attempt and cumulative prefix clocks overlap and must not be added. The11 worker-step-loss attempts add a measured CPU **lower bound3.13 core-hours**; their terminal rusage is unavailable and two starts were unsampled. A complete campaign CPU total is therefore not claimed. Actual CPU consumption and Slurm allocation are distinct; the original100 running-plus-pending campaign CPU envelope was never increased.

All ten original CFL100 N100 banks completed; final preparation attempts peaked56.7–65.0GiB, with earlier failed/resumed costs retained. All40 original fits were accepted resource-censored before a completed epoch under the unchanged12GiB fit guard. Batch32 padded vertex tensors alone measured12.09–13.51GiB in two banks before inputs/autograd. Their aggregate attempted CPU was{censors['total_attempt_CPU_seconds']:.2f}s;5-second guard sampling explains recorded overshoot. No further CFL100 attempt, cap/batch/loss change or resource escalation was made.

SP25 admission was corrected to16GiB per builder /3GiB per fit after measured peaks, while child termination guards remained24/6GiB and node/pool budgets230/96GiB. Nine continuation banks peaked8.87–9.00GiB; the final bank had8.99GiB peak and826.13wall/821.07CPU minutes. Healthy work was preserved. The later metadata-step lifetime fault interrupted11 attempts over9 fit IDs; its repair restored only absent workers, and successful fits were not rerun. The23 explicit startup-failure replacements and7 seed-RNG screen replacements remain separate records; the7 superseded CFL successes are excluded from all valid counts/tables here.

The closed, separately authorized loader diagnostic measured a6.48× pipeline ratio on three fixed SP25 N500 batches, not whole-fit acceleration: normal reading/decompression accounted for82.09percent and public LAVA forward/backward3.57percent of sampled CPU. Production loading and scientific settings were unchanged. [Diagnostic scope and measurements](ops/loader_profile_20261104/summary.md). The original larger-CFL qualifications had an unseeded transition-RNG limitation and remain supplementary resource observations only. [Current observed resource table](CFL_RESOURCE_COSTS.md) preserves these limits and dated projections separately from measurements.

## Provenance and retention

Scientific source commit`ed80f8b45fc291d27f9bf67b95b58fc9ee4642a9`; source bundle SHA256`1375e613d7c97c15402f31d5a65d16af91cb5be8748fbabd1ece31ced119f8f0`. The contract's original scientific cutoff9Nov03:57UTC and absolute deadline9Nov04:27UTC were not extended. This closeout adds no scientific audit, fit, qualification, solve or enumeration.

The verified22MB remote bundle contains9,035 compact files, plus the nested10,973 node07 and21,589 node09 seed/RNG records. Transfer size and each compact file's SHA256 were checked once. It contains source, original seeds, selected configurations, existing validation tables, per-seed measurements, clocks, failures, provenance and logs; no neural weights, optimizer states, prediction arrays or dense geometry caches. [Retrieval evidence](ops/FINAL_RETRIEVAL_VERIFIED.json), [compact bundle](retained_compact/remote_compact_final.tar.gz). GitHub/Drive backup and exact cleanup receipts are recorded separately in`ops/FINAL_BACKUP_VERIFIED.json` and`ops/MONITOR_CLOSED.json`; remote deletion follows verified backup. The readable report and scalar tables are durable alongside the bundle.
'''
(root / 'RESULTS_TIMING_RESOURCE_LIMITS.md').write_text(report)
print(json.dumps({'valid_successes': len(receipts), 'accepted_censors': len(censors['attempts']), 'outcomes': len(rows),
                  'selection_groups': len(selections), 'validation_rows': len(tables), 'confirmation_groups': len(aggregate),
                  'execution_finished_utc': finished, 'measured_valid_fit_core_hours': summary['measured_valid_whole_fit_CPU_seconds']/3600,
                  'SP25_ranges': fit_stats}, indent=2))
