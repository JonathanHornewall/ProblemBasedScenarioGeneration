"""Compile the completed one-seed LAVA replay; leave historical tables immutable."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EVIDENCE = ROOT / 'ops/cermics_20261107/closeout/evidence'
OUT = ROOT / 'results_20261108'


def main():
    manifest = json.loads((EVIDENCE / 'manifest.json').read_text())
    measured = {p.stem: json.loads(p.read_text()) for p in (EVIDENCE / 'results').glob('*.json')}
    assert set(measured) == set(manifest['jobs']) and len(measured) == 19
    out = []
    for fit_id, value in measured.items():
        original = manifest['jobs'][fit_id]
        assert value['status'] == 'complete' and value['epochs'] == 300 and value['test_access'] is False
        assert value['seeds'] == manifest['seeds']
        assert value['original_receipt_sha256'] == original['original_receipt_sha256']
        out.append(dict(route=value['route'], N=value['N'], depth=value['depth'],
                        panels=';'.join(original['panels']), original_fit_id=fit_id, measured_fits=1,
                        training_wall_seconds=value['training_wall_seconds'],
                        training_cpu_seconds=value['training_cpu_seconds'],
                        loader_wall_seconds=value['setup']['one_time_loader']['wall_seconds'],
                        old_loader_training_wall_seconds=original['original_training_wall_seconds'],
                        recorded_old_to_new_wall_ratio=original['original_training_wall_seconds']/value['training_wall_seconds'],
                        receipt=f'../ops/cermics_20261107/closeout/evidence/results/{fit_id}.json'))
    out.sort(key=lambda d: (d['route'], d['N'], d['depth']))
    OUT.mkdir(exist_ok=True)
    with (OUT / 'lava_training_replay_19.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(out[0]))
        writer.writeheader(); writer.writerows(out)
    previous = ROOT.parent / 'lava_inclusive_regret_20261104/training_times_300_epochs.csv'
    with previous.open(newline='') as f:
        reader = csv.DictReader(f); fields = reader.fieldnames; rows = list(reader)
    changes = 0
    for row in rows:
        value = measured.get(row['first_seed_fit_id']) if row['method'] == 'LAVA' else None
        if value is None:
            continue
        assert (row['route'], int(row['N']), int(row['depth'])) == (value['route'], value['N'], value['depth'])
        assert row['panel'] in manifest['jobs'][row['first_seed_fit_id']]['panels']
        row.update(measured_fits='1', first_seed_wall_seconds=str(value['training_wall_seconds']),
                   first_seed_cpu_seconds=str(value['training_cpu_seconds']),
                   mean_over_available_seeds_wall_seconds=str(value['training_wall_seconds']),
                   mean_over_available_seeds_cpu_seconds=str(value['training_cpu_seconds']),
                   between_fit_sd_wall_seconds='', timing_source='original_seed_resident_loader_replay_20261107',
                   status='measured')
        changes += 1
    assert changes == 36
    with (OUT / 'training_times_300_epochs_corrected.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    (OUT / 'TABLE_UPDATE.json').write_text(json.dumps({
        'previous_table': str(previous.relative_to(ROOT.parent)), 'new_table': 'training_times_300_epochs_corrected.csv',
        'updated_cells': changes, 'unique_replays': 19, 'seeds_per_replay': 1,
        'other_rows_unchanged': True, 'historical_table_overwritten': False,
        'regret_results_changed': False, 'old_loader_multi_seed_SD_reused': False,
        'sum_optimizer_wall_hours': sum(v['training_wall_seconds'] for v in measured.values())/3600,
        'sum_optimizer_CPU_hours': sum(v['training_cpu_seconds'] for v in measured.values())/3600,
    }, indent=2)+'\n')
    print(json.dumps({'unique_replays': len(out), 'updated_table_cells': changes, 'output': str(OUT)}))


if __name__ == '__main__':
    main()
