"""Package the 19 original winners and their inputs. No fitting or enumeration."""
import csv
import gzip
import hashlib
import io
import json
import pickletools
import re
import tarfile
from pathlib import Path


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
OLD = HERE.parent / 'lava_completion_20261102'
SNAPSHOT = OLD / 'retained_compact/remote_snapshot'
PILOT = PROJECT / 'ProximalSPO-parsimonious-pilot-20261019/runs'
BASE = PILOT / ('parsimonious_pilot_20261019/archival_evidence/'
                'terminal_snapshot_v5_20261019T195606Z_1792439766548107000/frozen_data')
FIRST = 'lava_complete__3612427e518ddbe8fdb8aa37929a8c00'
SEEDS = dict(data=151101, initialization=161101, shuffle=171101, perturbation=181101, structural=101)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def rows(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def unique_fits():
    table = rows(HERE.parent / 'lava_inclusive_regret_20261104/training_times_300_epochs.csv')
    outcomes = {r['id']: r for r in rows(OLD / 'artifacts/required_outcomes.csv')}
    jobs, banks = {}, {}
    for cell in table:
        if cell['method'] != 'LAVA' or cell['status'] != 'measured':
            continue
        outcome = outcomes.get(cell['first_seed_fit_id'])
        if not outcome:
            continue
        original = (OLD / outcome['source_receipt']).read_bytes()
        record = json.loads(original)
        geometry = record['summary']['geometry']
        if not geometry.get('lazy_node_local_vertices'):
            continue
        job = record['job']
        config = job['config']
        n = int(cell['N'])
        if (record['status'], outcome['outcome'], job['stage'], job['method'], config['seeds'],
            config['sizes']['train'], job['route'], job['architecture_depth'], record['summary']['epochs_completed']) != (
                'complete', 'valid_success', 'confirmation', 'lava', SEEDS, n, cell['route'], int(cell['depth']), 300):
            raise ValueError(f"Unmatched timing input {job['id']}")
        key = job['required_data_key']
        path = Path(geometry['separately_charged_build_receipt']).relative_to(
            '/home/jonathan.hornewall/runs/lava_completion_20261102')
        prefix = read(SNAPSHOT / path)
        if prefix['vertex_counts'] != geometry['vertex_counts'] or len(prefix['vertex_counts']) != n:
            raise ValueError(f'Prefix differs for {key}/{n}')
        banks.setdefault(key, {})[str(n)] = {k: prefix[k] for k in (
            'prefix_context_sha256', 'prefix_cost_sha256', 'metadata_sha256', 'vertex_counts',
            'public_source_sha256', 'adapter_sha256', 'vertex_storage_bytes')}
        if prefix.get('geometry_rng_protocol') != 'seeded_numpy_resume_v2':
            classification = read(OLD / 'ops/GEOMETRY_PROVENANCE.json').get(key, {})
            if classification.get('random_free_supplied_prefix', 0) < n:
                raise ValueError(f'No original RNG provenance for {key}/{n}')
        item = jobs.setdefault(job['id'], {'job': job, 'original_receipt_sha256': digest(original),
                              'original_training_wall_seconds': record['summary']['optimization_wall_seconds'],
                              'panels': []})
        item['panels'].append(cell['panel'])
    first = jobs[FIRST]['job']
    if (len(jobs), sum(len(v['panels']) for v in jobs.values()), len(banks),
        first['route'], first['config']['sizes']['train']) != (19, 36, 6, 'dys_shortest_path_25x25', 500):
        raise ValueError('Timing table and required replay set differ')
    return jobs, banks


def reference_vertices(keys):
    """Read only literal names/digests from the retained original RNG records."""
    result = {key: {} for key in keys}
    folder = SNAPSHOT / 'retained_compact'
    for archive in sorted(folder.glob('node*_seed_records.tar.gz')):
        with tarfile.open(archive) as tar:
            for member in tar:
                path = Path(member.name)
                key = path.parent.name.removesuffix('__init161101')
                if key not in keys or not re.fullmatch(r'rng_sample\d{6}\.pickle', path.name):
                    continue
                sample = int(path.stem.removeprefix('rng_sample'))
                if sample > max(map(int, keys[key])):
                    continue
                words = [arg for op, arg, _ in pickletools.genops(tar.extractfile(member).read())
                         if op.name in ('SHORT_BINUNICODE', 'BINUNICODE', 'UNICODE')]
                name = words[words.index('vertex_name') + 1]
                sha = words[words.index('vertex_sha256') + 1]
                if not re.fullmatch(r'[a-f0-9]{64}\.npy', name) or not re.fullmatch(r'[a-f0-9]{64}', sha):
                    raise ValueError(f'Invalid retained vertex identity at {path}')
                if result[key].setdefault(name, sha) != sha:
                    raise ValueError(f'Conflicting original vertex bytes at {path}')
    return result


def portable_builder():
    """Change only path/host metadata in the archived disk-backed geometry builder."""
    source = (OLD / 'ops/bank_geometry.py').read_text()
    edits = {
        "Path('/tmp/lava_completion_20261102/seeded_banks_v2')": "(Path(os.environ['LAVA_GEOMETRY_ROOT'])/'seeded_banks_v2')",
        "Path('/tmp/lava_completion_20261102/banks')": "(Path(os.environ['LAVA_GEOMETRY_ROOT'])/'banks')",
        "Path('/home/jonathan.hornewall/runs/scaled_data_neural_20261021/frozen_data')": "(root/'frozen_data')",
    }
    for old, new in edits.items():
        if source.count(old) != 1:
            raise ValueError(f'Geometry source changed near {old}')
        source = source.replace(old, new)
    return source.encode()


def main():
    jobs, banks = unique_fits()
    source = read(OLD / 'source_receipt.json')
    science = (OLD / 'source.tar.gz').read_bytes()
    if digest(science) != source['source_bundle_sha256']:
        raise ValueError('Archived numerical source differs')
    files = {'source.tar.gz': science, 'qualify.py': (OLD / 'ops/qualify.py').read_bytes(),
             'bank_geometry.py': portable_builder(), 'PLAN.json': (OLD / 'PLAN.json').read_bytes(),
             'requirements.lock': (PILOT / 'scaled_data_neural_20261021/ops/requirements.lock').read_bytes()}
    for key in sorted(banks):
        old_bank = BASE / key
        hashes = read(old_bank / 'receipt.json')['files_sha256']
        for name in ('observations.npz', 'structure.npz', 'structure.json'):
            payload = (old_bank / name).read_bytes()
            if digest(payload) != hashes[name]:
                raise ValueError(f'Original observed data or structure differs: {key}/{name}')
            files[f'base/{key}/{name}'] = payload
    manifest = {'first': FIRST, 'site': 'cermics', 'source': source,
                'seeds': SEEDS, 'jobs': jobs, 'banks': banks, 'original_vertex_hashes': reference_vertices(banks)}
    encoded = (json.dumps(manifest, sort_keys=True, indent=2) + '\n').encode()
    (HERE / 'manifest.json').write_bytes(encoded)
    files['manifest.json'] = encoded
    for name in ('replay.py', 'start_cermics.sh'):
        files[name] = (HERE / name).read_bytes()
    archive = HERE / 'replay_package.tar.gz'
    tmp = archive.with_suffix('.tmp')
    with tmp.open('wb') as raw, gzip.GzipFile(filename='', fileobj=raw, mode='wb', mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode='w|') as tar:
            for name, payload in sorted(files.items()):
                info = tarfile.TarInfo(name)
                info.size = len(payload)
                info.mode = 0o755 if name.endswith('.sh') else 0o644
                tar.addfile(info, io.BytesIO(payload))
    tmp.replace(archive)
    receipt = {'package': archive.name, 'bytes': archive.stat().st_size,
               'sha256': digest(archive.read_bytes()), 'manifest_sha256': digest(encoded),
               'unique_fits': len(jobs), 'table_entries': sum(len(x['panels']) for x in jobs.values()),
               'banks': len(banks), 'first': FIRST, 'numerical_source_commit': source['scientific_source_commit']}
    (HERE / 'PACKAGE.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
