"""One terminal compact transfer bundle; source, seeds and measured quantities only."""
import hashlib, io, json, os, pathlib, socket, tarfile, time

root = pathlib.Path('/home/jonathan.hornewall/runs/lava_completion_20261102')
scratch = pathlib.Path('/tmp/lava_completion_20261102')
assert socket.gethostname().startswith('node07')
terminal = json.loads((root / 'EXECUTION_OUTCOMES_COMPLETE.json').read_text())
assert terminal['successful_fits'] == 670 and terminal['resource_censored_fits'] == 40
assert terminal['grandfathered_active_fits_drained']
assert not root.is_symlink() and not scratch.is_symlink()
target = root / 'retained_compact'
target.mkdir(exist_ok=True)
receipt = root / 'ops/TERMINAL_COMPACT_RETENTION.json'
assert not receipt.exists(), 'Already attempted; reconcile existing receipt and archive'

def digest(raw):
    return hashlib.sha256(raw).hexdigest()

def walk(folder):
    for base, dirs, files in os.walk(folder, followlinks=False):
        dirs[:] = sorted(d for d in dirs if d != '__pycache__' and not pathlib.Path(base, d).is_symlink())
        for name in sorted(files):
            p = pathlib.Path(base, name)
            if p.is_file() and not p.is_symlink():
                yield p

def add(tar, path, name, records):
    raw = path.read_bytes()
    info = tarfile.TarInfo(name)
    info.size, info.mode, info.mtime = len(raw), 0o600, path.stat().st_mtime
    tar.addfile(info, io.BytesIO(raw))
    records.append({'path': name, 'bytes': len(raw), 'sha256': digest(raw)})

started = time.time()
seed_archive = target / 'node07_seed_records.tar.gz'
seed_records = []
with tarfile.open(seed_archive, 'x:gz', compresslevel=3) as tar:
    for base in ('banks', 'seeded_banks_v2'):
        for p in walk(scratch / base):
            if p.suffix == '.json' or (p.name.startswith('rng_sample') and p.suffix == '.pickle'):
                assert p.stat().st_size < 1024**2
                add(tar, p, str(p.relative_to(scratch)), seed_records)
seed_manifest = {'host': socket.gethostname(), 'files': seed_records,
                 'file_count': len(seed_records), 'uncompressed_bytes': sum(v['bytes'] for v in seed_records),
                 'archive_bytes': seed_archive.stat().st_size, 'archive_sha256': digest(seed_archive.read_bytes()),
                 'neural_weights': 0, 'dense_cache_files': 0}
(target / 'node07_seed_records_manifest.json').write_text(json.dumps(seed_manifest, separators=(',', ':')))

allowed = {'.json', '.jsonl', '.csv', '.tsv', '.md', '.txt', '.log', '.out', '.err',
           '.py', '.sh', '.sbatch', '.toml', '.yaml', '.yml'}
folders = ('architecture_choices', 'attempts', 'cfl100_requests', 'failures', 'jobs', 'logs', 'ops',
           'provenance', 'provenance_inputs', 'qualification', 'recovery', 'results', 'selections',
           'seeded_requests', 'implementation_checks', 'workers', 'banks', 'seeded_banks_v2')
paths = [p for p in root.iterdir() if p.is_file() and not p.is_symlink()
         and (p.suffix in {'.json', '.md', '.txt'} or p.name == 'source.tar.gz')]
for folder in folders:
    paths += [p for p in walk(root / folder) if p.suffix in allowed]
paths += [target / 'node09_seed_records.tar.gz', seed_archive, target / 'node07_seed_records_manifest.json']
archive = target / 'remote_compact_final.tar.gz'
records = []
with tarfile.open(archive, 'x:gz', compresslevel=3, dereference=False) as tar:
    for p in sorted(set(paths)):
        assert p.is_file() and not p.is_symlink()
        add(tar, p, str(p.relative_to(root)), records)
    manifest = json.dumps({'files': records, 'dense_caches': 0, 'neural_weights': 0,
                           'scope': 'source, original seeds/RNG, selected configs, scalar results, validation tables, clocks, failures and logs'},
                          separators=(',', ':')).encode()
    info = tarfile.TarInfo('COMPACT_MANIFEST.json'); info.size = len(manifest); info.mode = 0o600
    tar.addfile(info, io.BytesIO(manifest))
out = {'state': 'retained_on_shared_home', 'started_unix': started, 'finished_unix': time.time(),
       'archive': str(archive), 'archive_bytes': archive.stat().st_size,
       'archive_sha256': digest(archive.read_bytes()), 'file_count': len(records),
       'uncompressed_bytes': sum(v['bytes'] for v in records),
       'node07_seed_files': len(seed_records), 'node07_seed_archive_sha256': seed_manifest['archive_sha256'],
       'scratch_removed': False, 'weights': 0, 'dense_caches': 0,
       'slurm_job_id': os.environ.get('SLURM_JOB_ID')}
receipt.write_text(json.dumps(out, indent=2) + '\n')
print(json.dumps(out))
