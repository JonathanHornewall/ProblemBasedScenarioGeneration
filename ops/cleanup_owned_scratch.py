"""Delete only the exact campaign scratch after verified compact GitHub/Drive backup."""
import json, os, pathlib, shutil, socket, time

root = pathlib.Path('/home/jonathan.hornewall/runs/lava_completion_20261102')
scratch = pathlib.Path('/tmp/lava_completion_20261102')
host = socket.gethostname().split('.')[0]
assert host in {'node04', 'node05', 'node07', 'node08', 'node09'}
assert not root.is_symlink() and root.stat().st_uid == os.getuid()
backup = json.loads((root / 'ops/FINAL_BACKUP_VERIFIED.json').read_text())
assert backup['github']['verified'] and backup['drive']['verified'] and backup['retrieval']['verified']
terminal = json.loads((root / 'EXECUTION_OUTCOMES_COMPLETE.json').read_text())
assert terminal['successful_fits'] == 670 and terminal['resource_censored_fits'] == 40
assert shutil.rmtree.avoids_symlink_attacks
record = root / 'ops/terminal_scratch_cleanup' / (host + '.json')
assert not record.exists(), 'Cleanup already attempted; reconcile only'
record.parent.mkdir(exist_ok=True)
out = {'host': host, 'exact_path': str(scratch), 'started_unix': time.time(),
       'slurm_job_id': os.environ.get('SLURM_JOB_ID'), 'state': 'attempted',
       'follows_symlinks': False, 'scientific_work': False}
record.write_text(json.dumps(out, indent=2) + '\n')
if scratch.exists():
    assert not scratch.is_symlink() and scratch.stat().st_uid == os.getuid()
    assert scratch.resolve() == scratch
    shutil.rmtree(scratch)
    out['action'] = 'removed_exact_owned_scratch'
else:
    out['action'] = 'already_absent_no_deletion'
out.update(state='verified_absent', finished_unix=time.time(), verified_absent=not scratch.exists())
assert out['verified_absent']
record.write_text(json.dumps(out, indent=2) + '\n')
print(json.dumps(out))
