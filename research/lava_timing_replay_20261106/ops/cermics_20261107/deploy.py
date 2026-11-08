"""Verify and stage the approved replay package and the small site overlay once."""
import datetime as dt
import hashlib
import io
import json
import shlex
import tarfile
from pathlib import Path

from remote import cermics

OPS = Path(__file__).resolve().parent
CAMPAIGN = OPS.parents[1]
REMOTE = "/home/jonathan.hornewall/runs/lava_timing_replay_20261107"
SOURCE_HASH = "1375e613d7c97c15402f31d5a65d16af91cb5be8748fbabd1ece31ced119f8f0"

INSTALL = r"""
import datetime, hashlib, io, json, os, pathlib, shutil, sys, tarfile
root = pathlib.Path('/home/jonathan.hornewall/runs/lava_timing_replay_20261107')
package_sha, manifest_sha, source_sha = sys.argv[1:]
ops_rel = pathlib.Path('ops/cermics_20261107')
staging = root.with_name(root.name + '.staging-' + package_sha[:12])
assert not root.exists() and not root.is_symlink()
assert not staging.exists() and not staging.is_symlink()
os.umask(0o077)
staging.mkdir(mode=0o700)

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def extract(archive, target):
    target.mkdir(parents=True, exist_ok=True)
    for member in archive:
        rel = pathlib.Path(member.name)
        assert not rel.is_absolute() and '..' not in rel.parts
        out = target / rel
        if member.isdir():
            out.mkdir(parents=True, exist_ok=True)
        elif member.isfile():
            out.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(member) as source, out.open('xb') as dest:
                shutil.copyfileobj(source, dest)
        else:
            raise RuntimeError('Non-regular archive member: ' + member.name)

with tarfile.open(fileobj=sys.stdin.buffer, mode='r|') as outer:
    extract(outer, staging)
approved = json.loads((staging / ops_rel / 'DEPLOYMENT_INPUT.json').read_text())
for name, expected in approved['files_sha256'].items():
    assert sha(staging / name) == expected, name
package = staging / 'replay_package.tar.gz'
assert sha(package) == package_sha
with tarfile.open(package) as archive:
    extract(archive, staging)
assert sha(staging / 'manifest.json') == manifest_sha
assert sha(staging / 'source.tar.gz') == source_sha
with tarfile.open(staging / 'source.tar.gz') as archive:
    extract(archive, staging / 'code')
(staging / 'logs').mkdir()
out = {'state': 'verified_staged', 'remote_root': str(root),
       'package_sha256': package_sha, 'package_bytes': package.stat().st_size,
       'manifest_sha256': manifest_sha, 'source_sha256': source_sha,
       'ops_sha256': approved['files_sha256'], 'numerical_work_on_login': False,
       'observed_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
(staging / ops_rel / 'DEPLOYMENT_VERIFIED.json').write_text(json.dumps(out, indent=2) + '\n')
staging.rename(root)
print(json.dumps(out))
"""


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    ready_path = OPS / "CERMICS_PACKAGE_READY.json"
    ready = json.loads(ready_path.read_text())
    package = ready["package"]
    plan = json.loads((OPS / "RUN_PLAN.json").read_text())
    assert ready["status"] == "ready_for_cermics_submission"
    assert ready["remote_root"] == plan["remote_root"] == REMOTE
    assert ready["attempt_cutoff_utc"] == plan["cutoff_utc"]
    assert ready["node_scratch"] == plan["scratch"]
    assert package["manifest_sha256"] == plan["manifest_sha256"]
    assert package["sha256"] == plan["package_sha256"]
    assert package["unique_fits"] == 19 and package["banks"] == 6
    assert package["package"] == "replay_package.tar.gz"
    for name, expected in ready["files_sha256"].items():
        assert sha((CAMPAIGN / name).read_bytes()) == expected, name
    payload = (CAMPAIGN / "replay_package.tar.gz").read_bytes()
    assert len(payload) == package["bytes"] and sha(payload) == package["sha256"]
    files = {"replay_package.tar.gz": payload}
    for name in ("worker.sbatch", "slurm_replay.py", "RUN_PLAN.json", "CERMICS_PACKAGE_READY.json"):
        files[f"ops/cermics_20261107/{name}"] = (OPS / name).read_bytes()
    identity = {"files_sha256": {name: sha(data) for name, data in files.items()}}
    files["ops/cermics_20261107/DEPLOYMENT_INPUT.json"] = json.dumps(identity, indent=2).encode()
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as tar:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mode = 0o600
            tar.addfile(info, io.BytesIO(data))
    attempt = OPS / "DEPLOYMENT_ATTEMPT.json"
    with attempt.open("x") as stream:
        json.dump({**identity, "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                   "remote_root": REMOTE}, stream, indent=2)
    command = shlex.join(["python3", "-B", "-c", INSTALL, package["sha256"], package["manifest_sha256"], SOURCE_HASH])
    raw = cermics(command, mutation=True, input_data=buffer.getvalue(), timeout=150)
    receipt = json.loads(raw)
    assert receipt["state"] == "verified_staged" and receipt["remote_root"] == REMOTE
    assert receipt["package_sha256"] == package["sha256"] and receipt["ops_sha256"] == identity["files_sha256"]
    (OPS / "DEPLOYMENT_VERIFIED.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
