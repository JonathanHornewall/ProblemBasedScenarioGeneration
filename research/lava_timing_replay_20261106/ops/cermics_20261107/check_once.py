"""One read-only CERMICS snapshot of the replay's own queue and receipts."""
import argparse
import json
from pathlib import Path

from remote import cermics

OPS = Path(__file__).resolve().parent

PROBE = r"""
import datetime, json, os, pathlib, subprocess
r = pathlib.Path('/home/jonathan.hornewall/runs/lava_timing_replay_20261107')
ops = r / 'ops/cermics_20261107'
read = lambda p: json.loads(p.read_text())
out = {'observed_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
       'remote_root': str(r), 'root_exists': r.is_dir()}
if not r.is_dir():
    print(json.dumps(out))
    raise SystemExit

def command(args):
    p = subprocess.run(args, capture_output=True, text=True, timeout=10, env={**os.environ, 'TZ': 'UTC'})
    if p.returncode:
        raise RuntimeError(p.stderr[-600:])
    return p.stdout

plan = read(ops/'RUN_PLAN.json')
manifest = read(r/'manifest.json')
slots = [label if attempt == 1 else f'{label}-a{attempt}'
         for label, route in plan['routes'].items()
         for attempt in range(1, route.get('submission_attempt', 1) + 1)]
names = ','.join('lava-replay-1107-' + slot for slot in slots)
out['queue'] = command(['squeue','-h','-u','jonathan.hornewall','-n',names,
                        '-o','%i|%P|%T|%C|%m|%N|%j|%l|%r'])
out['submissions'] = {
    slot: {name:read(ops/'submissions'/slot/name)
            for name in ('INTENT.json','RECEIPT.json','SUBMISSION_ERROR.json')
            if (ops/'submissions'/slot/name).is_file()}
    for slot in slots
}
ids = sorted({p['RECEIPT.json']['job_id'] for p in out['submissions'].values() if 'RECEIPT.json' in p})
if ids:
    out['accounting'] = command(['sacct','-X','-n','-P','-j',','.join(ids),'--starttime','2026-11-07',
                                 '-o','JobIDRaw,JobName%96,State,ExitCode,ElapsedRaw,AllocCPUS,NodeList'])
out['markers'] = {name:read(r/name) for name in ('parity.json','COMPLETE.json','TERMINAL.json')
                  if (r/name).is_file()}
out['results'] = {p.stem:read(p) for p in (r/'results').glob('*.json')}
out['valid_results'] = [key for key,value in out['results'].items()
                        if key in manifest['jobs'] and value.get('status')=='complete'
                        and value.get('original_receipt_sha256')==manifest['jobs'][key]['original_receipt_sha256']
                        and value.get('epochs')==300 and value.get('test_access') is False]
out['first_fit_id'] = manifest['first']
out['shared_preparation'] = {str(p.relative_to(r)):read(p)
                             for p in (r/'seeded_banks_v2').glob('*/STATUS.json')}
out['attempts'] = {str(p.relative_to(r)):read(p)
                   for p in (ops/'attempts').glob('*/*.json')}
out['recent_fit_progress'] = {}
for folder in sorted((r/'attempts').glob('*')):
    candidates = sorted(folder.glob('*/progress.json'), reverse=True)
    if candidates:
        out['recent_fit_progress'][folder.name] = read(candidates[0])
out['runtime'] = {}
for p in (r/'runtime').glob('*.json'):
    info = read(p)
    desc = info.pop('cpu_description', '')
    info['cpu_model'] = next((line.split(':',1)[1].strip() for line in desc.splitlines()
                              if line.startswith('model name')), None)
    out['runtime'][p.name] = info
out['failures'] = {p.name:read(p) for p in (r/'failures').glob('*.json')}
if DETAILS:
    out['job_details_UTC'] = {key: command(['scontrol','show','job',key,'-o'])
                              for key in ids if any(line.startswith(key+'|') for line in out['queue'].splitlines())}
    out['log_tails'] = {}
    paths = sorted((r/'logs').glob('*.err')) + sorted((ops/'attempts').glob('*/*.log'))
    for p in paths:
        with p.open('rb') as stream:
            stream.seek(max(0, p.stat().st_size - 2500))
            out['log_tails'][str(p.relative_to(r))] = stream.read().decode(errors='replace')
print(json.dumps(out))
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--launch-readback", action="store_true")
    args = parser.parse_args()
    if (OPS / "MONITOR_CLOSED.json").is_file():
        print(json.dumps({"state": "closed_noop"}))
        return
    script = f"DETAILS={args.launch_readback!r}\n" + PROBE
    raw = cermics("python3 -B -", input_data=script.encode(), timeout=50)
    snapshot = json.loads(raw)
    folder = OPS / "snapshots"
    folder.mkdir(exist_ok=True)
    path = folder / (snapshot["observed_utc"].replace(":", "") + ".json")
    path.write_text(json.dumps(snapshot, indent=2) + "\n")
    (OPS / "LATEST.json").write_text(path.read_text())
    output = {"snapshot": str(path), **{key:snapshot.get(key) for key in (
        "observed_utc","root_exists","queue","accounting","markers","valid_results",
        "runtime","shared_preparation","attempts","failures","job_details_UTC","log_tails"
    )}}
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
