"""Fresh, bounded CERMICS transport; mutations are attempted once then reconciled."""
from __future__ import annotations
import argparse
import subprocess
import time
from pathlib import Path

PROXY = 'ssh -o BatchMode=yes -o ConnectTimeout=15 -o ConnectionAttempts=1 -o ControlMaster=no -o ControlPath=none -o StrictHostKeyChecking=yes -W %h:%p passerelle_enpc'
SSH = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', '-o', 'ConnectionAttempts=1',
       '-o', 'ControlMaster=no', '-o', 'ControlPath=none', '-o', 'StrictHostKeyChecking=yes',
       '-o', 'ProxyCommand=' + PROXY, 'jonathan.hornewall@master01-hpc']

def cermics(command, *, mutation=False, timeout=50, input_data=None):
    for index in range(1 if mutation else 3):
        try:
            p = subprocess.run(SSH + [command], input=input_data, capture_output=True, timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            if mutation:
                raise RuntimeError('Unknown mutation outcome; reconcile before retry') from exc
            retry = True
        else:
            if p.returncode == 0:
                return p.stdout
            error = p.stderr.decode(errors='replace')
            low = error.lower()
            retry = p.returncode == 255 and any(s in low for s in (
                'timed out', 'connection reset', 'connection refused', 'connection closed',
                'broken pipe', 'kex_exchange_identification')) and not any(s in low for s in (
                'permission denied', 'host key', 'authenticat', 'bad configuration', 'could not resolve'))
            if mutation or not retry:
                raise RuntimeError(f'Remote returncode {p.returncode}: {error[-1800:]}\n{p.stdout.decode(errors="replace")[-2500:]}')
        if index == 2:
            raise RuntimeError('Read-only transport exhausted; pause schedule and alert once')
        time.sleep(10 * (index + 1))
    raise RuntimeError('Unreconciled mutation')

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('command')
    p.add_argument('--mutation', action='store_true')
    p.add_argument('--input', type=Path)
    p.add_argument('--save', type=Path)
    args = p.parse_args()
    out = cermics(args.command, mutation=args.mutation,
                  input_data=args.input.read_bytes() if args.input else None)
    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        args.save.write_bytes(out)
    else:
        print(out.decode(errors='replace'))
