"""Keep a Slurm worker step alive across metadata-parent replacements."""
from pathlib import Path
import time

def campaign_active(root,deadline):
    root=Path(root)
    return time.time()<deadline and not any((root/name).exists() for name in
        ('ops/MONITOR_CLOSED.json','SCIENCE_COMPLETE.json','EXECUTION_OUTCOMES_COMPLETE.json'))

def keep_step_alive(root,deadline,children=(),*,interval=5):
    # A replaced original PID may be gone while its successor remains in this
    # step's cgroup. Exiting the launcher would let Slurm kill that successor.
    while campaign_active(root,deadline):
        for child in children:child.poll()
        time.sleep(interval)
