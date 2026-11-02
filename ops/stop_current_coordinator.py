"""Stop the exact metadata-only coordinator; numerical workers are untouched."""
import os,pathlib,time
import psutil
from qualify import read,write
r=pathlib.Path('/home/jonathan.hornewall/runs/lava_completion_20261102');old=read(r/'ops/COORDINATOR_ACTIVE.json')
p=psutil.Process(old['pid']);cmd=p.cmdline()
if len(cmd)<2 or cmd[1]!=str(r/'ops/coordinator.py') or p.environ().get('SLURM_JOB_ID')!='16223':raise RuntimeError('Coordinator identity mismatch')
record={'old':old,'cmdline':cmd,'observed_unix':time.time(),'reason':os.environ.get('LAVA_METADATA_RELOAD_REASON','Load explicit replacements for23 Torch startup failures; no worker is signalled.')}
write(r/'ops/coordinator_handoff'/(str(time.time_ns())+'__interop.json'),record)
p.terminate();p.wait(timeout=15)
print('Exact metadata coordinator exited; numerical workers preserved')
