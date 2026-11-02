"""One fresh coordinator; healthy existing numerical workers stay running."""
import fcntl,os,pathlib,time,signal
from completion import advance
from qualify import read,write
root=pathlib.Path(os.environ['LAVA_RUN_ROOT']);deadline=read(root/'CONTRACT.json')['scientific_cutoff_unix'];stopped=False
def stop(*_):
 global stopped
 stopped=True
signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
with (root/'ops/coordinator.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 write(root/'ops/COORDINATOR_ACTIVE.json',{'pid':os.getpid(),'host':os.uname().nodename,'started_unix':time.time(),
  'slurm_job_id':os.environ['SLURM_JOB_ID'],'fresh_receipt_cache':True})
 while not stopped and time.time()<deadline and not (root/'ops/MONITOR_CLOSED.json').exists():
  advance(root);time.sleep(10)
