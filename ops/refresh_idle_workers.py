"""Refresh each old worker only when idle; finite within its existing allocation."""
import os,pathlib,signal,socket,subprocess,sys,time
import psutil
from qualify import read,write
root=pathlib.Path(os.environ['LAVA_RUN_ROOT']);job=os.environ['SLURM_JOB_ID']
host=socket.gethostname();children=[];refreshed=[];indices=set();stopped=False
deadline=read(root/'CONTRACT.json')['scientific_cutoff_unix']
def stop(*_):
 global stopped
 stopped=True
signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
while not stopped and time.time()<deadline and len(indices)<16:
 for path in sorted((root/'workers').glob(host+'_*_'+job+'.json')):
  v=read(path)
  if v.get('state')!='waiting_for_dependencies_or_memory':continue
  try:proc=psutil.Process(v['pid']);cmd=proc.cmdline()
  except psutil.NoSuchProcess:continue
  if (len(cmd)<2 or cmd[1]!=str(root/'ops/completion.py') or '--worker' not in cmd
      or proc.environ().get('SLURM_JOB_ID')!=job):raise RuntimeError('Idle worker identity mismatch')
  index=cmd[cmd.index('--worker')+1]
  if index in indices:continue
  cores=proc.cpu_affinity()
  if len(cores)!=1:raise RuntimeError('Idle worker is not bound to one CPU')
  latest=read(path)
  if latest.get('state')!='waiting_for_dependencies_or_memory' or latest.get('pid')!=proc.pid:continue
  old_pid=proc.pid;proc.terminate();limit=time.monotonic()+15
  while proc.is_running() and proc.status()!=psutil.STATUS_ZOMBIE and time.monotonic()<limit:time.sleep(.25)
  if proc.is_running() and proc.status()!=psutil.STATUS_ZOMBIE:raise RuntimeError('Idle worker did not exit')
  child=subprocess.Popen(['taskset','-c',str(cores[0]),sys.executable,str(root/'ops/completion.py'),
   '--root',str(root),'--worker',index])
  children.append(child);indices.add(index);refreshed.append({'index':index,'old_pid':old_pid,'new_pid':child.pid,'core':cores[0]})
  write(root/'ops/IDLE_WORKER_REFRESH.json',{'slurm_job_id':job,'host':host,'refreshed':refreshed,
   'refreshed_unix':time.time(),'active_preparation_preserved':True,'pending_indices':16-len(indices)})
 time.sleep(5)
while not stopped and time.time()<deadline and any(c.poll() is None for c in children):time.sleep(5)
for c in children:
 if c.poll() is None:c.terminate()
for c in children:c.wait(timeout=30)
