"""One five-second CPU measurement plus current exact process/receipt metadata."""
import fcntl,json,os,pathlib,socket,time
import psutil
from qualify import read
r=pathlib.Path(os.environ['LAVA_RUN_ROOT']);host=socket.gethostname();now=time.time();procs=[];other=[]
for p in psutil.process_iter(['pid','name','cmdline','status']):
 try:
  cmd=p.info['cmdline']or[]
  if not(p.info['name']or'').startswith('python')or p.info['status']=='zombie':continue
  entry=next((f for f in('prepare_bank.py','bank_geometry.py','run_fit.py','coordinator.py')if str(r/'ops'/f)in cmd),None)
  if not entry:continue
  v={'pid':p.pid,'ppid':p.ppid(),'entry':entry,'job':p.environ().get('SLURM_JOB_ID'),'birth_unix':p.create_time(),'command':cmd,'CPU_seconds_before':sum(p.cpu_times()[:2]),'affinity':p.cpu_affinity(),'status':p.status()}
  if entry=='coordinator.py':other.append(v)
  else:procs.append(v)
 except psutil.NoSuchProcess:continue
time.sleep(5);ended=time.time()
for v in procs+other:
 try:
  p=psutil.Process(v['pid']);assert p.create_time()==v['birth_unix']
  v['CPU_seconds_after']=sum(p.cpu_times()[:2]);v['effective_cores_interval']=(v['CPU_seconds_after']-v['CPU_seconds_before'])/(ended-now)
  v['RSS_bytes']=p.memory_info().rss
  v['VmHWM_bytes']=int(next(line for line in pathlib.Path('/proc/'+str(p.pid)+'/status').read_text().splitlines()if line.startswith('VmHWM:')).split()[1])*1024
  if v['entry']=='run_fit.py':
   job=read(pathlib.Path(v['command'][v['command'].index('--manifest')+1]))['jobs'][0];v['id']=job['id'];v['n']=job['config']['sizes']['train'];v['data_seed']=job['config']['seeds']['data'];v['depth']=job['config']['model']['depth']
   progress=r/'results/progress'/(job['id']+'.json');v['progress']=read(progress)if progress.exists()else None
 except psutil.NoSuchProcess:v['naturally_exited_during_sample']=True
local=pathlib.Path('/tmp/lava_completion_20261102');held=read(local/'reservations.json')if(local/'reservations.json').exists()else{}
with(r/'ops/coordinator.lock').open('a')as lock:
 try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);coordinator_lock_held=False
 except BlockingIOError:coordinator_lock_held=True
print(json.dumps({'host':host,'observed_unix':ended,'CPU_interval_seconds':ended-now,'numeric':procs,'coordinators':other,'coordinator_lock_held':coordinator_lock_held,'coordinator_record':read(r/'ops/COORDINATOR_ACTIVE.json'),'reservations':held,'available_bytes':psutil.virtual_memory().available,'workers':{p.name:read(p)for p in(r/'workers').glob(host+'*.json')}}))
