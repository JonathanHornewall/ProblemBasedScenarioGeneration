"""Finite owner-local queue for diagnosed seeded-bank construction."""
import fcntl,os,pathlib,signal,socket,subprocess,sys,time
import psutil
from qualify import read,write
from bank_geometry import directory
from safe_geometry import release_unstarted_holds
from scope_control import preparation_maximum
root=pathlib.Path(os.environ['LAVA_RUN_ROOT']);host=socket.gethostname();job=os.environ['SLURM_JOB_ID']
deadline=read(root/'CONTRACT.json')['scientific_cutoff_unix'];stopped=False
def stop(*_):
 global stopped
 stopped=True
signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
local=pathlib.Path('/tmp/lava_completion_20261102');local.mkdir(exist_ok=True)
token=host+'_seeded_'+job;reservation=local/'reservations.json';guard=local/'reservations.lock'
request_folder=root/os.environ.get('LAVA_REQUEST_FOLDER','seeded_requests')
while not stopped and time.time()<deadline and not (root/'ops/MONITOR_CLOSED.json').exists() and not (root/'SCIENCE_COMPLETE.json').exists():
 chosen=None;claim=None
 for request in sorted(request_folder.glob('*.json')):
  value=read(request)
  if value['host']!=host:continue
  key=value['key'];seed=value['initialization_seed'];n=preparation_maximum(root,key,value['maximum']);ident=key+'__init'+str(seed)
  if not n:continue
  value={**value,'maximum':n}
  policy=root/'ops/CFL100_N100_RESOURCE_POLICY.json';resource_policy=read(policy) if policy.exists() else {}
  if 'cfl_100x10' in key and resource_policy and job not in resource_policy.get('authorized_jobs',[]):continue
  if (directory(key,seed)/f'prefix{n}.json').exists():
   release_unstarted_holds(root,key,seed,n);continue
  failure=root/'failures'/('seeded__'+ident+'.json')
  if failure.exists():
   retry='cfl_100x10' in key and job in resource_policy.get('authorized_jobs',[]) and read(failure).get('finished_unix',0)<resource_policy.get('activated_unix',0)
   if not retry:continue
  candidate=(root/'claims'/('seeded__'+ident+'.lock')).open('a')
  try:fcntl.flock(candidate,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:candidate.close();continue
  gib=96. if '50x50' in key or 'cfl_100x10' in key else 32. if 'cfl' in key else 24. if '25x25' in key else 6.
  with guard.open('a') as g:
   fcntl.flock(g,fcntl.LOCK_EX);held=read(reservation) if reservation.exists() else {}
   for k,v in list(held.items()):
    try:os.kill(v['worker_pid'],0)
    except ProcessLookupError:held.pop(k)
   if sum(v['gib'] for v in held.values())+gib>230:
    candidate.close();continue
   held[token]={'gib':gib,'worker_pid':os.getpid(),'id':'seeded__'+ident};write(reservation,held)
  chosen=value;claim=candidate;break
 if chosen is None:time.sleep(10);continue
 key=chosen['key'];seed=chosen['initialization_seed'];n=chosen['maximum'];ident=key+'__init'+str(seed)
 # Preserve valid old rows only when their consumed prefix has an exact
 # random-free proof. No neighbor construction is run by this check.
 legacy=root/'banks'/key
 prefixes=[int(p.stem[6:]) for p in legacy.glob('prefix*.json') if int(p.stem[6:])<=n]
 if prefixes:
  from geometry_provenance import certify
  certify(root,key,max(prefixes))
 opened=time.monotonic();reason=None
 with (root/'logs'/('seeded__'+ident+'.log')).open('ab') as log:
  child=subprocess.Popen([sys.executable,str(root/'ops/prepare_bank.py'),'--root',str(root),
   '--key',key,'--maximum',str(n),'--initialization-seed',str(seed)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  while True:
   pid,status,usage=os.wait4(child.pid,os.WNOHANG)
   if pid:child.returncode=os.waitstatus_to_exitcode(status);break
   try:
    proc=psutil.Process(child.pid);rss=proc.memory_info().rss;t=proc.cpu_times();cpu=t.user+t.system
   except psutil.Error:rss=0;cpu=0
   if stopped or time.time()>=deadline or time.monotonic()-opened>86400 or rss>gib*2**30 or psutil.virtual_memory().available<12*2**30:
    reason='allocation_stop' if stopped else 'absolute_deadline' if time.time()>=deadline else 'finite_wall_budget' if time.monotonic()-opened>86400 else 'memory_headroom_censor'
    os.killpg(child.pid,signal.SIGKILL);_,status,usage=os.wait4(child.pid,0);child.returncode=os.waitstatus_to_exitcode(status);break
   write(root/'workers'/(token+'.json'),{'state':'active','item':{'kind':'seeded_prepare','key':key,'n':n},
    'pid':child.pid,'worker_pid':os.getpid(),'rss_bytes':rss,'actual_process_cpu_seconds':cpu,
    'updated_unix':time.time(),'slurm_job_id':job})
   time.sleep(5)
 outcome={'key':key,'initialization_seed':seed,'maximum':n,'returncode':child.returncode,'reason':reason,
  'elapsed_wall_seconds':time.monotonic()-opened,'process_cpu_seconds':usage.ru_utime+usage.ru_stime,
  'peak_rss_bytes':usage.ru_maxrss*1024,'finished_unix':time.time(),'host':host,'slurm_job_id':job}
 write(root/'attempts'/('seeded__'+ident+'__'+str(time.time_ns())+'.json'),outcome)
 if child.returncode:write(root/'failures'/('seeded__'+ident+'.json'),outcome)
 else:release_unstarted_holds(root,key,seed,n)
 with guard.open('a') as g:
  fcntl.flock(g,fcntl.LOCK_EX);held=read(reservation);held.pop(token,None);write(reservation,held)
 claim.close()
 write(root/'workers'/(token+'.json'),{'state':'waiting_for_seeded_requests','updated_unix':time.time(),'slurm_job_id':job,'pid':os.getpid()})
write(root/'workers'/(token+'.json'),{'state':'ended','updated_unix':time.time(),'slurm_job_id':job})
