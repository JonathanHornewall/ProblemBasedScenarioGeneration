"""Reprice exact live claims without signalling children; reserve an idle metadata slot."""
import fcntl,json,os,pathlib,signal,socket,time
import psutil
from qualify import read,write
from completion import memory

r=pathlib.Path(os.environ['LAVA_RUN_ROOT']);policy=read(r/'ops/SP25_ADMISSION_POLICY_20261104.json')
assert os.environ['SLURM_JOB_ID']=='16726'and socket.gethostname().startswith('node07')
path=r/'ops/SP25_ADMISSION_REBALANCE_NODE07.json'
if path.exists():raise RuntimeError('Mutation already attempted; reconcile its receipt')
out={'started_unix':time.time(),'claims':[],'frozen_idle_workers':[],'preserved_active_workers':[],'healthy_children_signalled':0,'new_allocations':0}
write(path,out)
local=pathlib.Path('/tmp/lava_completion_20261102');slot=policy['reserved_coordinator_worker']
worker_file=r/'workers'/(socket.gethostname()+'_'+slot['index']+'_'+slot['job']+'.json')
v=read(worker_file);p=psutil.Process(v['pid']);cmd=p.cmdline()
assert v['state']=='waiting_for_dependencies_or_memory'and time.time()-v['updated_unix']<30
assert str(r/'ops/completion.py')in cmd and cmd[-2:]==['--worker',slot['index']]
assert p.environ()['SLURM_JOB_ID']==slot['job']and p.cpu_affinity()==[slot['core']]
paused=False
try:
 p.send_signal(signal.SIGSTOP);paused=True
 end=time.monotonic()+2
 while p.status()!=psutil.STATUS_STOPPED and time.monotonic()<end:time.sleep(.01)
 assert p.status()==psutil.STATUS_STOPPED
 with(local/'reservations.lock').open('a')as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);held=read(local/'reservations.json')
  assert read(worker_file)['state']=='waiting_for_dependencies_or_memory'
  assert not any(v['worker_pid']==p.pid for v in held.values())
  assert not[c for c in p.children(recursive=True)if c.status()!=psutil.STATUS_ZOMBIE]
  out['coordinator_slot']={'old_idle_worker_pid':p.pid,'old_birth_unix':p.create_time(),**slot}
  write(path,out);p.kill();paused=False
  write(worker_file,{'state':'reserved_for_metadata_coordinator','updated_unix':time.time(),'slurm_job_id':slot['job'],'core':slot['core']})
finally:
 if paused:
  try:p.send_signal(signal.SIGCONT)
  except psutil.NoSuchProcess:pass
# Freeze only proven idle metadata while claims are repriced. Active children
# and their parents continue; the fresh workers below replace these idle PIDs.
for file in sorted((r/'workers').glob(socket.gethostname()+'_*.json')):
 v=read(file)
 if v.get('state')!='waiting_for_dependencies_or_memory' or v.get('slurm_job_id')not in('16726','16740'):continue
 if time.time()-v.get('updated_unix',0)>30:continue
 proc=psutil.Process(v['pid']);cmd=proc.cmdline()
 assert str(r/'ops/completion.py')in cmd and '--worker'in cmd
 proc.send_signal(signal.SIGSTOP);frozen=False
 try:
  end=time.monotonic()+2
  while proc.status()!=psutil.STATUS_STOPPED and time.monotonic()<end:time.sleep(.01)
  assert proc.status()==psutil.STATUS_STOPPED
  with(local/'reservations.lock').open('a')as lock:
   fcntl.flock(lock,fcntl.LOCK_EX);held=read(local/'reservations.json')
   active=[c.pid for c in proc.children(recursive=True)if c.status()!=psutil.STATUS_ZOMBIE]
   reserved=any(z['worker_pid']==proc.pid for z in held.values())
   if active or reserved or read(file).get('state')!='waiting_for_dependencies_or_memory':
    out['preserved_active_workers'].append({'pid':proc.pid,'children':active,'reserved':reserved})
   else:
    out['frozen_idle_workers'].append({'pid':proc.pid,'birth_unix':proc.create_time(),'index':cmd[-1],'job':v['slurm_job_id'],'core':proc.cpu_affinity()})
    frozen=True;write(path,out)
 finally:
  if not frozen:proc.send_signal(signal.SIGCONT)
with(local/'reservations.lock').open('a')as lock:
 fcntl.flock(lock,fcntl.LOCK_EX);held=read(local/'reservations.json')
 out['reservations_before']=held.copy()
 for token,entry in list(held.items()):
  parent=psutil.Process(entry['worker_pid'])
  children=[c for c in parent.children()if c.status()!=psutil.STATUS_ZOMBIE and any(str(r/'ops'/f)in c.cmdline()for f in('prepare_bank.py','run_fit.py'))]
  if len(children)!=1:raise RuntimeError('Active claim needs one exact child: '+token)
  child=children[0];cmd=child.cmdline();is_bank=str(r/'ops/prepare_bank.py')in cmd
  if is_bank:
   key=cmd[cmd.index('--key')+1];route=key.split('__data')[0];item={'kind':'prepare','route':route,'n':int(cmd[cmd.index('--maximum')+1])}
   assert entry['id']=='bank__'+key
  else:
   job=read(pathlib.Path(cmd[cmd.index('--manifest')+1]))['jobs'][0]
   assert job['id']==entry['id'];item={'kind':'fit','route':job['route'],'n':job['config']['sizes']['train']}
  if item['route']!=policy['route']:raise RuntimeError('Unexpected non-SP25 claim')
  new=memory(item);limit=24 if is_bank else 6
  highwater=int(next(line for line in pathlib.Path('/proc/'+str(child.pid)+'/status').read_text().splitlines()if line.startswith('VmHWM:')).split()[1])*1024
  heartbeat=read(r/'workers'/(token+'.json'));highwater=max(highwater,heartbeat.get('rss_bytes',0))
  assert highwater<new*2**30/1.5 and entry['gib']in(new,limit)
  out['claims'].append({'token':token,'child_pid':child.pid,'child_birth_unix':child.create_time(),'parent_pid':parent.pid,'before_GiB':entry['gib'],'after_GiB':new,'retained_child_guard_GiB':limit,'measured_peak_bytes':highwater,'CPU_seconds':sum(child.cpu_times()[:2]),'id':entry['id']})
  held[token]={**entry,'gib':new,'child_memory_limit_gib':limit,'admission_revision':policy['revision'],'repriced_unix':time.time()}
 assert sum(v['gib']for v in held.values())<=230
 assert sum(v['gib']for k,v in held.items()if k.endswith('_16740'))<=96
 write(path,out);write(local/'reservations.json',held)
 out['reservations_after']=held;out['finished_unix']=time.time();out['state']='applied'
 write(path,out)
print(json.dumps(out))
