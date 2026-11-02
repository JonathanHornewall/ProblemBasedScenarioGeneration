"""Stop only snapshot-owned excluded preprocessing after retained inputs commit."""
import os,pathlib,socket,time,signal
import psutil
from qualify import read,write
from scope_control import preparation_maximum,retained_prefix_durable
r=pathlib.Path('/home/jonathan.hornewall/runs/lava_completion_20261102');host=socket.gethostname();actions=[]
snap=read(r/'ops/SCOPE_CHANGE_SNAPSHOT.json');grand=read(r/'ops/SCOPE_GRANDFATHERED_FITS.json')
for node in snap['nodes']:
 if node['host']!=host:continue
 for v in node['builders']:
  target=preparation_maximum(r,v['key'],v['maximum'])
  if target==v['maximum']:continue
  durable=retained_prefix_durable(r,v['key'],v['initialization_seed']) if target==100 else None
  if target==100 and not durable:
   actions.append({'pid':v['pid'],'action':'continue_until_N100_durable'});continue
  # Every healthy grandfathered fit already has its complete supplied prefix.
  # Keep those files; never stop a builder still needed for an absent prefix.
  dependency=False
  for fit in grand:
   if fit['job']['required_data_key']!=v['key']:continue
   n=fit['job']['config']['sizes']['train'];key=v['key'];seed=v['initialization_seed']
   paths=[pathlib.Path('/tmp/lava_completion_20261102/seeded_banks_v2')/(key+'__init'+str(seed)),pathlib.Path('/tmp/lava_completion_20261102/banks')/key]
   if not any((p/f'prefix{n}.json').exists() and (p/f'prefix{n}.pickle').exists() for p in paths):dependency=True
  if dependency:actions.append({'pid':v['pid'],'action':'retained_for_healthy_fit_dependency'});continue
  try:p=psutil.Process(v['pid'])
  except psutil.NoSuchProcess:actions.append({'pid':v['pid'],'action':'already_exited'});continue
  if abs(p.create_time()-v['birth_unix'])>.02 or p.cmdline()!=v['cmdline'] or os.getpgid(p.pid)!=p.pid or p.environ().get('SLURM_JOB_ID')!=v['slurm_job_id']:
   raise RuntimeError('Excluded builder exact identity mismatch')
  cpu=p.cpu_times();record={**v,'stopped_unix':time.time(),'stop_reason':'human_scope_excluded_suffix',
   'process_CPU_seconds_before_stop':cpu.user+cpu.system,'RSS_bytes_before_stop':p.memory_info().rss,
   'retained_N100_input':durable,'all_files_preserved':True,'scientific_failure':False}
  dest=r/'ops/scope_stopped_builders'/(v['key']+'__pid'+str(p.pid)+'.json');write(dest,record)
  os.killpg(p.pid,signal.SIGTERM);record['signal']='SIGTERM';write(dest,record);actions.append(record)
write(r/'ops/scope_stop_receipts'/(host+'.json'),{'observed_unix':time.time(),'actions':actions})
print(actions)
