"""Exact owned live identities and durable retained inputs; metadata only."""
import os,pathlib,socket,time,json
import psutil
from qualify import read
from scope_control import required_fit,amendment
r=pathlib.Path('/home/jonathan.hornewall/runs/lava_completion_20261102');host=socket.gethostname()
proof=read(r/'ops/GEOMETRY_PROVENANCE.json');workers={p.name:read(p) for p in (r/'workers').glob('*.json')}
out={'host':host,'observed_unix':time.time(),'fits':[],'builders':[],
 'CPU_model':next((s.split(':',1)[1].strip() for s in pathlib.Path('/proc/cpuinfo').read_text().splitlines() if s.startswith('model name')),None),
 'RAM_total_bytes':psutil.virtual_memory().total}
for proc in psutil.process_iter(['pid','cmdline','create_time','username']):
 cmd=proc.info['cmdline'] or []
 if len(cmd)<2 or cmd[1] not in (str(r/'ops/run_fit.py'),str(r/'ops/prepare_bank.py'),str(r/'ops/bank_geometry.py')):continue
 try:
  if proc.status()==psutil.STATUS_ZOMBIE:continue
  env=proc.environ();cpu=proc.cpu_times();base={'pid':proc.pid,'birth_unix':proc.info['create_time'],'host':host,
   'cmdline':cmd,'slurm_job_id':env.get('SLURM_JOB_ID'),'current_CPU_seconds':cpu.user+cpu.system,
   'current_RSS_bytes':proc.memory_info().rss,'elapsed_wall_seconds':time.time()-proc.info['create_time']}
 except psutil.NoSuchProcess:continue
 if cmd[1]==str(r/'ops/run_fit.py'):
  manifest=pathlib.Path(cmd[cmd.index('--manifest')+1])
  if not manifest.is_relative_to(r):continue
  job=read(manifest)['jobs'][0];receipt=r/'results/receipts'/(job['id']+'.json');progress=r/'results/progress'/(job['id']+'.json')
  v=read(receipt) if receipt.exists() else read(progress) if progress.exists() else {}
  if v.get('status') in ('failed','resource_censored'):continue
  owner=next((name for name,w in workers.items() if w.get('state')=='active' and w.get('pid')==proc.pid),None)
  out['fits'].append({**base,'id':job['id'],'job':job,'required_after_amendment':required_fit(r,job),
   'worker_record':owner,'status':v.get('status'),'phase':v.get('phase'),'epochs':v.get('epochs_completed',(v.get('summary')or{}).get('epochs_completed'))})
 elif '--key' in cmd:
  key=cmd[cmd.index('--key')+1];maximum=int(cmd[cmd.index('--maximum')+1]);route,data=key.split('__data')
  if 'cfl' not in route and route not in (amendment(r)or{}).get('excluded_routes',[]):continue
  seed=int(cmd[cmd.index('--initialization-seed')+1]) if '--initialization-seed' in cmd else None
  if seed is None:
   plan=read(r/'PLAN.json');s=next((s for s in plan['cells'][route]['confirmation'] if str(s['data'])==data),plan['cells'][route]['screen']);seed=s['initialization']
  candidates=[pathlib.Path('/tmp/lava_completion_20261102/seeded_banks_v2')/(key+'__init'+str(seed)),pathlib.Path('/tmp/lava_completion_20261102/banks')/key]
  prefixes=[]
  for where in candidates:
   p=where/'prefix100.json';meta=read(p) if p.exists() else None;pick=where/'prefix100.pickle'
   if not meta or not pick.is_file() or pick.stat().st_size!=meta.get('metadata_bytes'):continue
   seeded=meta.get('geometry_rng_protocol')=='seeded_numpy_resume_v2' and meta.get('initialization_seed')==seed and (where/'rng_sample000100.pickle').is_file()
   independent=proof.get(key,{}).get('random_free_supplied_prefix',0)>=100
   if seeded or independent:prefixes.append({'where':str(where),'prefix100':meta,'admissible_seeded':seeded,'admissible_random_free':independent})
  out['builders'].append({**base,'key':key,'route':route,'maximum':maximum,'initialization_seed':seed,
   'owned_process_group':os.getpgid(proc.pid)==proc.pid,'durable_N100_inputs':prefixes})
print(json.dumps(out))
