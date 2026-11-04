"""Applied, bounded worker-service and healthy-child continuity readback."""
import json,pathlib,subprocess,time
from qualify import read,write
r=pathlib.Path('/home/jonathan.hornewall/runs/lava_completion_20261102');path=r/'ops/WORKER_SERVICE_REPAIR_APPLIED_20261104.json'
out={'observed_unix':time.time(),'status':'readback_started'}
def node_step(entry=None,code=None):
 cmd=['srun','--overlap','--jobid=16726','--nodelist=node07','--ntasks=1','--cpus-per-task=1','--time=00:00:20','--cpu-bind=cores']
 cmd+=['bash',str(r/'ops/coordinator_env.sh'),str(r),entry]if entry else['/home/jonathan.hornewall/runs/scaled_data_neural_20261021/venv/bin/python','-B','-c',code]
 p=subprocess.run(cmd,capture_output=True,text=True,timeout=23)
 if p.returncode:raise RuntimeError(p.stderr+p.stdout)
 return json.loads(p.stdout)
out['node']=node_step('sp25_resource_snapshot.py');write(path,out)
code=r'''
import pathlib,json,psutil
r=pathlib.Path('/home/jonathan.hornewall/runs/lava_completion_20261102');workers=[];services=[]
for p in psutil.process_iter(['pid','cmdline','status']):
 try:
  cmd=p.info['cmdline']or[]
  if p.info['status']=='zombie'or len(cmd)<2:continue
  if cmd[1]==str(r/'ops/completion.py')and'--worker'in cmd:workers.append({'pid':p.pid,'job':p.environ().get('SLURM_JOB_ID'),'step':p.environ().get('SLURM_STEP_ID'),'index':cmd[-1],'core':p.cpu_affinity()})
  if cmd[1]==str(r/'ops/restore_dead_workers.py'):services.append({'pid':p.pid,'job':p.environ().get('SLURM_JOB_ID'),'step':p.environ().get('SLURM_STEP_ID'),'status':p.status()})
 except psutil.NoSuchProcess:continue
print(json.dumps({'workers':workers,'services':services}))
'''
out['service_metadata']=node_step(code=code)
out['restoration']={str(j):read(r/'ops'/('DEAD_WORKERS_RESTORED_'+str(j)+'_20261104.json'))for j in(16726,16740)}
out['interrupted_attempts']={p.name:read(p)for p in(r/'recovery/worker_step_loss').glob('*.json')}
ready=read(r/'READY.json');out['ready']={k:v for k,v in ready.items()if k not in('fits','preparation')};write(path,out)
old=json.loads(read(r/'ops/WORKER_STEP_LOSS_DIAGNOSIS_20261104.json')['node'])['processes'];ps=out['node']['numeric'];by={v['pid']:v for v in ps};continuity=[]
for before in old:
 cmd=before['cmd']
 if len(cmd)<2 or cmd[1]not in(str(r/'ops/run_fit.py'),str(r/'ops/prepare_bank.py')):continue
 if before['pid']in by:
  after=by[before['pid']];assert after['birth_unix']==before['birth_unix']and after['CPU_seconds_after']>before['CPU_seconds']
  continuity.append({'pid':before['pid'],'state':'same_PID_birth_advancing'})
 else:
  assert cmd[1]==str(r/'ops/run_fit.py');job=read(pathlib.Path(cmd[cmd.index('--manifest')+1]))['jobs'][0]
  v=read(r/'results/receipts'/(job['id']+'.json'));assert v['status']=='complete'and v['summary']['epochs_completed']==300
  continuity.append({'pid':before['pid'],'state':'naturally_completed300epochs','id':job['id']})
out['healthy_continuity']=continuity
assert len(out['service_metadata']['workers'])==31 and len(out['service_metadata']['services'])==2
assert sum(len(v['restored'])for v in out['restoration'].values())==11
assert ready['resource_censored_SP_CFL_fits']==40 and len(out['node']['coordinators'])==1 and out['node']['coordinators'][0]['pid']==2281033
assert len({v['affinity'][0]for v in ps})==len(ps)
assert sum(v['gib']for v in out['node']['reservations'].values())<=230
assert sum(v['gib']for k,v in out['node']['reservations'].items()if k.endswith('_16740'))<=96
active={p.get('id')for p in ps if p['entry']=='run_fit.py'};blocked=[];unstarted=[]
for item in ready['fits']:
 if item['id']in active or(r/'results/receipts'/(item['id']+'.json')).exists():continue
 job=read(r/'jobs'/(item['id']+'.json'))['jobs'][0];key=item['key']+'__init'+str(job['config']['seeds']['initialization'])
 ((unstarted if(r/'seeded_banks_v2'/key/('prefix'+str(item['n'])+'.json')).exists()else blocked)).append(item)
out['unstarted_dispatchable']=unstarted;out['blocked_prefix_dependencies']=blocked
assert not unstarted
out['actual_interval_numerical_cores']=sum(v['effective_cores_interval']for v in ps);out['status']='repaired_verified';out['finished_unix']=time.time();write(path,out)
print(json.dumps(out))
