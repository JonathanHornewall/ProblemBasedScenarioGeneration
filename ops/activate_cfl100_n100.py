"""Required N100 bank recovery: original ten paired banks, observed-memory repair."""
import pathlib,subprocess,time,json
from qualify import read,write
r=pathlib.Path('/home/jonathan.hornewall/runs/lava_completion_20261102');plan=read(r/'PLAN.json');reg=read(r/'ops/JOB_REGISTRY.json')
policy=r/'ops/CFL100_N100_RESOURCE_POLICY.json'
if policy.exists():raise RuntimeError('Resource policy already exists; reconcile rather than resubmit')
write(policy,{'revision':'CFL100_N100_96GiB_20261102','authorized_jobs':[],
 'observed_prior_guard_GiB':48,'observed_prior_peak_RSS_bytes':52020649984,
 'preparation_reservation_GiB':96,'node_aggregate_GiB':230,'new_parameters_are_resource_limits_only':True})
requests=[]
for i,seeds in enumerate(plan['cells']['dys_cfl_100x10']['confirmation']):
 key='dys_cfl_100x10__data'+str(seeds['data']);status=r/'banks'/key/'STATUS.json'
 owner=read(status).get('host') if status.exists() else None
 owner=owner or ('node09.cluster.local' if i%2==0 else 'node07.cluster.local')
 value={'key':key,'initialization_seed':seeds['initialization'],'maximum':100,'host':owner,
  'source_job_ids':[],'reason':'Required retained N100 completion; resume exact checkpoints with96GiB guard', 'updated_unix':time.time()}
 write(r/'cfl100_requests'/(key+'__init'+str(seeds['initialization'])+'.json'),value);requests.append(value)
 for old in (r/'seeded_requests').glob(key+'__init*.json'):
  write(r/'ops/scope_saved_requests'/(str(time.time_ns())+'__'+old.name),read(old));old.rename(r/'ops/scope_saved_requests'/(str(time.time_ns())+'__moved__'+old.name))
ad=read(r/'ADMISSION.json');ad['dedicated_preparation_routes']=['dys_cfl_100x10'];write(r/'ADMISSION.json',ad)
jobs=[];remaining=int(read(r/'CONTRACT.json')['deadline_unix']-time.time()-60)
d,rem=divmod(remaining,86400);h,rem=divmod(rem,3600);m,s=divmod(rem,60);limit=f'{d}-{h:02}:{m:02}:{s:02}'
for node in ('node07','node09'):
 q=subprocess.run(['squeue','-h','-r','-u','jonathan.hornewall','-o','%C|%j'],capture_output=True,text=True,timeout=5)
 slots=sum(int(v.split('|')[0])for v in q.stdout.splitlines()if '20261102' in v)
 if slots+1>100:raise RuntimeError('100CPU envelope')
 p=subprocess.run(['sbatch','--parsable','--partition=defq-gpu','--nodelist='+node,'--nice=10000','--requeue',
  '--job-name=lava_CFL100_N100_20261102','--ntasks=1','--cpus-per-task=1','--mem=96G','--time='+limit,
  '--export=ALL,LAVA_REQUEST_FOLDER=cfl100_requests','--output='+str(r/'logs/CFL100-N100-%j.out'),str(r/'ops/seeded_preparation.sbatch'),str(r)],capture_output=True,text=True,timeout=10)
 if p.returncode:raise RuntimeError(p.stderr+p.stdout)
 job=int(p.stdout.strip());jobs.append(job);reg['cfl100_'+node+'_pool']=job;reg['largest_total_requested_cpus_so_far']=max(reg.get('largest_total_requested_cpus_so_far',0),slots+1);write(r/'ops/JOB_REGISTRY.json',reg)
v=read(policy);v.update({'authorized_jobs':[str(j)for j in jobs],'activated_unix':time.time()});write(policy,v)
print(json.dumps({'jobs':jobs,'requests':requests,'policy':v,'registry':reg}))
