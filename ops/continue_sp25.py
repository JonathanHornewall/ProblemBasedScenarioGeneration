"""Admit already-required SP25 prefixes under unchanged measured resource guards."""
import json,os,pathlib,subprocess,time
from qualify import read,write
from completion import memory
from scope_control import required_fit

r=pathlib.Path('/home/jonathan.hornewall/runs/lava_completion_20261102')
receipt=r/'ops/SP25_N1000_CONTINUATION.json'
if receipt.exists():raise RuntimeError('Continuation already attempted; reconcile receipt')
contract=read(r/'CONTRACT.json');plan=read(r/'PLAN.json');admission=read(r/'ADMISSION.json')
assert admission['prepare_maximum_by_route']['dys_shortest_path_25x25']==100
assert memory({'kind':'prepare','route':'dys_shortest_path_25x25','n':1000})==24
assert memory({'kind':'fit','route':'dys_shortest_path_25x25','n':1000})==6
prefixes=[]
for seed in plan['cells']['dys_shortest_path_25x25']['confirmation']:
 key='dys_shortest_path_25x25__data'+str(seed['data'])
 meta=read(r/'seeded_banks_v2'/(key+'__init'+str(seed['initialization']))/'prefix100.json')
 assert meta['status']=='complete' and meta['supplied_training_samples']==100
 assert meta['geometry_rng_protocol']=='seeded_numpy_resume_v2' and meta['initialization_seed']==seed['initialization']
 prefixes.append(meta)
remaining=contract['scientific_cutoff_unix']-time.time()
projection=max(p['build_wall_seconds_cumulative']for p in prefixes)*10
assert projection<86400 and 2*projection<remaining
snapshot=read(r/'ops/SP25_CONTINUATION_INPUTS.json')
free=snapshot['node07']['scratch_available_GiB']*2**30
extra=sum(p['vertex_storage_bytes']for p in prefixes)*9
assert extra+30*2**30<free
out={'started_unix':time.time(),'original_banks':10,'original_maximum':100,'continued_maximum':1000,
 'existing_preparation_GiB':24,'existing_fit_GiB':6,'existing_node_GiB':230,'new_CPU_allocations':0,
 'max_observed_N100_wall_seconds':projection/10,'projected_N1000_wall_seconds_per_bank':projection,
 'projected_additional_vertex_bytes':extra,'projection_is_not_a_bound':True,
 'finite_preparation_seconds':86400,'deadline_utc':contract['deadline_utc'],
 'guard':'Unchanged24GiB preparation,6GiB fit,230GiB node and original96GiB resource-pool aggregate; original disk30GiB reserve and finite wall/cutoff.',
 'scientific_parameters_changed':False,'launchers':[]}
write(receipt,out)
admission['prepare_maximum_by_route']['dys_shortest_path_25x25']=1000
admission['sp25_expansion']={k:v for k,v in out.items()if k!='launchers'}
write(r/'ADMISSION.json',admission)
# Only idle workers are adopted so their cached6GiB pre-repair bank guard
# cannot misclassify the permitted24GiB continuation as a new resource censor.
for node,job,cpus in [('node07',16726,24),('node07',16740,8)]:
 log=(r/'logs'/('SP25-N1000-idle-refresh-'+str(job)+'.out')).open('ab')
 child=subprocess.Popen(['srun','--overlap','--jobid='+str(job),'--nodelist='+node,'--ntasks=1',
  '--cpus-per-task='+str(cpus),'--cpu-bind=cores','env',
  'LAVA_IDLE_REFRESH_LABEL=SP25_N1000_20261103',
  'LAVA_IDLE_REFRESH_REASON=Required SP25 continuation under existing24GiB guard; preserve per-pool budgets',
  'bash',str(r/'ops/coordinator_env.sh'),str(r),'refresh_idle_once.py'],
  stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 out['launchers'].append({'allocation':job,'pid':child.pid,'workers_maximum':cpus});write(receipt,out)
out['finished_unix']=time.time();write(receipt,out);print(json.dumps(out))
