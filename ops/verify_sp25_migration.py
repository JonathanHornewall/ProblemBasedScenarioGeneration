"""Applied resource/continuity verification of the authorized migration."""
import concurrent.futures,json,pathlib,subprocess,time
from qualify import read,write
r=pathlib.Path('/home/jonathan.hornewall/runs/lava_completion_20261102');out={'observed_unix':time.time()}
def inspect(pair):
 node,job,cpus=pair
 p=subprocess.run(['srun','--overlap','--jobid='+str(job),'--nodelist='+node,'--ntasks=1','--cpus-per-task='+str(cpus),'--time=00:00:20','--cpu-bind=cores','bash',str(r/'ops/coordinator_env.sh'),str(r),'sp25_resource_snapshot.py'],capture_output=True,text=True,timeout=23)
 if p.returncode:raise RuntimeError(p.stderr+p.stdout)
 return node,json.loads(p.stdout)
with concurrent.futures.ThreadPoolExecutor(max_workers=2)as ex:out['nodes']=dict(ex.map(inspect,[('node07',16726,1),('node09',16223,1)]))
ready=read(r/'READY.json');out['ready']={k:v for k,v in ready.items()if k not in('fits','preparation')}
out['refresh_receipts']={str(j):read(r/'ops'/('IDLE_REFRESH_ONCE_'+str(j)+'_SP25_MEASURED_20261104.json'))for j in(16726,16740)}
out['rebalance']=read(r/'ops/SP25_ADMISSION_REBALANCE_NODE07.json');out['migration']=read(r/'ops/SP25_ADMISSION_MIGRATION_20261104.json')
out['queue']=subprocess.run(['squeue','-h','-r','-u','jonathan.hornewall','-o','%i|%C|%T|%N|%j'],capture_output=True,text=True,timeout=8).stdout
before=read(r/'ops/SP25_ADMISSION_POLICY_20261104.json')['active_numeric_processes_before'];after=out['nodes']['node07']['numeric'];by_pid={v['pid']:v for v in after}
out['healthy_process_continuity']=[]
for old in before:
 new=by_pid.get(old['pid']);assert new and new['birth_unix']==old['birth_unix']
 assert new['CPU_seconds_after']>old['CPU_seconds']
 out['healthy_process_continuity'].append({'pid':old['pid'],'birth_unix':old['birth_unix'],'CPU_seconds_before':old['CPU_seconds'],'CPU_seconds_after':new['CPU_seconds_after'],'unchanged_PID_birth':True})
coordinators=[v for node in out['nodes'].values()for v in node['coordinators']]
assert len(coordinators)==1 and coordinators[0]['job']=='16726'
assert all(v['coordinator_lock_held']for v in out['nodes'].values())
assert ready['updated_unix']>out['nodes']['node07']['coordinator_record']['started_unix']
assert not out['nodes']['node09']['numeric']
assert ready['resource_censored_SP_CFL_fits']==40
assert sum(v['gib']for v in out['nodes']['node07']['reservations'].values())<=230
assert sum(v['gib']for k,v in out['nodes']['node07']['reservations'].items()if k.endswith('_16740'))<=96
assert any(v['entry']=='prepare_bank.py'and'dys_shortest_path_25x25__data151119'in v['command']for v in after)
assert sum(v['entry']=='run_fit.py'for v in after)>2
cores=[v['affinity'][0]for v in after+coordinators];assert len(cores)==len(set(cores))
out['productive_numerical_cores']=sum(v['effective_cores_interval']for v in after)
out['numeric_count']=len(after);out['builders']=sum(v['entry']=='prepare_bank.py'for v in after);out['fits']=sum(v['entry']=='run_fit.py'for v in after)
out['new_failed_attempts']={p.name:read(p)for p in(r/'attempts').glob('*.json')if p.stat().st_mtime>out['migration']['started_unix']and read(p).get('returncode')}
assert not out['new_failed_attempts']
out['status']='passed';out['finished_unix']=time.time();write(r/'ops/SP25_ADMISSION_APPLIED_20261104.json',out)
print(json.dumps({k:v for k,v in out.items()if k not in('nodes','refresh_receipts','rebalance','migration')}))
