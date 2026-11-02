"""One launch/reconciliation transaction, bounded by the existing contract."""
import hashlib,json,pathlib,subprocess,time
from qualify import read,write
from safe_geometry import release_unstarted_holds
root=pathlib.Path('/home/jonathan.hornewall/runs/lava_completion_20261102')
assert read(root/'ops/SEEDED_GEOMETRY_CHECK.json')['bank_geometry_sha256']==hashlib.sha256((root/'ops/bank_geometry.py').read_bytes()).hexdigest()
registry=read(root/'ops/JOB_REGISTRY.json');admission=read(root/'ADMISSION.json')
admission['resource_recovery_failure_sha256']={}
for failure in (root/'failures').glob('bank__dys_shortest_path_*.json'):
 if any(part in failure.stem for part in ('25x25','50x50')):
  admission['resource_recovery_failure_sha256'][failure.stem.removeprefix('bank__')]=hashlib.sha256(failure.read_bytes()).hexdigest()
write(root/'ADMISSION.json',admission)
released=[]
for key,value in read(root/'ops/GEOMETRY_PROVENANCE.json').items():
 released+=release_unstarted_holds(root,key,maximum=value.get('random_free_supplied_prefix',0))
requests=[('dys_cfl_50x5__data151101',161101,1000,'node09.cluster.local','Known affected unseeded random-search attempt3120354'),
 ('dys_cfl_100x10__data151101',161101,100,'node09.cluster.local','Bounded original N100 bank for conditional CFL100 admission')]
for key,seed,n,host,reason in requests:
 write(root/'seeded_requests'/(key+'__init'+str(seed)+'.json'),{'key':key,'initialization_seed':seed,
  'maximum':n,'host':host,'reason':reason,'source_job_ids':[],'updated_unix':time.time()})
submitted={}
for label,node,cpus,mem,script,args in (
 ('resource_repair_pool','node07',8,'96G','completion.sbatch',[str(root),'8','230']),
 ('seeded_node07_pool','node07',1,'96G','seeded_preparation.sbatch',[str(root)]),
 ('seeded_node09_pool','node09',1,'96G','seeded_preparation.sbatch',[str(root)])):
 if label in registry:submitted[label]={'existing_job':registry[label]};continue
 q=subprocess.run(['squeue','-h','-r','-u','jonathan.hornewall','-o','%C|%j'],capture_output=True,text=True,timeout=5)
 campaign=sum(int(row.split('|')[0]) for row in q.stdout.splitlines() if '20261102' in row)
 if campaign+cpus>100:raise RuntimeError('Campaign100-CPU running+pending envelope reached')
 cmd=['sbatch','--parsable','--partition=defq-gpu','--nodelist='+node,'--nice=10000','--requeue',
  '--job-name=lava_'+label+'_20261102','--ntasks=1','--cpus-per-task='+str(cpus),'--mem='+mem,
  '--time=6-06:00:00','--output='+str(root/'logs'/('repair-%j.out'))]
 if label=='resource_repair_pool':cmd+=['--export=ALL,LAVA_POOL_MEMORY_GIB=96']
 cmd+=[str(root/'ops'/script)]+args
 p=subprocess.run(cmd,capture_output=True,text=True,timeout=10)
 if p.returncode:raise RuntimeError(p.stderr+p.stdout)
 number=int(p.stdout.strip());registry[label]=number
 registry['largest_total_requested_cpus_so_far']=max(registry.get('largest_total_requested_cpus_so_far',0),campaign+cpus)
 write(root/'ops/JOB_REGISTRY.json',registry);submitted[label]={'job_id':number,'node':node,'cpus':cpus}
print(json.dumps({'submitted_unix':time.time(),'submitted':submitted,'released_unstarted_provenance_holds':released}))
