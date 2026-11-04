"""One authorized admission/control migration; release awaits verified local retrieval."""
import json,os,pathlib,subprocess,time
from qualify import read,write
r=pathlib.Path('/home/jonathan.hornewall/runs/lava_completion_20261102')
receipt=r/'ops/SP25_ADMISSION_MIGRATION_20261104.json'
if receipt.exists():raise RuntimeError('Migration already attempted; reconcile recorded steps')
assert read(r/'ops/SP25_ADMISSION_SOURCE_DEPLOYMENT_20261104.json')['focused_test']['returncode']==0
out={'started_unix':time.time(),'steps':[],'new_allocations':0,'healthy_numerical_children_signalled':0}
write(receipt,out)
def run(job,node,cpus,entry,extra=(),timeout=25):
 command=['srun','--overlap','--jobid='+str(job),'--nodelist='+node,'--ntasks=1','--cpus-per-task='+str(cpus),'--time=00:01:00','--cpu-bind=cores']
 if node=='node07':command+=['taskset','-c','46']
 command+=['env',*extra,'bash',str(r/'ops/coordinator_env.sh'),str(r),entry]
 p=subprocess.run(command,capture_output=True,text=True,timeout=timeout)
 out['steps'].append({'entry':entry,'job':job,'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr,'finished_unix':time.time()});write(receipt,out)
 if p.returncode:raise RuntimeError('Mutation failed; reconcile '+entry+': '+p.stderr+p.stdout)
run(16726,'node07',24,'rebalance_sp25_claims.py')
run(16223,'node09',1,'retain_node09_compact.py',timeout=55)
old=read(r/'ops/COORDINATOR_ACTIVE.json');assert old['slurm_job_id']=='16223'
run(16223,'node09',1,'stop_current_coordinator.py',('LAVA_METADATA_RELOAD_REASON=Move singleton coordinator to the reserved idle node07 core; healthy numerical children unchanged.',))
limit=str(int((read(r/'CONTRACT.json')['scientific_cutoff_unix']-time.time())//60))
def launch(job,cpus,entry,extra=(),core=None):
 command=['srun','--overlap','--jobid='+str(job),'--nodelist=node07','--ntasks=1','--cpus-per-task='+str(cpus),'--time='+limit,'--cpu-bind=cores']
 if core is not None:command+=['taskset','-c',str(core)]
 command+=['env',*extra,'bash',str(r/'ops/coordinator_env.sh'),str(r),entry]
 log=(r/'logs'/('sp25_measured_'+str(job)+'_'+entry+'.out')).open('ab')
 child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 out['steps'].append({'entry':entry,'job':job,'launcher_pid':child.pid,'launched_unix':time.time(),'command':command});write(receipt,out)
launch(16726,24,'coordinator.py',core=46)
launch(16740,8,'refresh_idle_once.py',('LAVA_IDLE_REFRESH_LABEL=SP25_MEASURED_20261104','LAVA_POOL_MEMORY_GIB=96','LAVA_FIRST_PREPARATION_KEY_BY_INDEX={"4":"dys_shortest_path_25x25__data151119"}','LAVA_IDLE_REFRESH_REASON=Measured SP25 admission16GiB preparation/3GiB fit; existing96GiB pool limit.'))
launch(16726,24,'refresh_idle_once.py',('LAVA_IDLE_REFRESH_LABEL=SP25_MEASURED_20261104','LAVA_IDLE_REFRESH_SKIP_INDICES=23','LAVA_IDLE_REFRESH_REASON=Measured SP25 admission16GiB preparation/3GiB fit; child guards24/6GiB unchanged.'))
out['state']='launched_pending_applied_verification';out['finished_unix']=time.time();write(receipt,out)
print(json.dumps(out))
