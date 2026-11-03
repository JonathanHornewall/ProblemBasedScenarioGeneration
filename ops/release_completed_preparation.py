"""Release only completed preparation queues; reconcile required SP25 continuation once."""
import concurrent.futures,json,pathlib,subprocess,time
from qualify import read,write
r=pathlib.Path('/home/jonathan.hornewall/runs/lava_completion_20261102');reg=read(r/'ops/JOB_REGISTRY.json')
receipt=r/'ops/COMPLETED_PREPARATION_RELEASE.json'
if receipt.exists():raise RuntimeError('Release already attempted; reconcile rather than resubmit')
keys=('seeded_node07_pool','seeded_node09_pool','seeded_node09_repair_pool','cfl100_node07_pool','cfl100_node09_pool')
jobs=sorted(int(reg[k])for k in keys);assert jobs==[16741,16742,16744,16745,16746]
assert all(read(p)['status']=='complete'for p in(r/'seeded_banks_v2').glob('dys_cfl_100x10__*/prefix100.json'))
assert len(list((r/'seeded_banks_v2').glob('dys_cfl_100x10__*/prefix100.json')))==10
code=r'''
import json,pathlib,psutil,time
r=pathlib.Path('/home/jonathan.hornewall/runs/lava_completion_20261102');out={'observed_unix':time.time(),'numeric':[],'reservations':json.loads(pathlib.Path('/tmp/lava_completion_20261102/reservations.json').read_text())}
for p in psutil.process_iter(['pid','name','cmdline','cpu_times','memory_info','status']):
 cmd=p.info['cmdline']or[]
 if not p.info['name'].startswith('python')or p.info['status']=='zombie':continue
 entry=next((f for f in('prepare_bank.py','bank_geometry.py','run_fit.py')if str(r/'ops'/f)in cmd),None)
 if not entry:continue
 env=p.environ();out['numeric'].append({'pid':p.pid,'entry':entry,'job':env.get('SLURM_JOB_ID'),'command':cmd,'CPU_seconds':sum(p.info['cpu_times'][:2]),'RSS_bytes':p.info['memory_info'].rss,'birth_unix':p.create_time()})
print(json.dumps(out))
'''
def inspect(pair):
 node,job=pair;p=subprocess.run(['srun','--overlap','--jobid='+str(job),'--nodelist='+node,'--ntasks=1','--cpus-per-task=1','--time=00:00:20','--cpu-bind=cores','/home/jonathan.hornewall/runs/scaled_data_neural_20261021/venv/bin/python','-B','-c',code],capture_output=True,text=True,timeout=23)
 if p.returncode:raise RuntimeError(p.stderr+p.stdout)
 return node,json.loads(p.stdout)
with concurrent.futures.ThreadPoolExecutor(max_workers=2)as ex:before=dict(ex.map(inspect,[('node07',16726),('node09',16223)]))
assert not any(p['job']in{str(j)for j in jobs}for n in before.values()for p in n['numeric'])
out={'started_unix':time.time(),'released_jobs':jobs,'mutation_attempted_unix':time.time(),'pre_release':before,'reason':'All preparation queues have completed every retained request; required SP25 continuation uses existing common pools','new_CPU_allocations':0}
write(receipt,out)
p=subprocess.run(['scancel']+[str(j)for j in jobs],capture_output=True,text=True,timeout=8)
out['scancel_returncode']=p.returncode;out['scancel_stderr']=p.stderr;write(receipt,out)
if p.returncode:raise RuntimeError('Release attempted once; reconcile scheduler outcome:'+p.stderr)
time.sleep(12)
q=subprocess.run(['squeue','-h','-r','-u','jonathan.hornewall','-o','%i|%C|%T|%N|%j'],capture_output=True,text=True,timeout=5)
out['queue']=q.stdout;out['released_jobs_still_visible']=[line for line in q.stdout.splitlines()if int(line.split('|')[0].split('_')[0])in jobs]
with concurrent.futures.ThreadPoolExecutor(max_workers=2)as ex:out['applied_node_state']=dict(ex.map(inspect,[('node07',16726),('node09',16223)]))
reg['released_completed_preparation_jobs']=jobs;reg['completed_KP_serial_job']=16727;reg['current_requested_CPUs_after_release']=48
write(r/'ops/JOB_REGISTRY.json',reg);out['finished_unix']=time.time();write(receipt,out);print(json.dumps(out))
