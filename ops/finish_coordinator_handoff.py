import json,pathlib,subprocess,time,hashlib
from qualify import read,write
r=pathlib.Path('/home/jonathan.hornewall/runs/lava_completion_20261102');actions=[]
for job,node in ((16223,'node09'),(16726,'node07'),(16740,'node07')):
 p=subprocess.run(['srun','--overlap','--jobid='+str(job),'--nodelist='+node,'--ntasks=1','--cpus-per-task=1',
  '--time=00:00:10','--cpu-bind=cores','/home/jonathan.hornewall/runs/scaled_data_neural_20261021/venv/bin/python',str(r/'ops/resume_control_wrappers.py')],capture_output=True,text=True,timeout=12)
 actions.append({'job':job,'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr})
 if p.returncode:raise RuntimeError(p.stdout+p.stderr)
a=read(r/'ADMISSION.json');a['prepare_maximum_by_route']['dys_shortest_path_15x15']=1000
a['sp15_expansion']={'admitted_unix':time.time(),'observed_N100_wall_seconds_range':[944,990],
 'observed_N100_vertex_bytes_max':44307680,'inference':'10x extrapolation ~2.75h/1000-bank and0.45GB/bank, before contingency; ten banks fit remaining151h and actual >300GB scratch headroom. Not a bound; retain existing per-bank wall/RSS/storage guards.','original_protocol_unchanged':True}
write(r/'ADMISSION.json',a)
out={'observed_unix':time.time(),'wrapper_reconciliation':actions,
 'coordinator':read(r/'ops/COORDINATOR_ACTIVE.json') if (r/'ops/COORDINATOR_ACTIVE.json').exists() else None,
 'coordinator_log':(r/'logs/coordinator-current.out').read_text()[-4000:],
 'replacement_jobs':{p.stem:read(p)['jobs'][0]['replaces_job_id'] for p in (r/'jobs').glob('*.json') if read(p)['jobs'][0].get('replaces_job_id')},
 'CFL_selections':[str(p) for p in (r/'selections').glob('*/lava__cfl.json')],
 'registry':read(r/'ops/JOB_REGISTRY.json'),'admission':a}
print(json.dumps(out))
