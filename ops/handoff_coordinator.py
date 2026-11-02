"""One exact coordinator handoff; unaffected numerical children keep running."""
import json,pathlib,subprocess,time
from qualify import read,write
r=pathlib.Path('/home/jonathan.hornewall/runs/lava_completion_20261102');results=[]
affected=set(read(r/'ops/AFFECTED_JOB_IDS.json'))
for p in (r/'selections').glob('*/lava__cfl.json'):
 if any(v['id'] in affected for v in read(p).get('validation_table',[])):
  raise RuntimeError('Affected CFL selection exists; keep hold for explicit dependency repair')
for job,node in ((16223,'node09'),(16726,'node07'),(16740,'node07')):
 p=subprocess.run(['srun','--overlap','--jobid='+str(job),'--nodelist='+node,'--ntasks=1',
  '--cpus-per-task=1','--time=00:00:10','--cpu-bind=cores',
  '/home/jonathan.hornewall/runs/scaled_data_neural_20261021/venv/bin/python',str(r/'ops/pause_old_coordinators.py')],capture_output=True,text=True,timeout=12)
 results.append({'job':job,'node':node,'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr})
 write(r/'ops/COORDINATOR_HANDOFF.json',{'observed_unix':time.time(),'steps':results,'workers_signalled':False})
 if p.returncode:raise RuntimeError(p.stdout+p.stderr)
log=(r/'logs/coordinator-current.out').open('ab')
child=subprocess.Popen(['srun','--overlap','--jobid=16223','--nodelist=node09','--ntasks=1',
 '--cpus-per-task=1','--cpu-bind=cores','bash',str(r/'ops/coordinator_env.sh'),str(r)],
 stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
write(r/'ops/COORDINATOR_LAUNCHER.json',{'pid':child.pid,'launched_unix':time.time(),'allocation':16223,
 'finite_inherited_allocation':True,'software_cutoff_unix':read(r/'CONTRACT.json')['scientific_cutoff_unix']})
hold=r/'ops/HOLD_CFL_COORDINATOR_REFRESH.json';archive=r/'ops/resolved_holds';archive.mkdir(exist_ok=True)
if hold.exists():hold.replace(archive/(str(time.time_ns())+'__'+hold.name))
print(json.dumps({'steps':results,'new_coordinator_launcher_pid':child.pid,'CFL_dispatch_hold_resolved':True}))
