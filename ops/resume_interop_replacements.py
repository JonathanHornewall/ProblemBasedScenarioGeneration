"""Reload the metadata coordinator only after the startup regression passes."""
import pathlib,subprocess,time,json
from qualify import read,write
r=pathlib.Path('/home/jonathan.hornewall/runs/lava_completion_20261102')
assert read(r/'ops/INTEROP_ISOLATION_CHECK.json')['status']=='passed'
assert len(read(r/'ops/JOB_REPAIRS.json'))==23
p=subprocess.run(['srun','--overlap','--jobid=16223','--nodelist=node09','--ntasks=1','--cpus-per-task=1',
 '--time=00:00:20','--cpu-bind=cores','/home/jonathan.hornewall/runs/scaled_data_neural_20261021/venv/bin/python',str(r/'ops/stop_current_coordinator.py')],capture_output=True,text=True,timeout=22)
if p.returncode:raise RuntimeError(p.stdout+p.stderr)
log=(r/'logs/coordinator-current.out').open('ab')
child=subprocess.Popen(['srun','--overlap','--jobid=16223','--nodelist=node09','--ntasks=1','--cpus-per-task=1',
 '--cpu-bind=cores','bash',str(r/'ops/coordinator_env.sh'),str(r)],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
result={'observed_unix':time.time(),'launcher_pid':child.pid,'old_coordinator_stop':p.stdout,
 'replacement_jobs':23,'scientific_settings_changed':False,'new_CPU_allocations':0,
 'original_failed_receipts_retained':True,'regression':read(r/'ops/INTEROP_ISOLATION_CHECK.json')}
write(r/'ops/INTEROP_REPAIR_RESUMPTION.json',result);print(json.dumps(result))
