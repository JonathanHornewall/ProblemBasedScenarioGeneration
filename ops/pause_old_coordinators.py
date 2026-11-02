"""Pause exact old metadata coordinators; never signal numerical children."""
import os,pathlib,time,sys
import psutil
from qualify import read,write
root=pathlib.Path('/home/jonathan.hornewall/runs/lava_completion_20261102')
job=os.environ['SLURM_JOB_ID'];expected={'16223','16726','16740'}
if job not in expected:raise RuntimeError('Unexpected allocation')
paused=[]
for proc in psutil.process_iter(['pid','cmdline','username']):
 cmd=proc.info['cmdline'] or []
 if not cmd or cmd[0]!=sys.executable:continue
 if str(root/'ops/completion.py') not in cmd or '--pool' not in cmd:continue
 if proc.environ().get('SLURM_JOB_ID')!=job:continue
 if proc.status()==psutil.STATUS_STOPPED:continue
 paused.append({'pid':proc.pid,'cmdline':cmd,'created_unix':proc.create_time(),'cpu_seconds':sum(proc.cpu_times()[:2])})
 proc.suspend()
write(root/'ops/coordinator_handoff'/(job+'.json'),{'paused_unix':time.time(),'slurm_job_id':job,
 'paused_metadata_coordinators':paused,'workers_signalled':False,'reason':'Drop cached affected v1 screen records; fresh coordinator uses explicit affected-job replacements.'})
print(paused)
