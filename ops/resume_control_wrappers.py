"""Resume only srun control wrappers included in the first coordinator match."""
import os,pathlib,time
import psutil
from qualify import read,write
r=pathlib.Path('/home/jonathan.hornewall/runs/lava_completion_20261102');job=os.environ['SLURM_JOB_ID'];actions=[]
for v in read(r/'ops/coordinator_handoff'/(job+'.json'))['paused_metadata_coordinators']:
 if v['cmdline'][0]!='srun':continue
 p=psutil.Process(v['pid'])
 if p.cmdline()!=v['cmdline'] or abs(p.create_time()-v['created_unix'])>.01:raise RuntimeError('Control-wrapper identity changed')
 if p.status()==psutil.STATUS_STOPPED:p.resume()
 actions.append({'pid':p.pid,'action':'resumed_srun_control_wrapper'})
write(r/'ops/coordinator_handoff'/(job+'__control_wrappers_resumed.json'),{'observed_unix':time.time(),'actions':actions,'numerical_workers_signalled':False})
print(actions)
