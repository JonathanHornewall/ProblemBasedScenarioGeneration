import hashlib,pathlib,subprocess,time,json
from qualify import read,write
r=pathlib.Path('/home/jonathan.hornewall/runs/lava_completion_20261102')
check=read(r/'ops/SEEDED_GEOMETRY_CHECK.json')
assert check['status']=='passed' and check['bank_geometry_sha256']==hashlib.sha256((r/'ops/bank_geometry.py').read_bytes()).hexdigest()
a=read(r/'ADMISSION.json')
for p in (r/'failures').glob('bank__*.json'):
 if '15x15' in p.stem or p.stem=='bank__dys_cfl_50x5__data17':
  a.setdefault('resource_recovery_failure_sha256',{})[p.stem.removeprefix('bank__')]=hashlib.sha256(p.read_bytes()).hexdigest()
write(r/'ADMISSION.json',a)
registry=read(r/'ops/JOB_REGISTRY.json')
registry.update({'node07_completion_pool':16726,'node07_completion_workers':24,'node07_completion_cancelled_attempt':16224,
 'kp_serial_job':16727,'qualification_active':[],'qualification_terminal':[16208,16209,16210],
 'provenance_check_array':16728,'seed_check_jobs':[16739,16743]})
write(r/'ops/JOB_REGISTRY.json',registry)
log=(r/'logs/idle-worker-refresh.out').open('ab')
child=subprocess.Popen(['srun','--overlap','--jobid=16223','--nodelist=node09','--ntasks=1',
 '--cpus-per-task=16','--cpu-bind=cores','bash',str(r/'ops/coordinator_env.sh'),str(r),'refresh_idle_workers.py'],
 stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
write(r/'ops/IDLE_REFRESH_LAUNCHER.json',{'pid':child.pid,'started_unix':time.time(),'allocation':16223,'new_CPU_allocations':0})
out={'observed_unix':time.time(),'launcher_pid':child.pid,'registry':registry,'seed_check':check}
print(json.dumps(out))
