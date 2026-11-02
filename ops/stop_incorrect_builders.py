"""End exact known-affected or newly unseeded continuations; retain prefixes."""
import os,pathlib,signal,time
import psutil
from qualify import read,write
r=pathlib.Path('/home/jonathan.hornewall/runs/lava_completion_20261102');actions=[]
for p in psutil.process_iter(['pid','cmdline']):
 cmd=p.info['cmdline'] or []
 if len(cmd)<2 or cmd[1]!=str(r/'ops/bank_geometry.py') or '--initialization-seed' in cmd or '--key' not in cmd:continue
 key=cmd[cmd.index('--key')+1]
 if not (key=='dys_cfl_50x5__data17' or key.startswith('dys_shortest_path_15x15__')):continue
 if p.environ().get('SLURM_JOB_ID') not in {'16223','16726','16740'} or os.getpgid(p.pid)!=p.pid:raise RuntimeError('Incorrect builder identity mismatch')
 record={'pid':p.pid,'cmdline':cmd,'birth_unix':p.create_time(),'cpu_seconds':sum(p.cpu_times()[:2]),
  'rss_bytes':p.memory_info().rss,'key':key,'observed_unix':time.time(),'bank_status':read(r/'banks'/key/'STATUS.json'),
  'reason':'Known random-transition affected development bank' if 'cfl' in key else 'Unseeded larger-bank continuation started by a pre-refresh worker; preserve independently proved prefix100, resume seeded.',
  'RNG_dependence_proved':True if 'cfl' in key else False}
 target=r/'recovery/affected_attempts'/(key+'__pid'+str(p.pid)+'.json');write(target,record)
 os.killpg(p.pid,signal.SIGKILL);record['action']='ended_exact_owned_builder_process_group';write(target,record);actions.append(record)
print(actions)
