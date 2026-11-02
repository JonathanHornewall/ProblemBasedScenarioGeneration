"""Save current ownership/admission and leave bounded workers progressing."""
import pathlib,subprocess,time,json,os
from qualify import read,write
r=pathlib.Path('/home/jonathan.hornewall/runs/lava_completion_20261102');reg=read(r/'ops/JOB_REGISTRY.json')
reg.update({'node07_completion_pool':16726,'node07_completion_workers':24,'node07_completion_cancelled_attempt':16224,
 'kp_serial_job':16727,'qualification_active':[],'qualification_terminal':[16208,16209,16210],
 'provenance_check_array':16728,'seed_check_jobs':[16739,16743]})
reg['largest_total_requested_cpus_so_far']=max(52,reg.get('largest_total_requested_cpus_so_far',0))
a=read(r/'ADMISSION.json');sp20=list((r/'banks').glob('dys_shortest_path_20x20__*/prefix100.json'))
if len(sp20)==10:
 values=[read(p) for p in sp20];wall=max(v['build_wall_seconds_cumulative'] for v in values)
 projected=sum(v['vertex_storage_bytes']*10 for v in values)
 if wall<=7200 and projected<=50*2**30:
  a['prepare_maximum_by_route']['dys_shortest_path_20x20']=1000
  a['sp20_expansion']={'admitted_unix':time.time(),'N100_banks':10,'max_N100_wall_seconds':wall,
   'projected_total_vertex_bytes_at_10x':projected,'projection_is_not_a_bound':True,
   'guard':'Original per-bank time/RSS, node230GiB ledger and30GiB disk reserve remain. Full original grid/seeds unchanged.'}
write(r/'ADMISSION.json',a)
launched={};remaining=int(read(r/'CONTRACT.json')['deadline_unix']-time.time()-60)
d,rem=divmod(remaining,86400);h,rem=divmod(rem,3600);m,s=divmod(rem,60);limit=f'{d}-{h:02}:{m:02}:{s:02}'
if 'seeded_node09_repair_pool' not in reg:
 q=subprocess.run(['squeue','-h','-r','-u','jonathan.hornewall','-o','%C|%j'],capture_output=True,text=True,timeout=5)
 slots=sum(int(v.split('|')[0]) for v in q.stdout.splitlines() if '20261102' in v)
 if slots+1>100:raise RuntimeError('Campaign CPU envelope')
 p=subprocess.run(['sbatch','--parsable','--partition=defq-gpu','--nodelist=node09','--nice=10000','--requeue',
  '--job-name=lava_seeded_repair_20261102','--ntasks=1','--cpus-per-task=1','--mem=96G','--time='+limit,
  '--output='+str(r/'logs/repair-%j.out'),str(r/'ops/seeded_preparation.sbatch'),str(r)],capture_output=True,text=True,timeout=10)
 if p.returncode:raise RuntimeError(p.stderr+p.stdout)
 reg['seeded_node09_repair_pool']=int(p.stdout.strip());launched['parallel_seeded_owner']=reg['seeded_node09_repair_pool']
 reg['largest_total_requested_cpus_so_far']=max(reg['largest_total_requested_cpus_so_far'],slots+1)
write(r/'ops/JOB_REGISTRY.json',reg)
# The previous refresh found no idle worker and ended. A finite rolling refresh
# now adopts each old worker only after it naturally becomes idle.
launcher=r/'ops/IDLE_REFRESH_LAUNCHER.json';old=read(launcher) if launcher.exists() else {}
cmd=pathlib.Path('/proc')/str(old.get('pid',-1))/'cmdline'
still_running=cmd.exists() and b'refresh_idle_workers.py' in cmd.read_bytes()
if not still_running:
 if old:write(r/'ops/idle_refresh_history'/(str(time.time_ns())+'.json'),old)
 log=(r/'logs/idle-worker-refresh.out').open('ab')
 child=subprocess.Popen(['srun','--overlap','--jobid=16223','--nodelist=node09','--ntasks=1','--cpus-per-task=16',
  '--cpu-bind=cores','bash',str(r/'ops/coordinator_env.sh'),str(r),'refresh_idle_workers.py'],
  stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 write(launcher,{'pid':child.pid,'started_unix':time.time(),'allocation':16223,'new_CPU_allocations':0,'rolling_idle_refresh':True})
 launched['rolling_refresh_launcher_pid']=child.pid
out={'observed_unix':time.time(),'registry':reg,'launched':launched,'admission':a,
 'coordinator':read(r/'ops/COORDINATOR_ACTIVE.json'),'seed_check':read(r/'ops/SEEDED_GEOMETRY_CHECK.json'),
 'provenance':read(r/'ops/GEOMETRY_PROVENANCE.json'),
 'queue':subprocess.run(['squeue','-h','-r','-u','jonathan.hornewall','-o','%i|%C|%T|%N|%j'],capture_output=True,text=True,timeout=5).stdout}
write(r/'ops/CURRENT_OPERATIONAL_STATE.json',out);print(json.dumps(out))
