"""Bounded detail read to diagnose the known interrupted banks and KP fits."""
import json,pathlib,subprocess,time
r=pathlib.Path('/home/jonathan.hornewall/runs/lava_completion_20261102')
out={'observed_unix':time.time()}
def command(args,timeout=12):
 p=subprocess.run(args,capture_output=True,text=True,timeout=timeout)
 return {'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr[-1000:]}
out['accounting']=command(['sacct','-j','16208,16209,16210,16211,16223,16224','--starttime','2026-11-02','-P','-o','JobID,JobName,State,ExitCode,Elapsed,TotalCPU,MaxRSS,NodeList,AllocCPUS'])
out['nodes']=command(['sinfo','-N','-h','-o','%N|%T|%C|%m|%e'])
out['guards']={p.name:json.loads(p.read_text()) for p in (r/'qualification').glob('*.memory_guard.json')}
out['receipts']={}
for p in (r/'results/receipts').glob('lava_recovery__*.json'):
 v=json.loads(p.read_text());out['receipts'][p.name]={'status':v.get('status'),'summary':v.get('summary'),'error':v.get('error'),'resource_usage':v.get('resource_usage')}
out['progress']={p.name:json.loads(p.read_text()) for p in (r/'results/progress').glob('lava_recovery__*.json')}
out['node09']=command(['srun','--overlap','--jobid=16223','--nodelist=node09','--ntasks=1','--cpus-per-task=1','--time=00:00:20','--cpu-bind=cores','bash','-c','free -b; df -B1 /tmp; ps -u jonathan.hornewall -o pid=,pcpu=,cputimes=,rss=,args=; cat /tmp/lava_completion_20261102/reservations.json'],timeout=25)
out['bank_prefixes']={}
for p in (r/'banks').glob('*/prefix1000.json'):
 v=json.loads(p.read_text());out['bank_prefixes'][p.parent.name]={k:v.get(k) for k in ('build_wall_seconds_cumulative','build_cpu_seconds_cumulative','vertex_storage_bytes','unique_anchor_cache_files')}
print(json.dumps(out))
