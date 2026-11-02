"""One reconciliation snapshot after known campaign mutations."""
import json,pathlib,subprocess,time
from collections import Counter
r=pathlib.Path('/home/jonathan.hornewall/runs/lava_completion_20261102')
out={'observed_unix':time.time()}
def command(args,timeout=12):
 p=subprocess.run(args,capture_output=True,text=True,timeout=timeout)
 return {'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr[-2000:]}
out['queue']=command(['squeue','-h','-r','-u','jonathan.hornewall','-o','%i|%C|%T|%N|%j'])
out['workers']={p.name:json.loads(p.read_text()) for p in (r/'workers').glob('*.json')}
out['banks']={p.parent.name:json.loads(p.read_text()) for p in (r/'banks').glob('*/STATUS.json')}
out['failures']={p.name:json.loads(p.read_text()) for p in (r/'failures').glob('*.json')}
out['resolved_banks']={p.name:json.loads(p.read_text()) for p in (r/'recovery/resolved_banks').glob('*.json')}
out['serial_live']={p.name:json.loads(p.read_text()) for p in (r/'recovery/live').glob('*.json')}
for name in ('ops/IDLE_WORKER_REFRESH.json','ops/KP_BANK_GEOMETRY_CHECK.json'):
 if (r/name).exists():out[name]=json.loads((r/name).read_text())
out['refresh_log']=(r/'logs/idle-worker-refresh.out').read_text()[-2500:] if (r/'logs/idle-worker-refresh.out').exists() else None
out['ready']={k:v for k,v in json.loads((r/'READY.json').read_text()).items() if k not in ('fits','preparation')}
out['receipt_counts']=dict(Counter(json.loads(p.read_text()).get('status') for p in (r/'results/receipts').glob('*.json')))
out['telemetry']={}
for job,node in ((16223,'node09'),(16726,'node07'),(16727,'node08')):
 try:out['telemetry'][node]=command(['srun','--overlap','--jobid='+str(job),'--nodelist='+node,
  '--ntasks=1','--cpus-per-task=1','--time=00:00:15','--cpu-bind=cores','bash','-c',
  'free -b; df -B1 /tmp; ps -u jonathan.hornewall -o pid=,pcpu=,cputimes=,rss=,args='],timeout=18)
 except subprocess.TimeoutExpired:out['telemetry'][node]={'error':'bounded step telemetry timeout'}
print(json.dumps(out))
