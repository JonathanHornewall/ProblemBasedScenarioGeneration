"""One compact read-only monitoring snapshot; no polling loop or remote mutation."""
from __future__ import annotations
import json
from pathlib import Path
import shlex
import time
from remote import cermics

SOURCE=r'''
import json,pathlib,subprocess,time
r=pathlib.Path('/home/jonathan.hornewall/runs/lava_completion_20261102')
read=lambda p:json.loads(p.read_text())
out={'observed_unix':time.time()}
for name in ('READY.json','SCIENCE_COMPLETE.json'):
 if (r/name).exists():
  v=read(r/name)
  if name=='READY.json':v={k:v[k] for k in v if k not in ('fits','preparation')}|{'ready_fit_count':len(v['fits']),'preparation_count':len(v['preparation'])}
  out[name]=v
out['qualification']={p.name:read(p) for p in sorted((r/'qualification').glob('*__seed*__sample*.json'))}
out['banks']={p.parent.name:read(p) for p in sorted((r/'banks').glob('*/STATUS.json'))}
out['workers']={p.name:read(p) for p in sorted((r/'workers').glob('*.json'))}
out['failures']={p.name:read(p) for p in sorted((r/'failures').glob('*.json'))}
out['fits']=[]
for folder in ('progress','receipts'):
 for p in sorted((r/'results'/folder).glob('*.json')):
  v=read(p);j=v.get('job',{});s=v.get('summary',{})
  out['fits'].append({'id':p.stem,'folder':folder,'status':v.get('status'),'phase':v.get('phase'),
   'host':v.get('host'),'pid':v.get('pid'),
   'route':j.get('route'),'stage':j.get('stage'),'campaign_phase':j.get('campaign_phase'),
   'depth':j.get('config',{}).get('model',{}).get('depth'),'data_seed':j.get('config',{}).get('seeds',{}).get('data'),
   'epochs_completed':s.get('epochs_completed',v.get('epochs_completed')),
   'geometry_samples_processed':v.get('geometry_samples_processed'),'elapsed_seconds':v.get('elapsed_seconds'),
   'optimization_wall_seconds':s.get('optimization_wall_seconds'),'optimization_cpu_seconds':s.get('optimization_cpu_seconds'),
   'validation_wall_seconds':s.get('validation_wall_seconds'),'test_evaluation_wall_seconds':s.get('test_evaluation_wall_seconds'),
   'error':v.get('error'),'resource_usage':v.get('resource_usage')})
out['squeue']=subprocess.run(['squeue','-h','-r','-u','jonathan.hornewall','-o','%i|%C|%T|%N|%j'],capture_output=True,text=True,timeout=10).stdout
out['shared_home_df']=subprocess.run(['df','-B1','/home'],capture_output=True,text=True,timeout=10).stdout
groups={}
for v in list(out['qualification'].values())+out['fits']:
 if v.get('status')=='running' and v.get('host') and v.get('pid'):
  groups.setdefault(v['host'],set()).add(int(v['pid']))
for name,v in out['workers'].items():
 if v.get('state')=='active' and v.get('pid'):
  groups.setdefault(name.split('_')[0],set()).add(int(v['pid']))
out['actual_cpu_rss']={}
owned={16208,16209,16210,16211,16223}
if (r/'ops/JOB_REGISTRY.json').exists():
 registry=read(r/'ops/JOB_REGISTRY.json')
 for key,value in registry.items():
  if key.endswith('_pool') or key in ('kp_recovery_array','kp_serial_job'):owned.add(int(value))
allocations={}
for row in out['squeue'].splitlines():
 parts=row.split('|')
 if len(parts)>=4 and int(parts[0].split('_')[0]) in owned and parts[2]=='RUNNING':
  allocations.setdefault(parts[3],parts[0])
for host,pids in groups.items():
 try:
  node=host.split('.')[0]
  if node not in allocations:continue
  jobid=allocations[node]
  if '_' in jobid:
   describe=subprocess.run(['scontrol','show','job',jobid,'-o'],capture_output=True,text=True,timeout=5)
   jobid=next(token.split('=',1)[1] for token in describe.stdout.split() if token.startswith('JobId='))
  command=['srun','--overlap','--jobid='+jobid,'--nodelist='+node,'--nodes=1',
   '--ntasks=1','--cpus-per-task=1','--time=00:00:10','--cpu-bind=cores',
   'ps','-p',','.join(str(p) for p in sorted(pids)),'-o','pid=,pcpu=,cputimes=,rss=,stat=']
  p=subprocess.run(command,capture_output=True,text=True,timeout=8)
  out['actual_cpu_rss'][host]={'returncode':p.returncode,'processes':p.stdout,'error':p.stderr[-500:]}
 except subprocess.TimeoutExpired:
  out['actual_cpu_rss'][host]={'error':'bounded process-telemetry transport timeout'}
print(json.dumps(out,separators=(',',':')))
'''

def check(root):
    root=Path(root)
    if (root/'ops/MONITOR_CLOSED.json').exists():return {'state':'closed_noop'}
    raw=cermics(shlex.join(['python3','-B','-c',SOURCE]),timeout=50)
    data=json.loads(raw)
    (root/'monitor_snapshots').mkdir(exist_ok=True)
    name=str(int(data['observed_unix']))+'.json'
    (root/'monitor_snapshots'/name).write_bytes(raw)
    (root/'monitor_snapshots/LATEST.json').write_bytes(raw)
    from collections import Counter
    print(json.dumps({'snapshot':str(root/'monitor_snapshots'/name),
        'qualification':{k:v['status'] for k,v in data['qualification'].items()},
        'bank_states':dict(Counter(v['status'] for v in data['banks'].values())),
        'fit_receipts':dict(Counter(v['status'] for v in data['fits'] if v['folder']=='receipts')),
        'failures':list(data['failures']),'queue':data['squeue'],'ready':data.get('READY.json')},indent=2))
    return data

if __name__=='__main__':check(Path(__file__).resolve().parents[1])
