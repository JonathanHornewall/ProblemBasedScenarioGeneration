"""Single bounded state/impact reconciliation after the correctness repair."""
import json,pathlib,subprocess,time,collections
r=pathlib.Path('/home/jonathan.hornewall/runs/lava_completion_20261102')
out={'observed_unix':time.time(),'registry':json.loads((r/'ops/JOB_REGISTRY.json').read_text()),
 'provenance':json.loads((r/'ops/GEOMETRY_PROVENANCE.json').read_text())}
out['queue']=subprocess.run(['squeue','-h','-r','-u','jonathan.hornewall','-o','%i|%C|%T|%N|%j'],capture_output=True,text=True,timeout=5).stdout
out['counts']={};out['affected_completed_ids']=[];out['unresolved_completed_ids']=[]
count=collections.Counter();routes=collections.Counter()
for p in (r/'results/receipts').glob('*.json'):
 v=json.loads(p.read_text());j=v.get('job',{});s=v.get('summary',{})
 if v.get('status')!='complete':continue
 n=j.get('config',{}).get('sizes',{}).get('train',0);key=j.get('required_data_key')
 if j.get('lava_completion_revision'):
  proof=out['provenance'].get(key,{})
  if proof.get('random_free_supplied_prefix',0)>=n:state='retained_random_free'
  elif s.get('geometry',{}).get('geometry_rng_protocol')=='seeded_numpy_resume_v2':state='seeded_v2'
  elif proof.get('random_transition_first_sample') and proof['random_transition_first_sample']<=n:
   state='affected';out['affected_completed_ids'].append(p.stem)
  else:state='unresolved';out['unresolved_completed_ids'].append(p.stem)
 else:state='original_seeded_KP_recovery'
 count[state]+=1;routes[(j.get('route'),j.get('stage'),state)]+=1
out['counts']=dict(count);out['counts_by_route']=[{'route':k[0],'stage':k[1],'provenance':k[2],'count':v} for k,v in sorted(routes.items())]
out['workers']={p.name:json.loads(p.read_text()) for p in (r/'workers').glob('*.json')}
out['banks']={p.parent.name:json.loads(p.read_text()) for p in (r/'banks').glob('*/STATUS.json')}
out['seeded_banks']={p.parent.name:json.loads(p.read_text()) for p in (r/'seeded_banks_v2').glob('*/STATUS.json')}
out['serial_live']={p.name:json.loads(p.read_text()) for p in (r/'recovery/live').glob('*.json')}
out['failures']={p.name:json.loads(p.read_text()) for p in (r/'failures').glob('*.json')}
out['resource_repair_log']={p.name:p.read_text()[-2200:] for p in (r/'logs').glob('repair-*.out')}
out['seeded_logs']={p.name:p.read_text()[-2500:] for p in (r/'logs').glob('seeded__*.log')}
out['prefixes']={}
for p in (r/'banks').glob('*/prefix100.json'):
 v=json.loads(p.read_text());out['prefixes'][p.parent.name]={k:v.get(k) for k in ('build_wall_seconds_cumulative','build_cpu_seconds_cumulative','vertex_storage_bytes','host')}
out['telemetry']={}
for job,node in ((16223,'node09'),(16726,'node07'),(16727,'node08')):
 if not any(row.startswith(str(job)+'|') for row in out['queue'].splitlines()):continue
 p=subprocess.run(['srun','--overlap','--jobid='+str(job),'--nodelist='+node,'--ntasks=1','--cpus-per-task=1',
  '--time=00:00:10','--cpu-bind=cores','bash','-c','free -b; df -B1 /tmp; ps -u jonathan.hornewall -o pid=,pcpu=,cputimes=,rss=,args='],capture_output=True,text=True,timeout=12)
 out['telemetry'][node]={'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr[-1000:]}
print(json.dumps(out))
