"""Stage compact consumed-anchor metadata and launch a finite branch replay."""
import json,pathlib,subprocess,time
r=pathlib.Path('/home/jonathan.hornewall/runs/lava_completion_20261102');groups={}
for p in (r/'results/receipts').glob('*.json'):
 v=json.loads(p.read_text());j=v.get('job',{})
 if v.get('status')!='complete' or not j.get('lava_completion_revision'):continue
 key=j['required_data_key'];n=j['config']['sizes']['train']
 groups[key]=max(n,groups.get(key,0))
targets=[{'key':k,'n':n} for k,n in sorted(groups.items())]
(r/'ops/PROVENANCE_TARGETS.json').write_text(json.dumps(targets,indent=2)+'\n')
owners={}
for v in targets:
 meta=json.loads((r/'banks'/v['key']/f'prefix{v["n"]}.json').read_text())
 owners.setdefault(meta['host'].split('.')[0],[]).append(v)
jobs={'node09':16223,'node07':16726,'node08':16727};staged=[]
for node,items in owners.items():
 source='''import pathlib,shutil,hashlib,json
r=pathlib.Path('/home/jonathan.hornewall/runs/lava_completion_20261102')
for v in ITEMS:
 key,n=v['key'],v['n'];p=pathlib.Path('/tmp/lava_completion_20261102/banks')/key/f'prefix{n}.pickle'
 meta=json.loads((r/'banks'/key/f'prefix{n}.json').read_text())
 assert hashlib.sha256(p.read_bytes()).hexdigest()==meta['metadata_sha256']
 destination=r/'provenance_inputs'/key;destination.mkdir(parents=True,exist_ok=True)
 shutil.copyfile(p,destination/p.name)
 print(key,n,p.stat().st_size)
'''.replace('ITEMS',repr(items))
 cmd=['srun','--overlap','--jobid='+str(jobs[node]),'--nodelist='+node,'--ntasks=1',
  '--cpus-per-task=1','--time=00:00:30','--cpu-bind=cores','python3','-c',source]
 p=subprocess.run(cmd,capture_output=True,text=True,timeout=35)
 if p.returncode:raise RuntimeError(p.stderr+p.stdout)
 staged.append({'node':node,'stdout':p.stdout})
if not targets:raise RuntimeError('No consumed prefixes found')
cmd=['sbatch','--parsable','--partition=defq','--nodelist=node04','--nice=10000','--requeue',
 '--job-name=lavaprov_20261102','--ntasks=1','--cpus-per-task=1','--mem=4G','--time=00:30:00',
 f'--array=0-{len(targets)-1}%4','--output='+str(r/'logs/provenance-%A_%a.out'),str(r/'ops/provenance.sbatch'),str(r)]
p=subprocess.run(cmd,capture_output=True,text=True,timeout=15)
if p.returncode:raise RuntimeError(p.stderr+p.stdout)
print(json.dumps({'submitted_unix':time.time(),'job_id':p.stdout.strip(),'targets':targets,'staged':staged}))
