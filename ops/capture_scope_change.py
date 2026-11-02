"""Snapshot grandfathered live work and clamp existing preparation requests."""
import pathlib,subprocess,time,json
from qualify import read,write
from scope_control import preparation_maximum
r=pathlib.Path('/home/jonathan.hornewall/runs/lava_completion_20261102');snap={'observed_unix':time.time(),'nodes':[]}
for job,node in ((16223,'node09'),(16726,'node07'),(16727,'node08')):
 p=subprocess.run(['srun','--overlap','--jobid='+str(job),'--nodelist='+node,'--ntasks=1','--cpus-per-task=1',
  '--time=00:00:15','--cpu-bind=cores','/home/jonathan.hornewall/runs/scaled_data_neural_20261021/venv/bin/python',str(r/'ops/snapshot_scope_node.py')],capture_output=True,text=True,timeout=18)
 if p.returncode:raise RuntimeError(p.stderr+p.stdout)
 snap['nodes'].append(json.loads(p.stdout))
write(r/'ops/SCOPE_CHANGE_SNAPSHOT.json',snap)
grand=[v for node in snap['nodes'] for v in node['fits'] if not v['required_after_amendment']]
write(r/'ops/SCOPE_GRANDFATHERED_FITS.json',grand)
a=read(r/'SCOPE_AMENDMENT_20261102_CFL_N100.json');a['grandfathered_active_fit_ids']=[v['id'] for v in grand];a['grandfathered_count']=len(grand)
write(r/'SCOPE_AMENDMENT_20261102_CFL_N100.json',a)
changed=[]
for p in (r/'seeded_requests').glob('*.json'):
 v=read(p);n=preparation_maximum(r,v['key'],v['maximum'])
 if n==v['maximum']:continue
 archive=r/'ops/scope_saved_requests'/(str(time.time_ns())+'__'+p.name);write(archive,v)
 if n==0:p.replace(archive.with_name(archive.stem+'__excluded.json'))
 else:write(p,{**v,'maximum':n,'prior_maximum':v['maximum'],'scope_amendment':a['revision'],'updated_unix':time.time()})
 changed.append({'key':v['key'],'old_maximum':v['maximum'],'new_maximum':n})
ad=read(r/'ADMISSION.json');ad['routes']=[v for v in ad['routes'] if 'cfl' not in v]+a['retained_cfl_routes']
for route in a['retained_cfl_routes']:ad.setdefault('prepare_maximum_by_route',{})[route]=100
ad['prepare_seed_limit_by_route']={k:v for k,v in ad.get('prepare_seed_limit_by_route',{}).items() if 'cfl' not in k}
ad['scope_amendment']=a['revision'];write(r/'ADMISSION.json',ad)
print(json.dumps({'grandfathered_count':len(grand),'grandfathered':[{k:v[k] for k in ('id','host','pid')} for v in grand],
 'builders':[{k:v[k] for k in ('key','host','pid','maximum')}for node in snap['nodes']for v in node['builders']],
 'requests_clamped':changed,'admission':ad,'snapshot':snap}))
