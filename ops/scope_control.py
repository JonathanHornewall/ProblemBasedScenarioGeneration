"""Current human scope; fresh admission is independent of stale worker queues."""
import os,pathlib,socket,time
from qualify import read,write
NAME='SCOPE_AMENDMENT_20261102_CFL_N100.json'
def amendment(root):
 path=pathlib.Path(root)/NAME
 return read(path) if path.exists() else None
def resource_censors(root):
 path=pathlib.Path(root)/'ops/RESOURCE_CENSORED_CELLS_20261103.json'
 return read(path) if path.exists() else {}
def censored_ids(root):
 value=resource_censors(root)
 return {v['id']for v in value.get('attempts',[])}if value.get('status')=='resource_censored' else set()
def required_fit(root,job):
 a=amendment(root)
 if a and job.get('route') in a.get('excluded_routes',[]):return False
 if not a or 'cfl' not in job.get('route',''):return True
 c=job.get('config',{});s=c.get('sizes',{})
 return (job['route'] in a['retained_cfl_routes'] and s.get('train')==100
  and s.get('validation')==32 and s.get('test')==100 and c.get('model',{}).get('depth') in (0,1,2,3))
def runtime_fit_allowed(root,job):
 if job.get('route') in resource_censors(root).get('no_further_attempt_routes',[]):return False
 if job.get('id') in censored_ids(root):return False
 if required_fit(root,job):return True
 file=pathlib.Path(root)/'ops/SCOPE_GRANDFATHERED_FITS.json'
 return file.exists() and any(v['id']==job['id'] and v['pid']==os.getpid()
  and v['host']==socket.gethostname() for v in read(file))
def preparation_maximum(root,key,maximum):
 a=amendment(root);route=key.split('__data')[0]
 if a and route in a.get('excluded_routes',[]):return 0
 if not a or 'cfl' not in route:return maximum
 return min(maximum,100) if route in a['retained_cfl_routes'] else 0
def retained_prefix_durable(root,key,seed):
 root=pathlib.Path(root);proof=read(root/'ops/GEOMETRY_PROVENANCE.json')
 paths=[pathlib.Path('/tmp/lava_completion_20261102/seeded_banks_v2')/(key+'__init'+str(seed)),pathlib.Path('/tmp/lava_completion_20261102/banks')/key]
 for where in paths:
  meta=where/'prefix100.json';data=where/'prefix100.pickle'
  if not meta.is_file() or not data.is_file():continue
  v=read(meta)
  if data.stat().st_size!=v.get('metadata_bytes'):continue
  seeded=v.get('geometry_rng_protocol')=='seeded_numpy_resume_v2' and v.get('initialization_seed')==seed and (where/'rng_sample000100.pickle').is_file()
  if seeded or proof.get(key,{}).get('random_free_supplied_prefix',0)>=100:return {'where':str(where),'metadata':v}
 return None
def phase_required(root,family,phase):
 return not amendment(root) or family!='cfl' or phase['train']==100
def route_required(root,route):
 a=amendment(root)
 if a and route in a.get('excluded_routes',[]):return False
 return not a or 'cfl' not in route or route in a['retained_cfl_routes']
def drain_status(root):
 root=pathlib.Path(root);path=root/'ops/SCOPE_GRANDFATHERED_FITS.json';entries=read(path) if path.exists() else []
 result={'active':[],'drained':[],'unresolved':[]}
 for v in entries:
  record=root/'workers'/v['worker_record'] if v.get('worker_record') else None
  worker=read(record) if record and record.exists() else {}
  if worker.get('state')=='active' and worker.get('pid')==v['pid']:
   result['active'].append(v['id']);continue
  receipt=root/'results/receipts'/(v['id']+'.json')
  if receipt.exists() or worker.get('updated_unix',0)>=v['birth_unix'] and worker.get('state') in ('active','waiting_for_dependencies_or_memory','ended'):
   result['drained'].append(v['id'])
  else:result['unresolved'].append(v['id'])
 return result
def kp_status(root):
 root=pathlib.Path(root);values=[]
 for i in range(6):
  path=root/'recovery/jobs'/('job'+str(i)+('_serial' if i in (4,5) else '')+'.json')
  if not path.exists():path=root/'recovery/jobs'/('job'+str(i)+'.json')
  job=read(path)['jobs'][0];receipt=root/'results/receipts'/(job['id']+'.json')
  v=read(receipt) if receipt.exists() else {}
  values.append({'id':job['id'],'complete':v.get('status')=='complete' and (v.get('summary')or{}).get('epochs_completed')==300})
 return values
def scope_skip(root,kind,identity,detail):
 write(pathlib.Path(root)/'ops/scope_admission_skips'/(identity+'__'+str(time.time_ns())+'.json'),
  {'kind':kind,'identity':identity,'detail':detail,'observed_unix':time.time(),
   'pid':os.getpid(),'host':socket.gethostname(),'scientific_failure':False})
