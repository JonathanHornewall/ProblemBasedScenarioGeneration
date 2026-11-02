"""Admit proven random-free legacy prefixes or exact seeded-v2 caches."""
import fcntl,os,pathlib,socket,time
from qualify import read,write,sha
from bank_geometry import directory,get_geometry as load

def decision(root,job):
 root=pathlib.Path(root);key=job['required_data_key'];n=int(job['config']['sizes']['train'])
 if 'cfl' in job.get('route','') and (root/'ops/HOLD_CFL_COORDINATOR_REFRESH.json').exists():
  return {'ready':False,'reason':'CFL affected-score coordinator refresh'}
 registry=root/'ops/GEOMETRY_PROVENANCE.json';values=read(registry) if registry.exists() else {}
 entry=values.get(key,{})
 if entry.get('random_free_supplied_prefix',0)>=n:return {'ready':True,'initialization_seed':None}
 seed=int(job['config']['seeds']['initialization']);where=directory(key,seed)
 check=root/'ops/SEEDED_GEOMETRY_CHECK.json'
 checked=check.exists() and read(check).get('status')=='passed' and read(check).get('bank_geometry_sha256')==sha(root/'ops/bank_geometry.py')
 if checked and (where/f'prefix{n}.json').exists():
  meta=read(where/f'prefix{n}.json')
  if meta.get('geometry_rng_protocol')!='seeded_numpy_resume_v2' or meta.get('initialization_seed')!=seed:
   raise RuntimeError('Seeded cache identity mismatch')
  return {'ready':True,'initialization_seed':seed}
 if (directory(key)/f'prefix{n}.json').exists():
  from geometry_provenance import certify
  entry=certify(root,key,n)
  if entry.get('random_free_supplied_prefix',0)>=n:return {'ready':True,'initialization_seed':None}
 request=root/'seeded_requests'/(key+'__init'+str(seed)+'.json');request.parent.mkdir(exist_ok=True)
 with request.with_suffix('.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);old=read(request) if request.exists() else {}
  write(request,{'key':key,'initialization_seed':seed,'maximum':max(n,old.get('maximum',0)),
   'host':socket.gethostname(),'source_job_ids':sorted(set(old.get('source_job_ids',[])+[job['id']])),
   'updated_unix':time.time(),'reason':'Actual random-transition branch reachable; seeded construction required.'})
 return {'ready':False,'reason':'seeded_v2_cache_requested','request':str(request)}

def get_geometry(job,model,contexts,costs,update):
 result=decision(os.environ['LAVA_RUN_ROOT'],job)
 if not result['ready']:raise RuntimeError('Seeded geometry required before fit admission')
 return load(job,model,contexts,costs,update,initialization_seed=result['initialization_seed'])

def release_unstarted_holds(root,key,seed=None,maximum=None):
 root=pathlib.Path(root);moved=[]
 for failure in (root/'failures').glob('lava_complete__*.json'):
  v=read(failure)
  if v.get('returncode')!=75:continue
  manifest=root/'jobs'/(failure.stem+'.json')
  if not manifest.exists():continue
  job=read(manifest)['jobs'][0]
  if job['required_data_key']!=key:continue
  n=job['config']['sizes']['train']
  if maximum is not None and n>maximum:continue
  if seed is not None and job['config']['seeds']['initialization']!=seed:continue
  destination=root/'recovery/provenance_holds'/(failure.stem+'__'+str(time.time_ns())+'.json')
  destination.parent.mkdir(parents=True,exist_ok=True);failure.replace(destination);moved.append(failure.stem)
 return moved
