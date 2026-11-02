"""Two original CFL observations; test real resume with explicit RNG draws.

The extra draws are instrumentation for this implementation fixture only.
They make a skipped enumeration observably shift the next draw unless the
saved RNG position is restored. No fixture cache is admitted to science.
"""
import hashlib,json,os,pathlib,pickle,time,subprocess,sys
import numpy as np
from bank_geometry import prepare,directory
from proximal_spo.public_highs import public_geometry
from qualify import read,write,sha
root=pathlib.Path(os.environ['LAVA_RUN_ROOT']);key='dys_cfl_50x5__data17';seed=8201
where=pathlib.Path('/tmp/lava_completion_20261102/seed_resume_check')/os.environ['SLURM_JOB_ID']
records=root/'implementation_checks/seed_resume'/os.environ['SLURM_JOB_ID']
public=public_geometry(os.environ['SOLVER_FREE_DFL_REPO']);draws=[]
originals={name:getattr(public,name) for name in ('get_adjacent_vertices_degenerate_case','get_adjacent_vertices_non_degenerate_case')}
def wrap(fn):
 def instrumented(*a,**k):
  draws.append(int(np.random.randint(0,2**31-1)));return fn(*a,**k)
 return instrumented
for name,fn in originals.items():setattr(public,name,wrap(fn))
started=time.perf_counter()
cli_seeds={}
for test_key,expected in ((key,8201),('dys_shortest_path_10x10__data1103',201),('dys_shortest_path_20x20__data151101',161101)):
 p=subprocess.run([sys.executable,str(root/'ops/bank_geometry.py'),'--root',str(root),'--key',test_key,
  '--print-original-initialization-seed'],capture_output=True,text=True,check=True)
 assert int(p.stdout.strip())==expected;cli_seeds[test_key]=expected
try:
 prepare(root,key,2,initialization_seed=seed,where_override=where/'full',record_override=records/'full',reuse_random_free=False)
 full_draws=draws.copy();draws.clear()
 prepare(root,key,1,initialization_seed=seed,where_override=where/'split',record_override=records/'split',reuse_random_free=False)
 first_draws=draws.copy();draws.clear()
 prepare(root,key,2,initialization_seed=seed,where_override=where/'split',record_override=records/'split',reuse_random_free=False)
 resumed_draws=draws.copy()
 assert len(full_draws)>=2 and full_draws==first_draws+resumed_draws
 for n in (1,2):
  values=[]
  for part in ('full','split'):
   with (where/part/f'rng_sample{n:06d}.pickle').open('rb') as f:values.append(pickle.load(f))
  for field in ('numpy_before','numpy_after'):
   assert pickle.dumps(values[0][field],protocol=5)==pickle.dumps(values[1][field],protocol=5)
  assert values[0]['vertex_name']==values[1]['vertex_name']
  assert sha(where/'full'/values[0]['vertex_name'])==sha(where/'split'/values[1]['vertex_name'])
 assert directory(key,seed)!=directory(key,seed+1)
 result={'status':'passed','scientific_fits':False,'supplied_original_training_samples':2,
  'key':key,'initialization_seed':seed,'fixture_only_injected_draws':True,
  'full_draws':full_draws,'split_first_draws':first_draws,'resumed_draws':resumed_draws,
  'exact_before_after_rng_states':True,'exact_resumed_rows':True,'seed_separates_cache_namespace':True,
  'CLI_original_initialization_seeds':cli_seeds,
  'elapsed_wall_seconds':time.perf_counter()-started,'slurm_job_id':os.environ['SLURM_JOB_ID'],
  'bank_geometry_sha256':sha(root/'ops/bank_geometry.py'),'check_sha256':sha(pathlib.Path(__file__))}
 write(root/'ops/SEEDED_GEOMETRY_CHECK.json',result);print(json.dumps(result))
finally:
 for name,fn in originals.items():setattr(public,name,fn)
