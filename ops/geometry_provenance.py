"""Classify the released random-transition branch without neighbor enumeration.

Replay the original objective/basis sequence. Compare each consumed anchor to
the retained metadata and evaluate the exact initial public transition test.
The retry after AssertionError repeats that same test, so an initially valid
transition column cannot reach search_for_transition_column on either call.
"""
import argparse,fcntl,hashlib,json,os,pathlib,pickle,socket,time
from qualify import read,write,sha

class RandomTransition(Exception):pass

def certify(root,key,n,metadata=None):
 root=pathlib.Path(root);lockfile=root/'provenance'/key/'classification.lock';lockfile.parent.mkdir(parents=True,exist_ok=True)
 with lockfile.open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  return _certify(root,key,n,metadata)

def _certify(root,key,n,metadata=None):
 import numpy as np
 import scipy.linalg
 import torch
 from physical128_data import array_hash
 from simplified_data import load_structure
 import proximal_spo.public_highs as public
 root=pathlib.Path(root);registry=root/'ops/GEOMETRY_PROVENANCE.json'
 existing=read(registry).get(key,{}) if registry.exists() else {}
 if existing.get('random_free_supplied_prefix',0)>=n:return existing
 if existing.get('random_transition_first_sample') and existing['random_transition_first_sample']<=n:return existing
 torch.set_num_threads(1)
 if torch.get_num_interop_threads()!=1:torch.set_num_interop_threads(1)
 where=pathlib.Path('/tmp/lava_completion_20261102/banks')/key
 meta=read(root/'banks'/key/f'prefix{n}.json')
 data=pathlib.Path(metadata) if metadata else where/f'prefix{n}.pickle'
 if sha(data)!=meta['metadata_sha256']:raise RuntimeError('Retained anchor metadata differs')
 with data.open('rb') as f:fields=pickle.load(f)
 with np.load(meta['source_training_file'],allow_pickle=False) as f:
  contexts=f['train__contexts'][:n].copy();costs=f['train__physical_costs'][:n].copy()
 if array_hash(contexts)!=meta['prefix_context_sha256'] or array_hash(costs)!=meta['prefix_cost_sha256']:
  raise RuntimeError('Strict consumed training prefix differs')
 source=pathlib.Path('/home/jonathan.hornewall/runs/scaled_data_neural_20261021/frozen_data')/key
 model=public.HighsOptModel(load_structure(source).continuous_relaxation())
 completed=0;unique=0;first=None;first_anchor=None
 class BranchOnly(public.HighsLAVADataset):
  def _get_adjacent_vertices(self,solution):
   nonlocal unique,first,first_anchor
   unique+=1
   matrix,basic,nonbasic=self._basis(solution)
   if np.sum(solution[basic]<1e-10)!=0:
    # Exact first test from released dataset_augmented.py, use_scipy=True.
    basic=np.array(sorted(basic));nonbasic=np.array(sorted(nonbasic))
    dirs=scipy.linalg.solve(matrix[:,basic],matrix[:,nonbasic])
    zero=np.where(solution[basic]==0)[0]
    transitions=np.where(np.all(dirs[zero,:]<=0,axis=0))[0]
    if len(transitions)==0:
     first=completed+1;first_anchor=hashlib.sha256(solution[:model.num_cost].tobytes()).hexdigest()
     raise RandomTransition()
   return [solution]
 def progress(count,dataset):
  nonlocal completed
  anchor=np.asarray(dataset.slack_model._model.getSolution().col_value,dtype=np.float64)[:model.num_cost].copy()
  anchor[np.abs(anchor)<1e-10]=0.
  if not np.array_equal(anchor,fields['sols'][count-1]):raise RuntimeError('Replayed consumed anchor differs')
  completed=count
 started,cpu=time.perf_counter(),time.process_time()
 original=public.binding_normals
 public.binding_normals=lambda p,s:np.empty((0,p.variable_count),dtype=np.float64)
 try:
  try:BranchOnly(model,contexts,costs,public_repo=os.environ['SOLVER_FREE_DFL_REPO'],vertex_tolerance=0.,progress=progress)
  except RandomTransition:pass
 finally:public.binding_normals=original
 result={'key':key,'supplied_training_samples':n,'status':'affected' if first else 'random_free',
  'random_free_supplied_prefix':completed,'random_transition_first_sample':first,
  'first_random_anchor_sha256':first_anchor,'unique_anchor_tests':unique,
  'all_preceding_replayed_anchors_equal':True,'neighbor_enumerations':0,
  'adapter_retry_initial_transition_test_identical':True,
  'wall_seconds':time.perf_counter()-started,'process_cpu_seconds':time.process_time()-cpu,
  'finished_unix':time.time(),'host':socket.gethostname(),'slurm_job_id':os.environ['SLURM_JOB_ID'],
  'classifier_sha256':sha(pathlib.Path(__file__)),'metadata_sha256':sha(data),
  'prefix_cost_sha256':meta['prefix_cost_sha256'],'prefix_context_sha256':meta['prefix_context_sha256'],
  'source_training_file':meta['source_training_file']}
 write(root/'provenance'/key/f'prefix{n}.json',result)
 with (root/'ops/GEOMETRY_PROVENANCE.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);values=read(registry) if registry.exists() else {}
  prior=values.get(key,{})
  result['random_free_supplied_prefix']=max(prior.get('random_free_supplied_prefix',0),result['random_free_supplied_prefix'])
  if prior.get('random_transition_first_sample'):
   result['random_transition_first_sample']=min(first or n+1,prior['random_transition_first_sample'])
  values[key]=result;write(registry,values)
 return result

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--key',required=True)
 p.add_argument('--n',type=int,required=True);p.add_argument('--metadata')
 a=p.parse_args();print(json.dumps(certify(a.root,a.key,a.n,a.metadata)))
