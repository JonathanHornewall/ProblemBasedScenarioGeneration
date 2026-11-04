"""One isolated SP25 loader/loss profile; never edits a campaign input or process."""
from __future__ import annotations
import gc,hashlib,json,os,pathlib,resource,signal,sys,time,traceback

ROOT=pathlib.Path('/home/jonathan.hornewall/runs/lava_completion_20261102')
OUT=ROOT/'ops/loader_profile_20261104'
START=time.perf_counter();CPU_START=time.process_time();DEADLINE=START+260
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
 os.environ[key]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
os.environ['SOLVER_FREE_DFL_REPO']=str(ROOT/'code/vendor/public')
os.environ['LAVA_RUN_ROOT']=str(ROOT);os.environ['LAVA_CODE']=str(ROOT/'code')
for directory in ('ops','code/scripts','code/src','code/vendor/external/src','code/vendor/public'):
 sys.path.insert(0,str(ROOT/directory))
raw={'status':'starting','started_unix':time.time(),'budget':{'wall_seconds':300,'internal_wall_seconds':260,'CPU_seconds':275,'RSS_GiB':6,'address_space_GiB':10,'CPUs':1,'numerical_threads':1},
 'scientific_fits':0,'optimizer_updates':0,'enumerations':0,'live_profiler_attachments':0,
 'case':'SP25,N500,data151101,initialization161101','runs':[],'preloads':[],'samples':[]}
def write():
 temp=OUT/('raw_timing.json.'+str(os.getpid())+'.tmp')
 temp.write_text(json.dumps(raw,indent=2,sort_keys=True,allow_nan=False)+'\n');temp.replace(OUT/'raw_timing.json')
def stop(*args):raise TimeoutError('Diagnostic finite time budget')
signal.signal(signal.SIGALRM,stop);signal.alarm(270)
resource.setrlimit(resource.RLIMIT_CPU,(275,280))
def digest(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def clock():return time.perf_counter(),time.process_time()
def delta(t):return {'wall_seconds':time.perf_counter()-t[0],'CPU_seconds':time.process_time()-t[1]}
def core_id(cpu):
 p=pathlib.Path('/sys/devices/system/cpu')/('cpu'+str(cpu))/'topology'
 return [int((p/'physical_package_id').read_text()),int((p/'core_id').read_text())]
def check():
 if time.perf_counter()>DEADLINE:raise TimeoutError('Diagnostic internal wall budget')
 if proc.memory_info().rss>6*2**30:raise MemoryError('Diagnostic6GiB RSS guard')
try:
 import psutil
 proc=psutil.Process();allowed=sorted(os.sched_getaffinity(0));assert os.environ.get('SLURM_JOB_ID')
 assert len(allowed)==1,'A single Slurm-bound CPU is required'
 chosen=allowed[0];physical=core_id(chosen);active=[]
 for p in psutil.process_iter(['pid','cmdline','status']):
  cmd=p.info['cmdline']or[]
  if p.info['status']=='zombie' or not any(str(ROOT/'ops'/f)in cmd for f in('run_fit.py','prepare_bank.py','bank_geometry.py','coordinator.py')):continue
  affin=p.cpu_affinity();active.append({'pid':p.pid,'affinity':affin,'physical_cores':[core_id(c)for c in affin]})
  assert physical not in [core_id(c)for c in affin],'Diagnostic core overlaps a campaign process'
 raw['placement']={'job':os.environ['SLURM_JOB_ID'],'host':os.uname().nodename,'CPU':chosen,'physical_socket_core':physical,'campaign_affinities':active,'isolated_from_campaign':True}
 write()
 import numpy as np,torch
 torch.set_num_threads(1);torch.set_num_interop_threads(1)
 resource.setrlimit(resource.RLIMIT_AS,(10*2**30,10*2**30))
 from proximal_spo.public_highs import HighsBindingDataset,public_geometry
 from proximal_spo.seeds import derive_seed
 from bank_geometry import LazyNormals,LazyVertices
 from methods.lava import LAVA
 key='dys_shortest_path_25x25__data151101';bank=pathlib.Path('/tmp/lava_completion_20261102/seeded_banks_v2')/(key+'__init161101')
 meta=json.loads((bank/'prefix500.json').read_text())
 import pickle
 with (bank/'prefix500.pickle').open('rb')as f:fields=pickle.load(f)
 inputs=pathlib.Path('/home/jonathan.hornewall/runs/scaled_data_neural_20261021/frozen_data')/key/'observations_n500.npz'
 with np.load(inputs,allow_pickle=False)as f:
  contexts=f['train__contexts'].copy();costs=f['train__physical_costs'].copy()
 assert len(contexts)==500 and meta['initialization_seed']==161101
 train=HighsBindingDataset.__new__(HighsBindingDataset)
 train.model=None;train.feats=contexts;train.costs=costs;train.sols=fields['sols'];train.objs=fields['objs']
 train.relaxed_sols=train.sols;train.relaxed_objs=train.objs
 normals=LazyNormals(bank,fields['ctrs']);vertices=LazyVertices(bank,fields['vertices'])
 train.ctrs=normals;train.adjacent_verts=vertices
 collate=public_geometry(str(ROOT/'code/vendor/public')).collate_fn_2
 loss=LAVA(None,threshold=-1.0)
 raw['identities']={'input_file':str(inputs),'input_sha256':digest(inputs),'bank':str(bank),'prefix500_sha256':digest(bank/'prefix500.json'),'prefix_pickle_sha256':digest(bank/'prefix500.pickle'),'geometry_identity':meta['geometry_identity_sha256'],
  'source_sha256':{str(p.relative_to(ROOT)):digest(p)for p in [ROOT/'ops/bank_geometry.py',ROOT/'code/src/proximal_spo/public_highs.py',ROOT/'code/src/proximal_spo/seeds.py',ROOT/'code/vendor/public/data/dataset_augmented.py',ROOT/'code/vendor/public/methods/lava.py',ROOT/'code/scripts/run_central_path_final.py',OUT/'profile.py']}}
 raw['versions']={'python':sys.version,'torch':torch.__version__,'numpy':np.__version__,'torch_threads':torch.get_num_threads(),'torch_interop_threads':torch.get_num_interop_threads()}
 raw['initialization']={'wall_seconds':time.perf_counter()-START,'CPU_seconds':time.process_time()-CPU_START}
 raw['protocol']={'original_shuffle_seed':171101,'epoch':1,'derived_order_seed':derive_seed(171101,'epoch',1),'full_batch_positions':[0,7,14],'batch_size':32,'repeats_per_batch':4,'path_order':'AB,BA,AB,BA','threshold':-1.0,'mm':-1.0,
  'prediction':'fixed float32 sin(column*0.017+observation*0.11)+0.5*cos(column*0.031-observation*0.07); no model/optimizer',
  'resident':'sampled raw float64 normal matrices and vertices in RAM; identical public float32 conversion/collation/loss',
  'I/O_caveat':'No page-cache flushing; lazy vertex timer covers file open/mmap creation. Page faults/materialization can be charged to conversion. Normal read timer combines ZIP reading/decompression.',
  'scope':'Data assembly plus public LAVA forward and prediction-gradient backward only; excludes model forward/backward, optimizer, evaluation, diagnostics, persistence and one-time shared geometry construction.'}
 order=np.random.default_rng(raw['protocol']['derived_order_seed']).permutation(500)
 method_globals=HighsBindingDataset.__getitem__.__globals__;original_torch=method_globals['torch']
 pad_globals=collate.__globals__;original_pad=pad_globals['pad_sequence'];original_as_tensor=torch.as_tensor
 class Resident:
  def __init__(self,values):self.values=values
  def __getitem__(self,index):return self.values[index]
 class Timed:
  def __init__(self,source,name,parts):self.source,self.name,self.parts=source,name,parts
  def __getitem__(self,index):
   t=clock();value=self.source[index];a=delta(t);add(self.parts,self.name,a);return value
 def add(parts,name,a):
  value=parts.setdefault(name,{'wall_seconds':0.,'CPU_seconds':0.})
  for k in value:value[k]+=a[k]
 def subtract(total,*pieces):return {k:total[k]-sum(p[k]for p in pieces)for k in total}
 zero={'wall_seconds':0.,'CPU_seconds':0.}
 def pipeline(path,indices,resident,prediction):
  check();parts={};conversion_calls=0;pad_calls=0
  class TorchProxy:
   float32=original_torch.float32
   @staticmethod
   def as_tensor(value,**kwargs):
    nonlocal conversion_calls
    field=conversion_calls%8;conversion_calls+=1;t=clock();answer=original_as_tensor(value,**kwargs)
    add(parts,('normals_conversion'if field==6 else'vertices_conversion'if field==7 else'other_conversion'),delta(t));return answer
  def timed_pad(values,*args,**kwargs):
   nonlocal pad_calls
   name='normal_padding'if pad_calls==0 else'vertex_padding';pad_calls+=1;t=clock();answer=original_pad(values,*args,**kwargs);add(parts,name,delta(t));return answer
  if path=='A':train.ctrs=Timed(normals,'normal_read_decompress',parts);train.adjacent_verts=Timed(vertices,'vertex_open_mmap',parts)
  else:train.ctrs=resident[0];train.adjacent_verts=resident[1]
  pred=prediction.detach().clone().requires_grad_(True)
  method_globals['torch']=TorchProxy;pad_globals['pad_sequence']=timed_pad
  total=clock()
  try:
   t=clock();items=[train[int(i)]for i in indices];get_total=delta(t)
   method_globals['torch']=original_torch
   t=clock();batch=collate(items);collate_total=delta(t)
   pad_globals['pad_sequence']=original_pad
   t=clock();value=loss(pred,batch[7],batch[4],-1.0);parts['loss_forward']=delta(t)
   t=clock();value.backward();parts['loss_backward']=delta(t)
   whole=delta(total)
   parts['getitem_other']=subtract(get_total,*[parts.get(k,zero)for k in('normal_read_decompress','vertex_open_mmap','normals_conversion','vertices_conversion','other_conversion')])
   parts['collate_other']=subtract(collate_total,parts['normal_padding'],parts['vertex_padding'])
   parts['pipeline_other']=subtract(whole,get_total,collate_total,parts['loss_forward'],parts['loss_backward'])
   shape={'normal_padded':list(batch[6].shape),'vertices_padded':list(batch[7].shape),'dtype':str(batch[7].dtype),'normal_tensor_bytes':batch[6].numel()*batch[6].element_size(),'vertices_tensor_bytes':batch[7].numel()*batch[7].element_size()}
   scalar=float(value.detach());gradient=pred.grad.detach().clone()
   assert np.isfinite(scalar)and torch.isfinite(gradient).all()
   return {'path':path,'total':whole,'components':parts,'shapes':shape,'loss':scalar,'gradient_L2':float(torch.linalg.vector_norm(gradient))},gradient
  finally:
   method_globals['torch']=original_torch;pad_globals['pad_sequence']=original_pad
   train.ctrs=normals;train.adjacent_verts=vertices
 for batch_position in (0,7,14):
  check();indices=order[batch_position*32:(batch_position+1)*32].tolist();assert len(indices)==32
  t=clock();resident_normals={};resident_vertices={};pre_parts={}
  for i in indices:
   t0=clock();resident_normals[i]=normals[i];add(pre_parts,'normal_read_decompress',delta(t0))
   t0=clock();resident_vertices[i]=np.array(vertices[i],copy=True);add(pre_parts,'vertex_open_read_copy',delta(t0));check()
  preload=delta(t);resident=(Resident(resident_normals),Resident(resident_vertices))
  raw['preloads'].append({'batch_position':batch_position,'total':preload,'components':pre_parts,'normal_RAM_bytes':sum(v.nbytes for v in resident_normals.values()),'vertex_RAM_bytes':sum(v.nbytes for v in resident_vertices.values())})
  raw['samples'].append({'batch_position':batch_position,'original_epoch_indices':indices,'vertex_counts':[int(fields['vertex_counts'][i])for i in indices],
   'normal_dtypes':sorted({str(v.dtype)for v in resident_normals.values()}),'vertex_dtypes':sorted({str(v.dtype)for v in resident_vertices.values()}),'normal_shapes':[list(resident_normals[i].shape)for i in indices]})
  rows=torch.as_tensor(indices,dtype=torch.float32)[:,None];cols=torch.arange(costs.shape[1],dtype=torch.float32)[None,:]
  prediction=torch.sin(cols*.017+rows*.11)+.5*torch.cos(cols*.031-rows*.07)
  for repeat in range(4):
   equality={}
   for path in ('A','B')if repeat%2==0 else('B','A'):
    result,gradient=pipeline(path,indices,resident,prediction);result.update({'batch_position':batch_position,'repeat':repeat});raw['runs'].append(result)
    if batch_position==0 and repeat==0:equality[path]=(result['loss'],gradient)
    del gradient;write();check()
   if equality:
    a,b=equality['A'],equality['B'];raw['equality']={'scalar_exact':a[0]==b[0],'gradient_exact':torch.equal(a[1],b[1]),'gradient_max_absolute_difference':float((a[1]-b[1]).abs().max()),'gradient_sha256':hashlib.sha256(a[1].numpy().tobytes()).hexdigest()}
    assert raw['equality']['scalar_exact']and raw['equality']['gradient_exact']
   gc.collect()
  del resident,resident_normals,resident_vertices;gc.collect()
 raw['status']='complete'
except BaseException as exc:
 raw['status']='budget_censored'if isinstance(exc,(TimeoutError,MemoryError))else'failed'
 raw['error']=repr(exc);raw['traceback']=traceback.format_exc()
finally:
 signal.alarm(0);raw['finished_unix']=time.time();raw['total_wall_seconds']=time.perf_counter()-START;raw['total_CPU_seconds']=time.process_time()-CPU_START
 raw['peak_RSS_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024;write()
 print(json.dumps({k:raw[k]for k in('status','total_wall_seconds','total_CPU_seconds','peak_RSS_bytes')}),flush=True)
