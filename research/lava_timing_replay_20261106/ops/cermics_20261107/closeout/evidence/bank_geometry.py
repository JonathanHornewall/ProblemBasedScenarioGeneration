"""One owner builds exact public geometry; strict prefix views serve each N.

The maintained HighsLAVADataset performs every solve and enumeration unchanged.
The callback writes completed anchors and prefix metadata to node-local scratch.
Training loads only its supplied prefix, and vertex rows are never coalesced.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import pickle
import random
import shutil
import socket
import sys
import time

from qualify import read,sha,write

def directory(key,initialization_seed=None):
    if initialization_seed is not None:
        return (Path(os.environ['LAVA_GEOMETRY_ROOT'])/'seeded_banks_v2')/(key+'__init'+str(initialization_seed))
    return (Path(os.environ['LAVA_GEOMETRY_ROOT'])/'banks')/key

def vertex_name(anchor):
    return hashlib.sha256(anchor.tobytes()).hexdigest()+'.npy'

def prepare(root,key,maximum=1000,force=False,initialization_seed=None,where_override=None,record_override=None,reuse_random_free=True):
    # The released recursive lexicographic search exceeds Python's default
    # 1000 frames on SP50. Raise only this interpreter resource limit; its
    # original three-second search timeout and basis cap stay unchanged.
    sys.setrecursionlimit(max(sys.getrecursionlimit(),20000))
    import numpy as np
    import highspy
    import torch
    from physical128_data import array_hash
    from simplified_data import load_structure
    from proximal_spo.public_highs import HighsLAVADataset,HighsOptModel,binding_normals
    import proximal_spo.public_highs as public_highs
    torch.set_num_threads(1)
    if torch.get_num_interop_threads()!=1:torch.set_num_interop_threads(1)
    root=Path(root);where=Path(where_override) if where_override else directory(key,initialization_seed);where.mkdir(parents=True,exist_ok=True)
    record=Path(record_override)/'STATUS.json' if record_override else root/('banks' if initialization_seed is None else 'seeded_banks_v2')/(key if initialization_seed is None else key+'__init'+str(initialization_seed))/'STATUS.json'
    shared=record.parent
    old=read(record) if record.exists() else {}
    if not force and old.get('status')=='complete' and (where/f'prefix{maximum}.json').exists():return
    if old:write(record.parent/'build_history'/(str(time.time_ns())+'.json'),old)
    source=(root/'frozen_data')/key
    meta=read(source/'receipt.json');route,seed=key.split('__data')
    development=seed not in [str(s['data']) for s in read(root/'PLAN.json')['cells'][route]['confirmation']]
    train_source=source.parent/'retuning_development'/key if development and maximum>100 else source
    stored_n=100 if maximum<=100 else 500 if maximum<=500 else 1000
    training_file=train_source/f'observations_n{stored_n}.npz'
    source_receipt=read(train_source/'receipt.json')
    if sha(training_file)!=source_receipt['files_sha256'][training_file.name]:
        raise RuntimeError('Frozen training file changed')
    with np.load(training_file,allow_pickle=False) as f:
        # No validation, test, conditional moments or generator is read here.
        contexts=f['train__contexts'][:maximum].copy()
        costs=f['train__physical_costs'][:maximum].copy()
    if len(costs)!=maximum:raise ValueError('Exact supplied training prefix missing')
    problem=load_structure(source).continuous_relaxation();model=HighsOptModel(problem)
    seeded=initialization_seed is not None
    if seeded:
        random.seed(initialization_seed);np.random.seed(initialization_seed);torch.manual_seed(initialization_seed)
    numpy_before=np.random.get_state();completed_samples=0;reused_random_free=0
    provenance=read(root/'ops/GEOMETRY_PROVENANCE.json') if (root/'ops/GEOMETRY_PROVENANCE.json').exists() else {}
    independent_prefix=provenance.get(key,{}).get('random_free_supplied_prefix',0) if reuse_random_free else 0
    def rng_hash(state):return hashlib.sha256(pickle.dumps(state,protocol=5)).hexdigest()
    info={'key':key,'route':route,'data_seed':int(seed),'source_observation_sha256':sha(training_file),
          'source_training_file':str(training_file),'host':socket.gethostname(),
          'scratch':str(where),'slurm_job_id':os.environ['SLURM_JOB_ID'],'status':'running',
          'started_unix':time.time(),'maximum_training_samples':maximum,
          'validation_opened':False,'test_opened':False,'generator_called':False,
          'adapter_sha256':sha(Path(os.environ['LAVA_CODE'])/'src/proximal_spo/public_highs.py'),
          'highs_version':highspy.Highs().version(),'basis_enumeration_cap':250,'vertex_tolerance':0.,
          'binding_normals_storage':'lossless NPZ per exact anchor; lazy original arrays at training',
          'python_recursion_limit':sys.getrecursionlimit(),
          'geometry_rng_protocol':'seeded_numpy_resume_v2' if seeded else 'legacy_unseeded_v1',
          'initialization_seed':initialization_seed,
          'builder_identity':{'schema':2 if seeded else 1,'initialization_seed':initialization_seed,
              'adapter_sha256':sha(Path(os.environ['LAVA_CODE'])/'src/proximal_spo/public_highs.py'),
              'basis_enumeration_cap':250,'vertex_tolerance':0.},
          'reused_completed_prefixes':sorted(int(p.stem[6:]) for p in where.glob('prefix*.json'))}
    write(record,info)
    started,cpu_started=time.perf_counter(),time.process_time()
    sols,objs,ctrs,names,counts=[],[],[],[],[]
    unique_names=set()
    class Retained(HighsLAVADataset):
        def _get_adjacent_vertices(self,solution):
            nonlocal reused_random_free
            anchor=np.asarray(solution[:model.num_cost],dtype=np.float64)
            path=where/vertex_name(anchor)
            checkpoint=where/f'rng_sample{completed_samples+1:06d}.pickle'
            if seeded and checkpoint.exists():
                with checkpoint.open('rb') as f:state=pickle.load(f)
                if state['vertex_name']!=path.name or rng_hash(state['numpy_before'])!=rng_hash(np.random.get_state()):
                    raise RuntimeError('Seeded resume RNG position/anchor differs')
                if not path.exists() or sha(path)!=state['vertex_sha256']:raise RuntimeError('Seeded completed anchor changed')
                np.random.set_state(state['numpy_after'])
            elif seeded and not path.exists() and completed_samples<independent_prefix:
                original=directory(key)/path.name
                if original.exists() and not original.is_symlink():
                    os.link(original,path);reused_random_free+=1
            elif seeded and path.exists():
                # A crash between row storage and the RNG commit leaves an
                # uncommitted anchor. Preserve it, then enumerate from the
                # exact restored before-state rather than trusting it.
                archive=where/'uncommitted_attempts';archive.mkdir(exist_ok=True)
                path.replace(archive/(str(time.time_ns())+'__'+path.name))
            if path.exists():
                # Rebuild an interrupted dataset in original solver order;
                # already completed unique-anchor enumerations are reused.
                vertices=np.load(path,allow_pickle=False)
                return np.pad(vertices,((0,0),(0,len(solution)-model.num_cost)))
            return super()._get_adjacent_vertices(solution)
    def progress(n,d):
        nonlocal numpy_before,completed_samples
        solution=np.asarray(d.slack_model._model.getSolution().col_value,dtype=np.float64)
        solution[np.abs(solution)<1e-10]=0.
        sol=solution[:model.num_cost].copy();vertices=d.adjacent_vertices_cache[tuple(sol)]
        name=vertex_name(sol);path=where/name
        if not path.exists():
            if shutil.disk_usage(where).free < vertices.nbytes+30*2**30:
                raise RuntimeError('Node-local storage reserve reached; retain completed prefix')
            temp=path.with_suffix(f'.{os.getpid()}.tmp')
            with temp.open('wb') as f:np.save(f,vertices,allow_pickle=False)
            temp.replace(path)
        normals_name=Path(name).stem+'.normals.npz';normals_path=where/normals_name
        if not normals_path.exists():
            normals=binding_normals(problem,sol)
            tmp=normals_path.with_suffix(f'.{os.getpid()}.tmp')
            with tmp.open('wb') as f:np.savez_compressed(f,normals=normals)
            tmp.replace(normals_path)
        unique_names.add(name);sols.append(sol);objs.append([float(costs[n-1]@sol)])
        ctrs.append(normals_name);names.append(name);counts.append(len(vertices))
        if seeded:
            checkpoint=where/f'rng_sample{n:06d}.pickle'
            state={'sample':n,'vertex_name':name,'vertex_sha256':sha(path),
                   'numpy_before':numpy_before,'numpy_after':np.random.get_state()}
            if checkpoint.exists():
                with checkpoint.open('rb') as f:previous=pickle.load(f)
                if (rng_hash(previous['numpy_before'])!=rng_hash(state['numpy_before'])
                    or rng_hash(previous['numpy_after'])!=rng_hash(state['numpy_after'])):
                    raise RuntimeError('Seeded completed-sample RNG state differs')
            else:
                tmp=checkpoint.with_suffix(f'.{os.getpid()}.tmp')
                with tmp.open('wb') as f:pickle.dump(state,f,protocol=5)
                tmp.replace(checkpoint)
            numpy_before=np.random.get_state()
        completed_samples=n
        elapsed=time.perf_counter()-started;cpu=time.process_time()-cpu_started
        if n in (100,300,500,1000,maximum):
            existing=where/f'prefix{n}.json'
            if existing.exists():
                previous=read(existing)
                if (previous['prefix_context_sha256']!=array_hash(contexts[:n])
                    or previous['prefix_cost_sha256']!=array_hash(costs[:n])
                    or sha(where/f'prefix{n}.pickle')!=previous['metadata_sha256']):
                    raise RuntimeError('Retained exact prefix differs; refuse overwrite')
                # Preserve the original completed-prefix build timing. The
                # worker's separate attempt receipt charges replay/recovery.
                return
            fields={'sols':np.asarray(sols),'objs':np.asarray(objs),'ctrs':ctrs.copy(),
                    'vertices':names.copy(),'vertex_counts':counts.copy()}
            data=where/f'prefix{n}.pickle';tmp=data.with_suffix(f'.{os.getpid()}.tmp')
            with tmp.open('wb') as f:pickle.dump(fields,f,protocol=5)
            tmp.replace(data)
            prefix={**info,'status':'complete','supplied_training_samples':n,
                'prefix_context_sha256':array_hash(contexts[:n]),'prefix_cost_sha256':array_hash(costs[:n]),
                'build_wall_seconds_cumulative':elapsed,'build_cpu_seconds_cumulative':cpu,
                'unique_anchor_cache_files':len(unique_names),'vertex_counts':counts.copy(),
                'vertex_storage_bytes':sum((where/v).stat().st_size for v in unique_names),
                'metadata_sha256':sha(data),'metadata_bytes':data.stat().st_size,
                'public_source_sha256':d.public.source_sha256,'finished_unix':time.time(),
                'completeness_certified':False}
            prefix['seeded_rng_checkpoints']=n if seeded else 0
            prefix['reused_random_free_unique_anchors']=reused_random_free
            prefix['geometry_identity_sha256']=hashlib.sha256(json.dumps({
                'builder_identity':info['builder_identity'],'public_source_sha256':d.public.source_sha256,
                'context_sha256':prefix['prefix_context_sha256'],'cost_sha256':prefix['prefix_cost_sha256']},sort_keys=True).encode()).hexdigest()
            write(where/f'prefix{n}.json',prefix);write(shared/f'prefix{n}.json',prefix)
        if n%10==0 or n==1:
            write(record,{**info,'samples_completed':n,'unique_anchors':len(unique_names),
                'elapsed_wall_seconds':elapsed,'process_cpu_seconds':cpu,'updated_unix':time.time()})
    # The constructor's binding-normal list is bookkeeping only. Retaining
    # every dense matrix can exceed RAM for SP50. The callback stores the
    # exact maintained arrays losslessly; training restores those same arrays.
    # Solves, bases, public enumeration, projection and duplicates are untouched.
    original_binding=public_highs.binding_normals
    public_highs.binding_normals=lambda p,s:np.empty((0,p.variable_count),dtype=np.float64)
    try:
        d=Retained(model,contexts,costs,public_repo=os.environ['SOLVER_FREE_DFL_REPO'],
                   vertex_tolerance=0.,progress=progress)
    finally:
        public_highs.binding_normals=original_binding
    write(record,{**info,'status':'complete','samples_completed':maximum,
        'elapsed_wall_seconds':time.perf_counter()-started,
        'process_cpu_seconds':time.process_time()-cpu_started,'finished_unix':time.time(),
        'public_source_sha256':d.public.source_sha256})

class LazyVertices:
    def __init__(self,where,names):self.where,self.names=where,names
    def __len__(self):return len(self.names)
    def __getitem__(self,index):
        import numpy as np
        return np.load(self.where/self.names[index],mmap_mode='c',allow_pickle=False)

class LazyNormals:
    def __init__(self,where,names):self.where,self.names=where,names
    def __len__(self):return len(self.names)
    def __getitem__(self,index):
        import numpy as np
        with np.load(self.where/self.names[index],allow_pickle=False) as f:return f['normals']

def get_geometry(job,model,contexts,costs,update,initialization_seed=None):
    import numpy as np
    from physical128_data import array_hash
    from proximal_spo.public_highs import HighsBindingDataset
    t,cpu=time.perf_counter(),time.process_time()
    key=job['required_data_key'];where=directory(key,initialization_seed);n=len(costs)
    meta=read(where/f'prefix{n}.json')
    if initialization_seed is not None and (meta.get('geometry_rng_protocol')!='seeded_numpy_resume_v2' or meta.get('initialization_seed')!=initialization_seed):
        raise RuntimeError('Seeded cache identity differs')
    if meta['prefix_context_sha256']!=array_hash(contexts) or meta['prefix_cost_sha256']!=array_hash(costs):
        raise RuntimeError('Strict supplied training prefix differs from prepared geometry')
    with (where/f'prefix{n}.pickle').open('rb') as f:fields=pickle.load(f)
    d=HighsBindingDataset.__new__(HighsBindingDataset)
    d.model,d.feats,d.costs=model,contexts,costs
    d.sols,d.objs=fields['sols'],fields['objs']
    d.ctrs=LazyNormals(where,fields['ctrs']) if isinstance(fields['ctrs'][0],str) else fields['ctrs']
    d.relaxed_sols,d.relaxed_objs=d.sols,d.objs
    d.adjacent_verts=LazyVertices(where,fields['vertices'])
    d.geometry_cache_receipt={'hit':True,'key':key if initialization_seed is None else key+'__init'+str(initialization_seed)+'__rngv2','supplied_training_samples':n,
        'strict_prefix_context_sha256':meta['prefix_context_sha256'],
        'strict_prefix_cost_sha256':meta['prefix_cost_sha256'],
        'cache_load_wall_seconds':time.perf_counter()-t,'cache_load_cpu_seconds':time.process_time()-cpu,
        'cache_build_wall_seconds_this_call':0.,'cache_build_cpu_seconds_this_call':0.,
        'separately_charged_build_receipt':str(Path(os.environ['LAVA_RUN_ROOT'])/('banks' if initialization_seed is None else 'seeded_banks_v2')/(key if initialization_seed is None else key+'__init'+str(initialization_seed))/f'prefix{n}.json'),
        'geometry_rng_protocol':meta.get('geometry_rng_protocol','legacy_unseeded_v1'),
        'initialization_seed':initialization_seed,'builder_identity':meta.get('builder_identity'),
        'geometry_identity_sha256':meta.get('geometry_identity_sha256'),
        'vertex_counts':fields['vertex_counts'],'basis_enumeration_cap':250,'vertex_tolerance':0.,
        'completeness_certified':False,'lazy_node_local_vertices':True}
    update('preprocessing',geometry_cache_key=key,geometry_cache_hit=True)
    return d

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--key',required=True)
    p.add_argument('--maximum',type=int,default=1000);p.add_argument('--initialization-seed',type=int)
    p.add_argument('--print-original-initialization-seed',action='store_true')
    a=p.parse_args()
    if a.print_original_initialization_seed:
        plan=read(Path(a.root)/'PLAN.json');route,data=a.key.split('__data')
        seeds=next((v for v in plan['cells'][route]['confirmation'] if str(v['data'])==data),plan['cells'][route]['screen'])
        if str(seeds['data'])!=data:raise RuntimeError('Original bank seed identity missing')
        print(seeds['initialization'])
    elif a.initialization_seed is None:
        # Existing worker processes still use this CLI. New constructions must
        # enter the seeded owner path even before those workers are refreshed.
        import runpy
        runpy.run_path(str(Path(a.root)/'ops/prepare_bank.py'),run_name='__main__')
    else:prepare(a.root,a.key,a.maximum,initialization_seed=a.initialization_seed)
