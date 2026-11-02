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
import shutil
import socket
import time

from qualify import read,sha,write

def directory(key):
    return Path('/tmp/lava_completion_20261102/banks')/key

def vertex_name(anchor):
    return hashlib.sha256(anchor.tobytes()).hexdigest()+'.npy'

def prepare(root,key,maximum=1000):
    import numpy as np
    import highspy
    import torch
    from physical128_data import array_hash
    from simplified_data import load_structure
    from proximal_spo.public_highs import HighsLAVADataset,HighsOptModel,binding_normals
    torch.set_num_threads(1)
    if torch.get_num_interop_threads()!=1:torch.set_num_interop_threads(1)
    root=Path(root);where=directory(key);where.mkdir(parents=True,exist_ok=True)
    record=root/'banks'/key/'STATUS.json'
    old=read(record) if record.exists() else {}
    if old.get('status')=='complete' and (where/f'prefix{maximum}.json').exists():return
    source=Path('/home/jonathan.hornewall/runs/scaled_data_neural_20261021/frozen_data')/key
    meta=read(source/'receipt.json');route,seed=key.split('__data')
    development=seed not in [str(s['data']) for s in read(root/'PLAN.json')['cells'][route]['confirmation']]
    train_source=source.parent/'retuning_development'/key if development else source
    training_file=train_source/f'observations_n{maximum if maximum in (100,500,1000) else 1000}.npz'
    source_receipt=read(train_source/'receipt.json')
    if sha(training_file)!=source_receipt['files_sha256'][training_file.name]:
        raise RuntimeError('Frozen training file changed')
    with np.load(training_file,allow_pickle=False) as f:
        # No validation, test, conditional moments or generator is read here.
        contexts=f['train__contexts'][:maximum].copy()
        costs=f['train__physical_costs'][:maximum].copy()
    if len(costs)!=maximum:raise ValueError('Exact supplied training prefix missing')
    problem=load_structure(source).continuous_relaxation();model=HighsOptModel(problem)
    info={'key':key,'route':route,'data_seed':int(seed),'source_observation_sha256':sha(training_file),
          'source_training_file':str(training_file),'host':socket.gethostname(),
          'scratch':str(where),'slurm_job_id':os.environ['SLURM_JOB_ID'],'status':'running',
          'started_unix':time.time(),'maximum_training_samples':maximum,
          'validation_opened':False,'test_opened':False,'generator_called':False,
          'adapter_sha256':sha(Path(os.environ['LAVA_CODE'])/'src/proximal_spo/public_highs.py'),
          'highs_version':highspy.Highs().version(),'basis_enumeration_cap':250,'vertex_tolerance':0.,
          'reused_completed_prefixes':sorted(int(p.stem[6:]) for p in where.glob('prefix*.json'))}
    write(record,info)
    started,cpu_started=time.perf_counter(),time.process_time()
    sols,objs,ctrs,names,counts=[],[],[],[],[]
    unique_names=set()
    class Retained(HighsLAVADataset):
        def _get_adjacent_vertices(self,solution):
            anchor=np.asarray(solution[:model.num_cost],dtype=np.float64)
            path=where/vertex_name(anchor)
            if path.exists():
                # Rebuild an interrupted dataset in original solver order;
                # already completed unique-anchor enumerations are reused.
                vertices=np.load(path,allow_pickle=False)
                return np.pad(vertices,((0,0),(0,len(solution)-model.num_cost)))
            return super()._get_adjacent_vertices(solution)
    def progress(n,d):
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
        unique_names.add(name);sols.append(sol);objs.append([float(costs[n-1]@sol)])
        ctrs.append(binding_normals(problem,sol));names.append(name);counts.append(len(vertices))
        elapsed=time.perf_counter()-started;cpu=time.process_time()-cpu_started
        if n in (100,300,500,1000,maximum):
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
            write(where/f'prefix{n}.json',prefix);write(root/'banks'/key/f'prefix{n}.json',prefix)
        if n%10==0 or n==1:
            write(record,{**info,'samples_completed':n,'unique_anchors':len(unique_names),
                'elapsed_wall_seconds':elapsed,'process_cpu_seconds':cpu,'updated_unix':time.time()})
    d=Retained(model,contexts,costs,public_repo=os.environ['SOLVER_FREE_DFL_REPO'],
               vertex_tolerance=0.,progress=progress)
    write(record,{**info,'status':'complete','samples_completed':maximum,
        'elapsed_wall_seconds':time.perf_counter()-started,
        'process_cpu_seconds':time.process_time()-cpu_started,'finished_unix':time.time(),
        'public_source_sha256':d.public.source_sha256})

class LazyVertices:
    def __init__(self,where,names):self.where,self.names=where,names
    def __len__(self):return len(self.names)
    def __getitem__(self,index):
        import numpy as np
        return np.load(self.where/self.names[index],mmap_mode='r',allow_pickle=False)

def get_geometry(job,model,contexts,costs,update):
    import numpy as np
    from physical128_data import array_hash
    from proximal_spo.public_highs import HighsBindingDataset
    t,cpu=time.perf_counter(),time.process_time()
    key=job['required_data_key'];where=directory(key);n=len(costs)
    meta=read(where/f'prefix{n}.json')
    if meta['prefix_context_sha256']!=array_hash(contexts) or meta['prefix_cost_sha256']!=array_hash(costs):
        raise RuntimeError('Strict supplied training prefix differs from prepared geometry')
    with (where/f'prefix{n}.pickle').open('rb') as f:fields=pickle.load(f)
    d=HighsBindingDataset.__new__(HighsBindingDataset)
    d.model,d.feats,d.costs=model,contexts,costs
    d.sols,d.objs,d.ctrs=fields['sols'],fields['objs'],fields['ctrs']
    d.relaxed_sols,d.relaxed_objs=d.sols,d.objs
    d.adjacent_verts=LazyVertices(where,fields['vertices'])
    d.geometry_cache_receipt={'hit':True,'key':key,'supplied_training_samples':n,
        'strict_prefix_context_sha256':meta['prefix_context_sha256'],
        'strict_prefix_cost_sha256':meta['prefix_cost_sha256'],
        'cache_load_wall_seconds':time.perf_counter()-t,'cache_load_cpu_seconds':time.process_time()-cpu,
        'cache_build_wall_seconds_this_call':0.,'cache_build_cpu_seconds_this_call':0.,
        'separately_charged_build_receipt':str(Path(os.environ['LAVA_RUN_ROOT'])/'banks'/key/f'prefix{n}.json'),
        'vertex_counts':fields['vertex_counts'],'basis_enumeration_cap':250,'vertex_tolerance':0.,
        'completeness_certified':False,'lazy_node_local_vertices':True}
    update('preprocessing',geometry_cache_key=key,geometry_cache_hit=True)
    return d

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--key',required=True)
    p.add_argument('--maximum',type=int,default=1000)
    a=p.parse_args();prepare(a.root,a.key,a.maximum)
