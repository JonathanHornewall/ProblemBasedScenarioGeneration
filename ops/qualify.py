"""Two original training observations per route, isolated with wait4 accounting.

No validation/test array is read. The original objective sequence is replayed
before the target sample so HiGHS sees the same warm simplex basis as a bank.
The released enumeration and exact-float duplicate removal remain unchanged.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import signal
import socket
import subprocess
import sys
import time

def read(path):
    return json.loads(Path(path).read_text())

def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temp.replace(path)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 2**20), b''):
            h.update(block)
    return h.hexdigest()

def child(args):
    import numpy as np
    import highspy
    import torch
    from physical128_data import array_hash
    from simplified_data import load_structure
    from proximal_spo.public_highs import HighsLAVADataset, HighsOptModel

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    source = Path(args.bank)
    meta = read(source/'receipt.json')
    for name in ('observations.npz', 'structure.npz', 'structure.json'):
        if sha(source/name) != meta['files_sha256'][name]:
            raise ValueError('Original supplied input changed: '+name)
    with np.load(source/'observations.npz', allow_pickle=False) as f:
        # Deliberately do not materialize any validation arrays in this NPZ.
        x = f['train__contexts'][:args.sample+1].copy()
        costs = f['train__physical_costs'][:args.sample+1].copy()
    problem = load_structure(source).continuous_relaxation()
    model = HighsOptModel(problem)
    prefix_available = {}
    for i in range(args.sample):
        cache = Path(args.scratch)/f'sample{i}.pickle'
        if cache.exists():
            with cache.open('rb') as f:
                prefix_available[i] = pickle.load(f)
    state = {'phase':'preprocessing', 'sample':args.sample,
        'route':args.route, 'data_seed':args.seed, 'source_bank':str(source),
        'source_observation_sha256':meta['files_sha256']['observations.npz'],
        'source_receipt_sha256':sha(source/'receipt.json'),
        'cost_sha256':array_hash(costs[-1]), 'context_sha256':array_hash(x[-1]),
        'highs_version':highspy.Highs().version(), 'threads':1,
        'basis_enumeration_cap':250, 'vertex_tolerance':0.,
        'duplicate_behavior':'released exact float tuple equality',
        'physical_lp_relaxation':True, 'validation_opened':False, 'test_opened':False,
        'solver_objective_prefix_replayed':args.sample,
        'earlier_geometry_unavailable':[i for i in range(args.sample) if i not in prefix_available],
        'host':socket.gethostname(), 'slurm_job_id':os.environ['SLURM_JOB_ID'],
        'adapter_sha256':sha(Path(os.environ['LAVA_CODE'])/'src/proximal_spo/public_highs.py')}
    write(args.progress,state)
    class Isolated(HighsLAVADataset):
        calls = 0
        def _get_adjacent_vertices(self, solution):
            index = self.calls
            self.calls += 1
            if index < args.sample:
                previous = prefix_available.get(index)
                # Public code projects first, then removes exact duplicates.
                # Supply retained geometry with zero slack padding solely for
                # prefix-cache reconstruction; target enumeration is untouched.
                if previous is not None:
                    vertices = previous['adjacent_verts']
                    return np.pad(vertices, ((0,0),(0,len(solution)-model.num_cost)))
                return [solution]
            return super()._get_adjacent_vertices(solution)
    def progress(n, dataset):
        if n <= args.sample and n-1 not in prefix_available:
            dataset.adjacent_vertices_cache.clear()
    t, cpu = time.perf_counter(), time.process_time()
    d = Isolated(model,x,costs,public_repo=os.environ['SOLVER_FREE_DFL_REPO'],
                 vertex_tolerance=0.,progress=progress)
    cache = Path(args.scratch)/f'sample{args.sample}.pickle'
    fields = {'sols':d.sols[-1], 'objs':d.objs[-1], 'ctrs':d.ctrs[-1],
              'adjacent_verts':d.adjacent_verts[-1]}
    with cache.open('wb') as f:
        pickle.dump(fields,f,protocol=5)
    state.update(phase='complete',status='complete',wall_seconds=time.perf_counter()-t,
        process_cpu_seconds=time.process_time()-cpu,
        raw_neighbour_rows=len(d.adjacent_verts[-1]),
        float64_vertex_bytes=d.adjacent_verts[-1].nbytes,
        tnp_retries=d.adjacency_tnp_retry_count,
        basis_completions=d.basis_completion_count,
        anchor_reselections=d.anchor_reselection_count,
        public_source_sha256=d.public.source_sha256,
        sample_cache={'path':str(cache),'bytes':cache.stat().st_size,'sha256':sha(cache)},
        completeness_certified=False)
    write(args.progress,state)

def parent(args):
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Numerical work must run in Slurm')
    if len(os.sched_getaffinity(0)) != 1:
        raise RuntimeError('Qualification must be bound to one CPU')
    root = Path(args.root)
    if time.time() >= read(root/'CONTRACT.json')['deadline_unix']:
        return
    scratch = Path('/tmp')/'lava_completion_20261102'/('qualification_'+args.route)
    scratch.mkdir(parents=True, exist_ok=True)
    write(root/'ops'/('scratch_'+args.route+'.json'),
          {'host':socket.gethostname(),'path':str(scratch),'slurm_job_id':os.environ['SLURM_JOB_ID']})
    for sample in range(2):
        out = root/'qualification'/f'{args.route}__seed{args.seed}__sample{sample}.json'
        if out.exists() and read(out).get('status') == 'complete':
            continue
        if out.exists():
            previous = read(out)
            archive = root/'qualification'/'attempts'/(out.stem+'__job'+previous['slurm_job_id']+'.json')
            write(archive,previous)
        progress = root/'qualification'/f'{args.route}__sample{sample}.progress.json'
        logpath = root/'logs'/f'{args.route}__sample{sample}.log'
        logpath.parent.mkdir(exist_ok=True)
        cmd=[sys.executable,str(Path(__file__).resolve()),'--child','--route',args.route,
             '--bank',args.bank,'--seed',str(args.seed),'--sample',str(sample),
             '--scratch',str(scratch),'--progress',str(progress)]
        opened=time.monotonic(); reason=None
        with logpath.open('ab') as log:
            p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            write(out,{'status':'running','pid':p.pid,'route':args.route,'data_seed':args.seed,
                'sample':sample,'started_unix':time.time(),'timeout_seconds':args.timeout,
                'host':socket.gethostname(),'slurm_job_id':os.environ['SLURM_JOB_ID']})
            while True:
                pid,status,usage=os.wait4(p.pid,os.WNOHANG)
                if pid:
                    p.returncode=os.waitstatus_to_exitcode(status)
                    break
                if time.monotonic()-opened >= args.timeout or time.time()>=read(root/'CONTRACT.json')['deadline_unix']:
                    reason='per_observation_wall_timeout_censored'
                    os.killpg(p.pid,signal.SIGKILL)
                    _,status,usage=os.wait4(p.pid,0)
                    p.returncode=os.waitstatus_to_exitcode(status)
                    break
                time.sleep(1)
        record=read(progress) if progress.exists() else {}
        successful=p.returncode==0 and record.get('status')=='complete'
        record.update(status='complete' if successful else 'censored' if reason else 'failed',
            outcome_reason=reason or ('complete' if successful else 'child_exit'),
            returncode=p.returncode, elapsed_wall_seconds=time.monotonic()-opened,
            process_cpu_seconds_with_imports=usage.ru_utime+usage.ru_stime,
            peak_rss_bytes=int(usage.ru_maxrss)*1024,
            finished_unix=time.time(),route=args.route,data_seed=args.seed,sample=sample,
            log=str(logpath),owned_child_process_group=p.pid,timeout_seconds=args.timeout,
            timeout_is_not_infeasibility=True)
        write(out,record)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root');p.add_argument('--route',required=True)
    p.add_argument('--bank',required=True);p.add_argument('--seed',type=int,default=17)
    p.add_argument('--timeout',type=float,default=7200)
    p.add_argument('--child',action='store_true');p.add_argument('--sample',type=int)
    p.add_argument('--scratch');p.add_argument('--progress')
    a=p.parse_args()
    child(a) if a.child else parent(a)
