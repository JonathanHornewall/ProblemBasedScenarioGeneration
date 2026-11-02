"""Run one maintained fit with run-owned cache reuse and separate load timing."""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*2**20),b''):h.update(b)
    return h.hexdigest()

def install_geometry():
    import final_geometry_cache as cache
    from pilot_common import write
    original=cache.get_geometry
    def get(job,model,contexts,costs,update):
        key=cache.geometry_cache_key(job,contexts,costs)
        target=Path(os.environ['CENTRAL_PATH_GEOMETRY_CACHE']);target.mkdir(parents=True,exist_ok=True)
        reuse=None
        # Existing campaigns are read-only: never open their lock files.
        old=Path('/home/jonathan.hornewall/runs/scaled_data_neural_20261021/lava_geometry_cache')
        with (target/(key+'.copy.lock')).open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if not (target/(key+'.json')).exists() and (old/(key+'.json')).exists():
                meta=json.loads((old/(key+'.json')).read_text());data=old/(key+'.pickle')
                if (meta.get('status')=='complete' and data.exists()
                    and meta.get('geometry',{}).get('basis_enumeration_cap')==250
                    and meta.get('geometry',{}).get('vertex_tolerance')==0.):
                    t=time.perf_counter();before=sha(data)
                    tmp=target/(key+f'.{os.getpid()}.tmp')
                    shutil.copyfile(data,tmp)
                    if sha(tmp)!=before:raise RuntimeError('Existing geometry copy differs')
                    tmp.replace(target/(key+'.pickle'))
                    write(target/(key+'.json'),meta)
                    reuse={'path':str(data),'sha256':before,'bytes':data.stat().st_size,
                        'copy_wall_seconds':time.perf_counter()-t,'prior_build_wall_seconds':meta.get('seconds')}
                    write(target/(key+'.reused.json'),reuse)
        t,cpu=time.perf_counter(),time.process_time()
        result=original(job,model,contexts,costs,update)
        elapsed=time.perf_counter()-t;used_cpu=time.process_time()-cpu
        receipt=result.geometry_cache_receipt
        result.geometry_cache_receipt={**receipt,'bank_identity':key,
            'cache_load_wall_seconds':elapsed if receipt.get('hit') else 0.,
            'cache_load_cpu_seconds':used_cpu if receipt.get('hit') else 0.,
            'cache_build_wall_seconds_this_call':0. if receipt.get('hit') else elapsed,
            'cache_build_cpu_seconds_this_call':0. if receipt.get('hit') else used_cpu,
            'existing_build_reuse':reuse or (json.loads((target/(key+'.reused.json')).read_text())
                                           if (target/(key+'.reused.json')).exists() else None)}
        return result
    cache.get_geometry=get

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output-root',type=Path,required=True)
    args=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or len(os.sched_getaffinity(0))!=1:
        raise RuntimeError('Single-core Slurm fit required')
    job=json.loads(args.manifest.read_text())['jobs'][0]
    from scope_control import runtime_fit_allowed,scope_skip
    scope_root=Path(os.environ.get('LAVA_RUN_ROOT',Path(__file__).resolve().parents[1]))
    if not runtime_fit_allowed(scope_root,job):
        scope_skip(scope_root,'fit',job['id'],'Excluded by the direct human CFL N100 amendment')
        raise SystemExit(76)
    if job.get('lava_completion_revision') or job.get('lava_completion_recovery',{}).get('lazy_exact_bank'):
        from safe_geometry import decision
        admission=decision(Path(os.environ['LAVA_RUN_ROOT']),job)
        if not admission['ready']:
            print('Operational hold: '+admission['reason'],flush=True)
            raise SystemExit(75)
    if job.get('lava_completion_revision') or job.get('lava_completion_recovery',{}).get('lazy_exact_bank'):
        import final_geometry_cache
        from safe_geometry import get_geometry
        final_geometry_cache.get_geometry=get_geometry
        import scaled_data_neural_campaign as campaign
        if not any(p['id']=='linear_n100' for p in campaign.PHASES):
            campaign.PHASES.append({'id':'linear_n100','train':100,'epochs':300,'depth':0,'tune':True})
    else:
        install_geometry()
    import run_central_path_final as maintained
    args.job_index=0;args.smoke=False
    if not runtime_fit_allowed(scope_root,job):
        scope_skip(scope_root,'fit',job['id'],'Scope refreshed immediately before maintained fitting')
        raise SystemExit(76)
    maintained.run(args)
