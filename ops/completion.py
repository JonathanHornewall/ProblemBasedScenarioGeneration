"""Finite, validation-only completion DAG and node-local worker pool."""
from __future__ import annotations
import argparse
import collections
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

from qualify import read,write,sha
_RECORDS={}

PHASES=[{'id':'linear_n100','train':100,'depth':0,'epochs':300,'tune':True}]+[
    {'id':f'mlp_depth{d}','train':100,'depth':d,'epochs':300,'tune':True} for d in (1,2,3)]+[
    {'id':f'linear_retune_n{n}','train':n,'validation':v,'depth':0,'epochs':300,'tune':True}
    for n,v in ((300,96),(500,160),(1000,320))]+[
    {'id':f'nonlinear_retune_n{n}','train':n,'validation':v,'depth':None,'epochs':300,
     'tune':True,'frozen_nonlinear_depth':True} for n,v in ((300,96),(500,160),(1000,320))]

def make_job(root,plan,route,seeds,phase,choices,stage):
    import scaled_data_neural_campaign as old
    job=old.make_job(plan['templates']['lava'],'lava',route,seeds,choices,phase,stage=stage)
    job['lava_completion_revision']='20261102_v1'
    job['config']['training']['maximum_optimization_seconds']=None
    job['resource_limits']={'preprocessing_cap_seconds':1800,'job_wall_cap_seconds':43200}
    job.pop('id',None);job['config'].pop('name',None)
    jid='lava_complete__'+hashlib.sha256(json.dumps(job,sort_keys=True).encode()).hexdigest()[:32]
    job['id']=jid;job['config']['name']=jid
    affected=root/'ops/AFFECTED_JOB_IDS.json'
    if affected.exists() and jid in read(affected):
        job['replaces_job_id']=jid;job['lava_geometry_repair']='seeded_numpy_resume_v2'
        job.pop('id');job['config'].pop('name')
        jid='lava_complete__'+hashlib.sha256(json.dumps(job,sort_keys=True).encode()).hexdigest()[:32]
        job['id']=jid;job['config']['name']=jid
    return job

def require(root,job):
    path=root/'jobs'/(job['id']+'.json')
    if not path.exists():write(path,{'protocol':{'lava_completion_revision':'20261102_v1'},'jobs':[job]})
    result=root/'results/receipts'/(job['id']+'.json')
    if job['id'] not in _RECORDS and result.exists():_RECORDS[job['id']]=read(result)
    return _RECORDS.get(job['id'])

def advance(root):
    from pilot_common import SMALLEST,family
    import simplified_campaign as linear
    import frozen_nonlinear_retuning as nonlinear
    root=Path(root);plan=read(root/'PLAN.json');admission=read(root/'ADMISSION.json')
    routes=admission['routes'];ready=[];expected=[];selections={};pending=[]
    for fam in ('sp','cfl'):
        smallest=SMALLEST[fam]
        for base_phase in PHASES:
            phase=copy.deepcopy(base_phase)
            if phase['depth'] is None:
                phase['runtime_source_commit']=read(root/'FROZEN_NONLINEAR_RETUNING_AMENDMENT_20261022.json')['runtime_source_commit']
                phase=nonlinear.resolve_phase(root,phase,'lava',fam,freeze=True)
                if phase is None:
                    pending.append(fam+'__'+base_phase['id']+'__depth_dependency');continue
            jobs=[make_job(root,plan,smallest,plan['cells'][smallest]['screen'],phase,q,'screen') for q in linear.grid('lava')]
            records=[require(root,j) for j in jobs];expected.extend(j['id'] for j in jobs)
            for j,r in zip(jobs,records):
                if r is None:ready.append({'kind':'fit','id':j['id'],'key':j['required_data_key'],
                    'n':j['config']['sizes']['train'],'route':j['route'],'stage':'screen'})
            selection_path=root/'selections'/phase['id']/('lava__'+fam+'.json')
            if selection_path.exists():selection=read(selection_path)
            elif all(r is not None for r in records):
                eligible=[];table=[]
                for i,(j,r) in enumerate(zip(jobs,records)):
                    score=linear.score(r)
                    table.append({'id':j['id'],'choices':j['choices'],'status':r['status'],
                        'epochs_completed':r.get('summary',{}).get('epochs_completed'),'validation_score':score})
                    if score is not None:eligible.append((score,i,j))
                if not eligible:
                    pending.append(fam+'__'+phase['id']+'__no_successful_screen');continue
                score,_,template=min(eligible,key=lambda v:(v[0],v[1]))
                selection={'method':'lava','family':fam,'phase':phase['id'],'score':score,
                    'selected_template':template,'screen_fits':16,'validation_table':table,
                    'test_scores_read':False,'tie_rule':'validation score then original Cartesian grid order',
                    'final_checkpoint_rule':'each confirmation selects on its own supplied observed validation'}
                write(selection_path,selection)
            else:continue
            selections[fam+'__'+phase['id']]=selection
            for route in [r for r in plan['routes'] if family(r)==fam]:
                for seed in plan['cells'][route]['confirmation']:
                    j=make_job(root,plan,route,seed,phase,selection['selected_template']['choices'],'confirmation')
                    r=require(root,j);expected.append(j['id'])
                    if r is None and route in routes:ready.append({'kind':'fit','id':j['id'],
                        'key':j['required_data_key'],'n':j['config']['sizes']['train'],'route':route,'stage':'confirmation'})
    keys=sorted({r+'__data'+str(s['data']) for r in routes for s in plan['cells'][r]['confirmation'][:admission.get('prepare_seed_limit_by_route',{}).get(r,10)]}|
        {r+'__data'+str(plan['cells'][r]['screen']['data']) for r in ('dys_shortest_path_10x10','dys_cfl_50x5')})
    prep=[]
    development_keys={r+'__data'+str(plan['cells'][r]['screen']['data']) for r in ('dys_shortest_path_10x10','dys_cfl_50x5')}
    keys.sort(key=lambda key:(key not in development_keys,key))
    for key in keys:
        status=root/'banks'/key/'STATUS.json'
        v=read(status) if status.exists() else {}
        maximum=admission.get('prepare_maximum_by_route',{}).get(key.split('__data')[0],1000)
        # The owner must check its actual node-local prefix even when the
        # shared status says complete. A missing local cache stays dispatchable.
        prep.append({'kind':'prepare','id':'bank__'+key,'key':key,'n':maximum,
            'route':key.split('__data')[0],'stage':'upstream_preprocessing',
            'recorded_owner_host':v.get('host'),'shared_status':v.get('status')})
    # Ready training comes first; missing geometry owners are a separate queue,
    # so other bank/phase work fills all usable memory and CPU slots.
    write(root/'READY.json',{'updated_unix':time.time(),'fits':ready,'preparation':prep,
        'selected_groups':len(selections),'expected_created_jobs':len(set(expected)),
        'expected_complete_scope':{'screens':320,'confirmations':1000,'kp_recoveries':6},
        'admitted_routes':routes,'pending_dependencies':pending,
        'all_routes':sorted(r for r in plan['routes'] if family(r) in ('sp','cfl'))})
    if len(set(expected))==1320 and all(jid in _RECORDS and linear.successful(_RECORDS[jid]) for jid in set(expected)):
        write(root/'SCIENCE_COMPLETE.json',{'finished_unix':time.time(),'screens':320,'confirmations':1000,
            'note':'KP recovery scope is accounted separately before terminal closeout'})

def memory(item):
    if item['kind']=='prepare':return 32. if item['route']=='dys_cfl_50x5' else 48. if 'cfl' in item['route'] else 96. if '50x50' in item['route'] else 24. if '25x25' in item['route'] else 6.
    return 24. if '50x50' in item['route'] else 12. if 'cfl' in item['route'] else 6.

def work(root,index):
    root=Path(root)
    from bank_geometry import directory
    token=socket.gethostname()+'_'+str(index)+'_'+os.environ['SLURM_JOB_ID']
    deadline=read(root/'CONTRACT.json')['scientific_cutoff_unix']
    plan=read(root/'PLAN.json')
    def available(key,n):
        route,data=key.split('__data')
        seeds=next((v for v in plan['cells'][route]['confirmation'] if str(v['data'])==data),plan['cells'][route]['screen'])
        return (directory(key)/f'prefix{n}.json').exists() or (directory(key,seeds['initialization'])/f'prefix{n}.json').exists()
    development_keys={r+'__data'+str(plan['cells'][r]['screen']['data']) for r in ('dys_shortest_path_10x10','dys_cfl_50x5')}
    heartbeat=root/'workers'/(token+'.json');attempts=root/'attempts';attempts.mkdir(exist_ok=True)
    local=Path('/tmp/lava_completion_20261102');local.mkdir(exist_ok=True)
    reservation=local/'reservations.json';guard=local/'reservations.lock'
    stopped=False
    def stop(*_):
        nonlocal stopped
        stopped=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    while not stopped and time.time()<deadline and not (root/'ops/MONITOR_CLOSED.json').exists() and not (root/'SCIENCE_COMPLETE.json').exists():
        queue=read(root/'READY.json') if (root/'READY.json').exists() else {'fits':[],'preparation':[]}
        chosen=None;claim=None
        # A shared completion marker does not promise that the owner's local
        # prefix still exists. Derive missing preparation from ready fits too,
        # including when an older coordinator writes a coarse READY snapshot.
        queued_keys={item['key'] for item in queue['preparation']}
        for item in queue['fits']:
            state=root/'banks'/item['key']/'STATUS.json'
            owner=read(state).get('host') if state.exists() else None
            if (owner in (None,socket.gethostname()) and item['key'] not in queued_keys
                and not available(item['key'],item['n'])):
                maximum=read(root/'ADMISSION.json').get('prepare_maximum_by_route',{}).get(item['route'],1000)
                queue['preparation'].append({'kind':'prepare','id':'bank__'+item['key'],
                    'key':item['key'],'n':maximum,'route':item['route'],'stage':'upstream_preprocessing'})
                queued_keys.add(item['key'])
        preparation=sorted(queue['preparation'],key=lambda item:(item['key'] not in development_keys,item['key']))
        for item in queue['fits']+preparation:
            bank_status=root/'banks'/item['key']/'STATUS.json'
            if bank_status.exists():
                owner=read(bank_status).get('host')
                if owner and owner!=socket.gethostname():continue
            if item['kind']=='fit' and not available(item['key'],item['n']):continue
            failure=root/'failures'/(item['id']+'.json')
            if failure.exists():
                admission=read(root/'ADMISSION.json')
                approved=admission.get('bank_recovery_failure_sha256',{}).get(item['key']) or admission.get('resource_recovery_failure_sha256',{}).get(item['key'])
                pointer=read(failure).get('reason')=='seeded_construction_owner'
                if item['kind']!='prepare' or not pointer and (not approved or sha(failure)!=approved):continue
            claim_path=root/'claims'/(item['id']+'.lock');claim_path.parent.mkdir(exist_ok=True)
            candidate=claim_path.open('a')
            try:fcntl.flock(candidate,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:candidate.close();continue
            if item['kind']=='fit' and (root/'results/receipts'/(item['id']+'.json')).exists():candidate.close();continue
            if item['kind']=='prepare' and available(item['key'],item['n']):candidate.close();continue
            with guard.open('a') as g:
                fcntl.flock(g,fcntl.LOCK_EX);held=read(reservation) if reservation.exists() else {}
                # Reap only dead local worker reservations, never other jobs.
                for key,v in list(held.items()):
                    try:os.kill(v['worker_pid'],0)
                    except ProcessLookupError:held.pop(key)
                if sum(v['gib'] for v in held.values())+memory(item)>float(os.environ['LAVA_NODE_MEMORY_GIB']):
                    candidate.close();continue
                own=sum(v['gib'] for key,v in held.items() if key.endswith('_'+os.environ['SLURM_JOB_ID']))
                if own+memory(item)>float(os.environ.get('LAVA_POOL_MEMORY_GIB',os.environ['LAVA_NODE_MEMORY_GIB'])):
                    candidate.close();continue
                held[token]={'gib':memory(item),'worker_pid':os.getpid(),'id':item['id']};write(reservation,held)
            chosen=item;claim=candidate;break
        if chosen is None:
            write(heartbeat,{'state':'waiting_for_dependencies_or_memory','updated_unix':time.time(),
                'slurm_job_id':os.environ['SLURM_JOB_ID'],'pid':os.getpid()})
            time.sleep(10);continue
        item=chosen
        path=root/'logs'/(item['id']+'.log');path.parent.mkdir(exist_ok=True)
        if item['kind']=='fit':cmd=[sys.executable,str(root/'ops/run_fit.py'),'--manifest',str(root/'jobs'/(item['id']+'.json')),'--output-root',str(root/'results')]
        else:cmd=[sys.executable,str(root/'ops/prepare_bank.py'),'--root',str(root),'--key',item['key'],'--maximum',str(item['n'])]
        opened=time.monotonic();cap=min(43200 if item['kind']=='fit' else 86400,deadline-time.time())
        reason=None;peak_sampled=0;cpu_sampled=0.
        import psutil
        with path.open('ab') as log:
            child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            while True:
                pid,status,usage=os.wait4(child.pid,os.WNOHANG)
                if pid:
                    child.returncode=os.waitstatus_to_exitcode(status);break
                try:
                    proc=psutil.Process(child.pid);rss=proc.memory_info().rss
                    peak_sampled=max(peak_sampled,rss);t=proc.cpu_times();cpu_sampled=t.user+t.system
                except psutil.Error:pass
                if stopped or time.monotonic()-opened>=cap or peak_sampled>memory(item)*2**30:
                    reason='allocation_stop' if stopped else 'finite_wall_budget' if time.monotonic()-opened>=cap else 'memory_reservation_exceeded'
                    os.killpg(child.pid,signal.SIGKILL);_,status,usage=os.wait4(child.pid,0)
                    child.returncode=os.waitstatus_to_exitcode(status);break
                write(heartbeat,{'state':'active','item':item,'pid':child.pid,'worker_pid':os.getpid(),
                    'updated_unix':time.time(),'rss_bytes':peak_sampled,'actual_process_cpu_seconds':cpu_sampled,
                    'slurm_job_id':os.environ['SLURM_JOB_ID']})
                time.sleep(5)
        outcome={**item,'returncode':child.returncode,'reason':reason,'elapsed_wall_seconds':time.monotonic()-opened,
            'process_cpu_seconds':usage.ru_utime+usage.ru_stime,'peak_rss_bytes':usage.ru_maxrss*1024,
            'finished_unix':time.time(),'host':socket.gethostname(),'slurm_job_id':os.environ['SLURM_JOB_ID']}
        write(attempts/(item['id']+'__'+str(time.time_ns())+'.json'),outcome)
        if child.returncode:
            write(root/'failures'/(item['id']+'.json'),outcome)
        elif item['kind']=='prepare' and (root/'failures'/(item['id']+'.json')).exists():
            write(root/'recovery/resolved_banks'/(item['key']+'.json'),{
                'resolved_unix':time.time(),'successful_attempt':outcome,
                'preserved_failed_attempt_sha256':sha(root/'failures'/(item['id']+'.json'))})
        with guard.open('a') as g:
            fcntl.flock(g,fcntl.LOCK_EX);held=read(reservation);held.pop(token,None);write(reservation,held)
        claim.close()
    write(heartbeat,{'state':'ended','updated_unix':time.time(),'slurm_job_id':os.environ['SLURM_JOB_ID']})

def pool(root,workers):
    root=Path(root)
    if not os.environ.get('SLURM_JOB_ID') or workers>int(os.environ.get('SLURM_CPUS_PER_TASK','0')):
        raise RuntimeError('Bounded Slurm allocation required')
    cores=sorted(os.sched_getaffinity(0))[:workers]
    if len(cores)!=workers:raise RuntimeError('Insufficient private CPU affinity')
    children=[]
    for i,core in enumerate(cores):
        children.append(subprocess.Popen(['taskset','-c',str(core),sys.executable,str(Path(__file__).resolve()),
            '--root',str(root),'--worker',str(i)]))
    stopped=False
    def stop(*_):
        nonlocal stopped
        stopped=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    deadline=read(root/'CONTRACT.json')['scientific_cutoff_unix']
    while not stopped and time.time()<deadline and any(c.poll() is None for c in children):
        with (root/'ops/coordinator.lock').open('a') as control:
            try:fcntl.flock(control,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:pass
            else:advance(root)
        time.sleep(10)
    for child in children:
        if child.poll() is None:child.terminate()
    for child in children:child.wait(timeout=30)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--worker',type=int);p.add_argument('--pool',type=int);p.add_argument('--advance',action='store_true')
    a=p.parse_args()
    if a.advance:advance(a.root)
    elif a.worker is not None:work(a.root,a.worker)
    else:pool(a.root,a.pool)
