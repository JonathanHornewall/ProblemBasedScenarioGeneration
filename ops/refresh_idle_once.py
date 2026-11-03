"""Replace exact idle workers once within an existing allocation.

Stop the worker briefly before checking descendants and reservations, so an
idle-to-active race cannot signal a scientific child. Numerical children and
their original resource guards remain untouched. No allocation is added.
"""
import json,os,pathlib,signal,socket,subprocess,sys,time
import psutil
from qualify import read,write

root=pathlib.Path(os.environ['LAVA_RUN_ROOT']);host=socket.gethostname();job=os.environ['SLURM_JOB_ID']
label=os.environ.get('LAVA_IDLE_REFRESH_LABEL','CFL50_6GiB')
receipt=root/'ops'/('IDLE_REFRESH_ONCE_'+job+'_'+label+'.json')
if receipt.exists():raise RuntimeError('Refresh already attempted; reconcile its receipt')
deadline=read(root/'CONTRACT.json')['scientific_cutoff_unix']
if time.time()>=deadline:raise RuntimeError('Scientific cutoff passed')
record={'started_unix':time.time(),'host':host,'slurm_job_id':job,'refreshed':[],'preserved':[],
        'new_CPU_allocations':0,'reason':os.environ.get('LAVA_IDLE_REFRESH_REASON','Measured CFL50 N100 fit reservation12 to6GiB')}
write(receipt,record)
children=[]
for file in sorted((root/'workers').glob(host+'_*_'+job+'.json')):
    v=read(file)
    if v.get('state')!='waiting_for_dependencies_or_memory' or time.time()-v.get('updated_unix',0)>30:continue
    proc=None;paused=False
    try:
        proc=psutil.Process(v['pid']);cmd=proc.cmdline()
        if len(cmd)<2 or cmd[1]!=str(root/'ops/completion.py') or '--worker' not in cmd:continue
        prior_env=proc.environ()
        if prior_env.get('SLURM_JOB_ID')!=job:raise RuntimeError('Worker allocation differs')
        cores=proc.cpu_affinity();birth=proc.create_time()
        if len(cores)!=1:raise RuntimeError('Single bound core required')
        proc.send_signal(signal.SIGSTOP);paused=True
        stop_limit=time.monotonic()+2
        while proc.status()!=psutil.STATUS_STOPPED and time.monotonic()<stop_limit:time.sleep(.01)
        if proc.status()!=psutil.STATUS_STOPPED:raise RuntimeError('Idle worker did not stop')
        latest=read(file)
        reservation=pathlib.Path('/tmp/lava_completion_20261102/reservations.json')
        held=read(reservation) if reservation.exists() else {}
        active_children=[p.pid for p in proc.children(recursive=True) if p.status()!=psutil.STATUS_ZOMBIE]
        reserved=any(z['worker_pid']==proc.pid for z in held.values())
        if latest.get('state')!='waiting_for_dependencies_or_memory' or latest.get('pid')!=proc.pid or active_children or reserved:
            record['preserved'].append({'pid':proc.pid,'active_children':active_children,'reserved':reserved})
            proc.send_signal(signal.SIGCONT);paused=False;continue
        if proc.create_time()!=birth:raise RuntimeError('Process identity changed')
        index=cmd[cmd.index('--worker')+1]
        # No scientific process exists or can be created by this stopped idle
        # worker. Killing only that process releases its claims atomically.
        proc.kill();paused=False
        child_env=os.environ.copy()
        for key in ('LAVA_NODE_MEMORY_GIB','LAVA_POOL_MEMORY_GIB'):
            if key in prior_env:child_env[key]=prior_env[key]
            else:child_env.pop(key,None)
        child=subprocess.Popen(['taskset','-c',str(cores[0]),sys.executable,str(root/'ops/completion.py'),
                                '--root',str(root),'--worker',index],start_new_session=True,env=child_env)
        children.append(child)
        record['refreshed'].append({'index':index,'old_pid':proc.pid,'old_birth_unix':birth,
                                    'new_pid':child.pid,'core':cores[0],
                                    'preserved_pool_memory_GiB':prior_env.get('LAVA_POOL_MEMORY_GIB'),
                                    'preserved_node_memory_GiB':prior_env.get('LAVA_NODE_MEMORY_GIB')})
        write(receipt,record)
    except psutil.NoSuchProcess:pass
    finally:
        if paused and proc is not None:
            try:proc.send_signal(signal.SIGCONT)
            except psutil.NoSuchProcess:pass
record['finished_unix']=time.time();write(receipt,record);print(json.dumps(record),flush=True)
# Keep these workers in their original finite Slurm step; they retain their
# own cutoff and per-fit guards. This is a dispatcher process, not chat polling.
while time.time()<deadline and any(c.poll() is None for c in children):time.sleep(5)
for child in children:
    if child.poll() is None:child.terminate()
