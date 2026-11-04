"""Adopt the current dispatcher at an unstarted fit's admission boundary."""
import fcntl,os,pathlib,signal,socket,sys,time
import psutil
from qualify import read,write

def handoff_if_stale(root,job,*,reservation_path=None):
    if job['route']=='dys_shortest_path_25x25':expected,previous=3,6
    elif job['route']=='dys_cfl_50x5' and job['config']['sizes']['train']==100:expected,previous=6,12
    else:return False
    root=pathlib.Path(root)
    reservation=pathlib.Path(reservation_path or '/tmp/lava_completion_20261102/reservations.json')
    if not reservation.exists():return False
    # The current program is still before classification, Torch and fitting.
    parent=psutil.Process(os.getppid());cmd=parent.cmdline();allocation=os.environ['SLURM_JOB_ID']
    if len(cmd)<2 or cmd[1]!=str(root/'ops/completion.py') or '--worker' not in cmd:return False
    index=cmd[cmd.index('--worker')+1];token=socket.gethostname()+'_'+index+'_'+allocation
    entry=read(reservation).get(token)
    if not entry or entry.get('worker_pid')!=parent.pid or entry.get('id')!=job['id']:return False
    if entry['gib']==expected:return False
    if entry['gib']!=previous:raise RuntimeError('Unexpected fit admission reservation')
    if parent.environ().get('SLURM_JOB_ID')!=allocation:raise RuntimeError('Parent allocation differs')
    if parent.cpu_affinity()!=sorted(os.sched_getaffinity(0)) or len(os.sched_getaffinity(0))!=1:
        raise RuntimeError('Boundary handoff must retain the same single core')
    if os.getpgid(0)!=os.getpid():raise RuntimeError('Unstarted admission must own its process group')
    birth=parent.create_time();paused=False
    path=root/'ops/worker_boundary_handoffs'/(token+'__'+str(time.time_ns())+'.json')
    try:
        parent.send_signal(signal.SIGSTOP);paused=True
        limit=time.monotonic()+2
        while parent.status()!=psutil.STATUS_STOPPED and time.monotonic()<limit:time.sleep(.01)
        if parent.status()!=psutil.STATUS_STOPPED:raise RuntimeError('Parent did not stop at admission')
        children=[p.pid for p in parent.children(recursive=True) if p.status()!=psutil.STATUS_ZOMBIE]
        if children!=[os.getpid()] or parent.create_time()!=birth:
            raise RuntimeError('A different active child exists; preserve the parent')
        with reservation.with_suffix('.lock').open('a') as guard:
            fcntl.flock(guard,fcntl.LOCK_EX);held=read(reservation)
            if held.get(token)!=entry:raise RuntimeError('Admission reservation changed')
            record={'observed_unix':time.time(),'slurm_job_id':allocation,'host':socket.gethostname(),
                    'old_worker_pid':parent.pid,'old_worker_birth_unix':birth,'new_worker_pid':os.getpid(),
                    'worker_index':index,'core':next(iter(os.sched_getaffinity(0))),
                    'unstarted_job_id':job['id'],'old_reservation_GiB':previous,'new_reservation_GiB':expected,
                    'scientific_fit_started':False,'healthy_active_fits_signalled':0,
                    'new_CPU_allocations':0,'state':'admission_handoff_prepared'}
            write(path,record)
            # The parent's preceding scientific child has already finished.
            # Its sole current child is this unstarted admission. Retain this
            # PID/core/cgroup and replace only the obsolete metadata worker.
            parent.kill();paused=False
            held.pop(token);write(reservation,held)
        record['state']='exec_current_worker';write(path,record)
        os.execv(sys.executable,[sys.executable,str(root/'ops/completion.py'),'--root',str(root),'--worker',index])
    finally:
        if paused:
            try:parent.send_signal(signal.SIGCONT)
            except psutil.NoSuchProcess:pass

