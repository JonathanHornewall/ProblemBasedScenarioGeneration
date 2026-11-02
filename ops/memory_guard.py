"""Finite qualification resource backstop inside its existing Slurm allocation.

This is a fit resource guard, not a chat monitor or a new numerical experiment.
Only process groups created by qualify.py for this exact route can be signaled.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import time
import psutil
from qualify import read,write

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
p.add_argument('--route',required=True);p.add_argument('--maximum-gib',type=float,required=True)
p.add_argument('--minimum-available-gib',type=float,default=2.)
a=p.parse_args();end=min(time.time()+4*3600,read(a.root/'CONTRACT.json')['deadline_unix'])
closed=set()
while time.time()<end:
    terminal=0
    for sample in range(2):
        path=a.root/'qualification'/f'{a.route}__seed17__sample{sample}.json'
        if not path.exists():continue
        v=read(path)
        if v.get('slurm_job_id')!=os.environ['SLURM_JOB_ID']:continue
        if v.get('status') in ('complete','failed','censored'):
            terminal+=1;continue
        pid=v.get('pid')
        if not pid or pid in closed:continue
        try:
            proc=psutil.Process(pid);command=proc.cmdline()
            if (str(a.root/'ops/qualify.py') not in command or a.route not in command
                or os.getpgid(pid)!=pid or abs(proc.create_time()-v['started_unix'])>10):
                raise RuntimeError('Owned qualification process identity differs')
            rss=proc.memory_info().rss;available=psutil.virtual_memory().available
            if rss>a.maximum_gib*2**30 or available<a.minimum_available_gib*2**30:
                cpu=proc.cpu_times()
                write(a.root/'qualification'/f'{a.route}__sample{sample}.memory_guard.json',
                    {'status':'resource_censored','reason':'native_preprocessing_memory_headroom',
                     'pid':pid,'rss_bytes':rss,'available_host_bytes':available,
                     'maximum_gib':a.maximum_gib,'minimum_available_gib':a.minimum_available_gib,
                     'cpu_seconds_before_signal':cpu.user+cpu.system,'observed_unix':time.time(),
                     'claim':'Memory censor; not a proof of scientific infeasibility.'})
                os.killpg(pid,signal.SIGKILL);closed.add(pid)
        except psutil.NoSuchProcess:pass
    if terminal==2:break
    time.sleep(5)
write(a.root/'ops'/('memory_guard_'+a.route+'.json'),{'status':'ended','ended_unix':time.time()})
