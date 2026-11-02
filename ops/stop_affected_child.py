"""End only the identified unseeded CFL50 random-search child, preserving clocks."""
import json,os,pathlib,signal,time
import psutil
from qualify import read,write,sha
root=pathlib.Path('/home/jonathan.hornewall/runs/lava_completion_20261102')
pid=3120354;key='dys_cfl_50x5__data151101'
record={'observed_unix':time.time(),'pid':pid,'key':key,'reason':'Known affected unseeded public random-transition preprocessing attempt; Main-directed correctness repair.'}
try:proc=psutil.Process(pid)
except psutil.NoSuchProcess:
 record['action']='already_exited'
else:
 cmd=proc.cmdline()
 if (str(root/'ops/bank_geometry.py') not in cmd or key not in cmd
     or proc.environ().get('SLURM_JOB_ID')!='16223' or os.getpgid(pid)!=pid):
  raise RuntimeError('Exact affected child identity mismatch')
 record.update({'cmdline':cmd,'birth_unix':proc.create_time(),'cpu_seconds':sum(proc.cpu_times()[:2]),
  'rss_bytes':proc.memory_info().rss,'bank_status':read(root/'banks'/key/'STATUS.json')})
 for n in (100,):
  meta=read(root/'banks'/key/f'prefix{n}.json')
  data=pathlib.Path('/tmp/lava_completion_20261102/banks')/key/f'prefix{n}.pickle'
  if sha(data)!=meta['metadata_sha256']:raise RuntimeError('Completed prefix changed before preservation')
  record[f'preserved_prefix{n}']=meta
 write(root/'recovery/affected_attempts'/f'{key}__pid{pid}.json',record)
 os.killpg(pid,signal.SIGKILL);record['action']='signalled_exact_process_group_SIGKILL'
write(root/'recovery/affected_attempts'/f'{key}__pid{pid}.json',record)
print(json.dumps(record))
