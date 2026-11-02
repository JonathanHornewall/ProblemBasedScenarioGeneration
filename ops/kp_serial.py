"""One finite, serial repair attempt for the two remaining seed151101 fits."""
import json,os,pathlib,signal,subprocess,sys,time
import psutil
from qualify import read,write

root=pathlib.Path(os.environ['LAVA_RUN_ROOT'])
if not os.environ.get('SLURM_JOB_ID') or len(os.sched_getaffinity(0))!=1:
 raise RuntimeError('One Slurm-bound CPU required')
deadline=read(root/'CONTRACT.json')['scientific_cutoff_unix']
stopped=False
def stop(*_):
 global stopped
 stopped=True
signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)

def run(label,command,cap):
 opened=time.monotonic();reason=None;peak=0;cpu=0.
 with (root/'logs'/('kp_serial__'+label+'.log')).open('ab') as log:
  child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  while True:
   pid,status,usage=os.wait4(child.pid,os.WNOHANG)
   if pid:child.returncode=os.waitstatus_to_exitcode(status);break
   try:
    proc=psutil.Process(child.pid);rss=proc.memory_info().rss
    peak=max(peak,rss);t=proc.cpu_times();cpu=t.user+t.system
   except psutil.Error:rss=0
   available=psutil.virtual_memory().available
   if stopped or time.time()>=deadline or time.monotonic()-opened>=cap or rss>210*2**30 or available<12*2**30:
    reason='allocation_stop' if stopped else 'absolute_deadline' if time.time()>=deadline else 'finite_wall_budget' if time.monotonic()-opened>=cap else 'memory_headroom_censor'
    os.killpg(child.pid,signal.SIGKILL);_,status,usage=os.wait4(child.pid,0)
    child.returncode=os.waitstatus_to_exitcode(status);break
   write(root/'recovery/live'/(label+'.json'),{'status':'running','pid':child.pid,'host':os.uname().nodename,
     'slurm_job_id':os.environ['SLURM_JOB_ID'],'observed_unix':time.time(),
     'actual_process_cpu_seconds':cpu,'rss_bytes':rss,'peak_sampled_rss_bytes':peak,'available_host_bytes':available})
   time.sleep(5)
 result={'status':'complete' if child.returncode==0 else 'resource_censored' if reason else 'failed',
  'returncode':child.returncode,'reason':reason,'elapsed_wall_seconds':time.monotonic()-opened,
  'process_cpu_seconds':usage.ru_utime+usage.ru_stime,'peak_rss_bytes':usage.ru_maxrss*1024,
  'finished_unix':time.time(),'host':os.uname().nodename,'slurm_job_id':os.environ['SLURM_JOB_ID']}
 write(root/'recovery/serial_attempts'/(label+'.json'),result)
 write(root/'recovery/live'/(label+'.json'),result)
 return child.returncode==0

os.environ['LAVA_GEOMETRY_CHECK_KEYS']='dys_knapsack_1000x2__data151101'
os.environ['LAVA_GEOMETRY_CHECK_RECEIPT']='KP_BANK_GEOMETRY_CHECK.json'
if not run('exact_bank_regression',[sys.executable,str(root/'ops/check_bank_geometry.py')],1800):sys.exit(1)
if not run('bank_dys_knapsack_1000x2__data151101',[sys.executable,str(root/'ops/bank_geometry.py'),
 '--root',str(root),'--key','dys_knapsack_1000x2__data151101','--maximum','1000'],14400):sys.exit(1)
for name in ('job5_serial.json','job4_serial.json'):
 manifest=root/'recovery/jobs'/name;job=read(manifest)['jobs'][0]
 receipt=root/'results/receipts'/(job['id']+'.json')
 if receipt.exists():
  if read(receipt)['status']!='complete':raise RuntimeError('Existing adverse receipt requires a diagnosed new attempt')
  continue
 if stopped or time.time()>=deadline:break
 run(job['id'],[sys.executable,str(root/'ops/run_fit.py'),'--manifest',str(manifest),'--output-root',str(root/'results')],43200)
write(root/'recovery/SERIAL_ATTEMPT_ENDED.json',{'finished_unix':time.time(),'stopped':stopped,
 'note':'Execution end is not scientific completion; reconcile exact two fit receipts.'})
