"""Restore exact proven-dead slots on their existing cores; signal nobody."""
import os,pathlib,socket,subprocess,sys,time
import psutil
from qualify import read,write
from worker_service import keep_step_alive

root=pathlib.Path(os.environ['LAVA_RUN_ROOT']);job=os.environ['SLURM_JOB_ID'];host=socket.gethostname()
assert job in('16726','16740')and host.startswith('node07')
plan=read(root/'ops/WORKER_SLOT_RESTORE_PLAN_20261104.json');receipt=root/'ops'/('DEAD_WORKERS_RESTORED_'+job+'_20261104.json')
if receipt.exists():raise RuntimeError('Restoration already attempted; reconcile receipt')
deadline=read(root/'CONTRACT.json')['scientific_cutoff_unix'];assert time.time()<deadline
out={'started_unix':time.time(),'job':job,'step':os.environ.get('SLURM_STEP_ID'),'restored':[],'healthy_processes_signalled':0,'new_allocations':0};write(receipt,out);children=[]
for slot in plan['slots']:
 if slot['job']!=job:continue
 token=slot['token'];file=root/'workers'/(token+'.json');old=read(file)
 assert old['pid']==slot['dead_child_pid']and old['worker_pid']==slot['dead_worker_pid']
 for pid in(slot['dead_child_pid'],slot['dead_worker_pid']):
  try:p=psutil.Process(pid);assert p.status()==psutil.STATUS_ZOMBIE,'A live PID must be preserved'
  except psutil.NoSuchProcess:pass
 for p in psutil.process_iter(['pid','cmdline','status']):
  cmd=p.info['cmdline']or[]
  if p.info['status']=='zombie' or len(cmd)<2 or cmd[1]!=str(root/'ops/completion.py')or'--worker'not in cmd:continue
  assert not(p.environ().get('SLURM_JOB_ID')==job and cmd[cmd.index('--worker')+1]==slot['index']),'Worker index already live'
 assert slot['core']in os.sched_getaffinity(0)
 for p in psutil.process_iter(['pid','cmdline','status']):
  cmd=p.info['cmdline']or[]
  if p.info['status']!='zombie'and any(str(root/'ops'/f)in cmd for f in('run_fit.py','prepare_bank.py','coordinator.py')):
   assert slot['core']not in p.cpu_affinity(),'Target core has a live campaign child'
 write(root/'recovery/worker_step_loss'/(token+'.json'),{'worker_snapshot':old,**slot,
   'status':'interrupted_by_step_lifetime','terminal_returncode_observed':False,
   'process_CPU_seconds_lower_bound':old.get('actual_process_cpu_seconds',0),
   'sampled_RSS_peak_bytes_lower_bound':old.get('rss_bytes',0),
   'successful_fit_rerun':False,'no_canonical_failure_written_over_healthy_replacement':True})
 env=os.environ.copy();env['LAVA_NODE_MEMORY_GIB']='230';env['LAVA_POOL_MEMORY_GIB']='96'if job=='16740'else'230'
 env.pop('LAVA_PRIORITY_PREPARATION_KEY',None)
 child=subprocess.Popen(['taskset','-c',str(slot['core']),sys.executable,str(root/'ops/completion.py'),'--root',str(root),'--worker',slot['index']],env=env,start_new_session=True)
 children.append(child);out['restored'].append({**slot,'new_worker_pid':child.pid});write(receipt,out)
out['finished_admission_unix']=time.time();out['state']='slots_restored_service_held_until_campaign_terminal';write(receipt,out);print(__import__('json').dumps(out),flush=True)
keep_step_alive(root,deadline,children)
for child in children:
 if child.poll()is None:child.terminate()
