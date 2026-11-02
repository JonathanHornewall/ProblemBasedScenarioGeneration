"""New construction uses each original bank's initialization seed."""
import argparse,pathlib,socket,time,fcntl,os
import psutil
from qualify import read,write
from bank_geometry import prepare,directory
p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--key',required=True);p.add_argument('--maximum',type=int,required=True);p.add_argument('--initialization-seed',type=int)
a=p.parse_args();root=pathlib.Path(a.root);plan=read(root/'PLAN.json');route,data=a.key.split('__data')
seeds=next((v for v in plan['cells'][route]['confirmation'] if str(v['data'])==data),plan['cells'][route]['screen'])
if str(seeds['data'])!=data:raise RuntimeError('Original bank seed identity missing')
if a.initialization_seed is not None and a.initialization_seed!=seeds['initialization']:raise RuntimeError('Requested initialization differs from original bank')
failure=root/'failures'/('bank__'+a.key+'.json')
if not failure.exists():write(failure,{'key':a.key,'reason':'seeded_construction_owner','host':socket.gethostname(),'created_unix':time.time(),'scientific_failure':False})
status=root/'banks'/a.key/'STATUS.json';old=read(status) if status.exists() else {'key':a.key,'host':socket.gethostname(),'status':'seeded_constructing'}
write(status,{**old,'seeded_cache_owner_host':socket.gethostname(),'seeded_initialization_seed':seeds['initialization'],'seeded_requested_maximum':a.maximum})
# Dispatch queues use different claims; actual construction has one common
# lock. Adopt older in-flight builders without racing their saved RNG states.
birth=psutil.Process().create_time();deadline=read(root/'CONTRACT.json')['scientific_cutoff_unix']
def older_builder():
 for proc in psutil.process_iter(['pid','cmdline','create_time']):
  cmd=proc.info['cmdline'] or []
  if proc.pid==os.getpid() or len(cmd)<2 or cmd[1] not in (str(root/'ops/prepare_bank.py'),str(root/'ops/bank_geometry.py')):continue
  if '--key' not in cmd or cmd[cmd.index('--key')+1]!=a.key:continue
  if (proc.info['create_time'],proc.pid)<(birth,os.getpid()) and proc.status()!=psutil.STATUS_ZOMBIE:return True
 return False
while older_builder() and time.time()<deadline:time.sleep(1)
if time.time()>=deadline:raise SystemExit(75)
with (root/'claims'/('seeded_build__'+a.key+'__init'+str(seeds['initialization'])+'.lock')).open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 prefixes=[int(v.stem[6:]) for v in (root/'banks'/a.key).glob('prefix*.json') if int(v.stem[6:])<=a.maximum]
 if prefixes:
  from geometry_provenance import certify
  certify(root,a.key,max(prefixes))
 prepare(root,a.key,a.maximum,initialization_seed=seeds['initialization'])
 current=read(status)
 write(status,{**current,'seeded_available_prefixes':sorted(int(p.stem[6:]) for p in directory(a.key,seeds['initialization']).glob('prefix*.json')),'seeded_updated_unix':time.time()})
