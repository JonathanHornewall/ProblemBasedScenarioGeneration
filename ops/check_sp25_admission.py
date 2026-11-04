"""One metadata/process regression; no fit, dataset, solver or enumeration."""
import json,os,pathlib,subprocess,sys,tempfile,time
from completion import memory,memory_limit

prepare={'route':'dys_shortest_path_25x25','kind':'prepare','n':1000}
fit={**prepare,'kind':'fit'}
assert (memory(prepare),memory_limit(prepare))==(16,24)
assert (memory(fit),memory_limit(fit))==(3,6)
assert 10*memory(prepare)+21*memory(fit)==223<230
assert 4*memory(prepare)+4*memory(fit)==76<96
base=pathlib.Path(tempfile.mkdtemp(prefix='lava_sp25_admission_fixture_'))
old_program=r'''
import json,os,pathlib,socket,subprocess,sys
r=pathlib.Path(sys.argv[sys.argv.index('--root')+1])
if '--fixture-gib' not in sys.argv:
 (r/'new_worker.json').write_text(json.dumps({'pid':os.getpid(),'affinity':list(os.sched_getaffinity(0))}))
 raise SystemExit(0)
gib=int(sys.argv[sys.argv.index('--fixture-gib')+1]);token=socket.gethostname()+'_0_'+os.environ['SLURM_JOB_ID']
(r/'reservations.json').write_text(json.dumps({token:{'gib':gib,'worker_pid':os.getpid(),'id':'boundary_fixture','child_memory_limit_gib':6}}))
p=subprocess.Popen([sys.executable,str(r/'request.py'),str(r)],start_new_session=True)
raise SystemExit(p.wait(timeout=10))
'''
request=r'''
import json,pathlib,sys
from worker_admission import handoff_if_stale
r=pathlib.Path(sys.argv[1]);job={'id':'boundary_fixture','route':'dys_shortest_path_25x25','config':{'sizes':{'train':1000}}}
result=handoff_if_stale(r,job,reservation_path=r/'reservations.json')
(r/'unchanged.json').write_text(json.dumps({'returned':result}))
'''
cases=[]
for gib in (3,6):
 root=base/str(gib);(root/'ops').mkdir(parents=True)
 (root/'ops/completion.py').write_text(old_program);(root/'request.py').write_text(request)
 child=subprocess.Popen([sys.executable,str(root/'ops/completion.py'),'--root',str(root),'--worker','0','--fixture-gib',str(gib)])
 code=child.wait(timeout=12)
 if gib==3:
  assert code==0 and json.loads((root/'unchanged.json').read_text())['returned']is False
  assert not(root/'new_worker.json').exists()
  assert next(iter(json.loads((root/'reservations.json').read_text()).values()))['child_memory_limit_gib']==6
 else:
  assert code==-9
  end=time.monotonic()+5
  while not(root/'new_worker.json').exists()and time.monotonic()<end:time.sleep(.02)
  new=json.loads((root/'new_worker.json').read_text())
  receipt=json.loads(next((root/'ops/worker_boundary_handoffs').glob('*.json')).read_text())
  assert receipt['new_worker_pid']==new['pid']and receipt['healthy_active_fits_signalled']==0
  assert receipt['old_reservation_GiB']==6 and receipt['new_reservation_GiB']==3
  assert len(new['affinity'])==1 and json.loads((root/'reservations.json').read_text())=={}
 cases.append({'admission_GiB':gib,'old_worker_exit':code,'status':'passed'})
print(json.dumps({'status':'passed','cases':cases,'fixture':str(base),'scientific_fits':0,'enumerations':0,
 'bank_admission_GiB':16,'bank_guard_GiB':24,'fit_admission_GiB':3,'fit_guard_GiB':6,
 'bound_cpu_count':len(os.sched_getaffinity(0))}))
