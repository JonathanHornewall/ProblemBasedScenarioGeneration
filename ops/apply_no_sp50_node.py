"""One exact scope snapshot; stop only excluded SP50 builders with no fit dependency."""
import contextlib,io,json,os,pathlib,runpy,signal,socket,time
import psutil
from qualify import read,write

r=pathlib.Path(os.environ['LAVA_RUN_ROOT']);host=socket.gethostname();receipt=r/'ops'/('SP50_SCOPE_NODE_'+host+'.json')
if receipt.exists():raise RuntimeError('Node scope mutation already attempted; reconcile its receipt')
buf=io.StringIO()
with contextlib.redirect_stdout(buf):runpy.run_path(str(r/'ops/snapshot_scope_node.py'),run_name='__main__')
snapshot=json.loads(buf.getvalue());out={'snapshot':snapshot,'stopped_builders':[],'preserved_builder_dependencies':[],
 'healthy_fits_signalled':0,'new_CPU_allocations':0,'observed_unix':time.time()}
write(receipt,out)
for v in snapshot['builders']:
 if v['route']!='dys_shortest_path_50x50':continue
 if any(j['job'].get('required_data_key')==v['key'] for j in snapshot['fits']):
  out['preserved_builder_dependencies'].append(v);write(receipt,out);continue
 proc=psutil.Process(v['pid'])
 if proc.create_time()!=v['birth_unix'] or proc.cmdline()!=v['cmdline'] or proc.environ().get('SLURM_JOB_ID')!=v['slurm_job_id'] or os.getpgid(proc.pid)!=proc.pid:
  raise RuntimeError('Exact excluded builder identity differs')
 record={**v,'signal':'SIGTERM','signal_attempt_unix':time.time(),'reason':'Human excluded all SP50 preparation and fits; no healthy fit depends on this builder'}
 out['stopped_builders'].append(record);write(receipt,out)
 os.killpg(proc.pid,signal.SIGTERM)
out['finished_unix']=time.time();write(receipt,out);print(json.dumps(out))
