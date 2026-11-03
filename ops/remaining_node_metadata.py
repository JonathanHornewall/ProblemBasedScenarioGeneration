"""One metadata snapshot of owned workers, prefix costs and array headers."""
import json,pathlib,pickle,socket,time,zipfile,math
import psutil,numpy as np
r=pathlib.Path('/home/jonathan.hornewall/runs/lava_completion_20261102');local=pathlib.Path('/tmp/lava_completion_20261102');host=socket.gethostname()
out={'host':host,'observed_unix':time.time(),'available_GiB':psutil.virtual_memory().available/2**30,'scratch_available_GiB':__import__('shutil').disk_usage('/tmp').free/2**30,'reservations':json.loads((local/'reservations.json').read_text()),'numeric_processes':[],'SP25_prefixes':[],'CFL100_shape':None}
for p in psutil.process_iter(['pid','name','cmdline','cpu_times','memory_info','status']):
 cmd=p.info['cmdline']or[]
 if not p.info['name'].startswith('python')or p.info['status']=='zombie':continue
 if not any(str(r/'ops'/f)in cmd for f in('run_fit.py','prepare_bank.py','bank_geometry.py')):continue
 out['numeric_processes'].append({'pid':p.pid,'command':cmd,'CPU_seconds':sum(p.info['cpu_times'][:2]),'RSS_bytes':p.info['memory_info'].rss})
for p in(r/'seeded_banks_v2').glob('dys_shortest_path_25x25__*/prefix100.json'):
 v=json.loads(p.read_text())
 if v['host']==host:out['SP25_prefixes'].append(v)
def header(f):
 version=np.lib.format.read_magic(f)
 if version==(1,0):shape,order,dtype=np.lib.format.read_array_header_1_0(f)
 elif version==(2,0):shape,order,dtype=np.lib.format.read_array_header_2_0(f)
 else:raise ValueError('Unexpected metadata-only NPY version')
 return {'shape':shape,'dtype':str(dtype),'bytes':int(math.prod(shape)*dtype.itemsize)}
for p in(r/'seeded_banks_v2').glob('dys_cfl_100x10__*/prefix100.json'):
 meta=json.loads(p.read_text())
 if meta['host']!=host:continue
 where=local/'seeded_banks_v2'/p.parent.name
 with(where/'prefix100.pickle').open('rb')as f:fields=pickle.load(f)
 with(where/fields['vertices'][0]).open('rb')as f:vert=header(f)
 with zipfile.ZipFile(where/fields['ctrs'][0])as z:
  with z.open('normals.npy')as f:norm=header(f)
 out['CFL100_shape']={'key':meta['key'],'sols_shape':list(fields['sols'].shape),'vertex_counts':fields['vertex_counts'],'first_vertex':vert,'first_normals':norm,'vertex_storage_bytes':meta['vertex_storage_bytes']}
 break
print(json.dumps(out))
