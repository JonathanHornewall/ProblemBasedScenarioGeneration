"""Retain only small seed/RNG/JSON records from exact owned node09 scratch."""
import hashlib,io,json,os,pathlib,socket,tarfile,time
import psutil
from qualify import read,write
r=pathlib.Path(os.environ['LAVA_RUN_ROOT']);scratch=pathlib.Path('/tmp/lava_completion_20261102')
assert os.environ['SLURM_JOB_ID']=='16223'and socket.gethostname().startswith('node09')
receipt=r/'ops/NODE09_COMPACT_RETENTION_20261104.json'
if receipt.exists():raise RuntimeError('Retention already attempted; reconcile receipt')
for p in psutil.process_iter(['pid','cmdline','status']):
 cmd=p.info['cmdline']or[]
 if p.info['status']!='zombie'and any(str(r/'ops'/f)in cmd for f in('prepare_bank.py','bank_geometry.py','run_fit.py')):
  raise RuntimeError('Preserve healthy numerical child before releasing node09')
target=r/'retained_compact';target.mkdir(exist_ok=True);archive=target/'node09_seed_records.tar.gz'
assert not archive.exists()and not scratch.is_symlink()
out={'started_unix':time.time(),'host':socket.gethostname(),'archive':str(archive),'files':[],'dense_cache_files':0,'neural_weights':0}
write(receipt,out)
with tarfile.open(archive,'x:gz',compresslevel=3,dereference=False)as tar:
 for base in('banks','seeded_banks_v2'):
  for bank in sorted((scratch/base).glob('*')):
   if bank.is_symlink()or not bank.is_dir():continue
   for f in sorted(bank.rglob('*')):
    if not f.is_file()or f.is_symlink()or any(p.is_symlink()for p in f.parents if p.is_relative_to(scratch)):continue
    if f.suffix!='.json'and not(f.name.startswith('rng_sample')and f.suffix=='.pickle'):continue
    assert f.stat().st_size<1024**2
    raw=f.read_bytes();name=str(f.relative_to(scratch));digest=hashlib.sha256(raw).hexdigest()
    info=tarfile.TarInfo(name);info.size=len(raw);info.mode=0o600;info.mtime=f.stat().st_mtime
    tar.addfile(info,io.BytesIO(raw));out['files'].append({'path':name,'bytes':len(raw),'sha256':digest})
out.update({'finished_unix':time.time(),'archive_bytes':archive.stat().st_size,'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'file_count':len(out['files']),'uncompressed_bytes':sum(v['bytes']for v in out['files']),'state':'retained_on_shared_home','scratch_removed':False})
write(receipt,out)
print(json.dumps({k:v for k,v in out.items()if k!='files'}))
