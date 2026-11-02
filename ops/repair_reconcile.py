import json,pathlib,subprocess,time,hashlib
r=pathlib.Path('/home/jonathan.hornewall/runs/lava_completion_20261102')
out={'observed_unix':time.time()}
for name in ('ops/IDLE_WORKER_REFRESH.json','logs/idle-worker-refresh.out'):
 out[name]=(r/name).read_text()[-7000:] if (r/name).exists() else None
out['deployed_completion_sha256']=hashlib.sha256((r/'ops/completion.py').read_bytes()).hexdigest()
out['failed_logs']={}
for p in (r/'failures').glob('bank__dys_shortest_path_50x50*.json'):
 log=r/'logs'/(p.stem+'.log');out['failed_logs'][p.name]={'failure':json.loads(p.read_text()),'log':log.read_text()[-5000:]}
out['refresh_processes']=subprocess.run(['srun','--overlap','--jobid=16223','--nodelist=node09','--ntasks=1',
 '--cpus-per-task=1','--time=00:00:10','--cpu-bind=cores','pgrep','-af','refresh_idle_workers'],capture_output=True,text=True,timeout=12).stdout
print(json.dumps(out))
