import json,pathlib,subprocess,time
r=pathlib.Path('/home/jonathan.hornewall/runs/lava_completion_20261102')
pid=3120354
cmd=['srun','--overlap','--jobid=16223','--nodelist=node09','--ntasks=1','--cpus-per-task=1',
 '--time=00:00:40','--cpu-bind=cores','bash','-c',
 'ps -p 3120354 -o pid=,etime=,pcpu=,cputimes=,rss=,stat=,wchan=,args=; '
 'cat /proc/3120354/io; command -v py-spy || true; command -v gdb || true; '
 'find /tmp/lava_completion_20261102/banks/dys_cfl_50x5__data151101 -maxdepth 1 -type f -printf "%T@ %s %f\\n" | sort -nr | head -12; '
 'tail -c 10000 /home/jonathan.hornewall/runs/lava_completion_20261102/logs/bank__dys_cfl_50x5__data151101.log; '
 'if command -v py-spy >/dev/null; then timeout 10 py-spy dump --pid 3120354; elif command -v gdb >/dev/null; then timeout 10 gdb --batch -ex "set pagination off" -ex "thread apply all bt 8" -ex detach -p 3120354; fi']
p=subprocess.run(cmd,capture_output=True,text=True,timeout=45)
print(json.dumps({'observed_unix':time.time(),'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}))
