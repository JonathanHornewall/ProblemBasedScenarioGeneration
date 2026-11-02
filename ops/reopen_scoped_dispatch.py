"""Reload scoped metadata accounting, reconcile, then reopen retained CFL cells."""
import pathlib,time,subprocess,json
from qualify import read,write
from scope_control import required_fit,drain_status,kp_status
r=pathlib.Path('/home/jonathan.hornewall/runs/lava_completion_20261102')
assert read(r/'ops/SCOPE_DISPATCH_CHECK.json')['status']=='passed'
assert read(r/'SCOPE_AMENDMENT_20261102_CFL_N100.json')['required_counts']['total']==810
p=subprocess.run(['srun','--overlap','--jobid=16223','--nodelist=node09','--ntasks=1','--cpus-per-task=1',
 '--time=00:00:20','--cpu-bind=cores','env','LAVA_METADATA_RELOAD_REASON=Apply direct human CFL50/100 N100 scope; preserve all numerical workers',
 '/home/jonathan.hornewall/runs/scaled_data_neural_20261021/venv/bin/python',str(r/'ops/stop_current_coordinator.py')],capture_output=True,text=True,timeout=22)
if p.returncode:raise RuntimeError(p.stderr+p.stdout)
log=(r/'logs/coordinator-current.out').open('ab')
child=subprocess.Popen(['srun','--overlap','--jobid=16223','--nodelist=node09','--ntasks=1','--cpus-per-task=1',
 '--cpu-bind=cores','bash',str(r/'ops/coordinator_env.sh'),str(r)],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
deadline=time.monotonic()+20;ready=None
while time.monotonic()<deadline:
 v=read(r/'READY.json')
 if v.get('expected_complete_scope',{}).get('total')==810:
  ready=v;break
 time.sleep(.25)
if ready is None:raise RuntimeError('Scoped READY not observed; CFL hold remains for reconciliation')
excluded=[]
for item in ready['fits']:
 if not required_fit(r,read(r/'jobs'/(item['id']+'.json'))['jobs'][0]):excluded.append(item['id'])
if excluded:raise RuntimeError('Excluded fit entered scoped READY; hold remains')
hold=r/'ops/HOLD_CFL_COORDINATOR_REFRESH.json'
if hold.exists():hold.rename(r/'ops/resolved_holds'/('scope_N100__'+str(time.time_ns())+'.json'))
# Release only retained startup holds with already-admissible prefix metadata.
proof=read(r/'ops/GEOMETRY_PROVENANCE.json');released=[]
for file in (r/'failures').glob('lava_complete__*.json'):
 v=read(file)
 if v.get('returncode') not in (75,76):continue
 manifest=r/'jobs'/(file.stem+'.json')
 if not manifest.exists():continue
 job=read(manifest)['jobs'][0]
 if 'cfl' not in job['route'] or not required_fit(r,job):continue
 key=job['required_data_key'];seed=job['config']['seeds']['initialization'];n=job['config']['sizes']['train']
 meta=r/'seeded_banks_v2'/(key+'__init'+str(seed))/f'prefix{n}.json'
 if proof.get(key,{}).get('random_free_supplied_prefix',0)<n and not meta.exists():continue
 destination=r/'recovery/provenance_holds'/(file.stem+'__scope_reopen__'+str(time.time_ns())+'.json')
 file.rename(destination);released.append(file.stem)
bad=set(read(r/'ops/AFFECTED_JOB_IDS.json'));counts={'SP':0,'CFL':0,'KP':sum(v['complete']for v in kp_status(r))};supplementary=0;unresolved=0
for file in (r/'results/receipts').glob('*.json'):
 v=read(file);job=v.get('job',{});s=v.get('summary')or{}
 if not job.get('lava_completion_revision') or v.get('status')!='complete' or file.stem in bad:continue
 n=job['config']['sizes']['train'];geo=s.get('geometry')or{}
 valid=proof.get(job['required_data_key'],{}).get('random_free_supplied_prefix',0)>=n or geo.get('geometry_rng_protocol')=='seeded_numpy_resume_v2'
 if not valid:unresolved+=1;continue
 if required_fit(r,job):counts['CFL' if 'cfl' in job['route'] else 'SP']+=1
 else:supplementary+=1
out={'observed_unix':time.time(),'required_counts':{'SP':660,'CFL':144,'KP':6,'total':810},'valid_required_completed':counts,
 'completed_total':sum(counts.values()),'valid_completed_supplementary':supplementary,'unresolved_completed_records':unresolved,
 'superseded_RNG_originals':len(bad),'excluded_new_dispatch':excluded,'CFL_allowed_in_READY':sorted(set((v['route'],v['n'])for v in ready['fits']if 'cfl' in v['route'])),
 'grandfathered_drain':drain_status(r),'retained_holds_released':released,'new_CPU_allocations_for_reload':0,
 'coordinator':read(r/'ops/COORDINATOR_ACTIVE.json'),'registry':read(r/'ops/JOB_REGISTRY.json'),
 'queue':subprocess.run(['squeue','-h','-r','-u','jonathan.hornewall','-o','%i|%C|%T|%N|%j'],capture_output=True,text=True,timeout=5).stdout,
 'deadline_utc':read(r/'CONTRACT.json')['deadline_utc'],'CFL_dispatch_reopened_retained_only':True}
write(r/'ops/SCOPE_DEPLOYMENT_READBACK.json',out);print(json.dumps(out))
