"""Read existing measured costs; no hashes, solves, or archive recovery."""
import pathlib,json,time
r=pathlib.Path('/home/jonathan.hornewall/runs/lava_completion_20261102');read=lambda p:json.loads(p.read_text())
out={'observed_unix':time.time(),'qualification':{},'prefixes':[],'preprocessing_attempts':[],'fit_costs':[]}
for p in (r/'qualification').glob('*__seed*__sample*.json'):out['qualification'][p.name]=read(p)
proof=read(r/'ops/GEOMETRY_PROVENANCE.json');bad=set(read(r/'ops/AFFECTED_JOB_IDS.json'))
for folder in ('banks','seeded_banks_v2'):
 for p in (r/folder).glob('*cfl*/prefix*.json'):
  v=read(p)
  out['prefixes'].append({'file':str(p),**{k:v.get(k) for k in ('key','route','data_seed','initialization_seed','host','slurm_job_id','supplied_training_samples','geometry_rng_protocol','build_wall_seconds_cumulative','build_cpu_seconds_cumulative','vertex_storage_bytes','metadata_bytes','highs_version','adapter_sha256','public_source_sha256')},
   'maximum_neighbor_rows':max(v.get('vertex_counts',[0])),
   'N100_admissible':v.get('geometry_rng_protocol')=='seeded_numpy_resume_v2' or proof.get(v.get('key'),{}).get('random_free_supplied_prefix',0)>=100})
for p in (r/'attempts').glob('*.json'):
 v=read(p);key=v.get('key','')
 if 'cfl' in key and (v.get('kind')=='prepare' or p.name.startswith('seeded__')):out['preprocessing_attempts'].append({'file':str(p),**v})
for p in (r/'results/receipts').glob('*.json'):
 v=read(p);j=v.get('job',{});s=v.get('summary')or{}
 if 'cfl' not in j.get('route',''):continue
 out['fit_costs'].append({'id':p.stem,'route':j['route'],'n':j['config']['sizes']['train'],'seeds':j['config']['seeds'],
  'status':v.get('status'),'superseded_RNG_attempt':p.stem in bad,'stage':j.get('stage'),
  **{k:s.get(k)for k in ('optimization_wall_seconds','optimization_cpu_seconds','validation_wall_seconds','test_evaluation_wall_seconds')},'geometry':s.get('geometry')})
out['scope_stops']={p.name:read(p)for p in (r/'ops/scope_stopped_builders').glob('*.json')}
print(json.dumps(out))
