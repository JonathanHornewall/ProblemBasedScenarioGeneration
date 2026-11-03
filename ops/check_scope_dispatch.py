"""Focused metadata-only dispatch/counting checks; no optimization or enumeration."""
import pathlib,os,json
from qualify import read,write
from completion import PHASES,make_job,outcome_coverage
from scope_control import phase_required,route_required,required_fit,preparation_maximum,censored_ids,runtime_fit_allowed
r=pathlib.Path(os.environ['LAVA_RUN_ROOT']);plan=read(r/'PLAN.json')
from simplified_campaign import grid
from pilot_common import family,SMALLEST
assert SMALLEST['cfl']=='dys_cfl_50x5'
counts={}
for fam in ('sp','cfl'):
 phases=[p for p in PHASES if phase_required(r,fam,p)]
 routes=[route for route in plan['routes'] if family(route)==fam and route_required(r,route)]
 counts[fam]={'screens':len(phases)*len(grid('lava')),'confirmations':sum(len(plan['cells'][route]['confirmation']) for route in routes)*len(phases)}
assert counts=={'sp':{'screens':160,'confirmations':400},'cfl':{'screens':64,'confirmations':80}}
for phase in [p for p in PHASES if p['train']==100]:
 for route,stage,seeds in [('dys_cfl_50x5','screen',plan['cells']['dys_cfl_50x5']['screen']),
                          ('dys_cfl_100x10','confirmation',plan['cells']['dys_cfl_100x10']['confirmation'][0])]:
  job=make_job(r,plan,route,seeds,phase,grid('lava')[0],stage)
  assert required_fit(r,job) and job['config']['sizes']=={'train':100,'validation':32,'test':100}
  assert job['config']['seeds']==seeds
for route in ('dys_cfl_50x5','dys_cfl_100x10','dys_cfl_200x10','dys_cfl_250x10','dys_cfl_500x10'):
 for n in (100,300,500,1000):
  sample={'route':route,'config':{'sizes':{'train':n,'validation':32,'test':100},'model':{'depth':0}}}
  assert required_fit(r,sample)==(route in ('dys_cfl_50x5','dys_cfl_100x10') and n==100)
  assert preparation_maximum(r,route+'__data151101',n)==(100 if route in ('dys_cfl_50x5','dys_cfl_100x10') else 0)
for n in (100,300,500,1000):
 assert not required_fit(r,{'route':'dys_shortest_path_50x50','config':{'sizes':{'train':n}}})
 assert preparation_maximum(r,'dys_shortest_path_50x50__data151101',n)==0
assert not route_required(r,'dys_shortest_path_50x50')
assert preparation_maximum(r,'dys_shortest_path_25x25__data151101',1000)==1000
assert preparation_maximum(r,'dys_knapsack_1000x2__data151101',1000)==1000
assert outcome_coverage({'ok','censor','missing'},{'ok'},{'censor'})[2]=={'missing'}
assert not outcome_coverage({'ok','censor'},{'ok'},{'censor'})[2]
assert outcome_coverage({'required'},set(),{'excluded'})[2]=={'required'}
assert len(outcome_coverage({'ok'},{'ok'},{'ok'})[0])==1 and not outcome_coverage({'ok'},{'ok'},{'ok'})[1]
declared=censored_ids(r);assert len(declared)==40
for jid in declared:
 job=read(r/'jobs'/(jid+'.json'))['jobs'][0]
 assert job['route']=='dys_cfl_100x10' and job['config']['sizes']=={'train':100,'validation':32,'test':100}
 assert required_fit(r,job) and not runtime_fit_allowed(r,job)
out={'status':'passed','counts':counts,'KP':6,'total_required':710,'neighbor_enumerations':0,
 'scientific_fits':0,'CFL100_uses_original_smallest_family_tuning':True,'original_seed_configurations_preserved':True,
 'stale_runtime_guard_checked':True,'preparation_caps_checked':True,'slurm_job_id':os.environ['SLURM_JOB_ID']}
out.update({'accepted_resource_censors':40,'terminal_coverage_regression':'passed','censors_are_scientific_successes':False})
write(r/'ops/SCOPE_DISPATCH_CHECK.json',out);print(json.dumps(out))
