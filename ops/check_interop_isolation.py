"""Reproduce the cold-classification/fit initialization boundary on two LPs."""
import os,pathlib,shutil,json,time
import torch
from safe_geometry import classify_in_subprocess
from qualify import read,write,sha
root=pathlib.Path(os.environ['LAVA_RUN_ROOT']);key='dys_cfl_50x5__data17'
fixture=root/'implementation_checks/interop_isolation'/str(time.time_ns())
(fixture/'ops').mkdir(parents=True)
source=root/'implementation_checks/seed_resume/16743/full/prefix2.json'
target=fixture/'banks'/key/'prefix2.json';target.parent.mkdir(parents=True)
shutil.copyfile(source,target)
metadata=pathlib.Path('/tmp/lava_completion_20261102/seed_resume_check/16743/full/prefix2.pickle')
result=classify_in_subprocess(fixture,key,2,metadata)
# This is the exact second initialization that failed after in-process
# classification. Cold classification must not consume the parent's call.
torch.set_num_interop_threads(1)
assert torch.get_num_interop_threads()==1 and torch.get_num_threads()==1 and len(os.sched_getaffinity(0))==1 and result['random_free_supplied_prefix']==2
write(root/'ops/INTEROP_ISOLATION_CHECK.json',{'status':'passed','scientific_fit':False,
 'neighbor_enumerations':0,'fixture_original_observations':2,'cold_classifier_subprocess':True,
 'parent_maintained_interop_initialization_succeeded':True,'finished_unix':time.time(),
 'torch_num_threads':torch.get_num_threads(),'torch_num_interop_threads':torch.get_num_interop_threads(),'CPU_affinity_count':len(os.sched_getaffinity(0)),
 'slurm_job_id':os.environ['SLURM_JOB_ID'],'safe_geometry_sha256':sha(root/'ops/safe_geometry.py')})
print('Cold classification followed by maintained Torch interop initialization: passed')
