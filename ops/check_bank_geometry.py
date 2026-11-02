"""Focused implementation regression, two supplied training samples per family."""
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from bank_geometry import prepare,get_geometry,directory
from simplified_data import load_structure
from proximal_spo.public_highs import HighsLAVADataset,HighsOptModel,public_geometry
from qualify import write

torch.set_num_threads(1);torch.set_num_interop_threads(1)
root=Path(os.environ['LAVA_RUN_ROOT']);results=[]
for key in ('dys_shortest_path_10x10__data1103','dys_cfl_50x5__data17'):
    t=time.perf_counter();prepare(root,key,2)
    source=Path('/home/jonathan.hornewall/runs/scaled_data_neural_20261021/frozen_data')/key
    with np.load(source/'observations_n100.npz',allow_pickle=False) as f:
        x=f['train__contexts'][:2];c=f['train__physical_costs'][:2]
    model=HighsOptModel(load_structure(source).continuous_relaxation())
    reference=HighsLAVADataset(model,x,c,public_repo=os.environ['SOLVER_FREE_DFL_REPO'],vertex_tolerance=0.)
    cached=get_geometry({'required_data_key':key},model,x,c,lambda *a,**k:None)
    assert np.array_equal(reference.sols,cached.sols)
    assert np.array_equal(reference.objs,cached.objs)
    for i in range(2):
        assert np.array_equal(reference.adjacent_verts[i],cached.adjacent_verts[i])
        assert np.array_equal(reference.ctrs[i],cached.ctrs[i])
    from methods.lava import LAVA
    collate=public_geometry(os.environ['SOLVER_FREE_DFL_REPO']).collate_fn_2
    outputs=[]
    for dataset in (reference,cached):
        batch=collate([dataset[i] for i in range(2)])
        prediction=torch.tensor(-c,dtype=torch.float32,requires_grad=True)
        value=LAVA(model,-.1)(prediction,batch[7],batch[4],-1.)
        value.backward();outputs.append((value.item(),prediction.grad.numpy().copy()))
    assert outputs[0][0]==outputs[1][0] and np.array_equal(outputs[0][1],outputs[1][1])
    try:get_geometry({'required_data_key':key},model,x,c+1.,lambda *a,**k:None)
    except RuntimeError:pass
    else:raise AssertionError('Mismatched supplied training costs were accepted')
    results.append({'key':key,'samples':2,'exact_anchor_rows':True,'exact_neighbour_rows_and_order':True,
        'exact_binding_normals':True,'exact_public_loss_and_gradient':True,
        'mismatched_costs_rejected':True,'elapsed_wall_seconds':time.perf_counter()-t,
        'vertex_counts':[len(v) for v in reference.adjacent_verts]})
write(root/'ops/BANK_GEOMETRY_CHECK.json',{'status':'passed','results':results,
    'scientific_fits':False,'source':'first two original supplied training costs only'})
print(json.dumps(results),flush=True)
