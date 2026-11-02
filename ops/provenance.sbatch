#!/bin/bash
set -euo pipefail
export LAVA_RUN_ROOT="$1" LAVA_CODE="$1/code"
export PYTHONPATH="$LAVA_RUN_ROOT/ops:$LAVA_CODE/scripts:$LAVA_CODE/src:$LAVA_CODE/vendor/external/src:$LAVA_CODE/vendor/public"
export SOLVER_FREE_DFL_REPO="$LAVA_CODE/vendor/public" DFL_EXTENSIONS_SRC="$LAVA_CODE/vendor/external/src"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
LAVA_PYTHON=/home/jonathan.hornewall/runs/scaled_data_neural_20261021/venv/bin/python
srun --cpu-bind=cores "$LAVA_PYTHON" -c 'import os,json,pathlib; from geometry_provenance import certify; r=pathlib.Path(os.environ["LAVA_RUN_ROOT"]); v=json.loads((r/"ops/PROVENANCE_TARGETS.json").read_text())[int(os.environ["SLURM_ARRAY_TASK_ID"])]; print(json.dumps(certify(r,v["key"],v["n"],r/"provenance_inputs"/v["key"]/("prefix"+str(v["n"])+".pickle"))))'
