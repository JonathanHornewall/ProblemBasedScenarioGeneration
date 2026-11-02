#!/bin/bash
set -euo pipefail
LAVA_RUN_ROOT="$1"
export LAVA_CODE="$LAVA_RUN_ROOT/code"
export PYTHONPATH="$LAVA_CODE/scripts:$LAVA_CODE/src:$LAVA_CODE/vendor/external/src:$LAVA_CODE/vendor/public"
export SOLVER_FREE_DFL_REPO="$LAVA_CODE/vendor/public"
export DFL_EXTENSIONS_SRC="$LAVA_CODE/vendor/external/src"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export ATLANTA_SOURCE_MANIFEST="$LAVA_RUN_ROOT/source_receipt.json"
export CENTRAL_PATH_FROZEN_DATA=/home/jonathan.hornewall/runs/scaled_data_neural_20261021/frozen_data
export SCALED_DATA_NEURAL_ROOT="$LAVA_RUN_ROOT"
export SIMPLIFIED_ROOT="$LAVA_RUN_ROOT"
export SCALED_DATA_NEURAL_SITE=cermics
export CENTRAL_PATH_GEOMETRY_CACHE="/tmp/lava_completion_20261102/recovery_geometry"
LAVA_PYTHON=/home/jonathan.hornewall/runs/scaled_data_neural_20261021/venv/bin/python
LAVA_MANIFEST="$LAVA_RUN_ROOT/recovery/jobs/job${SLURM_ARRAY_TASK_ID}.json"
hostname
date -u
df -h /tmp
srun --cpu-bind=cores "$LAVA_PYTHON" "$LAVA_RUN_ROOT/ops/run_fit.py" --manifest "$LAVA_MANIFEST" --output-root "$LAVA_RUN_ROOT/results"
