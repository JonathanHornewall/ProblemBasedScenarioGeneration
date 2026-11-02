#!/bin/bash
# Arguments: exact campaign root, route, original seed.
set -euo pipefail
LAVA_RUN_ROOT="$1"
LAVA_ROUTE="$2"
LAVA_DATA_SEED="$3"
export LAVA_CODE="$LAVA_RUN_ROOT/code"
export PYTHONPATH="$LAVA_CODE/scripts:$LAVA_CODE/src:$LAVA_CODE/vendor/external/src:$LAVA_CODE/vendor/public"
export SOLVER_FREE_DFL_REPO="$LAVA_CODE/vendor/public"
export DFL_EXTENSIONS_SRC="$LAVA_CODE/vendor/external/src"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
LAVA_PYTHON=/home/jonathan.hornewall/runs/scaled_data_neural_20261021/venv/bin/python
LAVA_BANK="/home/jonathan.hornewall/runs/parsimonious_pilot_20261019/frozen_data/${LAVA_ROUTE}__data${LAVA_DATA_SEED}"
date -u
hostname
df -h /tmp
srun --cpu-bind=cores "$LAVA_PYTHON" "$LAVA_RUN_ROOT/ops/qualify.py" --root "$LAVA_RUN_ROOT" --route "$LAVA_ROUTE" --bank "$LAVA_BANK" --seed "$LAVA_DATA_SEED"
