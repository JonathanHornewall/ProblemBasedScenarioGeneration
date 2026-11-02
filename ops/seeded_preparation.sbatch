#!/bin/bash
set -euo pipefail
export LAVA_RUN_ROOT="$1" LAVA_CODE="$1/code"
export PYTHONPATH="$LAVA_RUN_ROOT/ops:$LAVA_CODE/scripts:$LAVA_CODE/src:$LAVA_CODE/vendor/external/src:$LAVA_CODE/vendor/public"
export SOLVER_FREE_DFL_REPO="$LAVA_CODE/vendor/public" DFL_EXTENSIONS_SRC="$LAVA_CODE/vendor/external/src"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
srun --cpu-bind=cores /home/jonathan.hornewall/runs/scaled_data_neural_20261021/venv/bin/python "$LAVA_RUN_ROOT/ops/seeded_preparation.py"
