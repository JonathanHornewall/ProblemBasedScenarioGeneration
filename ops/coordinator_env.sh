#!/bin/bash
set -euo pipefail
export LAVA_RUN_ROOT="$1" LAVA_CODE="$1/code"
export PYTHONPATH="$LAVA_RUN_ROOT/ops:$LAVA_CODE/scripts:$LAVA_CODE/src:$LAVA_CODE/vendor/external/src:$LAVA_CODE/vendor/public"
export SOLVER_FREE_DFL_REPO="$LAVA_CODE/vendor/public" DFL_EXTENSIONS_SRC="$LAVA_CODE/vendor/external/src"
export CENTRAL_PATH_FROZEN_DATA=/home/jonathan.hornewall/runs/scaled_data_neural_20261021/frozen_data
export SCALED_DATA_NEURAL_ROOT="$LAVA_RUN_ROOT" SIMPLIFIED_ROOT="$LAVA_RUN_ROOT" SCALED_DATA_NEURAL_SITE=cermics
export ATLANTA_SOURCE_MANIFEST="$LAVA_RUN_ROOT/source_receipt.json" CENTRAL_PATH_GEOMETRY_CACHE=/tmp/lava_completion_20261102/banks
export LAVA_NODE_MEMORY_GIB="${LAVA_NODE_MEMORY_GIB:-230}" LAVA_POOL_MEMORY_GIB="${LAVA_POOL_MEMORY_GIB:-230}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
LAVA_ENTRY="${2:-coordinator.py}"
exec /home/jonathan.hornewall/runs/scaled_data_neural_20261021/venv/bin/python "$LAVA_RUN_ROOT/ops/$LAVA_ENTRY"
