#!/usr/bin/env bash
set -euo pipefail
umask 077
export LAVA_RUN_ROOT=/home/jonathan.hornewall/runs/lava_timing_replay_20261107
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
python=/home/jonathan.hornewall/runs/scaled_data_neural_20261021/venv/bin/python
[[ -n "${SLURM_JOB_ID:-}" && "$(hostname -s)" != master01-hpc ]]
exec srun --exact --ntasks=1 --cpus-per-task=1 --cpu-bind=cores \
  "$python" -B "$LAVA_RUN_ROOT/ops/cermics_20261107/slurm_replay.py" worker "${1:?route required}"
