#!/bin/bash
set -euo pipefail
cd /home/jonathan.hornewall/runs/lava_completion_20261102
scontrol update JobId=16210 MinMemoryNode=230G 2>&1 || true
scontrol update JobId=16209 MinMemoryNode=30G 2>&1 || true
nohup srun --overlap --jobid=16210 --nodelist=node07 --nodes=1 --ntasks=1 --cpus-per-task=1 --time=04:00:00 --cpu-bind=cores /home/jonathan.hornewall/runs/scaled_data_neural_20261021/venv/bin/python ops/memory_guard.py --root /home/jonathan.hornewall/runs/lava_completion_20261102 --route dys_cfl_500x10 --maximum-gib 220 --minimum-available-gib 12 > logs/memory_guard500.log 2>&1 < /dev/null &
echo $! > ops/MEMORY_GUARD500_LAUNCHER_PID.txt
cat ops/MEMORY_GUARD500_LAUNCHER_PID.txt
