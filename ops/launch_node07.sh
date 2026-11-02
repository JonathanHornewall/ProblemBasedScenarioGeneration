#!/bin/bash
set -euo pipefail
LAVA_ROOT=/home/jonathan.hornewall/runs/lava_completion_20261102
LAVA_KEY=dys_cfl_50x5__data17
mkdir -p "$LAVA_ROOT/transfer"
# Reuse the completed two-observation exact cache on node09. Both steps use
# already-owned Slurm cores; no compute-node SSH or host-key changes.
flock "$LAVA_ROOT/claims/bank__${LAVA_KEY}.lock" srun --overlap --jobid=16223 --nodelist=node09 --nodes=1 --ntasks=1 --cpus-per-task=1 --time=00:01:00 --cpu-bind=cores tar -C "/tmp/lava_completion_20261102/banks/$LAVA_KEY" -cf "$LAVA_ROOT/transfer/cfl50_development_prefix2.tar" .
srun --overlap --jobid=16210 --nodelist=node07 --nodes=1 --ntasks=1 --cpus-per-task=1 --time=00:01:00 --cpu-bind=cores bash -c 'set -e; test ! -e /tmp/lava_completion_20261102/banks/dys_cfl_50x5__data17; mkdir -p /tmp/lava_completion_20261102/banks/dys_cfl_50x5__data17; tar -xf /home/jonathan.hornewall/runs/lava_completion_20261102/transfer/cfl50_development_prefix2.tar -C /tmp/lava_completion_20261102/banks/dys_cfl_50x5__data17'
cd "$LAVA_ROOT"
sbatch --parsable --partition=defq-gpu --nodelist=node07 --nice=10000 --requeue --job-name=lavacomp_20261102_b --ntasks=1 --cpus-per-task=8 --mem=160G --time=6-22:00:00 --output=logs/completion-%j.out ops/completion.sbatch "$LAVA_ROOT" 8 160
