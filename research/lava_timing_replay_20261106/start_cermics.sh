#!/usr/bin/env bash
# Invoked by srun with one physical CPU inside the finite campaign allocation.
# Source is unpacked once at deployment; the shared environment is read-only.
set -euo pipefail
run_dir="${LAVA_RUN_ROOT:?isolated campaign directory required}"
[[ "$run_dir" == /home/jonathan.hornewall/runs/lava_timing_replay_20261107 ]] || exit 2
[[ -n "${SLURM_JOB_ID:-}" && -n "${SLURM_JOB_NODELIST:-}" ]] || exit 2
[[ "$(hostname -s)" != master01-hpc ]] || exit 2
cd -- "$run_dir"
export LAVA_CODE="$run_dir/code" LAVA_GEOMETRY_ROOT=/tmp/lava_timing_replay_20261107
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export CENTRAL_PATH_GEOMETRY_CACHE="$LAVA_GEOMETRY_ROOT/cache" SOLVER_FREE_DFL_REPO="$run_dir/code/vendor/public"
export DFL_EXTENSIONS_SRC="$run_dir/code/vendor/external/src"
export PYTHONPATH="$run_dir:$run_dir/code/scripts:$run_dir/code/src:$DFL_EXTENSIONS_SRC:$SOLVER_FREE_DFL_REPO"
LAVA_PYTHON=/home/jonathan.hornewall/runs/scaled_data_neural_20261021/venv/bin/python
[[ -x "$LAVA_PYTHON" && -f "$LAVA_CODE/scripts/run_central_path_final.py" ]] || exit 3
mkdir -p "$LAVA_GEOMETRY_ROOT" runtime
"$LAVA_PYTHON" - <<'PY'
import hashlib, highspy, importlib.metadata, json, numpy, os, pathlib, platform, scipy, shutil, sys, torch
expected = ((3,13,5), '1.12.0', '2.3.5', '1.16.3', '2.8.0+cpu', '1.0.2')
observed = (sys.version_info[:3], highspy.Highs().version(), numpy.__version__, scipy.__version__, torch.__version__, importlib.metadata.version('pyepo'))
if observed != expected:
    raise RuntimeError(f'Numerical environment differs: {observed}')
affinity = sorted(os.sched_getaffinity(0))
if len(affinity) != 1:
    raise RuntimeError(f'Expected one physical CPU, got {affinity}')
scratch = pathlib.Path(os.environ['LAVA_GEOMETRY_ROOT'])
free = shutil.disk_usage(scratch).free
if free < 30*2**30:
    raise RuntimeError('Less than the original 30 GiB node-local scratch reserve')
code = pathlib.Path(os.environ['LAVA_CODE'])
for rel, expected_sha in {
    'scripts/run_central_path_final.py': '2bd606c71c910bd7674a2456603b3bd4279b150cd8211d2d7249d51f63341d9f',
    'src/proximal_spo/public_highs.py': '1933a96df3236d548ba08440b6b5dda16ebdcbae95c7ca5983d611d0eeaaaf17',
    'vendor/public/methods/lava.py': 'c86af48cc221a6bb028a67f6f7def12c95c0b593cebbb62c0ff69fbb9b0fb9a1',
    'vendor/public/data/dataset_augmented.py': '9dba06da1514e7bf725ba0fc1f5bf86314425f643334a8ea951837c3a95dc0cd',
}.items():
    if hashlib.sha256((code/rel).read_bytes()).hexdigest() != expected_sha:
        raise RuntimeError(f'Frozen numerical source differs: {rel}')
record = {'slurm_job_id': os.environ['SLURM_JOB_ID'], 'slurm_step_id': os.environ.get('SLURM_STEP_ID'),
          'host': platform.node(), 'affinity': affinity, 'versions': observed, 'scratch_free_bytes': free,
          'cpu_description': pathlib.Path('/proc/cpuinfo').read_text(), 'environment_modified': False}
path = pathlib.Path('runtime') / (os.environ['SLURM_JOB_ID'] + '-' + str(os.getpid()) + '.json')
with path.open('x') as stream:
    json.dump(record, stream, indent=2)
PY
exec "$LAVA_PYTHON" replay.py "${1:?data key required}" "${2:?first or remaining required}"
