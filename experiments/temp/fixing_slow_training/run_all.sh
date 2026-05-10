#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

export JULIA_DEPOT_PATH="${PWD}/experiments/temp/fixing_slow_training/julia_depot:${JULIA_DEPOT_PATH:-${HOME}/.julia}"
export JULIA_PKG_PRECOMPILE_AUTO=0
export FIX_SLOW_N_TRAIN="${FIX_SLOW_N_TRAIN:-3}"
export FIX_SLOW_EPOCHS="${FIX_SLOW_EPOCHS:-1}"
export FIX_SLOW_BATCHSIZE="${FIX_SLOW_BATCHSIZE:-1}"
export FIX_SLOW_DISPLAY="${FIX_SLOW_DISPLAY:-0}"

julia --project=experiments/temp/fixing_slow_training -e 'import Pkg; Pkg.instantiate()'

julia --project=experiments/temp/fixing_slow_training experiments/temp/fixing_slow_training/old_benchmark.jl \
    2>&1 | tee experiments/temp/fixing_slow_training/old_stdout.log

julia --project=experiments/temp/fixing_slow_training experiments/temp/fixing_slow_training/current_benchmark.jl \
    2>&1 | tee experiments/temp/fixing_slow_training/current_stdout.log

FIX_SLOW_DISPLAY=1 FIX_SLOW_DISPLAY_SMOOTH=1 \
julia --project=experiments/temp/fixing_slow_training experiments/temp/fixing_slow_training/current_benchmark.jl \
    2>&1 | tee experiments/temp/fixing_slow_training/current_smooth_stdout.log
