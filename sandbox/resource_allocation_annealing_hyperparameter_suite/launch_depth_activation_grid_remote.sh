#!/usr/bin/env bash
set -euo pipefail

HOST="${HOST:-gcp-big}"
REMOTE_REPO="${REMOTE_REPO:-/home/rwl/ProblemBasedScenarioGeneration}"
REMOTE_JULIA="${REMOTE_JULIA:-/home/rwl/.juliaup/bin/julia}"
JOBS="${JOBS:-1}"

REMOTE_SUITE="$REMOTE_REPO/sandbox/resource_allocation_annealing_hyperparameter_suite"
REMOTE_PROJECT="$REMOTE_REPO/src/ContextualDFL/ContextualDFLTraining"
REMOTE_LOG="$REMOTE_SUITE/logs/depth_activation_grid_remote.log"
REMOTE_PID="$REMOTE_SUITE/logs/depth_activation_grid_remote.pid"

ssh -o BatchMode=yes "$HOST" -- bash -lc "
    mkdir -p '$REMOTE_SUITE/logs'
    cd '$REMOTE_REPO'
    nohup '$REMOTE_JULIA' \
        --project='$REMOTE_PROJECT' \
        '$REMOTE_SUITE/run_depth_activation_grid.jl' \
        --local-executor \
        --jobs '$JOBS' \
        >> '$REMOTE_LOG' 2>&1 < /dev/null &
    echo \$! > '$REMOTE_PID'
"

echo "Launched depth x activation grid on $HOST"
echo "Remote PID file: $REMOTE_PID"
echo "Remote log: $REMOTE_LOG"
