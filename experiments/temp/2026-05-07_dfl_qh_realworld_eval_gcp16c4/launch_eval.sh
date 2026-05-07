#!/usr/bin/env bash
set -euo pipefail

SANDBOX="/home/rwl/ProblemBasedScenarioGeneration/experiments/temp/2026-05-07_dfl_qh_realworld_eval_gcp16c4"
SOURCE_DIR="/home/rwl/ProblemBasedScenarioGeneration/src/ContextualDFL"
RUNNER="${SANDBOX}/evaluate_realworld_models.jl"
MODEL_ROOT="${SANDBOX}/trained_models"
ARTIFACT_DIR="${SOURCE_DIR}/ContextualDFLExperiments/experiments/baseline_benchmarks/artifacts/realworld_30ctx_20x1000_20260507"
OUTPUT_DIR="${SANDBOX}/results"
LOG="${SANDBOX}/eval.log"
PID_FILE="${SANDBOX}/eval.pid"

if [[ -f "${PID_FILE}" ]]; then
    existing_pid="$(cat "${PID_FILE}")"
    if [[ -n "${existing_pid}" ]] && kill -0 "${existing_pid}" 2>/dev/null; then
        echo "already running pid=${existing_pid}"
        echo "log=${LOG}"
        exit 0
    fi
fi

mkdir -p "${OUTPUT_DIR}"

(
    cd "${SOURCE_DIR}"
    export JULIA_NUM_THREADS=1
    export OPENBLAS_NUM_THREADS=1
    export BLIS_NUM_THREADS=1
    export MKL_NUM_THREADS=1
    export OMP_NUM_THREADS=1
    exec /home/rwl/.juliaup/bin/julia --project=ContextualDFLExperiments "${RUNNER}" \
        --model-root "${MODEL_ROOT}" \
        --artifact-dir "${ARTIFACT_DIR}" \
        --output-dir "${OUTPUT_DIR}" \
        --local-workers 4 \
        --job-batch-size 4 \
        --problems transshipment_h_and_q,random_yield,resource_allocation \
        --outputs q,h \
        --repeats 1:2
) > "${LOG}" 2>&1 &

pid="$!"
echo "${pid}" > "${PID_FILE}"
echo "started pid=${pid}"
echo "log=${LOG}"
