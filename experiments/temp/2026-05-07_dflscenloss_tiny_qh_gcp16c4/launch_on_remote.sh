#!/usr/bin/env bash
set -euo pipefail

SANDBOX="${SANDBOX:-/home/rwl/ProblemBasedScenarioGeneration/experiments/temp/2026-05-07_dflscenloss_tiny_qh_gcp16c4}"
SOURCE_DIR="${SOURCE_DIR:-/home/rwl/ProblemBasedScenarioGeneration/src/ContextualDFL}"
RUNNER="${SANDBOX}/benchmark_dflscenloss_tiny_qh.jl"
ARTIFACT_DIR="${ARTIFACT_DIR:-${SOURCE_DIR}/ContextualDFLExperiments/experiments/baseline_benchmarks/artifacts/tiny_30ctx_5x100_seed20260505}"
OUTPUT_DIR="${OUTPUT_DIR:-${SANDBOX}/results_final_eval_once}"
LOG="${LOG:-${SANDBOX}/dflscenloss_tiny_qh_final_eval_once.log}"
PID_FILE="${PID_FILE:-${SANDBOX}/dflscenloss_tiny_qh_final_eval_once.pid}"
WORKERS="${WORKERS:-12}"
JOB_BATCH_SIZE="${JOB_BATCH_SIZE:-12}"
PROBLEMS="${PROBLEMS:-transshipment_h,transshipment_h_and_q,random_yield,resource_allocation}"
OUTPUTS="${OUTPUTS:-q,h}"
SEEDS="${SEEDS:-20260505,20260506,20260507}"
MAX_EPOCHS="${MAX_EPOCHS:-130}"
JULIA_BIN="${JULIA_BIN:-/home/rwl/.juliaup/bin/julia}"

if [[ -f "${PID_FILE}" ]]; then
    existing_pid="$(cat "${PID_FILE}")"
    if [[ -n "${existing_pid}" ]] && kill -0 "${existing_pid}" 2>/dev/null; then
        echo "already running pid=${existing_pid}"
        echo "log=${LOG}"
        exit 0
    fi
fi

mkdir -p "${OUTPUT_DIR}" "$(dirname "${LOG}")"

(
    cd "${SOURCE_DIR}"
    export CDFL_SOURCE_DIR="${SOURCE_DIR}"
    export JULIA_NUM_THREADS=1
    export OPENBLAS_NUM_THREADS=1
    export BLIS_NUM_THREADS=1
    export MKL_NUM_THREADS=1
    export OMP_NUM_THREADS=1
    exec "${JULIA_BIN}" --project=ContextualDFLExperiments "${RUNNER}" \
        --artifact-dir "${ARTIFACT_DIR}" \
        --output-dir "${OUTPUT_DIR}" \
        --local-workers "${WORKERS}" \
        --job-batch-size "${JOB_BATCH_SIZE}" \
        --problems "${PROBLEMS}" \
        --outputs "${OUTPUTS}" \
        --seeds "${SEEDS}" \
        --max-epochs "${MAX_EPOCHS}"
) > "${LOG}" 2>&1 &

pid="$!"
echo "${pid}" > "${PID_FILE}"
echo "started pid=${pid}"
echo "log=${LOG}"
echo "output_dir=${OUTPUT_DIR}"
