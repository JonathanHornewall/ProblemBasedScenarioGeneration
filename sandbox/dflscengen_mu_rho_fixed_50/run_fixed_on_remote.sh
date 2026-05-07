#!/usr/bin/env bash
set -u -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PROJECT_DIR="${REPO_ROOT}/src/ContextualDFL/ContextualDFLExperiments"

SWEEP_ID="${CDFL_SANDBOX_SWEEP_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUTPUT_ROOT="${CDFL_SANDBOX_OUTPUT_ROOT:-${SCRIPT_DIR}/results/${SWEEP_ID}}"
GROUND_TRUTH_DIR="${CDFL_SANDBOX_GROUND_TRUTH_DIR:-${SCRIPT_DIR}/ground_truth}"
MAX_PARALLEL="${CDFL_SANDBOX_MAX_PARALLEL:-4}"
THREADS_PER_RUN="${CDFL_SANDBOX_THREADS_PER_RUN:-1}"
BLAS_THREADS_PER_RUN="${CDFL_SANDBOX_BLAS_THREADS_PER_RUN:-1}"
BASE_SEED="${CDFL_SANDBOX_BASE_SEED:-20260507}"
JULIA_CMD="${JULIA_CMD:-julia}"

mkdir -p "${OUTPUT_ROOT}/logs"
mkdir -p "${GROUND_TRUTH_DIR}"

MANIFEST_PATH="${OUTPUT_ROOT}/job_manifest.tsv"
STATUS_PATH="${OUTPUT_ROOT}/job_status.tsv"
SUMMARY_PATH="${OUTPUT_ROOT}/sweep_summary.csv"
COMMANDS_PATH="${OUTPUT_ROOT}/commands.tsv"

printf "job_id\tkind\tvalue\tseed\toutput_dir\tlog_path\n" > "${MANIFEST_PATH}"
printf "job_id\tkind\tvalue\tseed\texit_code\toutput_dir\tlog_path\n" > "${STATUS_PATH}"
printf "job_id\tcommand\n" > "${COMMANDS_PATH}"

echo "Preparing shared ground-truth train/test data..."
JULIA_NUM_THREADS="${THREADS_PER_RUN}" \
CDFL_SANDBOX_SEED="${BASE_SEED}" \
CDFL_SANDBOX_GROUND_TRUTH_DIR="${GROUND_TRUTH_DIR}" \
CDFL_SANDBOX_ONLY_PREPARE_DATA=1 \
"${JULIA_CMD}" --project="${PROJECT_DIR}" "${SCRIPT_DIR}/annealing_fixed_sandbox.jl" \
    > "${OUTPUT_ROOT}/logs/prepare_ground_truth.log" 2>&1 || {
        echo "Failed to prepare ground-truth data; inspect ${OUTPUT_ROOT}/logs/prepare_ground_truth.log." >&2
        exit 1
    }

sanitize_value() {
    local value="$1"
    value="${value//./p}"
    value="${value//-/m}"
    echo "${value}"
}

launch_job() {
    local job_id="$1"
    local kind="$2"
    local value="$3"
    local seed="$4"
    local label
    label="$(sanitize_value "${value}")"
    local run_dir="${OUTPUT_ROOT}/${kind}_${label}"
    local log_path="${OUTPUT_ROOT}/logs/${job_id}.log"
    mkdir -p "${run_dir}"

    printf "%s\t%s\t%s\t%s\t%s\t%s\n" \
        "${job_id}" "${kind}" "${value}" "${seed}" "${run_dir}" "${log_path}" >> "${MANIFEST_PATH}"

    local mu="0.0"
    local rho="0.0"
    if [ "${kind}" = "mu" ]; then
        mu="${value}"
    elif [ "${kind}" = "rho" ]; then
        rho="${value}"
    else
        echo "Unknown job kind: ${kind}" >&2
        exit 2
    fi

    local command="JULIA_NUM_THREADS=${THREADS_PER_RUN} OPENBLAS_NUM_THREADS=${BLAS_THREADS_PER_RUN} OMP_NUM_THREADS=${BLAS_THREADS_PER_RUN} CDFL_SANDBOX_SEED=${seed} CDFL_SANDBOX_PARAM_LIST=${mu} CDFL_SANDBOX_MU_REF=${CDFL_SANDBOX_MU_REF:-0.0} CDFL_SANDBOX_RHO=${rho} CDFL_SANDBOX_RHO_REF=${rho} CDFL_SANDBOX_POLICY_INFERENCE_RHO=${rho} CDFL_SANDBOX_OPTIMALITY_RHO=${CDFL_SANDBOX_OPTIMALITY_RHO:-0.0} CDFL_SANDBOX_FIRST_STAGE_EPOCHS=${CDFL_SANDBOX_FIRST_STAGE_EPOCHS:-50} CDFL_SANDBOX_DEFAULT_EPOCHS=${CDFL_SANDBOX_DEFAULT_EPOCHS:-50} CDFL_SANDBOX_RUN_FINAL_STAGE=0 CDFL_SANDBOX_DISPLAY_ITERATIONS=${CDFL_SANDBOX_DISPLAY_ITERATIONS:-0} CDFL_SANDBOX_DISPLAY_SMOOTH=${CDFL_SANDBOX_DISPLAY_SMOOTH:-0} CDFL_SANDBOX_GROUND_TRUTH_DIR=${GROUND_TRUTH_DIR} CDFL_SANDBOX_OUTPUT_DIR=${run_dir} ${JULIA_CMD} --project=${PROJECT_DIR} ${SCRIPT_DIR}/annealing_fixed_sandbox.jl"
    printf "%s\t%s\n" "${job_id}" "${command}" >> "${COMMANDS_PATH}"

    (
        set +e
        JULIA_NUM_THREADS="${THREADS_PER_RUN}" \
        OPENBLAS_NUM_THREADS="${BLAS_THREADS_PER_RUN}" \
        OMP_NUM_THREADS="${BLAS_THREADS_PER_RUN}" \
        CDFL_SANDBOX_SEED="${seed}" \
        CDFL_SANDBOX_PARAM_LIST="${mu}" \
        CDFL_SANDBOX_MU_REF="${CDFL_SANDBOX_MU_REF:-0.0}" \
        CDFL_SANDBOX_RHO="${rho}" \
        CDFL_SANDBOX_RHO_REF="${rho}" \
        CDFL_SANDBOX_POLICY_INFERENCE_RHO="${rho}" \
        CDFL_SANDBOX_OPTIMALITY_RHO="${CDFL_SANDBOX_OPTIMALITY_RHO:-0.0}" \
        CDFL_SANDBOX_FIRST_STAGE_EPOCHS="${CDFL_SANDBOX_FIRST_STAGE_EPOCHS:-50}" \
        CDFL_SANDBOX_DEFAULT_EPOCHS="${CDFL_SANDBOX_DEFAULT_EPOCHS:-50}" \
        CDFL_SANDBOX_RUN_FINAL_STAGE=0 \
        CDFL_SANDBOX_DISPLAY_ITERATIONS="${CDFL_SANDBOX_DISPLAY_ITERATIONS:-0}" \
        CDFL_SANDBOX_DISPLAY_SMOOTH="${CDFL_SANDBOX_DISPLAY_SMOOTH:-0}" \
        CDFL_SANDBOX_GROUND_TRUTH_DIR="${GROUND_TRUTH_DIR}" \
        CDFL_SANDBOX_OUTPUT_DIR="${run_dir}" \
        "${JULIA_CMD}" --project="${PROJECT_DIR}" "${SCRIPT_DIR}/annealing_fixed_sandbox.jl" > "${log_path}" 2>&1
        exit_code="$?"
        echo "${exit_code}" > "${run_dir}/exit_code.txt"
        printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
            "${job_id}" "${kind}" "${value}" "${seed}" "${exit_code}" "${run_dir}" "${log_path}" >> "${STATUS_PATH}"
        exit "${exit_code}"
    ) &
}

wait_for_slot() {
    while [ "$(jobs -rp | wc -l)" -ge "${MAX_PARALLEL}" ]; do
        wait -n || true
    done
}

job_index=0

for mu in ${CDFL_SANDBOX_MUS:-0.001 0.0001}; do
    launch_job "mu_$(sanitize_value "${mu}")" "mu" "${mu}" "${BASE_SEED}"
    wait_for_slot
done

for rho in ${CDFL_SANDBOX_RHOS:-0.001 0.0001}; do
    launch_job "rho_$(sanitize_value "${rho}")" "rho" "${rho}" "${BASE_SEED}"
    wait_for_slot
done

while [ "$(jobs -rp | wc -l)" -gt 0 ]; do
    wait -n || true
done

first_summary=1
while IFS=$'\t' read -r _job_id _kind _value _seed run_dir _log_path; do
    [ "${_job_id}" = "job_id" ] && continue
    run_summary="${run_dir}/run_summary.csv"
    if [ -f "${run_summary}" ]; then
        if [ "${first_summary}" -eq 1 ]; then
            cat "${run_summary}" > "${SUMMARY_PATH}"
            first_summary=0
        else
            tail -n +2 "${run_summary}" >> "${SUMMARY_PATH}"
        fi
    fi
done < "${MANIFEST_PATH}"

echo "Sandbox artifacts: ${OUTPUT_ROOT}"
echo "Ground truth data: ${GROUND_TRUTH_DIR}"
echo "Job status: ${STATUS_PATH}"
if [ -f "${SUMMARY_PATH}" ]; then
    echo "Sweep summary: ${SUMMARY_PATH}"
else
    echo "No sweep summary was produced; inspect logs under ${OUTPUT_ROOT}/logs."
fi

if awk -F '\t' 'NR > 1 && $5 != 0 { failed = 1 } END { exit failed ? 1 : 0 }' "${STATUS_PATH}"; then
    exit 0
else
    echo "At least one sandbox job failed; inspect ${STATUS_PATH}."
    exit 1
fi
