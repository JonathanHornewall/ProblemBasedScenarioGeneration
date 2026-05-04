#!/usr/bin/env bash
set -u -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PROJECT_DIR="${REPO_ROOT}/src/ContextualDFL/ContextualDFLExperiments"
ANNEALING_DIR="${PROJECT_DIR}/experiments/resource_allocation_annealing"

SWEEP_ID="${CDFL_SANDBOX_SWEEP_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUTPUT_ROOT="${CDFL_SANDBOX_OUTPUT_ROOT:-${ANNEALING_DIR}/results/quadratic_smoothing_sandbox/${SWEEP_ID}}"
MAX_PARALLEL="${CDFL_SANDBOX_MAX_PARALLEL:-4}"
THREADS_PER_RUN="${CDFL_SANDBOX_THREADS_PER_RUN:-1}"
BASE_SEED="${CDFL_SANDBOX_BASE_SEED:-20260504}"
JULIA_CMD="${JULIA_CMD:-julia}"

mkdir -p "${OUTPUT_ROOT}/logs"

MANIFEST_PATH="${OUTPUT_ROOT}/job_manifest.tsv"
STATUS_PATH="${OUTPUT_ROOT}/job_status.tsv"
SUMMARY_PATH="${OUTPUT_ROOT}/sweep_summary.csv"
COMMANDS_PATH="${OUTPUT_ROOT}/commands.tsv"

printf "job_id\trho\treplicate\tseed\toutput_dir\tlog_path\n" > "${MANIFEST_PATH}"
printf "job_id\trho\treplicate\tseed\texit_code\toutput_dir\tlog_path\n" > "${STATUS_PATH}"
printf "job_id\tcommand\n" > "${COMMANDS_PATH}"

sanitize_rho() {
    local value="$1"
    value="${value//./p}"
    value="${value//-/m}"
    echo "${value}"
}

launch_job() {
    local job_id="$1"
    local rho="$2"
    local replicate="$3"
    local seed="$4"
    local rho_label
    rho_label="$(sanitize_rho "${rho}")"
    local run_dir="${OUTPUT_ROOT}/rho_${rho_label}/rep_${replicate}"
    local log_path="${OUTPUT_ROOT}/logs/${job_id}.log"
    mkdir -p "${run_dir}"

    printf "%s\t%s\t%s\t%s\t%s\t%s\n" \
        "${job_id}" "${rho}" "${replicate}" "${seed}" "${run_dir}" "${log_path}" >> "${MANIFEST_PATH}"

    local command="CDFL_SANDBOX_SEED=${seed} CDFL_SANDBOX_RHO=${rho} CDFL_SANDBOX_RHO_REF=${rho} CDFL_SANDBOX_POLICY_INFERENCE_RHO=${rho} CDFL_SANDBOX_OPTIMALITY_RHO=${CDFL_SANDBOX_OPTIMALITY_RHO:-0.0} CDFL_SANDBOX_OUTPUT_DIR=${run_dir} ${JULIA_CMD} --project=${PROJECT_DIR} ${SCRIPT_DIR}/annealing_rho_sandbox.jl"
    printf "%s\t%s\n" "${job_id}" "${command}" >> "${COMMANDS_PATH}"

    (
        set +e
        JULIA_NUM_THREADS="${THREADS_PER_RUN}" \
        CDFL_SANDBOX_SEED="${seed}" \
        CDFL_SANDBOX_RHO="${rho}" \
        CDFL_SANDBOX_RHO_REF="${rho}" \
        CDFL_SANDBOX_POLICY_INFERENCE_RHO="${rho}" \
        CDFL_SANDBOX_OPTIMALITY_RHO="${CDFL_SANDBOX_OPTIMALITY_RHO:-0.0}" \
        CDFL_SANDBOX_OUTPUT_DIR="${run_dir}" \
        "${JULIA_CMD}" --project="${PROJECT_DIR}" "${SCRIPT_DIR}/annealing_rho_sandbox.jl" > "${log_path}" 2>&1
        exit_code="$?"
        echo "${exit_code}" > "${run_dir}/exit_code.txt"
        printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
            "${job_id}" "${rho}" "${replicate}" "${seed}" "${exit_code}" "${run_dir}" "${log_path}" >> "${STATUS_PATH}"
        exit "${exit_code}"
    ) &
}

wait_for_slot() {
    while [ "$(jobs -rp | wc -l)" -ge "${MAX_PARALLEL}" ]; do
        wait -n || true
    done
}

job_index=0

if [ -n "${CDFL_SANDBOX_BASELINE_RHO-0.0}" ]; then
    job_index=$((job_index + 1))
    baseline_rho="${CDFL_SANDBOX_BASELINE_RHO-0.0}"
    launch_job "rho_$(sanitize_rho "${baseline_rho}")_rep_1" "${baseline_rho}" "1" "$((BASE_SEED + job_index))"
    wait_for_slot
fi

for rho in ${CDFL_SANDBOX_RHOS:-0.01 0.05 0.1}; do
    for replicate in $(seq 1 "${CDFL_SANDBOX_REPLICATES:-3}"); do
        job_index=$((job_index + 1))
        launch_job "rho_$(sanitize_rho "${rho}")_rep_${replicate}" \
            "${rho}" "${replicate}" "$((BASE_SEED + job_index))"
        wait_for_slot
    done
done

while [ "$(jobs -rp | wc -l)" -gt 0 ]; do
    wait -n || true
done

first_summary=1
while IFS=$'\t' read -r _job_id _rho _replicate _seed run_dir _log_path; do
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

echo "Sandbox sweep artifacts: ${OUTPUT_ROOT}"
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
