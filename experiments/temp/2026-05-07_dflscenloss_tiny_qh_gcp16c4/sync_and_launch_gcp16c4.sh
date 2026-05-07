#!/usr/bin/env bash
set -euo pipefail

REMOTE="${REMOTE:-gcp-16c-4}"
LOCAL_SOURCE_DIR="${LOCAL_SOURCE_DIR:-/home/rwl/ProblemBasedScenarioGeneration/src/ContextualDFL}"
REMOTE_SOURCE_DIR="${REMOTE_SOURCE_DIR:-/home/rwl/ProblemBasedScenarioGeneration/src/ContextualDFL}"
LOCAL_SANDBOX="${LOCAL_SANDBOX:-/home/rwl/ProblemBasedScenarioGeneration/experiments/temp/2026-05-07_dflscenloss_tiny_qh_gcp16c4}"
REMOTE_SANDBOX="${REMOTE_SANDBOX:-/home/rwl/ProblemBasedScenarioGeneration/experiments/temp/2026-05-07_dflscenloss_tiny_qh_gcp16c4}"
TINY_ARTIFACT_DIR="${LOCAL_SOURCE_DIR}/ContextualDFLExperiments/experiments/baseline_benchmarks/artifacts/tiny_30ctx_5x100_seed20260505"

test -d "${LOCAL_SOURCE_DIR}" || {
    echo "missing local source dir: ${LOCAL_SOURCE_DIR}" >&2
    exit 1
}
test -d "${TINY_ARTIFACT_DIR}" || {
    echo "missing tiny artifact dir: ${TINY_ARTIFACT_DIR}" >&2
    exit 1
}
test -f "${LOCAL_SANDBOX}/benchmark_dflscenloss_tiny_qh.jl" || {
    echo "missing benchmark runner in ${LOCAL_SANDBOX}" >&2
    exit 1
}

ssh "${REMOTE}" "mkdir -p '${REMOTE_SOURCE_DIR}' '${REMOTE_SANDBOX}'"

rsync -az "${LOCAL_SOURCE_DIR}/ContextualDFL/" \
    "${REMOTE}:${REMOTE_SOURCE_DIR}/ContextualDFL/"

rsync -az \
    --exclude 'experiments/baseline_benchmarks/results/' \
    --exclude 'experiments/baseline_benchmarks/cache/' \
    --exclude 'experiments/baseline_benchmarks/**/*.log' \
    --exclude 'experiments/baseline_benchmarks/**/*.pid' \
    "${LOCAL_SOURCE_DIR}/ContextualDFLExperiments/" \
    "${REMOTE}:${REMOTE_SOURCE_DIR}/ContextualDFLExperiments/"

rsync -az "${LOCAL_SANDBOX}/" "${REMOTE}:${REMOTE_SANDBOX}/"

ssh "${REMOTE}" "chmod +x '${REMOTE_SANDBOX}/launch_on_remote.sh' && SOURCE_DIR='${REMOTE_SOURCE_DIR}' SANDBOX='${REMOTE_SANDBOX}' bash '${REMOTE_SANDBOX}/launch_on_remote.sh'"
