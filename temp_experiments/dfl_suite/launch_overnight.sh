#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PROJECT_DIR="${REPO_ROOT}/src/ContextualDFL/ContextualDFLTraining"
JOBS="${DFL_SUITE_JOBS:-14}"
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_DIR}"

cd "${REPO_ROOT}"
exec caffeinate -dimsu julia --project="${PROJECT_DIR}" \
  "${SCRIPT_DIR}/run_suite.jl" --jobs "${JOBS}" \
  > "${LOG_DIR}/controller_full.log" 2> "${LOG_DIR}/controller_full.err"
