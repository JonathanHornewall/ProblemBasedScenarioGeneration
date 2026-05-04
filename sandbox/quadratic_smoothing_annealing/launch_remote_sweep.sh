#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REMOTE="${CDFL_SANDBOX_REMOTE:-gcp-big}"
REMOTE_REPO="${CDFL_SANDBOX_REMOTE_REPO:-~/ProblemBasedScenarioGeneration}"
SWEEP_ID="${CDFL_SANDBOX_SWEEP_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
REMOTE_RUNNER="sandbox/quadratic_smoothing_annealing/run_sweep_on_remote.sh"

if [ "${CDFL_SANDBOX_SYNC:-1}" = "1" ]; then
    "${HOME}/sync-julia-code.sh"
fi

exports=""
for name in "${!CDFL_SANDBOX_@}"; do
    case "${name}" in
        CDFL_SANDBOX_REMOTE|CDFL_SANDBOX_REMOTE_REPO|CDFL_SANDBOX_SYNC)
            ;;
        *)
            printf -v quoted_value '%q' "${!name}"
            exports+="export ${name}=${quoted_value}; "
            ;;
    esac
done
if [ -n "${JULIA_CMD:-}" ]; then
    printf -v quoted_julia '%q' "${JULIA_CMD}"
    exports+="export JULIA_CMD=${quoted_julia}; "
fi
printf -v quoted_sweep_id '%q' "${SWEEP_ID}"
exports+="export CDFL_SANDBOX_SWEEP_ID=${quoted_sweep_id}; "

remote_cmd="cd ${REMOTE_REPO} && ${exports} bash ${REMOTE_RUNNER}"
ssh "${REMOTE}" -- "bash -lc $(printf '%q' "${remote_cmd}")"
