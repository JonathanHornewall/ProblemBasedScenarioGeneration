#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
REPO_ROOT="$(cd -- "${PROJECT_DIR}/.." && pwd)"

JULIA="${JULIA:-julia}"
RUNNER="${SCRIPT_DIR}/run_baselines.jl"
AGGREGATOR="${SCRIPT_DIR}/aggregate_tiny_full_baselines.jl"
ARTIFACT_DIR="${ARTIFACT_DIR:-${SCRIPT_DIR}/artifacts/tiny_30ctx_5x100_seed20260505}"
RESULT_DIR="${RESULT_DIR:-${SCRIPT_DIR}/results/tiny_30ctx_5x100_full_baselines_20260507}"
QCONV_RESULT_DIR="${QCONV_RESULT_DIR:-${SCRIPT_DIR}/results/tiny_30ctx_5x100_qconv_and_ra_decoders_20260507}"
CACHE_DIR="${CACHE_DIR:-${SCRIPT_DIR}/cache_tiny_30ctx_5x100_seed20260505}"
LOCAL_WORKERS="${LOCAL_WORKERS:-50}"
REPLICA_SEEDS="${REPLICA_SEEDS:-20260505,20260506,20260507}"
HOSTS="${HOSTS:-ibm-96c-1 ibm-96c-2}"

usage() {
  cat <<USAGE
Usage: $(basename "$0") <command>

Commands:
  sync-code            rsync this checkout to HOSTS
  generate-artifacts   create tiny 30ctx 5x100 artifacts
  sync-artifacts       rsync tiny artifacts to HOSTS
  sync-results         rsync tiny result directories back from HOSTS
  validate-artifacts   validate all tiny artifacts
  smoke                run the three required smoke rows
  full-ibm96c1         run resource/shipment/newsvendor full grid
  full-ibm96c2         run transshipment/random-yield full grid
  qconv-ibm96c1        run converted-q random-yield rows
  qconv-ibm96c2        run resource-allocation cost-decoder rows
  aggregate            merge and validate final result CSVs

Environment:
  JULIA, ARTIFACT_DIR, RESULT_DIR, QCONV_RESULT_DIR, CACHE_DIR, LOCAL_WORKERS, REPLICA_SEEDS, HOSTS
USAGE
}

julia_runner() {
  "$JULIA" --project="$PROJECT_DIR" "$RUNNER" "$@"
}

generate_artifacts() {
  julia_runner \
    --profile tiny \
    --export-data-artifacts "$ARTIFACT_DIR" \
    --cache-dir "$CACHE_DIR" \
    --local-workers "$LOCAL_WORKERS"
}

validate_artifacts() {
  julia_runner \
    --validate-tiny-data-artifacts \
    --tiny-artifact-dir "$ARTIFACT_DIR" \
    --cache-dir "$CACHE_DIR"
}

smoke() {
  mkdir -p "$RESULT_DIR/smoke"
  julia_runner \
    --use-tiny-data-artifacts \
    --tiny-artifact-dir "$ARTIFACT_DIR" \
    --cache-dir "$CACHE_DIR" \
    --output-dir "$RESULT_DIR/smoke/shipment_er_saa" \
    --benchmarks shipment_planning \
    --policies er_saa \
    --replica-seeds 20260505
  julia_runner \
    --use-tiny-data-artifacts \
    --tiny-artifact-dir "$ARTIFACT_DIR" \
    --cache-dir "$CACHE_DIR" \
    --output-dir "$RESULT_DIR/smoke/newsvendor_nn" \
    --benchmarks unreliable_newsvendor \
    --policies nn \
    --replica-seeds 20260505
  julia_runner \
    --use-tiny-data-artifacts \
    --tiny-artifact-dir "$ARTIFACT_DIR" \
    --cache-dir "$CACHE_DIR" \
    --output-dir "$RESULT_DIR/smoke/newsvendor_dfl_rho001" \
    --benchmarks unreliable_newsvendor \
    --policies dfl_mu0_rho0.01 \
    --replica-seeds 20260505
}

full_ibm96c1() {
  julia_runner \
    --tiny-full-baselines \
    --tiny-artifact-dir "$ARTIFACT_DIR" \
    --cache-dir "$CACHE_DIR" \
    --output-dir "$RESULT_DIR/ibm-96c-1" \
    --benchmarks resource_allocation,shipment_planning,unreliable_newsvendor \
    --local-workers "$LOCAL_WORKERS" \
    --replica-seeds "$REPLICA_SEEDS"
}

full_ibm96c2() {
  julia_runner \
    --tiny-full-baselines \
    --tiny-artifact-dir "$ARTIFACT_DIR" \
    --cache-dir "$CACHE_DIR" \
    --output-dir "$RESULT_DIR/ibm-96c-2" \
    --benchmarks transshipment_q,transshipment_h,transshipment_h_and_q,random_yield \
    --local-workers "$LOCAL_WORKERS" \
    --replica-seeds "$REPLICA_SEEDS"
}

qconv_ibm96c1() {
  mkdir -p "$QCONV_RESULT_DIR/ibm-96c-1"
  CDFL_BASELINE_QCONV_EPOCHS="${CDFL_BASELINE_QCONV_EPOCHS:-50}" \
  julia_runner \
    --use-tiny-data-artifacts \
    --tiny-artifact-dir "$ARTIFACT_DIR" \
    --cache-dir "$CACHE_DIR" \
    --output-dir "$QCONV_RESULT_DIR/ibm-96c-1" \
    --benchmarks random_yield \
    --policies spoplus_qconv,dfl_qconv \
    --local-workers "$LOCAL_WORKERS" \
    --replica-seeds "${QCONV_REPLICA_SEEDS:-20260505,20260506}"
}

qconv_ibm96c2() {
  mkdir -p "$QCONV_RESULT_DIR/ibm-96c-2"
  CDFL_BASELINE_RA_DECODER_EPOCHS="${CDFL_BASELINE_RA_DECODER_EPOCHS:-50}" \
  julia_runner \
    --use-tiny-data-artifacts \
    --tiny-artifact-dir "$ARTIFACT_DIR" \
    --cache-dir "$CACHE_DIR" \
    --output-dir "$QCONV_RESULT_DIR/ibm-96c-2" \
    --benchmarks resource_allocation \
    --policies dfl_ra_physical_cost,dfl_ra_full_cost,dfl_ra_economic_cost \
    --local-workers "$LOCAL_WORKERS" \
    --replica-seeds "${QCONV_REPLICA_SEEDS:-20260505,20260506}"
}

sync_code() {
  for host in $HOSTS; do
    rsync -az \
      --exclude '/ContextualDFLExperiments/experiments/baseline_benchmarks/results/' \
      --exclude '/ContextualDFLExperiments/experiments/baseline_benchmarks/cache*/' \
      "$REPO_ROOT/" "$host:$REPO_ROOT/"
  done
}

sync_artifacts() {
  for host in $HOSTS; do
    ssh "$host" "mkdir -p '$ARTIFACT_DIR'"
    rsync -az "$ARTIFACT_DIR/" "$host:$ARTIFACT_DIR/"
  done
}

sync_results() {
  mkdir -p "$RESULT_DIR"
  for host in $HOSTS; do
    rsync -az "$host:$RESULT_DIR/" "$RESULT_DIR/"
  done
}

aggregate() {
  "$JULIA" "$AGGREGATOR" --input "$RESULT_DIR" --output-dir "$RESULT_DIR"
}

command="${1:-}"
case "$command" in
  sync-code) sync_code ;;
  generate-artifacts) generate_artifacts ;;
  sync-artifacts) sync_artifacts ;;
  sync-results) sync_results ;;
  validate-artifacts) validate_artifacts ;;
  smoke) smoke ;;
  full-ibm96c1) full_ibm96c1 ;;
  full-ibm96c2) full_ibm96c2 ;;
  qconv-ibm96c1) qconv_ibm96c1 ;;
  qconv-ibm96c2) qconv_ibm96c2 ;;
  aggregate) aggregate ;;
  ""|help|-h|--help) usage ;;
  *) usage >&2; exit 2 ;;
esac
