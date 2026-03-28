#!/usr/bin/env bash
#
# Grid Search: Train models across 3 problems x 16 architecture configs = 48 runs
#
# Organized into 6 work units (2 batches per problem, 8 configs each).
# Each batch runs in a single Julia process to avoid repeated compilation.
#
# Usage:
#   ./run_grid_search.sh              # Run all 6 units sequentially
#   ./run_grid_search.sh --parallel   # Run all 6 units in parallel (needs ~6x memory)
#   ./run_grid_search.sh --unit 1     # Run only unit 1
#
set -euo pipefail
cd "$(dirname "$0")"

PROJECT="../src/ProblemBasedScenarioGeneration"

# --- Config strings (hidden_dim:n_layers:activation) ---
# Configs 1-8:  hidden_dim 64 (all combos) + 128 with 2 layers
BATCH_A="64:2:relu,64:2:tanh,64:3:relu,64:3:tanh,64:4:relu,64:4:tanh,128:2:relu,128:2:tanh"
# Configs 9-16: 128 with 3-4 layers + 256 with 2-3 layers (pruned 256x4)
BATCH_B="128:3:relu,128:3:tanh,128:4:relu,128:4:tanh,256:2:relu,256:2:tanh,256:3:relu,256:3:tanh"

# --- Per-problem training parameters ---
# newsvendor is fast (2-dim output), resource_allocation & shipment_planning are much slower
# due to higher-dimensional LP solves
NEWSVENDOR_EPOCHS=50
NEWSVENDOR_N_TRAIN=200
NEWSVENDOR_N_TEST=50

RA_EPOCHS=10
RA_N_TRAIN=50
RA_N_TEST=20

SP_EPOCHS=10
SP_N_TRAIN=50
SP_N_TEST=20

# --- Logging ---
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

run_unit() {
    local unit_num=$1
    local problem=$2
    local batch_name=$3
    local configs=$4
    local epochs=$5
    local n_train=$6
    local n_test=$7

    local logfile="$LOG_DIR/unit${unit_num}_${problem}_${batch_name}.log"

    echo "[Unit $unit_num] $problem / $batch_name — starting (epochs=$epochs, n_train=$n_train, log: $logfile)"
    julia --project="$PROJECT" grid_worker.jl \
        --problem "$problem" \
        --output-dir "$problem/$batch_name" \
        --configs "$configs" \
        --epochs "$epochs" \
        --n-train "$n_train" \
        --n-test "$n_test" \
        2>&1 | tee "$logfile"
    echo "[Unit $unit_num] $problem / $batch_name — done"
}

# --- Parse arguments ---
PARALLEL=false
ONLY_UNIT=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --parallel) PARALLEL=true; shift ;;
        --unit) ONLY_UNIT="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# --- Define units ---
# Unit 1: newsvendor batch_a (configs 1-8)
# Unit 2: newsvendor batch_b (configs 9-16)
# Unit 3: resource_allocation batch_a
# Unit 4: resource_allocation batch_b
# Unit 5: shipment_planning batch_a
# Unit 6: shipment_planning batch_b

UNIT_PROBLEMS=("newsvendor" "newsvendor" "resource_allocation" "resource_allocation" "shipment_planning" "shipment_planning")
UNIT_BATCHES=("batch_a" "batch_b" "batch_a" "batch_b" "batch_a" "batch_b")
UNIT_CONFIGS=("$BATCH_A" "$BATCH_B" "$BATCH_A" "$BATCH_B" "$BATCH_A" "$BATCH_B")
UNIT_EPOCHS=("$NEWSVENDOR_EPOCHS" "$NEWSVENDOR_EPOCHS" "$RA_EPOCHS" "$RA_EPOCHS" "$SP_EPOCHS" "$SP_EPOCHS")
UNIT_N_TRAIN=("$NEWSVENDOR_N_TRAIN" "$NEWSVENDOR_N_TRAIN" "$RA_N_TRAIN" "$RA_N_TRAIN" "$SP_N_TRAIN" "$SP_N_TRAIN")
UNIT_N_TEST=("$NEWSVENDOR_N_TEST" "$NEWSVENDOR_N_TEST" "$RA_N_TEST" "$RA_N_TEST" "$SP_N_TEST" "$SP_N_TEST")

echo "=========================================="
echo "Grid Search: 3 problems x 16 configs = 48 runs"
echo "=========================================="
echo "newsvendor:          epochs=$NEWSVENDOR_EPOCHS, n_train=$NEWSVENDOR_N_TRAIN"
echo "resource_allocation: epochs=$RA_EPOCHS, n_train=$RA_N_TRAIN"
echo "shipment_planning:   epochs=$SP_EPOCHS, n_train=$SP_N_TRAIN"
echo "LR: 1e-3 | Seed: 42"
echo "Hidden dims: 64, 128, 256"
echo "Layers: 2, 3, 4"
echo "Activations: relu, tanh"
echo "(Pruned: 256x4 configs)"
echo "=========================================="
date
echo ""

if [[ -n "$ONLY_UNIT" ]]; then
    # Run a single unit
    idx=$((ONLY_UNIT - 1))
    run_unit "$ONLY_UNIT" "${UNIT_PROBLEMS[$idx]}" "${UNIT_BATCHES[$idx]}" "${UNIT_CONFIGS[$idx]}" \
        "${UNIT_EPOCHS[$idx]}" "${UNIT_N_TRAIN[$idx]}" "${UNIT_N_TEST[$idx]}"

elif $PARALLEL; then
    # Run all 6 units in parallel
    PIDS=()
    for i in $(seq 0 5); do
        unit_num=$((i + 1))
        run_unit "$unit_num" "${UNIT_PROBLEMS[$i]}" "${UNIT_BATCHES[$i]}" "${UNIT_CONFIGS[$i]}" \
            "${UNIT_EPOCHS[$i]}" "${UNIT_N_TRAIN[$i]}" "${UNIT_N_TEST[$i]}" &
        PIDS+=($!)
    done

    echo "Launched ${#PIDS[@]} workers. Waiting..."
    FAILED=0
    for pid in "${PIDS[@]}"; do
        if ! wait "$pid"; then
            FAILED=$((FAILED + 1))
        fi
    done

    if [[ $FAILED -gt 0 ]]; then
        echo "WARNING: $FAILED unit(s) failed!"
    fi

else
    # Run all 6 units sequentially
    for i in $(seq 0 5); do
        unit_num=$((i + 1))
        run_unit "$unit_num" "${UNIT_PROBLEMS[$i]}" "${UNIT_BATCHES[$i]}" "${UNIT_CONFIGS[$i]}" \
            "${UNIT_EPOCHS[$i]}" "${UNIT_N_TRAIN[$i]}" "${UNIT_N_TEST[$i]}"
        echo ""
    done
fi

# --- Assemble results ---
echo ""
echo "Assembling results..."
julia assemble_results.jl

echo ""
echo "=========================================="
echo "Grid search complete!"
date
echo "Results: experiments/grid_search_results.csv"
echo "=========================================="
