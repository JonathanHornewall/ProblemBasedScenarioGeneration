# Prompt For Local Codex Agent

You are working on the user's local computer, likely macOS. The remote server is reachable as `rwl@reyland.dev`.

Goal: download the tiny benchmark data artifacts and cached rho-DFL model weights from the remote server, then evaluate only the rho-DFL models on the tiny data set using local parallel Julia workers. Do not mutate source code or git state unless needed to fix a clear local setup issue.

Assume the local repo root is `ProblemBasedScenarioGeneration/src/ContextualDFL`. If it is elsewhere, locate it first and run commands from that repo root.

Remote repo root:

`/home/rwl/ProblemBasedScenarioGeneration/src/ContextualDFL`

Relative paths to sync:

- Tiny data artifacts:
  `ContextualDFLExperiments/experiments/baseline_benchmarks/artifacts/tiny_30ctx_5x100_seed20260505/`
- Model cache directory:
  `ContextualDFLExperiments/experiments/baseline_benchmarks/cache_tiny_30ctx_5x100_seed20260505/models/`

Run these from the local repo root:

```bash
mkdir -p ContextualDFLExperiments/experiments/baseline_benchmarks/artifacts/tiny_30ctx_5x100_seed20260505
mkdir -p ContextualDFLExperiments/experiments/baseline_benchmarks/cache_tiny_30ctx_5x100_seed20260505/models

rsync -az --progress \
  rwl@reyland.dev:/home/rwl/ProblemBasedScenarioGeneration/src/ContextualDFL/ContextualDFLExperiments/experiments/baseline_benchmarks/artifacts/tiny_30ctx_5x100_seed20260505/ \
  ContextualDFLExperiments/experiments/baseline_benchmarks/artifacts/tiny_30ctx_5x100_seed20260505/

rsync -az --progress \
  rwl@reyland.dev:/home/rwl/ProblemBasedScenarioGeneration/src/ContextualDFL/ContextualDFLExperiments/experiments/baseline_benchmarks/cache_tiny_30ctx_5x100_seed20260505/models/ \
  ContextualDFLExperiments/experiments/baseline_benchmarks/cache_tiny_30ctx_5x100_seed20260505/models/
```

After syncing, verify the rho-DFL cache count. Expected final count is `63` `.jls` DFL model files:

```bash
find ContextualDFLExperiments/experiments/baseline_benchmarks/cache_tiny_30ctx_5x100_seed20260505/models \
  -type f -name '*dfl_mu0_rho0*.jls' | wc -l
```

If the count is less than `63`, do not start evaluation unless the user explicitly accepts local training for the missing models. Re-run the rsync later; the remote run may still be producing the last weights.

Validate the tiny artifacts:

```bash
cd ContextualDFLExperiments/experiments/baseline_benchmarks

julia --project=../.. run_baselines.jl \
  --validate-tiny-data-artifacts \
  --tiny-artifact-dir artifacts/tiny_30ctx_5x100_seed20260505 \
  --cache-dir cache_tiny_30ctx_5x100_seed20260505
```

Then evaluate only the rho-DFL policies using parallel local workers. Pick a safe worker count for the local machine, for example `min(number_of_cores - 2, 50)`, but at least `1`.

```bash
cd ContextualDFLExperiments/experiments/baseline_benchmarks

LOCAL_WORKERS=<set_this_to_a_safe_worker_count>
OUTDIR="results/tiny_30ctx_5x100_dfl_rho_eval_local_$(date +%Y%m%d_%H%M%S)"

julia --project=../.. run_baselines.jl \
  --tiny-full-baselines \
  --tiny-artifact-dir artifacts/tiny_30ctx_5x100_seed20260505 \
  --cache-dir cache_tiny_30ctx_5x100_seed20260505 \
  --output-dir "$OUTDIR" \
  --benchmarks resource_allocation,shipment_planning,transshipment_q,transshipment_h,transshipment_h_and_q,random_yield,unreliable_newsvendor \
  --policies dfl_mu0_rho0.1,dfl_mu0_rho0.01,dfl_mu0_rho0.001 \
  --replica-seeds 20260505,20260506,20260507 \
  --local-workers "$LOCAL_WORKERS"
```

Expected output: `63` successful result rows, corresponding to:

`7 benchmarks * 3 rho policies * 3 replicas = 63`

The output CSV should be:

`ContextualDFLExperiments/experiments/baseline_benchmarks/$OUTDIR/baseline_results_latest.csv`

After the run, verify row count and statuses:

```bash
python3 - <<'PY'
import csv
from pathlib import Path

outdir = sorted(Path("results").glob("tiny_30ctx_5x100_dfl_rho_eval_local_*"))[-1]
csv_path = outdir / "baseline_results_latest.csv"

rows = list(csv.DictReader(csv_path.open()))
ok = [r for r in rows if r.get("status") == "ok"]

print("csv:", csv_path)
print("rows:", len(rows))
print("ok:", len(ok))
print("statuses:", sorted({r.get("status") for r in rows}))

for field, expected in [
    ("test_contexts", "30"),
    ("test_scenarios_per_context", "500"),
    ("evaluation_batches", "5"),
]:
    bad = [r for r in rows if r.get(field) != expected]
    print(field, "bad:", len(bad))

bad_eval = [r for r in rows if r.get("rho_eval") not in ("0.0", "0")]
print("rho_eval_not_zero:", len(bad_eval))
PY
```

Important: the cached model files are serialized Julia `.jls` payloads under the cache `models/` directory. The `results/.../policy_histories/` CSVs are training histories, not model weights.
