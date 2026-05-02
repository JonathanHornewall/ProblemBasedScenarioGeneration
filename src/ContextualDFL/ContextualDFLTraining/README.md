# ContextualDFLTraining

Distributed grid-search project for `ContextualDFL`.

Run the real grid search from the current server with:

```bash
julia --project=ContextualDFLTraining ContextualDFLTraining/gridsearch.jl
```

The driver runs `~/sync-julia-code.sh`, starts only remote Julia workers on
`rwl@gcp-big` and `rwl@gcp-small`, and writes CSV results locally under
`ContextualDFLTraining/results/<timestamp>/`.

Useful environment variables:

- `GRIDSEARCH_SMOKE=1`: run one tiny remote configuration.
- `SKIP_SYNC=1`: skip `~/sync-julia-code.sh`.
- `GCP_BIG_WORKERS=8`: worker count for `rwl@gcp-big`.
- `GCP_SMALL_WORKERS=4`: worker count for `rwl@gcp-small`.
- `PMAP_BATCH_SIZE=1`: pmap batch size.
- `REMOTE_JULIA=/home/rwl/.juliaup/bin/julia`: remote Julia executable.
- `REMOTE_CONTEXTUAL_DFL_TRAINING_PROJECT=/home/rwl/ProblemBasedScenarioGeneration/src/ContextualDFL/ContextualDFLTraining`: remote project path.
- `MLFLOW_EXPERIMENT_ID=<id>`: enable MLflow logging for every candidate.
- `MLFLOW_TRACKING_URI=http://...`: tracking server URI passed to remote workers.
- `MLFLOW_ENABLED=0`: disable MLflow even if `MLFLOW_EXPERIMENT_ID` is set.

The training wrapper assumes the relevant `ContextualDFL.train`, loss,
decoder, and stochastic-programming methods are implemented before the real
grid search is run.
