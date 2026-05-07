# DFL-rho Tiny 30ctx 5x100 Summary

Result root: `ContextualDFLExperiments/experiments/baseline_benchmarks/results/tiny_30ctx_5x100_dfl_rho_asap_20260507/`

Validation: 63 rows, all `status=ok`; 3 replicas for each of 7 problems x 3 rho policies; `mu_eval=0`, `rho_eval=0`; `test_contexts=30`, `test_scenarios_per_context=500`, `evaluation_batches=5`.

## Mean Regret by Problem and rho

| problem | rho=0.001 | rho=0.01 | rho=0.1 | best rho |
|---|---:|---:|---:|---|
| `resource_allocation` | 154.595 +/- 5.71 | 204.038 +/- 3.29 | 420.934 +/- 24 | `0.001` |
| `shipment_planning` | 72.3899 +/- 40.5 | 33.9686 +/- 11.2 | 374.047 +/- 270 | `0.01` |
| `transshipment_q` | 3.80186e-08 +/- 5.63e-09 | 328.754 +/- 3.37 | 2330.42 +/- 6.65 | `0.001` |
| `transshipment_h` | 84.4144 +/- 2.74 | 902.369 +/- 10.6 | 2156.6 +/- 3.94 | `0.001` |
| `transshipment_h_and_q` | 85.777 +/- 6.45 | 637.301 +/- 12.1 | 2198.58 +/- 14 | `0.001` |
| `random_yield` | 3.65278 +/- 3.09e-08 | 3.68972 +/- 0.064 | 3.54599 +/- 0.0364 | `0.1` |
| `unreliable_newsvendor` | 0.0728765 +/- 0.0404 | 0.195039 +/- 0.257 | 0.205859 +/- 0.273 | `0.001` |

## Relative Regret Means

| problem | rho=0.001 | rho=0.01 | rho=0.1 |
|---|---:|---:|---:|
| `resource_allocation` | 0.181349 | 0.242094 | 0.481806 |
| `shipment_planning` | 0.0101387 | 0.00463745 | 0.05317 |
| `transshipment_q` | 1.66748e-11 | 0.14419 | 1.02211 |
| `transshipment_h` | 0.0276721 | 0.291957 | 0.702518 |
| `transshipment_h_and_q` | 0.0286088 | 0.207378 | 0.724804 |
| `random_yield` | 0.269089 | 0.271805 | 0.25905 |
| `unreliable_newsvendor` | 0.0355144 | 0.0925793 | 0.0972099 |
