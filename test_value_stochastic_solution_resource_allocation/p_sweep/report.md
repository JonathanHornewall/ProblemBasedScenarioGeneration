# Resource-Allocation Average-Scenario VSS p Sweep

This run regenerates resource-allocation test data for demand powers `p = 0.5, 1.0, 1.5`, solves benchmark optima, and evaluates one average-scenario decision per context.

| p | source | contexts | scenarios/context | batches | mean gap | mean relative gap | eval seconds |
|---:|:---|---:|---:|---:|---:|---:|---:|
| 0.5 | generated | 30 | 1000 | 20 | 11.729307211108422 | 0.014162484958088983 | 146.704936626 |
| 1.0 | generated | 30 | 1000 | 20 | 11.729487315762034 | 0.014153046132160051 | 147.8574615 |
| 1.5 | generated | 30 | 1000 | 20 | 11.729579648226741 | 0.013949430180262712 | 146.068646 |
| 2.0 | reference_p2 | 30 | 1000 | 20 | 11.729649602870433 | 0.013628308640135863 | 881.435946742 |

Generated at 2026-05-05T15:47:51.093Z.
