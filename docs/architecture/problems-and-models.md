# Unit 3: Problem Instances & Models

This document describes the `ProblemInstance` interface, the three concrete problem
formulations shipped with the package, and the neural network model architecture
used to generate scenario parameters from contextual features.

---

## 1. The `ProblemInstance` Interface

### Abstract type

```julia
abstract type ProblemInstance end
```

Every concrete problem must subtype `ProblemInstance` and implement four methods.

### Required methods

| Method | Signature | Returns |
|--------|-----------|---------|
| `first_stage_data` | `first_stage_data(prob::ProblemInstance)` | `(A::Matrix, b::Vector, c::Vector)` -- first-stage constraint matrix, RHS, and cost vector |
| `scenario_realization` | `scenario_realization(prob::ProblemInstance, param::AbstractVector)` | `Scenario` -- maps a scenario parameter vector to a fully specified `Scenario` struct |
| `generate_dataset` | `generate_dataset(prob::ProblemInstance, n::Int)` | `Vector{Tuple{Vector{Float64}, Scenario{Float64}}}` -- `n` (context, scenario) pairs for training or evaluation |
| `noise_pattern` | `noise_pattern(prob::ProblemInstance)` | `NoisePattern` -- an enum value indicating which second-stage matrices are stochastic |

### `NoisePattern` enum

```julia
@enum NoisePattern begin
    H_ONLY   # only h (RHS) varies
    Q_ONLY   # only q (cost) varies
    W_ONLY   # only W (constraint matrix) varies
    WH       # W and h vary
    WQ       # W and q vary
    WHQ      # W, h, and q all vary
end
```

The noise pattern governs how the neural network output is interpreted by the
output head (see Section 5).

### `Scenario` struct (reference)

Every `scenario_realization` call returns a `Scenario{T}` with four fields:

| Field | Dimensions | Description |
|-------|-----------|-------------|
| `W` | `(m2, n2)` | Second-stage constraint matrix |
| `T` | `(m2, n1)` | Coupling matrix linking first-stage decisions to second-stage constraints |
| `h` | `(m2,)` | Second-stage right-hand side |
| `q` | `(n2,)` | Second-stage cost vector |

---

## 2. Problem Instance: `UnreliableNewsvendorProblem`

**Noise pattern:** `WH` (both `W` and `h` are stochastic)

### 2.1 Problem description

A newsvendor faces uncertain demand `D` and unreliable supply with reliability
factor `U`, both drawn uniformly from `[0, 1]`. The first stage decides an
order quantity `z >= 0`; the second stage resolves overage, underage, and slack.

### 2.2 Parameters

| Symbol | Default | Meaning |
|--------|---------|---------|
| `p` | 5.0 | Selling price |
| `c` | 1.0 | Purchase cost |
| `pi` | 10.0 | Underage penalty (lost-sales cost) |
| `eta` | 0.5 | Overage penalty (holding cost) |

### 2.3 First-stage formulation

| Matrix | Value | Dimensions |
|--------|-------|------------|
| `A1` | `[0.0]` (1x1 zero matrix) | `(1, 1)` |
| `b1` | `[0.0]` | `(1,)` |
| `c1` | `[0.0]` | `(1,)` |

The first-stage cost is zero in this formulation because the purchase cost `c`
is embedded in the second-stage cost vector `q`.

Decision variable: `z` (scalar order quantity), `n1 = 1`.

### 2.4 Second-stage formulation

Second-stage decision variables `y = [y1, y2, y3]` where:

- `y1` -- overage (excess inventory)
- `y2` -- underage (lost sales)
- `y3` -- slack

The second-stage LP (for a given first-stage decision `z` and scenario `(D, U)`) is:

```
min  q' y
s.t. W y = h - T z
     y >= 0
```

**Fixed matrices:**

| Matrix | Value | Dimensions |
|--------|-------|------------|
| `W` | `[1 -1 -1; 0 0 1]` | `(2, 3)` |
| `q` | `[p + eta, pi, c - p]` = `[5.5, 10.0, -4.0]` | `(3,)` |

**Scenario-dependent matrices:**

| Matrix | Value | Dimensions |
|--------|-------|------------|
| `T` | `[0; -U]` (reshaped to 2x1) | `(2, 1)` |
| `h` | `[-D, 0]` | `(2,)` |

Both `T` and `h` depend on the random scenario parameters `(D, U)`, hence the
`WH` noise pattern (the coupling matrix `T` plays a role analogous to `W` in
the canonical two-stage form since the effective RHS is `h - T z`).

### 2.5 Scenario parameter vector

`param = [D, U]` where `D` is demand and `U` is supply reliability, both in `[0, 1]`.

**Context:** a 1-dimensional vector `[D]` (the demand observation only).

### 2.6 Dataset generation

Each sample draws `D ~ Uniform(0, 1)` and `U ~ Uniform(0, 1)` independently.
The context vector is `[D]`, so the generator must learn to predict both `D`
(which it observes) and `U` (which is latent).

---

## 3. Problem Instance: `ResourceAllocationProblem`

**Noise pattern:** `H_ONLY` (only `h` varies across scenarios)

### 3.1 Problem description

Allocate resources across `I = 20` clients, each served by a subset of
`J = 30` resources. First stage decides resource allocation levels; second
stage routes resources to meet stochastic demand, with unmet demand incurring
waste costs.

### 3.2 Parameters

| Symbol | Dimensions | Description |
|--------|-----------|-------------|
| `cz` | `(20,)` | First-stage allocation costs per client |
| `qw` | `(30,)` | Second-stage waste/penalty costs per resource |
| `rho` | `(20,)` | Yield parameters per client (values near 0.9--1.0) |
| `mu_ij` | `(20, 30)` | Service rate matrix (sparse; many entries are zero) |

All parameter values are hardcoded constants in the source.

### 3.3 First-stage formulation

| Matrix | Value | Dimensions |
|--------|-------|------------|
| `A1` | zeros | `(1, 20)` |
| `b1` | `[0.0]` | `(1,)` |
| `c1` | `cz` (allocation costs) | `(20,)` |

Decision variables: `z_i` for `i = 1..20` (allocation level per client), `n1 = 20`.

### 3.4 Second-stage formulation

Second-stage decision variables (total `n2 = J + I*J + I + J = 30 + 600 + 20 + 30 = 680`):

| Variable block | Symbol | Count | Description |
|----------------|--------|-------|-------------|
| 1 | `w_j` | `J = 30` | Waste/unmet demand per resource |
| 2 | `y_{ij}` | `I*J = 600` | Flow from client `i` to resource `j` |
| 3 | `s_i` | `I = 20` | Supply slack per client |
| 4 | `slack_j` | `J = 30` | Demand slack per resource |

Constraint rows (`m2 = I + J = 50`):

**Resource constraints** (rows `1..I`): for each client `i`:

```
sum_j y_{ij} + s_i = rho_i * z_i
```

These ensure that the total flow out of client `i` plus slack equals the
yield-adjusted allocation.

**Demand constraints** (rows `I+1..I+J`): for each resource `j`:

```
w_j + sum_i mu_{ij} * y_{ij} - slack_j = xi_j
```

where `xi_j` is the stochastic demand for resource `j`.

**LP matrices:**

| Matrix | Dimensions | Structure |
|--------|-----------|-----------|
| `W` | `(50, 680)` | Sparse; encodes resource and demand constraints |
| `T` | `(50, 20)` | Diagonal in rows 1..20: `T[i,i] = -rho_i`; rows 21..50 are zero |
| `q` | `(680,)` | Only first 30 entries are nonzero (`qw`); rest are zero |
| `h` | `(50,)` | `[zeros(20); demand]` -- only the demand portion (entries 21..50) varies |

### 3.5 Scenario parameter vector

`param` is a demand vector of length `J = 30`. The `scenario_realization`
function constructs `h = [zeros(I); param]` so that the demand appears in the
RHS of the demand constraints.

**Context:** a 3-dimensional vector `x in R^3`.

### 3.6 Dataset generation

Demand is generated via a polynomial model:

```
xi_j = A_j + sum_{l=1}^{L} B_{jl} * x_l^p + noise
```

where:
- `A_j ~ Normal(50, 5)` (intercept per resource)
- `B_{jl}` drawn from shifted uniforms (coefficients per resource per context dimension)
- `x ~ |MvNormal(0, Sigma)|` with a random correlation matrix `Sigma`
- `noise ~ Normal(0, sigma)` with `sigma = 5.0` by default
- `L = 3` context dimensions, `p = 1` (linear by default)

---

## 4. Problem Instance: `ShipmentPlanningProblem`

**Noise pattern:** `H_ONLY` (only `h` varies across scenarios)

### 4.1 Problem description

Plan production at `I = 12` warehouses to meet stochastic demand at `J = 4`
locations. First stage decides production quantities; second stage handles
emergency production and shipment routing.

### 4.2 Parameters

| Symbol | Dimensions | Description |
|--------|-----------|-------------|
| `production_costs` | `(12,)` | First-stage production cost per warehouse (all 5.0) |
| `emergency_costs` | `(12,)` | Emergency production cost per warehouse (all 100.0) |
| `shipment_costs` | `(12, 4)` | Shipment cost matrix = `10 * distances` |
| `distances` | `(12, 4)` | Distance matrix between warehouses and locations |
| `context_dim` | scalar | Context feature dimension (3) |

### 4.3 First-stage formulation

| Matrix | Value | Dimensions |
|--------|-------|------------|
| `A1` | zeros | `(1, 12)` |
| `b1` | `[0.0]` | `(1,)` |
| `c1` | `production_costs` | `(12,)` |

Decision variables: `z_i` for `i = 1..12` (production quantity at each warehouse), `n1 = 12`.

### 4.4 Second-stage formulation

Second-stage decision variables (total `n2 = I + I*J + J + I = 12 + 48 + 4 + 12 = 76`):

| Variable block | Symbol | Count | Description |
|----------------|--------|-------|-------------|
| 1 | `y^w_i` | `I = 12` | Emergency production at warehouse `i` |
| 2 | `y^s_{ij}` | `I*J = 48` | Shipment flow from warehouse `i` to location `j` |
| 3 | `d_slack_j` | `J = 4` | Demand slack at location `j` |
| 4 | `s_slack_i` | `I = 12` | Supply slack at warehouse `i` |

Constraint rows (`m2 = J + I = 4 + 12 = 16`):

**Demand constraints** (rows `1..J`): for each location `j`:

```
sum_i y^s_{ij} - d_slack_j = xi_j
```

where `xi_j` is the stochastic demand at location `j`.

**Supply constraints** (rows `J+1..J+I`): for each warehouse `i`:

```
-y^w_i + sum_j y^s_{ij} + s_slack_i = z_i
```

These ensure that shipments out of warehouse `i` are covered by planned
production `z_i` plus emergency production `y^w_i`.

**LP matrices:**

| Matrix | Dimensions | Structure |
|--------|-----------|-----------|
| `W` | `(16, 76)` | Encodes demand and supply constraints |
| `T` | `(16, 12)` | Rows 1..4 are zero; rows 5..16 have `T[J+i, i] = -1.0` |
| `q` | `(76,)` | Emergency costs in entries 1..12, shipment costs in entries 13..60, zeros elsewhere |
| `h` | `(16,)` | `[demand; zeros(12)]` -- only the demand portion (entries 1..4) varies |

### 4.5 Scenario parameter vector

`param` is a demand vector of length `J = 4`. The `scenario_realization`
function constructs `h = [param; zeros(I)]` so demand appears in the RHS
of the demand constraints (first `J` rows).

**Context:** a 3-dimensional vector `x in R^3`.

### 4.6 Dataset generation

Demand is generated via a polynomial model (same structure as resource
allocation but with smaller scale):

```
xi_j = A_j + sum_{l=1}^{L} B_{jl} * x_l^p + noise
```

where:
- `A_j ~ Normal(10, 2)` (intercept per location)
- `B_{jl}` drawn from shifted uniforms with smaller ranges
- `x ~ |Normal(0, 1)|` (absolute values of standard normals)
- `noise ~ Normal(0, sigma)` with `sigma = 2.0` by default
- Demand is clamped to non-negative: `max(demand, 0)`

---

## 5. Neural Network Architecture

### 5.1 `build_generator` -- the feed-forward network

Defined in `src/models/scenario_generator.jl`.

```julia
build_generator(input_dim, output_dim;
                hidden_dim=128, n_layers=3, activation=relu) -> Flux.Chain
```

The network is a standard multi-layer perceptron:

```
Dense(input_dim => hidden_dim, relu)      # input layer
Dense(hidden_dim => hidden_dim, relu)     # hidden layer 2
Dense(hidden_dim => hidden_dim, relu)     # hidden layer 3
Dense(hidden_dim => output_dim, softplus) # output layer
```

Key design choices:

- **`n_layers` hidden layers** (default 3): the first `Dense` layer maps from
  `input_dim` to `hidden_dim`, followed by `n_layers - 1` hidden-to-hidden
  layers, all using `relu` activation.
- **softplus output activation**: the final layer uses `softplus(x) = log(1 + exp(x))`
  to ensure all outputs are strictly positive. This is appropriate because
  scenario parameters (demands, reliabilities) must be non-negative.
- **`hidden_dim`** defaults to 128 neurons per hidden layer.

### 5.2 Problem-aware constructor

```julia
build_generator(prob::ProblemInstance; nr_of_scenarios=1, kw...) -> Flux.Chain
```

This overload infers `input_dim` and `output_dim` from the problem instance:

- `input_dim = _context_dim(prob)`
- `output_dim = _scenario_param_dim(prob) * nr_of_scenarios`

When generating multiple scenarios (`nr_of_scenarios > 1`), the output
dimension scales linearly -- the network produces all scenario parameters
as a single flat vector.

### 5.3 Dimension helpers

```julia
_context_dim(::ResourceAllocationProblem)    = 3
_context_dim(::ShipmentPlanningProblem)      = 3
_context_dim(::UnreliableNewsvendorProblem)  = 1

_scenario_param_dim(prob::ResourceAllocationProblem) = length(prob.qw)           # 30
_scenario_param_dim(prob::ShipmentPlanningProblem)   = size(prob.shipment_costs, 2)  # 4
_scenario_param_dim(::UnreliableNewsvendorProblem)   = 2                         # [D, U]
```

---

## 6. Output Heads

Defined in `src/models/output_heads.jl`.

An output head is a function that takes the raw network output vector and
converts it into a `Vector{Scenario}`. The head is selected based on the
problem's `NoisePattern`.

```julia
build_output_head(pattern::NoisePattern, prob::ProblemInstance;
                  nr_of_scenarios=1) -> Function
```

### 6.1 `H_ONLY` head (`_h_only_head`)

Used by: `ResourceAllocationProblem`, `ShipmentPlanningProblem`.

The raw output is interpreted directly as the stochastic RHS parameter `h`.
The head reshapes the flat output vector into `(sc_dim, nr_of_scenarios)` and
calls `scenario_realization` for each column. The `W`, `T`, and `q` matrices
are fixed and come from the problem struct.

```
raw_output (length = sc_dim * S) --> reshape to (sc_dim, S) --> scenario_realization per column
```

### 6.2 `WH` head (`_wh_head`)

Used by: `UnreliableNewsvendorProblem`.

The raw output encodes parameters that affect both `W` (via `T`) and `h`. For
the newsvendor, the 2-element parameter vector `[D, U]` is used inside
`scenario_realization` to construct both `T = [0; -U]` and `h = [-D, 0]`.

The reshaping logic is identical to the `H_ONLY` head -- the distinction is
semantic: each column of the reshaped output is passed to
`scenario_realization`, which uses the parameters to build scenario-dependent
versions of multiple matrices (not just `h`).

### 6.3 `Q_ONLY` head (`_q_only_head`)

Implemented but not currently used by any shipped problem. Same reshape-and-realize
pattern as the other heads.

### 6.4 `build_full_model` -- composing generator and head

```julia
build_full_model(prob::ProblemInstance;
                 nr_of_scenarios=1, hidden_dim=128, n_layers=3)
    -> (gen, head, model)
```

Returns three objects:

1. **`gen`** -- the `Flux.Chain` neural network (trainable parameters live here)
2. **`head`** -- the output head function (no trainable parameters)
3. **`model`** -- the composed function `x -> head(gen(x))` that maps a context
   vector to a `Vector{Scenario}`

---

## 7. Scenario Parameter Dimensions Summary

| Problem | Noise pattern | Context dim | Scenario param dim | Param meaning | `n1` | `n2` | `m2` |
|---------|--------------|-------------|-------------------|---------------|------|------|------|
| `UnreliableNewsvendorProblem` | `WH` | 1 | 2 | `[D, U]` (demand, reliability) | 1 | 3 | 2 |
| `ResourceAllocationProblem` | `H_ONLY` | 3 | 30 | Demand vector `xi` (one per resource) | 20 | 680 | 50 |
| `ShipmentPlanningProblem` | `H_ONLY` | 3 | 4 | Demand vector `xi` (one per location) | 12 | 76 | 16 |

### Network output dimension

For `S` scenarios, the network output dimension is `scenario_param_dim * S`:

| Problem | S=1 | S=5 | S=10 |
|---------|-----|-----|------|
| `UnreliableNewsvendorProblem` | 2 | 10 | 20 |
| `ResourceAllocationProblem` | 30 | 150 | 300 |
| `ShipmentPlanningProblem` | 4 | 20 | 40 |

---

## 8. File Reference

| File | Contents |
|------|----------|
| `src/problems/interface.jl` | `ProblemInstance` abstract type, `NoisePattern` enum, method stubs |
| `src/problems/newsvendor.jl` | `UnreliableNewsvendorProblem` struct and interface methods |
| `src/problems/resource_allocation.jl` | `ResourceAllocationProblem` struct, hardcoded parameters, interface methods |
| `src/problems/shipment_planning.jl` | `ShipmentPlanningProblem` struct and interface methods |
| `src/models/scenario_generator.jl` | `build_generator`, `_context_dim`, `_scenario_param_dim` |
| `src/models/output_heads.jl` | `build_output_head`, `_h_only_head`, `_wh_head`, `_q_only_head`, `build_full_model` |
| `src/core/scenario.jl` | `Scenario` struct |
| `src/core/two_stage_lp.jl` | `TwoStageSLP` struct |

All paths are relative to `src/ProblemBasedScenarioGeneration/src/`.
