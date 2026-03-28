# Unit 2: Mathematical Foundations

This document covers the mathematical structures, algorithms, and differentiation machinery at the core of ProblemBasedScenarioGeneration. Every type name and function name referenced below corresponds to an actual implementation in the package source.

---

## 1. Core Data Structures

### 1.1 `Scenario{T}`

**Source:** `src/core/scenario.jl`

A `Scenario{T<:Real}` represents the data for a single second-stage realization in a two-stage stochastic linear program. It contains four fields:

| Field | Dimensions | Mathematical meaning |
|-------|------------|---------------------|
| `W`   | `(m2, n2)` | Second-stage constraint (recourse) matrix |
| `T`   | `(m2, n1)` | Coupling matrix linking first-stage decisions to second-stage constraints |
| `h`   | `(m2,)`    | Second-stage right-hand side vector |
| `q`   | `(n2,)`    | Second-stage cost vector |

Given a first-stage decision `x1`, the second-stage (recourse) problem under this scenario is:

```
min  q' y
s.t. W y = h - T x1
     y >= 0
```

The constructor validates dimensional consistency: `W` and `T` must share the same row count `m2`, `h` must have length `m2`, and `q` must have length `n2`. The convenience function `scenario_from_tuple(W, T_mat, h, q)` promotes all inputs to a common element type before construction.

### 1.2 `TwoStageSLP{T}`

**Source:** `src/core/two_stage_lp.jl`

A `TwoStageSLP{T<:Real}` represents a complete two-stage stochastic linear program. Its fields are:

| Field | Dimensions | Mathematical meaning |
|-------|------------|---------------------|
| `A`   | `(m1, n1)` | First-stage constraint matrix |
| `b`   | `(m1,)`    | First-stage right-hand side |
| `c`   | `(n1,)`    | First-stage cost vector |
| `scenarios` | `Vector{Scenario{T}}`, length `S` | Second-stage scenario realizations |
| `p`   | `(S,)`     | Probability weights for each scenario |

The full two-stage stochastic LP is:

```
min   c' x1 + sum_{s=1}^{S} p_s * Q(x1, s)
s.t.  A x1 = b
      x1 >= 0
```

where `Q(x1, s)` is the optimal value of the recourse problem for scenario `s`:

```
Q(x1, s) = min  q_s' y_s
           s.t. W_s y_s = h_s - T_s x1
                y_s >= 0
```

The constructor enforces:
- The number of scenarios equals the length of the probability vector.
- Every scenario's coupling matrix `T_s` has `n1` columns (matching the first-stage dimension).
- Probabilities are strictly positive and sum to 1 (within tolerance `1e-10`).

An outer constructor promotes all inputs to a common element type. A convenience constructor `TwoStageSLP(A, b, c, scenarios)` assigns equiprobable weights `p_s = 1/S`.

---

## 2. Canonical Form Transformation

**Source:** `src/core/canonical_form.jl`

All LP solvers in this package operate on the canonical form:

```
min  c' x    s.t.  A x = b,   x >= 0
```

### 2.1 `to_canonical(A, b, c)`

Converts a standard-form LP (which may have free variables) to canonical form by variable splitting:

Each variable `x_i` is replaced by `x_i = x_i+ - x_i-` where `x_i+, x_i- >= 0`, and slack variables are added for the equality constraints.

The transformation produces:

```
A_can = [A | -A | I_m]         (m x (2n + m))
c_can = [c; -c; 0_m]           (2n + m)
b_can = b                       (m)
```

Zero rows (corresponding to unconstrained first-stage problems) are automatically removed.

### 2.2 `to_canonical_decision(x, A, b)`

Maps a decision vector from standard form to canonical form:

```
x_can = [x+; x-; slack]
```

where `x+ = max(x, 0)`, `x- = max(-x, 0)`, and `slack = max(b - A*x, 0)`.

### 2.3 `extensive_form(slp::TwoStageSLP)`

Builds the deterministic equivalent (extensive form) of the 2SLP. The decision vector is:

```
z = [x1; y1; y2; ...; y_S]
```

The constraint matrix has block-angular structure:

```
         x1    y1    y2   ...   y_S
      [  A     0     0    ...    0  ]   <- first-stage
      [  T1    W1    0    ...    0  ]   <- scenario 1
      [  T2    0     W2   ...    0  ]   <- scenario 2
      [  ...   ...   ...   ...  ... ]
      [  T_S   0     0    ...   W_S ]   <- scenario S
```

The cost vector is:

```
c_ext = [c1; p1*q1; p2*q2; ...; p_S*q_S]
```

The RHS vector is:

```
b_ext = [b; h1; h2; ...; h_S]
```

Zero rows are dropped. The construction is mutation-free (compatible with Zygote automatic differentiation).

---

## 3. LP Solver (Exact, via HiGHS/JuMP)

**Source:** `src/solvers/lp_solver.jl`

### 3.1 `solve_lp(A, b, c; tol=1e-9)`

Solves the canonical LP exactly using the HiGHS simplex/interior-point solver through JuMP:

```
min  c' x    s.t.  A x = b,   x >= 0
```

**Algorithm:**
1. Build a JuMP `Model` with the HiGHS optimizer (silenced).
2. Set primal and dual feasibility tolerances to `tol`.
3. Add non-negative variables `x[1:n] >= 0`.
4. Add equality constraints `A * x .== b`.
5. Set the linear objective `min c' x`.
6. Call `optimize!()` and check for optimality.
7. Return the primal solution `x_opt` and the dual variables `lambda_opt` for the equality constraints.
8. Perform a post-solve sanity check on primal feasibility.

The solver throws on infeasibility or unboundedness. The variant `solve_lp_primal(A, b, c)` returns only the primal solution.

This solver is used for both training (via the subgradient rrule) and evaluation.

---

## 4. Log-Barrier Newton Solver

**Source:** `src/solvers/barrier_solver.jl`

### 4.1 Problem Formulation

The barrier solver replaces the non-negativity constraints `x >= 0` with a logarithmic barrier term, solving:

```
min  c' x  -  mu * sum_i log(x_i)
s.t. A x = b,   x > 0
```

where `mu > 0` is the barrier parameter. As `mu -> 0`, the solution approaches the true LP optimum.

### 4.2 KKT Conditions

The Lagrangian is:

```
L(x, lambda) = c' x - mu * sum_i log(x_i) + lambda' (A x - b)
```

Setting partial derivatives to zero yields the KKT system:

```
Stationarity:   c - mu / x + A' lambda = 0     (n equations)
Feasibility:    A x - b = 0                      (m equations)
```

where `mu / x` denotes elementwise division, i.e., `(mu/x)_i = mu / x_i`.

### 4.3 Newton Step

The Newton system for the KKT conditions is:

```
[ D    A' ] [ dx      ]       [ c - mu/x + A' lambda ]
[ A    0  ] [ dlambda ]  = -  [ A x - b              ]
```

where `D = diag(mu / x_i^2)` is the Hessian of the barrier term with respect to `x`.

### 4.4 `BarrierCache{T}`

The solver returns a `BarrierCache{T}` struct that stores the solution and problem data for downstream implicit differentiation:

| Field    | Description |
|----------|-------------|
| `x`      | Primal solution `(n,)` |
| `lambda` | Dual variables for `Ax = b` `(m,)` |
| `mu`     | Barrier parameter |
| `A`      | Constraint matrix `(m, n)` |
| `b`      | RHS vector `(m,)` |
| `c`      | Cost vector `(n,)` |

### 4.5 `solve_barrier(A, b, c, mu; tol, max_iter, x0, lambda0)`

**Full algorithm:**

1. **Type promotion.** All inputs are promoted to a common floating-point type `T`.

2. **Fallback for mu = 0.** When the barrier parameter is zero, the solver delegates to `solve_lp(A, b, c)` and wraps the result in a `BarrierCache`.

3. **Initialization.** If no warm-start is provided, `_barrier_init(A, b)` computes an interior feasible point:
   - Compute the minimum-norm solution `x0 = pinv(A) * b` via the pseudo-inverse.
   - If all components are positive and the residual is small, accept immediately.
   - Otherwise, iterate up to 5 times: shift to ensure strict positivity, re-project onto `Ax = b` via `x0 = x0 - pinv(A) * (A*x0 - b)`, and clamp to `x >= 1e-6`.
   - The result is clamped to `x >= 1e-8` for safety.

4. **Newton iterations** (up to `max_iter`, default 200):

   a. Compute residuals:
   ```
   r_dual = c - mu ./ x + A' * lambda
   r_prim = A * x - b
   ```

   b. Check convergence: if `||[r_dual; r_prim]||_2 < tol`, stop.

   c. Assemble the `(n+m) x (n+m)` KKT matrix:
   ```
   K = [ diag(mu / x^2)    A' ]
       [ A                  0  ]
   ```

   d. Solve the Newton system `K * [dx; dlambda] = -[r_dual; r_prim]`.
      If the system is singular, add a small regularization `1e-12 * I`.

   e. **Line search** to maintain `x > 0`:
      - Compute the maximum step `alpha` such that `x + alpha * dx > 0`:
        for all components where `dx_i < 0`, set `alpha = min(alpha, 0.99 * (-x_i / dx_i))`.
      - Backtrack (halve alpha up to 50 times) until all components of `x + alpha * dx` are strictly positive.

   f. Update: `x = x + alpha * dx`, `lambda = lambda + alpha * dlambda`.

5. **Return** `BarrierCache(x, lambda, mu, A, b, c)`.

### 4.6 `kkt_residual(cache::BarrierCache)`

Computes the Euclidean norm of the KKT residual at the cached solution, used for testing convergence quality:

```
||[c - mu/x + A' lambda; A x - b]||_2
```

---

## 5. Implicit Differentiation

**Source:** `src/diff/implicit_diff.jl`

The key challenge in decision-focused learning is differentiating through the optimization solver. The package uses the implicit function theorem applied to the KKT conditions.

### 5.1 Theory

At the optimal `(x*, lambda*)`, the KKT conditions hold:

```
g1(x, lambda; b, c) = c - mu / x + A' lambda = 0
g2(x, lambda; b, c) = A x - b                 = 0
```

The Jacobian of `(g1, g2)` with respect to `(x, lambda)` is the KKT matrix:

```
K = [ D    A' ]       where D = diag(mu / x*^2)
    [ A    0  ]
```

By the implicit function theorem, the sensitivities of `(x*, lambda*)` with respect to parameters are obtained by solving linear systems involving `K`.

### 5.2 `_kkt_matrix(cache::BarrierCache)`

Assembles the `(n+m) x (n+m)` KKT matrix from the cached solution data.

> **Implementation note:** The matrix is constructed using `vcat`/`hcat` (i.e., `vcat(hcat(Diagonal(D_diag), A'), hcat(A, zeros(T, m, m)))`) rather than pre-allocating and filling indices. This mutation-free construction is required for compatibility with Zygote automatic differentiation, which does not support in-place array mutation.

### 5.3 `implicit_diff_h(cache::BarrierCache)` -- Sensitivity w.r.t. RHS `b`

Differentiating the KKT conditions with respect to `b`:

```
K * [ dx*/db; dlambda*/db ] = [ 0_n; I_m ]
```

The right-hand side comes from `dg2/db = -I_m` (negated in the implicit function theorem formula, yielding `+I_m` on the RHS after the sign convention).

Returns `dx*/db` as an `(n, m)` matrix: the first `n` rows of `K \ [0; I_m]`.

The RHS is constructed mutation-free as `vcat(zeros(T, n, m), Matrix{T}(I, m, m))`.

### 5.4 `implicit_diff_q(cache::BarrierCache)` -- Sensitivity w.r.t. cost `c`

Differentiating the KKT conditions with respect to `c`:

```
K * [ dx*/dc; dlambda*/dc ] = [ -I_n; 0_m ]
```

The right-hand side comes from `dg1/dc = I_n` (negated, giving `-I_n`).

Returns `dx*/dc` as an `(n, n)` matrix: the first `n` rows of `K \ [-I_n; 0]`.

The RHS is constructed mutation-free as `vcat(-Matrix{T}(I, n, n), zeros(T, m, n))`.

### 5.5 `recourse_multiplier(cache::BarrierCache)`

Returns the dual variable `lambda*` from a barrier-solved recourse subproblem. This is used to compute the sensitivity of the second-stage value function with respect to the first-stage decision:

```
dQ_s / dx1 = -T_s' * lambda_s*
```

This relationship follows from the LP duality: the dual variable of the constraint `W y = h - T x1` gives the marginal value of the right-hand side, and the chain rule with respect to `x1` introduces the factor `-T_s'`.

---

## 6. ChainRules rrules (Custom Reverse-Mode Differentiation)

The package provides two `rrule` implementations so that both solvers integrate seamlessly with reverse-mode AD (Zygote / ChainRules).

### 6.1 Barrier Solver rrule

**Source:** `src/diff/barrier_rrule.jl`

```julia
ChainRulesCore.rrule(::typeof(solve_barrier), A, b, c, mu; ...)
```

**Forward pass:** Calls `solve_barrier` normally, returning a `BarrierCache`.

**Pullback:** Given an upstream gradient `dx_bar` (the gradient of the loss with respect to the primal solution `x`), the pullback computes:

```
db_bar = (dx*/db)' * dx_bar     -- gradient w.r.t. b
dc_bar = (dx*/dc)' * dx_bar     -- gradient w.r.t. c
```

where `dx*/db` and `dx*/dc` are computed by `implicit_diff_h` and `implicit_diff_q` respectively. Both are wrapped in `@thunk` for lazy evaluation -- they are only computed if the downstream AD system actually needs them.

The tangents for `A` and `mu` are declared `NoTangent()` (gradients are not propagated through the constraint matrix or barrier parameter).

**Gradient extraction:** The pullback extracts `dx_bar` from the upstream `cache_bar` by checking for an `.x` property (matching the `BarrierCache` struct field).

> **Note:** The barrier solver's implicit differentiation rrule still exists in the codebase but is not on the default training path. It is available for use when `solve_barrier` is called directly (e.g., for experimentation or when smooth interior solutions are desired).

### 6.2 LP Solver rrule (Subgradient Approximation)

**Source:** `src/diff/subgradient_rrule.jl`

```julia
ChainRulesCore.rrule(::typeof(solve_lp), A, b, c; tol=1e-9)
```

When `mu = 0`, the LP solution lies at a vertex of the feasible polytope, where the standard implicit function theorem does not apply (the solution is non-smooth). The package uses a subgradient approximation:

**Forward pass:** Calls `solve_lp`, returning `(x_opt, lambda_opt)`.

**Pullback:** Given upstream gradient `dx_bar`:

```
db_bar = -lambda_opt * sum(dx_bar)
dc_bar =  x_opt      * sum(dx_bar)
```

The rationale:
- By LP duality, `dV/db = -lambda*` where `V(b) = min c'x s.t. Ax=b, x>=0` is the optimal value function. The dual variable serves as a subgradient for the primal-dual relationship.
- The cost sensitivity uses the optimal primal as a proxy.

This is the rrule used on the default training path, since both `surrogate_first_stage` and `recourse_cost` now call `solve_lp` exclusively.

---

## 7. Decision Regret Loss Function

**Source:** `src/loss/decision_regret.jl`

The decision regret loss measures how much worse the first-stage decision is when it is based on predicted scenarios rather than the true scenario. This is the training signal for the neural network scenario generator.

### 7.1 Surrogate First-Stage Solve

**`surrogate_first_stage(slp::TwoStageSLP, mu)`**

Solves the extensive-form 2SLP to obtain the first-stage decision:

1. Build the extensive form `(A_ext, b_ext, c_ext)` from the `TwoStageSLP`.
2. Solve the LP using `solve_lp(A_ext, b_ext, c_ext)` (HiGHS).
3. Return the first `n1` components of the solution (the first-stage decision `x1*`).

The `mu` parameter is retained in the function signature for API compatibility but does not affect solver choice -- `solve_lp` is always used regardless of the value of `mu`. The helper function `build_mu_vector` still exists in the codebase but is no longer called by `surrogate_first_stage`.

Gradients flow through this step via the subgradient rrule on `solve_lp` (see Section 6.2).

### 7.2 Recourse Cost

**`recourse_cost(x1, sc::Scenario, mu)`**

Evaluates the second-stage cost for a given first-stage decision `x1` under scenario `sc` by solving the pure LP:

```
Q(x1, sc) = min  q' y
            s.t. W y = h - T x1,   y >= 0
```

The function computes the recourse RHS as `b_rec = h - T * x1`, solves `solve_lp(W, b_rec, q)` using HiGHS, and returns `q' y*`.

As with `surrogate_first_stage`, the `mu` parameter is accepted for API compatibility but does not affect the computation -- there is no barrier term. Gradients flow through this step via the subgradient rrule on `solve_lp`.

### 7.3 Total Cost Evaluation

**`evaluate_cost(x1, slp::TwoStageSLP, mu)`**

Computes the total expected cost:

```
Z(x1) = c' x1 + sum_{s=1}^{S} p_s * Q(x1, s_s)
```

### 7.4 Absolute Decision Regret

**`decision_regret(prob, mu_surr, mu_prim, predicted_scenarios, actual_scenario)`**

The absolute decision regret is computed in two steps:

**Step 1 -- Surrogate decision.** Build a `TwoStageSLP` from the predicted scenarios (with equal probabilities) and solve it with barrier parameter `mu_surr` to get the surrogate first-stage decision `x1_hat`:

```
x1_hat = argmin_{x1}  c' x1 + sum_s (1/S) * Q_surr(x1, predicted_s; mu_surr)
         s.t.  A x1 = b,  x1 >= 0
```

**Step 2 -- Evaluation.** Evaluate `x1_hat` on the actual (true) scenario with barrier parameter `mu_prim`:

```
L_abs = c' x1_hat + Q(x1_hat, actual_scenario; mu_prim)
```

This quantity is the total cost incurred by the surrogate decision. Minimizing it during training pushes the scenario generator to produce scenarios that lead to good first-stage decisions.

The two barrier parameters are retained in the function signature for API compatibility. In practice, both `surrogate_first_stage` and `recourse_cost` now use `solve_lp` regardless of the barrier parameter values.

### 7.5 Relative Decision Regret

**`relative_decision_regret(prob, mu_surr, mu_prim, predicted_scenarios, actual_scenario)`**

The normalized version divides the gap by the optimal cost:

```
L_rel = (Z_surr - Z_opt) / (|Z_opt| + epsilon)
```

where:
- `Z_surr = c' x1_hat + Q(x1_hat, actual; mu_prim)` is the cost of the surrogate decision.
- `Z_opt = c' x1_opt + Q(x1_opt, actual; mu_prim)` is the cost of the decision that would be optimal if the actual scenario were known.
- `epsilon = 1e-10` prevents division by zero.

The optimal decision `x1_opt` is computed by solving a `TwoStageSLP` containing only the actual scenario with barrier parameter `mu_prim`.

A relative regret of 0 means the predicted scenarios led to the optimal decision. A positive value indicates the cost penalty from using predicted scenarios instead of knowing the true outcome.

---

## 8. End-to-End Gradient Flow

The full differentiable pipeline, from neural network output to loss, follows this path:

```
Neural network  -->  predicted (W, T, h, q)  -->  Scenario structs
                                                        |
                                                        v
                                            TwoStageSLP (surrogate)
                                                        |
                                                        v
                                           extensive_form(slp)
                                                        |
                                                        v
                                      solve_lp(A_ext, b_ext, c_ext)
                                         [rrule: subgradient]
                                                        |
                                                        v
                                                x1_hat (first-stage decision)
                                                        |
                                                        v
                                      recourse_cost(x1_hat, actual_scenario)
                                        -> solve_lp(W, b_rec, q)
                                           [rrule: subgradient]
                                                        |
                                                        v
                                              decision_regret loss
```

Gradients flow backward through this chain via ChainRules:

1. The loss provides `dL/dx1_hat`.
2. The recourse solver's subgradient rrule computes `dL/dh_rec = dL/d(h - T*x1)` using the LP dual variables as subgradients, which gives `dL/dx1_hat` via the chain rule through `h_rec = h - T * x1_hat`.
3. The surrogate solver's subgradient rrule computes `dL/db_ext` and `dL/dc_ext` from `dL/dx1_hat`, using the LP duals from HiGHS as approximate sensitivities.
4. Since `b_ext` and `c_ext` are functions of the scenario parameters `(W, T, h, q)`, Zygote propagates gradients through the `extensive_form` construction back to the neural network outputs.

Because both solvers now use `solve_lp` exclusively, the `mu` parameter is no longer critical for gradient flow. The subgradient rrule provides approximate gradients at any LP vertex solution without requiring a smooth interior point. The barrier solver's implicit differentiation rrule (Section 6.1) still exists in the codebase and can be used when `solve_barrier` is called directly, but it is not on the default training path.
