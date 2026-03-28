"""
Self-contained Newton log-barrier solver for canonical LPs.

Solves:  min  c'x - μ·Σᵢ log(xᵢ)   s.t. Ax = b,  x > 0

No JuMP or Ipopt dependency.
"""

"""
    BarrierCache{T}

Stores the solution and problem data from a barrier solve, used downstream
for implicit differentiation.

Fields:
- `x`:      primal solution (n,)
- `lambda`: dual variable for Ax=b (m,)
- `mu`:     barrier parameter
- `A`:      constraint matrix (m×n)
- `b`:      RHS vector (m,)
- `c`:      cost vector (n,)
"""
struct BarrierCache{T<:Real}
    x::Vector{T}
    lambda::Vector{T}
    mu::T
    A::Matrix{T}
    b::Vector{T}
    c::Vector{T}
end

# -----------------------------------------------------------------------
# Initialization
# -----------------------------------------------------------------------

"""
Find a strictly interior feasible point for Ax = b, x > 0.

Uses the pseudo-inverse for robustness with rank-deficient or
overdetermined A. Iteratively shifts and reprojects to obtain positivity.
"""
function _barrier_init(A::Matrix{T}, b::Vector{T}) where {T}
    m, n = size(A)

    # Minimum-norm solution of Ax = b via pseudo-inverse (robust to rank deficiency)
    x0 = pinv(A) * b

    # If all positive and feasible, we're done
    if all(x0 .> T(1e-6)) && norm(A * x0 - b) < T(1e-6) * (norm(b) + 1)
        return x0
    end

    # Shift to ensure strict positivity, then reproject
    for attempt in 1:5
        min_val = minimum(x0)
        if min_val <= T(1e-6)
            shift = abs(min_val) + T(1.0)
            x0 = x0 .+ shift
        end

        # Project back onto Ax = b using pseudo-inverse correction
        res = A * x0 - b
        correction = pinv(A) * res
        x0 = x0 - correction

        # Ensure positivity
        x0 = max.(x0, T(1e-6))

        # Check feasibility
        if norm(A * x0 - b) < T(1e-4) * (norm(b) + 1)
            break
        end
    end

    return x0
end

# -----------------------------------------------------------------------
# Main solver
# -----------------------------------------------------------------------

"""
    solve_barrier(A, b, c, mu; tol=1e-10, max_iter=200,
                  x0=nothing, lambda0=nothing) -> BarrierCache

Solve the log-barrier LP via Newton's method:

    min c'x - μ·Σ log(xᵢ)   s.t. Ax = b,  x > 0

The KKT conditions are:
    c - μ/x - A'λ = 0
    Ax - b        = 0

Newton step at (x, λ):
    [D  A'] [Δx ]   [c - μ/x - A'λ]
    [A  0 ] [Δλ ] = -[Ax - b        ]

where D = diag(μ/x²).

Supports warm-starting via `x0` and `lambda0`.

When μ = 0, falls back to `solve_lp`.
"""
function solve_barrier(
    A, b, c, mu::Real;
    tol::Real = 1e-10,
    max_iter::Int = 200,
    x0 = nothing,
    lambda0 = nothing
)
    T = promote_type(eltype(A), eltype(b), eltype(c), typeof(mu), Float64)
    A = T.(A)
    b = T.(b)
    c = T.(c)
    mu = T(mu)

    m, n = size(A)

    # Fall back to LP solver when barrier parameter is zero
    if iszero(mu)
        x_lp, lam_lp = solve_lp(A, b, c)
        return BarrierCache(T.(x_lp), T.(lam_lp), mu, A, b, c)
    end

    # --- Initialization ---
    x = x0 !== nothing ? T.(x0) : _barrier_init(A, b)
    x = max.(x, T(1e-8))   # ensure strict positivity

    lambda = lambda0 !== nothing ? T.(lambda0) : zeros(T, m)

    # --- Newton iterations ---
    #
    # Convention: Lagrangian L = c'x - μ Σ log(xᵢ) + λ'(Ax - b)
    # Stationarity: c - μ/x + A'λ = 0
    # Feasibility:  Ax - b = 0
    # KKT matrix:   K = [D A'; A 0]  where D = diag(μ/x²)
    #
    for iter in 1:max_iter
        # Primal and dual residuals
        r_dual = c - mu ./ x + A' * lambda   # stationarity residual
        r_prim = A * x - b                    # constraint violation

        res_norm = sqrt(sum(r_dual .^ 2) + sum(r_prim .^ 2))
        if res_norm < tol
            break
        end

        # Build KKT matrix K = [D A'; A 0] (no mutation for Zygote safety)
        D_diag = mu ./ (x .^ 2)
        K = vcat(
            hcat(Diagonal(D_diag), A'),
            hcat(A, zeros(T, m, m))
        )

        # Right-hand side: -[r_dual; r_prim]
        rhs = vcat(-r_dual, -r_prim)

        # Solve Newton system (with regularization for safety)
        step = try
            K \ rhs
        catch
            (K + T(1e-12) * I) \ rhs
        end

        dx      = step[1:n]
        dlambda = step[n+1:end]

        # --- Line search: maintain x > 0 ---
        alpha = T(1.0)

        # Maximum step before hitting the boundary
        neg_dx = dx .< 0
        if any(neg_dx)
            alpha = min(alpha, T(0.99) * minimum(-x[neg_dx] ./ dx[neg_dx]))
        end

        # Backtracking: ensure x > 0 after step (safety net)
        for _ in 1:50
            if all(x .+ alpha .* dx .> 0)
                break
            end
            alpha *= T(0.5)
        end

        x      = x      + alpha .* dx
        lambda = lambda + alpha .* dlambda
    end

    return BarrierCache(x, lambda, mu, A, b, c)
end

# -----------------------------------------------------------------------
# KKT residual helper (for testing)
# -----------------------------------------------------------------------

"""
    kkt_residual(cache::BarrierCache) -> Float64

Euclidean norm of the KKT conditions at the cached solution.
"""
function kkt_residual(cache::BarrierCache{T}) where {T}
    (; x, lambda, mu, A, b, c) = cache
    r_dual = c - mu ./ x + A' * lambda   # stationarity: c - μ/x + A'λ = 0
    r_prim = A * x - b                    # feasibility:  Ax - b = 0
    return sqrt(sum(r_dual .^ 2) + sum(r_prim .^ 2))
end
