"""
Implicit differentiation of the log-barrier LP optimum.

Given a cached KKT solution from `solve_barrier`, compute the sensitivity
of the primal optimum x* with respect to b and c using the implicit
function theorem applied to the KKT conditions.

KKT conditions (stationarity + feasibility):
    g₁(x,λ) = c - μ/x - A'λ = 0
    g₂(x,λ) = Ax - b         = 0

Jacobian of (g₁, g₂) w.r.t. (x, λ):
    K = [D  A']   where D = diag(μ/x²)
        [A  0 ]

Sensitivity w.r.t. b:  K [∂x/∂b; ∂λ/∂b] = [0; I_m]
Sensitivity w.r.t. c:  K [∂x/∂c; ∂λ/∂c] = [-I_n; 0]
"""

"""
    _kkt_matrix(cache::BarrierCache{T}) -> Matrix{T}

Assemble the (n+m) × (n+m) KKT matrix K = [D A'; A 0].
"""
function _kkt_matrix(cache::BarrierCache{T}) where {T}
    (; x, mu, A) = cache
    m, n = size(A)
    D_diag = mu ./ (x .^ 2)
    # Build K = [D A'; A 0] without in-place mutation (Zygote-safe)
    K = vcat(
        hcat(Diagonal(D_diag), A'),
        hcat(A, zeros(T, m, m))
    )
    return K
end

"""
    implicit_diff_h(cache::BarrierCache) -> Matrix{T}

Sensitivity of the primal optimum x* with respect to the RHS vector b (or h).

Returns ∂x*/∂b as an (n, m) matrix, where n = length(x), m = length(b).

Computed as: (K \\ [zeros(n,m); I_m])[1:n, :]
"""
function implicit_diff_h(cache::BarrierCache{T}) where {T}
    m, n = size(cache.A)
    K = _kkt_matrix(cache)

    # Right-hand side: [0_n; I_m] (no mutation for Zygote safety)
    RHS = vcat(zeros(T, n, m), Matrix{T}(I, m, m))

    sol = K \ RHS
    return sol[1:n, :]   # (n, m)
end

"""
    implicit_diff_q(cache::BarrierCache) -> Matrix{T}

Sensitivity of the primal optimum x* with respect to the cost vector c (or q).

Returns ∂x*/∂c as an (n, n) matrix.

Computed as: (K \\ [-I_n; zeros(m,n)])[1:n, :]
"""
function implicit_diff_q(cache::BarrierCache{T}) where {T}
    m, n = size(cache.A)
    K = _kkt_matrix(cache)

    # Right-hand side: [-I_n; 0_m] (no mutation for Zygote safety)
    RHS = vcat(-Matrix{T}(I, n, n), zeros(T, m, n))

    sol = K \ RHS
    return sol[1:n, :]   # (n, n)
end

"""
    recourse_multiplier(cache::BarrierCache) -> Vector{T}

Return the dual variable λ* from a barrier-solved recourse subproblem.

This is used to compute dV_s/dx₁ = -T_s' λ_s*, the sensitivity of the
second-stage value function with respect to the first-stage decision.
"""
recourse_multiplier(cache::BarrierCache) = cache.lambda
