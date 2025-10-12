module RefactoredLogbarLP

using LinearAlgebra

export InequalityEqualityLP, LogBarrierLP,
       slack,
       is_strictly_feasible,
       cost,
       diff_KKT_Y,
       diff_KKT_Aineq,
       diff_KKT_bineq,
       diff_KKT_Aeq,
       diff_KKT_beq,
       diff_KKT_c,
       diff_cache_computation,
       diff_opt_Aineq,
       diff_opt_bineq,
       diff_opt_Aeq,
       diff_opt_beq,
       diff_opt_c,
       diff_opt

struct InequalityEqualityLP{T<:Real}
    A_ineq::Matrix{T}
    b_ineq::Vector{T}
    A_eq::Matrix{T}
    b_eq::Vector{T}
    c::Vector{T}
    function InequalityEqualityLP(A_ineq::AbstractMatrix{T1},
                                  b_ineq::AbstractVector{T2},
                                  A_eq::AbstractMatrix{T3},
                                  b_eq::AbstractVector{T4},
                                  c::AbstractVector{T5}) where {T1<:Real,T2<:Real,T3<:Real,T4<:Real,T5<:Real}
        n = size(A_ineq, 2) > 0 ? size(A_ineq, 2) : (size(A_eq, 2) > 0 ? size(A_eq, 2) : length(c))
        size(A_ineq, 2) in (0, n) || error("Inequality matrix must have $n columns")
        size(A_eq, 2) in (0, n) || error("Equality matrix must have $n columns")
        length(c) == n || error("Cost vector must have length $n")
        size(A_ineq, 1) == length(b_ineq) || error("Inequality dimensions mismatch")
        size(A_eq, 1) == length(b_eq) || error("Equality dimensions mismatch")
        T = promote_type(T1, T2, T3, T4, T5)
        return new{T}(Matrix{T}(A_ineq), Vector{T}(b_ineq), Matrix{T}(A_eq), Vector{T}(b_eq), Vector{T}(c))
    end
end

struct LogBarrierLP{T<:Real}
    lp::InequalityEqualityLP{T}
    mu::Vector{T}
    function LogBarrierLP(lp::InequalityEqualityLP{T}, mu::AbstractVector{T}) where {T<:Real}
        size(lp.A_ineq, 1) == length(mu) || error("Regularisation parameters must match number of inequalities")
        return new{T}(lp, collect(mu))
    end
end

function LogBarrierLP(lp::InequalityEqualityLP{<:Real}, mu::Real)
    m = size(lp.A_ineq, 1)
    return LogBarrierLP(lp, fill(promote_type(eltype(lp.c), typeof(mu))(mu), m))
end

slack(instance::LogBarrierLP, x) = instance.lp.b_ineq - instance.lp.A_ineq * x

function is_strictly_feasible(instance::LogBarrierLP, x; tol=1e-9)
    lp = instance.lp
    mI = size(lp.A_ineq, 1)
    if mI > 0 && any(slack(instance, x) .<= tol)
        return false
    end
    if size(lp.A_eq, 1) > 0 && !all(isapprox.(lp.A_eq * x, lp.b_eq; atol=tol))
        return false
    end
    return true
end

function cost(instance::LogBarrierLP, x; tol=1e-8)
    lp = instance.lp
    mI = size(lp.A_ineq, 1)
    if mI == 0 || all(iszero.(instance.mu))
        size(lp.A_eq, 1) == 0 || all(isapprox.(lp.A_eq * x, lp.b_eq; atol=tol)) || error("Equality constraints violated")
        return dot(lp.c, x)
    end
    r = slack(instance, x)
    minimum(r) > tol || error("Point not strictly feasible")
    return dot(lp.c, x) - dot(instance.mu, log.(r))
end

function diff_KKT_Y(instance::LogBarrierLP, state, dual_state)
    lp = instance.lp
    x = state
    λ = dual_state
    n = length(lp.c)
    mE = size(lp.A_eq, 1)
    mI = size(lp.A_ineq, 1)
    T = promote_type(eltype(x), eltype(lp.c))
    K = zeros(T, n + mE, n + mE)
    if mI > 0 && !all(iszero.(instance.mu))
        r = slack(instance, x)
        inv_sq = instance.mu ./ (r .^ 2)
        K[1:n, 1:n] = lp.A_ineq' * (Diagonal(inv_sq) * lp.A_ineq)
    end
    if mE > 0
        K[1:n, n+1:end] .= lp.A_eq'
        K[n+1:end, 1:n] .= lp.A_eq
    end
    return Symmetric(K)
end

diff_KKT_Y(instance::LogBarrierLP, state) = diff_KKT_Y(instance, state, nothing)

function diff_KKT_Aineq(instance::LogBarrierLP, state, dual_state)
    lp = instance.lp
    x = state
    n = length(lp.c)
    mE = size(lp.A_eq, 1)
    mI = size(lp.A_ineq, 1)
    result = zeros(eltype(x), n + mE, mI, n)
    if mI == 0
        return result
    end
    r = slack(instance, x)
    w = instance.mu ./ r
    inv_sq = instance.mu ./ (r .^ 2)
    for p in 1:mI
        for q in 1:n
            for j in 1:n
                term = lp.A_ineq[p, j] * inv_sq[p] * x[q]
                result[j, p, q] += term
            end
            result[q, p, q] += w[p]
        end
    end
    return result
end

function diff_KKT_bineq(instance::LogBarrierLP, state, dual_state)
    lp = instance.lp
    x = state
    n = length(lp.c)
    mE = size(lp.A_eq, 1)
    mI = size(lp.A_ineq, 1)
    result = zeros(eltype(x), n + mE, mI)
    if mI == 0
        return result
    end
    r = slack(instance, x)
    inv_sq = instance.mu ./ (r .^ 2)
    for p in 1:mI
        for j in 1:n
            result[j, p] -= lp.A_ineq[p, j] * inv_sq[p]
        end
    end
    return result
end

function diff_KKT_Aeq(instance::LogBarrierLP, state, dual_state)
    lp = instance.lp
    λ = dual_state
    n = length(lp.c)
    mE = size(lp.A_eq, 1)
    T = eltype(state)
    if λ !== nothing
        T = promote_type(T, eltype(λ))
    end
    result = zeros(T, n + mE, mE, n)
    for p in 1:mE
        for q in 1:n
            if λ !== nothing
                result[q, p, q] += λ[p]
            end
            result[n + p, p, q] += state[q]
        end
    end
    return result
end

function diff_KKT_beq(instance::LogBarrierLP, state, dual_state)
    lp = instance.lp
    n = length(lp.c)
    mE = size(lp.A_eq, 1)
    result = zeros(eltype(state), n + mE, mE)
    if mE == 0
        return result
    end
    for p in 1:mE
        result[n + p, p] = -1
    end
    return result
end

function diff_KKT_c(instance::LogBarrierLP, state, dual_state)
    lp = instance.lp
    n = length(lp.c)
    mE = size(lp.A_eq, 1)
    result = zeros(eltype(state), n + mE, n)
    for j in 1:n
        result[j, j] = 1
    end
    return result
end

function diff_cache_computation(instance::LogBarrierLP,
                                optimal_state=[],
                                optimal_dual=[],
                                KKT_matrix=[],
                                solver=nothing)
    solver === nothing && error("A solver must be provided for cache computation")
    if optimal_state == []
        optimal_state, optimal_dual = solver(instance)
    end
    if KKT_matrix == []
        KKT_matrix = diff_KKT_Y(instance, optimal_state, optimal_dual)
    end
    return optimal_state, optimal_dual, KKT_matrix
end

function diff_opt_Aineq(instance::LogBarrierLP,
                        optimal_state=[],
                        optimal_dual=[],
                        KKT_matrix=[],
                        solver=nothing)
    solver === nothing && error("A solver must be provided")
    optimal_state, optimal_dual, KKT_matrix = diff_cache_computation(instance, optimal_state, optimal_dual, KKT_matrix, solver)
    n = length(instance.lp.c)
    mI = size(instance.lp.A_ineq, 1)
    if mI == 0
        return zeros(eltype(optimal_state), n, mI, n)
    end
    D = diff_KKT_Aineq(instance, optimal_state, optimal_dual)
    D = reshape(D, n + size(instance.lp.A_eq,1), :)
    sol = - (KKT_matrix \ D)
    sol = reshape(sol, n + size(instance.lp.A_eq,1), mI, n)
    return sol[1:n, :, :]
end

function diff_opt_bineq(instance::LogBarrierLP,
                        optimal_state=[],
                        optimal_dual=[],
                        KKT_matrix=[],
                        solver=nothing)
    solver === nothing && error("A solver must be provided")
    optimal_state, optimal_dual, KKT_matrix = diff_cache_computation(instance, optimal_state, optimal_dual, KKT_matrix, solver)
    n = length(instance.lp.c)
    D = diff_KKT_bineq(instance, optimal_state, optimal_dual)
    sol = - (KKT_matrix \ D)
    return sol[1:n, :]
end

function diff_opt_Aeq(instance::LogBarrierLP,
                      optimal_state=[],
                      optimal_dual=[],
                      KKT_matrix=[],
                      solver=nothing)
    solver === nothing && error("A solver must be provided")
    optimal_state, optimal_dual, KKT_matrix = diff_cache_computation(instance, optimal_state, optimal_dual, KKT_matrix, solver)
    n = length(instance.lp.c)
    D = diff_KKT_Aeq(instance, optimal_state, optimal_dual)
    D = reshape(D, n + size(instance.lp.A_eq,1), :)
    sol = - (KKT_matrix \ D)
    sol = reshape(sol, n + size(instance.lp.A_eq,1), size(instance.lp.A_eq,1), n)
    return sol[1:n, :, :]
end

function diff_opt_beq(instance::LogBarrierLP,
                      optimal_state=[],
                      optimal_dual=[],
                      KKT_matrix=[],
                      solver=nothing)
    solver === nothing && error("A solver must be provided")
    optimal_state, optimal_dual, KKT_matrix = diff_cache_computation(instance, optimal_state, optimal_dual, KKT_matrix, solver)
    n = length(instance.lp.c)
    D = diff_KKT_beq(instance, optimal_state, optimal_dual)
    sol = - (KKT_matrix \ D)
    return sol[1:n, :]
end

function diff_opt_c(instance::LogBarrierLP,
                    optimal_state=[],
                    optimal_dual=[],
                    KKT_matrix=[],
                    solver=nothing)
    solver === nothing && error("A solver must be provided")
    optimal_state, optimal_dual, KKT_matrix = diff_cache_computation(instance, optimal_state, optimal_dual, KKT_matrix, solver)
    n = length(instance.lp.c)
    D = diff_KKT_c(instance, optimal_state, optimal_dual)
    sol = - (KKT_matrix \ D)
    return sol[1:n, :]
end

function diff_opt(instance::LogBarrierLP;
                  optimal_state=[],
                  optimal_dual=[],
                  KKT_matrix=[],
                  solver=nothing,
                  params=[:A_ineq, :b_ineq, :A_eq, :b_eq, :c])
    allowed = Set([:A_ineq, :b_ineq, :A_eq, :b_eq, :c])
    all(p -> p in allowed, params) || error("Unsupported differentiation parameter")
    solver === nothing && error("A solver must be provided")
    optimal_state, optimal_dual, KKT_matrix = diff_cache_computation(instance, optimal_state, optimal_dual, KKT_matrix, solver)
    n = length(instance.lp.c)
    mI = size(instance.lp.A_ineq, 1)
    mE = size(instance.lp.A_eq, 1)
    D_Aineq = zeros(eltype(optimal_state), n, mI, n)
    D_bineq = zeros(eltype(optimal_state), n, mI)
    D_Aeq = zeros(eltype(optimal_state), n, mE, n)
    D_beq = zeros(eltype(optimal_state), n, mE)
    D_c = zeros(eltype(optimal_state), n, n)
    if :A_ineq in params
        D_Aineq = diff_opt_Aineq(instance, optimal_state, optimal_dual, KKT_matrix, solver)
    end
    if :b_ineq in params
        D_bineq = diff_opt_bineq(instance, optimal_state, optimal_dual, KKT_matrix, solver)
    end
    if :A_eq in params
        D_Aeq = diff_opt_Aeq(instance, optimal_state, optimal_dual, KKT_matrix, solver)
    end
    if :b_eq in params
        D_beq = diff_opt_beq(instance, optimal_state, optimal_dual, KKT_matrix, solver)
    end
    if :c in params
        D_c = diff_opt_c(instance, optimal_state, optimal_dual, KKT_matrix, solver)
    end
    return (A_ineq=D_Aineq, b_ineq=D_bineq, A_eq=D_Aeq, b_eq=D_beq, c=D_c)
end

end
