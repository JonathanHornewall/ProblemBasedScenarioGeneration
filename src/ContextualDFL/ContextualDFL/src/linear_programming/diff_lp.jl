import LinearAlgebra: Diagonal, I, Symmetric, bunchkaufman, factorize, rank
import SparseArrays: issparse, sparse, spzeros

function diff_solve(
    solver,
    lp::LP,
    μ;
    pre_computed=nothing,
    dc=zeros(eltype(lp.c), length(lp.c)),
    db_eq=zeros(eltype(lp.b_eq), length(lp.b_eq)),
    db_ineq=zeros(eltype(lp.b_ineq), length(lp.b_ineq)),
    tight_tol=1e-7,
    kwargs...,
)
    cache = _diff_precompute(solver, lp, μ, pre_computed, tight_tol; kwargs...)
    n = length(lp.c)
    m_eq = length(lp.b_eq)
    m_ineq = length(lp.b_ineq)

    length(dc) == n || throw(DimensionMismatch("dc must have length $n."))
    length(db_eq) == m_eq || throw(DimensionMismatch("db_eq must have length $m_eq."))
    length(db_ineq) == m_ineq || throw(DimensionMismatch("db_ineq must have length $m_ineq."))

    μ_vector = cache.μ
    T = promote_type(eltype(cache.z), eltype(lp.c), eltype(μ_vector))
    rhs_x = zeros(T, n)

    if _is_zero_barrier_parameter(μ_vector)
        all(iszero, db_ineq) ||
            (rhs_x .+= transpose(lp.A_ineq[cache.loose, :]) * (cache.d .* db_ineq[cache.loose]))
        rhs = vcat(rhs_x, db_eq, db_ineq[cache.tight])
    else
        rhs_eq = zeros(T, m_eq)
        all(iszero, dc) || (rhs_x .-= dc)
        all(iszero, db_eq) || (rhs_eq .= db_eq)
        all(iszero, db_ineq) ||
            (rhs_x .+= transpose(lp.A_ineq) * ((μ_vector .* cache.d) .* db_ineq))
        rhs = vcat(rhs_x, rhs_eq)
    end

    solution = cache.K_factorization \ rhs
    return solution[1:n]
end

function _diff_precompute(solver, lp::LP, μ, pre_computed, tight_tol; kwargs...)
    μ_vector = _barrier_parameter_vector(lp, μ)

    if !isnothing(pre_computed) && hasproperty(pre_computed, :K_factorization)
        pre_computed.μ == μ_vector ||
            throw(ArgumentError("pre_computed was built with a different μ."))
        return pre_computed
    end

    solve_result = if isnothing(pre_computed)
        if _is_zero_barrier_parameter(μ_vector)
            solve(solver, lp; kwargs...)
        else
            solve(solver, lp; μ=μ_vector, kwargs...)
        end
    else
        pre_computed
    end
    z = solve_result isa AbstractVector ? solve_result : solve_result.z
    length(z) == length(lp.c) ||
        throw(DimensionMismatch("The solution must have length $(length(lp.c))."))

    n = length(lp.c)
    m_eq = length(lp.b_eq)
    slack = lp.b_ineq - lp.A_ineq * z
    any(<(-tight_tol), slack) &&
        throw(DomainError(slack, "The solution violates inequality constraints."))

    if _is_zero_barrier_parameter(μ_vector)
        tight = findall(abs.(slack) .<= tight_tol)
        loose = findall(slack .> tight_tol)
        rank(Matrix(lp.A_eq)) == size(lp.A_eq, 1) ||
            throw(ArgumentError("Analytic-center differentiation requires A_eq to have full row rank."))

        F = Matrix(lp.A_eq)
        selected_tight = Int[]
        current_rank = rank(F)
        for index in tight
            candidate = [F; lp.A_ineq[index:index, :]]
            candidate_rank = rank(candidate)
            if candidate_rank > current_rank
                push!(selected_tight, index)
                F = candidate
                current_rank = candidate_rank
            end
        end

        tight = selected_tight
        A_loose = lp.A_ineq[loose, :]
        d = one(eltype(slack)) ./ (slack[loose] .^ 2)

        H = transpose(A_loose) * (Diagonal(d) * A_loose)
        T = promote_type(eltype(H), eltype(F), eltype(μ_vector))
        K = if issparse(H) || issparse(F)
            [
                sparse(H) sparse(transpose(F))
                sparse(F) spzeros(T, size(F, 1), size(F, 1))
            ]
        else
            [
                H transpose(F)
                F zeros(T, size(F, 1), size(F, 1))
            ]
        end

        K_factorization = issparse(K) ? factorize(K) : bunchkaufman(Symmetric(K))
        return (; z=z, d=d, K_factorization=K_factorization, μ=μ_vector, tight=tight, loose=loose)
    end

    all(>(zero(eltype(slack))), slack) ||
        throw(DomainError(slack, "The log-barrier solution must have positive inequality slack."))
    rank(Matrix(lp.A_eq)) == size(lp.A_eq, 1) ||
        throw(ArgumentError("Log-barrier differentiation requires A_eq to have full row rank."))

    d = one(eltype(slack)) ./ (slack .^ 2)
    H = transpose(lp.A_ineq) * (Diagonal(μ_vector .* d) * lp.A_ineq)
    T = promote_type(eltype(H), eltype(lp.A_eq), eltype(μ_vector))
    K = if issparse(H) || issparse(lp.A_eq)
        [
            sparse(H) sparse(transpose(lp.A_eq))
            sparse(lp.A_eq) spzeros(T, m_eq, m_eq)
        ]
    else
        [
            H transpose(lp.A_eq)
            lp.A_eq zeros(T, m_eq, m_eq)
        ]
    end

    K_factorization = issparse(K) ? factorize(K) : bunchkaufman(Symmetric(K))
    return (; z=z, d=d, K_factorization=K_factorization, μ=μ_vector, tight=Int[], loose=collect(1:length(lp.b_ineq)))
end
