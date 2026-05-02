import LinearAlgebra: lu

# Differentiates solve with respect to h_eq_array, h_ineq_array, and q_array.
# W_eq_array, W_ineq_array, T_eq_array, and T_ineq_array are treated as constants.
function ChainRulesCore.rrule(
    ::typeof(solve),
    solver::Solver,
    sp::StochasticProgram,
    W_eq_array,
    W_ineq_array,
    T_eq_array,
    T_ineq_array,
    h_eq_array,
    h_ineq_array,
    q_array;
    probabilities=nothing,
    μ=0,
    kwargs...,
)
    lp = construct_lp(
        sp,
        W_eq_array,
        W_ineq_array,
        T_eq_array,
        T_ineq_array,
        h_eq_array,
        h_ineq_array,
        q_array;
        probabilities=probabilities,
    )
    μ_vector =
        _stochastic_barrier_parameter_vector(lp, sp, W_ineq_array, μ; probabilities=probabilities)
    result = solve(solver, lp; μ=μ_vector, kwargs...)
    output = _split_stochastic_solution(sp, result, W_eq_array, W_ineq_array, q_array)

    p_vector = if isnothing(probabilities)
        fill(one(eltype(lp.c)) / size(W_eq_array, 3), size(W_eq_array, 3))
    else
        probabilities
    end

    function stochastic_solve_pullback(output_tangent)
        z, y, _, _, _, _ = output
        dz = _array_tangent(output_tangent, 1, z)
        dy = _array_tangent(output_tangent, 2, y)
        primal_tangent = vcat(dz, vec(dy))

        dc, db_eq, db_ineq = _lp_reverse_from_primal_tangent(
            solver,
            lp,
            μ_vector,
            result,
            primal_tangent;
            kwargs...,
        )

        first_stage_lp = sp.first_stage_lp
        K = size(W_eq_array, 3)
        nz = length(first_stage_lp.c)
        ny = size(q_array, 1)
        m1_eq = length(first_stage_lp.b_eq)
        m1_ineq = length(first_stage_lp.b_ineq)
        m2_eq = size(W_eq_array, 1)
        m2_ineq = size(W_ineq_array, 1)

        T = promote_type(eltype(dc), eltype(db_eq), eltype(db_ineq))
        dh_eq_array = zeros(T, size(h_eq_array))
        dh_ineq_array = zeros(T, size(h_ineq_array))
        dq_array = zeros(T, size(q_array))

        for k in 1:K
            y_cols = (nz + (k - 1) * ny + 1):(nz + k * ny)
            eq_rows = (m1_eq + (k - 1) * m2_eq + 1):(m1_eq + k * m2_eq)
            ineq_rows = (m1_ineq + (k - 1) * m2_ineq + 1):(m1_ineq + k * m2_ineq)

            dh_eq_array[:, k] = view(db_eq, eq_rows)
            dh_ineq_array[:, k] = view(db_ineq, ineq_rows)
            dq_array[:, k] = p_vector[k] .* view(dc, y_cols)
        end

        return (
            ChainRulesCore.NoTangent(),
            ChainRulesCore.NoTangent(),
            ChainRulesCore.NoTangent(),
            ChainRulesCore.NoTangent(),
            ChainRulesCore.NoTangent(),
            ChainRulesCore.NoTangent(),
            ChainRulesCore.NoTangent(),
            dh_eq_array,
            dh_ineq_array,
            dq_array,
        )
    end

    return output, stochastic_solve_pullback
end

function _lp_reverse_from_primal_tangent(
    solver,
    lp::LP,
    μ,
    result,
    primal_tangent;
    kwargs...,
)
    tight_tol = haskey(kwargs, :tight_tol) ? kwargs[:tight_tol] : 1e-7
    cache = _lp_reverse_precompute(lp, μ, result, tight_tol)

    n = length(lp.c)
    m_eq = length(lp.b_eq)
    m_ineq = length(lp.b_ineq)
    T = promote_type(eltype(lp.c), eltype(lp.b_eq), eltype(lp.b_ineq), eltype(primal_tangent))

    μ_vector = cache.μ
    kkt_size = _is_zero_barrier_parameter(μ_vector) ? n + m_eq + length(cache.tight) : n + m_eq
    rhs = zeros(T, kkt_size)
    rhs[1:n] = primal_tangent
    adjoint_solution = cache.K_factorization \ rhs
    adjoint_primal = adjoint_solution[1:n]

    dc = zeros(T, n)
    db_eq = zeros(T, m_eq)
    db_ineq = zeros(T, m_ineq)

    if _is_zero_barrier_parameter(μ_vector)
        adjoint_constraints = adjoint_solution[(n + 1):end]
        db_eq .= view(adjoint_constraints, 1:m_eq)
        db_ineq[cache.tight] .= view(adjoint_constraints, (m_eq + 1):length(adjoint_constraints))

        if !isempty(cache.loose)
            db_ineq[cache.loose] .=
                cache.d .* (view(lp.A_ineq, cache.loose, :) * adjoint_primal)
        end

        return dc, db_eq, db_ineq
    end

    dc .= .-adjoint_primal
    db_eq .= view(adjoint_solution, (n + 1):(n + m_eq))
    if m_ineq > 0
        db_ineq .= μ_vector .* cache.d .* (lp.A_ineq * adjoint_primal)
    end

    return dc, db_eq, db_ineq
end

function _lp_reverse_precompute(lp::LP, μ, result, tight_tol)
    μ_vector = _barrier_parameter_vector(lp, μ)

    z = result isa AbstractVector ? result : result.z
    n = length(lp.c)
    m_eq = length(lp.b_eq)
    length(z) == n || throw(DimensionMismatch("The solution must have length $(n)."))

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

        K_factorization = issparse(K) ? lu(K) : bunchkaufman(Symmetric(K))
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

    K_factorization = issparse(K) ? lu(K) : bunchkaufman(Symmetric(K))
    return (; z=z, d=d, K_factorization=K_factorization, μ=μ_vector, tight=Int[], loose=collect(1:length(lp.b_ineq)))
end

function _array_tangent(output_tangent, index, template)
    tangent = ChainRulesCore.unthunk(output_tangent)
    if tangent isa Tuple && length(tangent) >= index
        component = ChainRulesCore.unthunk(tangent[index])
        component isa AbstractArray && return component
    end

    return zeros(eltype(template), size(template))
end
