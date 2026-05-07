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
    ρ=0,
    rho=ρ,
    kwargs...,
)
    lp, μ_vector, ρ_vector, result = _solve_stochastic_extensive(
        solver,
        sp,
        W_eq_array,
        W_ineq_array,
        T_eq_array,
        T_ineq_array,
        h_eq_array,
        h_ineq_array,
        q_array;
        probabilities=probabilities,
        μ=μ,
        ρ=rho,
        kwargs...,
    )
    output = _split_stochastic_solution(sp, result, W_eq_array, W_ineq_array, q_array)

    p_vector = if isnothing(probabilities)
        fill(one(eltype(lp.c)) / size(W_eq_array, 3), size(W_eq_array, 3))
    else
        probabilities
    end

    function stochastic_solve_pullback(output_tangent)
        z, y, _, _, _, _ = output
        dz = _maybe_array_cotangent(output_tangent, 1; name=:z)
        dy = _maybe_array_cotangent(output_tangent, 2; name=:y)

        _assert_zero_cotangent_component(output_tangent, 3; name=:λ_b_eq)
        _assert_zero_cotangent_component(output_tangent, 4; name=:λ_b_ineq)
        _assert_zero_cotangent_component(output_tangent, 5; name=:λ_h_eq_array)
        _assert_zero_cotangent_component(output_tangent, 6; name=:λ_h_ineq_array)

        T = promote_type(
            eltype(lp.c),
            eltype(z),
            eltype(y),
            isnothing(dz) ? eltype(z) : eltype(dz),
            isnothing(dy) ? eltype(y) : eltype(dy),
        )
        primal_tangent = zeros(T, length(lp.c))
        nz = length(z)
        if !isnothing(dz)
            length(dz) == nz || throw(DimensionMismatch("z cotangent must have length $(nz)."))
            primal_tangent[1:nz] .= dz
        end
        if !isnothing(dy)
            length(dy) == length(y) ||
                throw(DimensionMismatch("y cotangent must have length $(length(y))."))
            offset = nz
            @inbounds for value in dy
                offset += 1
                primal_tangent[offset] = value
            end
        end

        dc, db_eq, db_ineq = _lp_reverse_from_primal_tangent(
            solver,
            lp,
            μ_vector,
            ρ_vector,
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
    ρ,
    result,
    primal_tangent;
    kwargs...,
)
    tight_tol = haskey(kwargs, :tight_tol) ? kwargs[:tight_tol] : 1e-7
    cache = _lp_reverse_precompute(lp, μ, ρ, result, tight_tol)

    n = length(lp.c)
    m_eq = length(lp.b_eq)
    m_ineq = length(lp.b_ineq)
    T = promote_type(
        eltype(lp.c),
        eltype(lp.b_eq),
        eltype(lp.b_ineq),
        eltype(cache.μ),
        eltype(cache.ρ),
        eltype(primal_tangent),
    )

    μ_vector = cache.μ
    ρ_vector = cache.ρ
    eq_basis_count = length(cache.eq_basis)
    kkt_size = _is_zero_barrier_parameter(μ_vector) ? n + eq_basis_count + length(cache.tight) : n + eq_basis_count
    rhs = zeros(T, kkt_size)
    rhs[1:n] = primal_tangent
    adjoint_solution = cache.K_factorization \ rhs
    adjoint_primal = adjoint_solution[1:n]

    dc = zeros(T, n)
    db_eq = zeros(T, m_eq)
    db_ineq = zeros(T, m_ineq)

    if _is_zero_barrier_parameter(μ_vector)
        adjoint_constraints = adjoint_solution[(n + 1):end]
        db_eq[cache.eq_basis] .= view(adjoint_constraints, 1:eq_basis_count)
        db_ineq[cache.tight] .= view(adjoint_constraints, (eq_basis_count + 1):length(adjoint_constraints))

        if _is_zero_quadratic_parameter(ρ_vector) && !isempty(cache.loose)
            db_ineq[cache.loose] .=
                cache.d .* (view(lp.A_ineq, cache.loose, :) * adjoint_primal)
        end
        if !_is_zero_quadratic_parameter(ρ_vector)
            dc .= .-adjoint_primal
        end

        return dc, db_eq, db_ineq
    end

    dc .= .-adjoint_primal
    db_eq[cache.eq_basis] .= view(adjoint_solution, (n + 1):(n + eq_basis_count))
    if m_ineq > 0
        db_ineq .= μ_vector .* cache.d .* (lp.A_ineq * adjoint_primal)
    end

    return dc, db_eq, db_ineq
end

function _lp_reverse_precompute(lp::LP, μ, result, tight_tol)
    return _lp_reverse_precompute(lp, μ, 0, result, tight_tol)
end

function _lp_reverse_precompute(lp::LP, μ, ρ, result, tight_tol)
    μ_vector = _barrier_parameter_vector(lp, μ)
    ρ_vector = _quadratic_parameter_vector(lp, ρ)

    z = result isa AbstractVector ? result : result.z
    n = length(lp.c)
    m_eq = length(lp.b_eq)
    length(z) == n || throw(DimensionMismatch("The solution must have length $(n)."))

    slack = hasproperty(result, :slack) ? result.slack : lp.b_ineq - lp.A_ineq * z
    any(<(-tight_tol), slack) &&
        throw(DomainError(slack, "The solution violates inequality constraints."))

    if _is_zero_barrier_parameter(μ_vector)
        tight = findall(abs.(slack) .<= tight_tol)
        loose = findall(slack .> tight_tol)

        eq_basis, F = _independent_constraint_rows(lp.A_eq)
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
        d = _is_zero_quadratic_parameter(ρ_vector) ?
            one(eltype(slack)) ./ (slack[loose] .^ 2) :
            zeros(promote_type(eltype(slack), eltype(ρ_vector)), length(loose))

        H = if _is_zero_quadratic_parameter(ρ_vector)
            A_loose = lp.A_ineq[loose, :]
            transpose(A_loose) * (Diagonal(d) * A_loose)
        else
            Diagonal(ρ_vector)
        end
        T = promote_type(eltype(H), eltype(F), eltype(μ_vector), eltype(ρ_vector))
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
        return (; z=z, d=d, K_factorization=K_factorization, μ=μ_vector, ρ=ρ_vector, eq_basis=eq_basis, tight=tight, loose=loose)
    end

    all(>(zero(eltype(slack))), slack) ||
        throw(DomainError(slack, "The log-barrier solution must have positive inequality slack."))

    d = one(eltype(slack)) ./ (slack .^ 2)
    H = transpose(lp.A_ineq) * (Diagonal(μ_vector .* d) * lp.A_ineq)
    H = _is_zero_quadratic_parameter(ρ_vector) ? H : H + Diagonal(ρ_vector)
    eq_basis, F = _independent_constraint_rows(lp.A_eq)
    T = promote_type(eltype(H), eltype(F), eltype(μ_vector), eltype(ρ_vector))
    K = if issparse(H) || issparse(lp.A_eq)
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
    return (; z=z, d=d, K_factorization=K_factorization, μ=μ_vector, ρ=ρ_vector, eq_basis=eq_basis, tight=Int[], loose=collect(1:length(lp.b_ineq)))
end

function _array_cotangent(output_tangent, index, template; name)
    component = _cotangent_component(output_tangent, index)
    if component isa AbstractArray
        return component
    end
    _is_zero_cotangent(component) && return zeros(eltype(template), size(template))

    throw(ArgumentError("Expected array or zero cotangent for $(name), got $(typeof(component))."))
end

function _maybe_array_cotangent(output_tangent, index; name)
    component = _cotangent_component(output_tangent, index)
    _is_zero_cotangent(component) && return nothing
    component isa AbstractArray && return component

    throw(ArgumentError("Expected array or zero cotangent for $(name), got $(typeof(component))."))
end

function _assert_zero_cotangent_component(output_tangent, index; name)
    component = _cotangent_component(output_tangent, index)
    _is_zero_cotangent(component) && return nothing

    throw(ArgumentError("The solve rrule does not support nonzero cotangents for $(name)."))
end

function _cotangent_component(output_tangent, index)
    tangent = ChainRulesCore.unthunk(output_tangent)
    if tangent isa ChainRulesCore.AbstractZero
        return ChainRulesCore.ZeroTangent()
    elseif tangent isa Tuple
        index > length(tangent) && return ChainRulesCore.ZeroTangent()
        return ChainRulesCore.unthunk(tangent[index])
    elseif tangent isa ChainRulesCore.Tangent
        index in propertynames(tangent) || return ChainRulesCore.ZeroTangent()
        return ChainRulesCore.unthunk(getproperty(tangent, index))
    end

    throw(ArgumentError("Expected tuple-like cotangent for solve output, got $(typeof(tangent))."))
end

_is_zero_cotangent(component::AbstractArray) = all(iszero, component)
_is_zero_cotangent(component::ChainRulesCore.AbstractZero) = true
_is_zero_cotangent(component::Number) = iszero(component)
_is_zero_cotangent(component) = false
