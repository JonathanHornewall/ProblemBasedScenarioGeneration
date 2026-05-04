function solve(
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
    _, _, _, result = _solve_stochastic_extensive(
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
    return _split_stochastic_solution(sp, result, W_eq_array, W_ineq_array, q_array)
end

function _solve_stochastic_extensive(
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
    ρ_vector =
        _stochastic_quadratic_parameter_vector(lp, sp, q_array, rho; probabilities=probabilities)

    result = try
        solve(solver, lp; μ=μ_vector, ρ=ρ_vector, kwargs...)
    catch error
        _throw_stochastic_program_failure(
            error,
            _stochastic_failure_location(W_eq_array),
            solver,
            sp,
            W_eq_array,
            W_ineq_array,
            T_eq_array,
            T_ineq_array,
            h_eq_array,
            h_ineq_array,
            q_array;
            μ=μ,
            ρ=rho,
            effective_μ=μ_vector,
            effective_ρ=ρ_vector,
            probabilities=probabilities,
            kwargs=(; kwargs...),
        )
    end

    return lp, μ_vector, ρ_vector, result
end

function _stochastic_barrier_parameter_vector(
    lp::LP,
    sp::StochasticProgram,
    W_ineq_array,
    μ;
    probabilities=nothing,
)
    μ isa AbstractVector && return _barrier_parameter_vector(lp, μ)

    K = size(W_ineq_array, 3)
    first_stage_inequalities = length(sp.first_stage_lp.b_ineq)
    second_stage_inequalities = size(W_ineq_array, 1)

    T = promote_type(
        typeof(μ),
        isnothing(probabilities) ? Float64 : eltype(probabilities),
    )
    μ_vector = zeros(T, length(lp.b_ineq))

    μ >= zero(μ) || throw(ArgumentError("μ must be non-negative."))
    μ_vector[1:first_stage_inequalities] .= μ

    probability_vector = if isnothing(probabilities)
        fill(one(T) / K, K)
    else
        length(probabilities) == K ||
            throw(DimensionMismatch("probabilities must have one entry per scenario."))
        probabilities
    end

    for k in 1:K
        first_row = first_stage_inequalities + (k - 1) * second_stage_inequalities + 1
        last_row = first_stage_inequalities + k * second_stage_inequalities
        rows = first_row:last_row
        μ_vector[rows] .= μ .* probability_vector[k]
    end

    return μ_vector
end

function _stochastic_quadratic_parameter_vector(
    lp::LP,
    sp::StochasticProgram,
    q_array,
    ρ;
    probabilities=nothing,
)
    ρ isa AbstractVector && return _quadratic_parameter_vector(lp, ρ)

    K = size(q_array, 2)
    first_stage_variables = length(sp.first_stage_lp.c)
    second_stage_variables = size(q_array, 1)

    T = promote_type(
        typeof(ρ),
        isnothing(probabilities) ? Float64 : eltype(probabilities),
    )
    ρ_vector = zeros(T, length(lp.c))

    ρ >= zero(ρ) || throw(ArgumentError("ρ must be non-negative."))
    ρ_vector[1:first_stage_variables] .= ρ

    probability_vector = if isnothing(probabilities)
        fill(one(T) / K, K)
    else
        length(probabilities) == K ||
            throw(DimensionMismatch("probabilities must have one entry per scenario."))
        probabilities
    end

    for k in 1:K
        first_col = first_stage_variables + (k - 1) * second_stage_variables + 1
        last_col = first_stage_variables + k * second_stage_variables
        cols = first_col:last_col
        ρ_vector[cols] .= ρ .* probability_vector[k]
    end

    return ρ_vector
end

function _split_stochastic_solution(sp::StochasticProgram, result, W_eq_array, W_ineq_array, q_array)
    first_stage_lp = sp.first_stage_lp
    K = size(W_eq_array, 3)
    nz = length(first_stage_lp.c)
    ny = size(q_array, 1)

    m1_eq = length(first_stage_lp.b_eq)
    m1_ineq = length(first_stage_lp.b_ineq)
    m2_eq = size(W_eq_array, 1)
    m2_ineq = size(W_ineq_array, 1)

    z = result.z[1:nz]
    y = reshape(result.z[(nz + 1):(nz + K * ny)], ny, K)

    λ_b_eq = result.dual_eq[1:m1_eq]
    λ_b_ineq = result.dual_ineq[1:m1_ineq]
    λ_h_eq_array = reshape(
        result.dual_eq[(m1_eq + 1):(m1_eq + K * m2_eq)],
        m2_eq,
        K,
    )
    λ_h_ineq_array = reshape(
        result.dual_ineq[(m1_ineq + 1):(m1_ineq + K * m2_ineq)],
        m2_ineq,
        K,
    )

    return (
        z,
        y,
        λ_b_eq,
        λ_b_ineq,
        λ_h_eq_array,
        λ_h_ineq_array,
    )
end
