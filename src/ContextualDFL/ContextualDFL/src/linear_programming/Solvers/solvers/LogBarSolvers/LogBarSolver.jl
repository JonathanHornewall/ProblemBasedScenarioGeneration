abstract type LogBarSolver end

solve(solver::LogBarSolver, lp::LP; μ=nothing, ρ=0, kwargs...) =
    error("Log-barrier LP solving is not defined for $(typeof(solver)).")

function _barrier_parameter_vector(n_inequalities::Integer, μ)
    isnothing(μ) &&
        throw(ArgumentError("A log-barrier parameter μ must be provided."))
    n_inequalities >= 0 || throw(ArgumentError("n_inequalities must be non-negative."))

    if μ isa Number
        μ >= zero(μ) || throw(ArgumentError("μ must be non-negative."))
        return fill(μ, n_inequalities)
    end

    μ isa AbstractVector ||
        throw(ArgumentError("μ must be a scalar or a vector with one entry per inequality."))
    length(μ) == n_inequalities ||
        throw(DimensionMismatch("μ must have one entry per inequality."))
    any(value -> value < zero(value), μ) &&
        throw(ArgumentError("μ entries must be non-negative."))

    return collect(μ)
end

_barrier_parameter_vector(lp::LP, μ) =
    _barrier_parameter_vector(length(lp.b_ineq), μ)

function _quadratic_parameter_vector(n_variables::Integer, ρ)
    n_variables >= 0 || throw(ArgumentError("n_variables must be non-negative."))

    if ρ isa Number
        ρ >= zero(ρ) || throw(ArgumentError("ρ must be non-negative."))
        return fill(ρ, n_variables)
    end

    ρ isa AbstractVector ||
        throw(ArgumentError("ρ must be a scalar or a vector with one entry per variable."))
    length(ρ) == n_variables ||
        throw(DimensionMismatch("ρ must have one entry per variable."))
    any(value -> value < zero(value), ρ) &&
        throw(ArgumentError("ρ entries must be non-negative."))

    return collect(ρ)
end

_quadratic_parameter_vector(lp::LP, ρ) =
    _quadratic_parameter_vector(length(lp.c), ρ)

_is_zero_barrier_parameter(μ) =
    μ isa Number ? iszero(μ) : all(iszero, μ)

_is_zero_quadratic_parameter(ρ) =
    ρ isa Number ? iszero(ρ) : all(iszero, ρ)
