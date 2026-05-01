abstract type Solver end
abstract type SolverConfig end

struct BarrierCache{T<:Real}
    x::Vector{T}
    lambda::Vector{T}
    mu::Vector{T}
    A::Matrix{T}
    b::Vector{T}
    c::Vector{T}
end

# The LPImplementation layer is planned, but intentionally inactive in the
# first version. Solver implementations can introduce this abstraction once
# in-place mutable LP representations are needed.
#=
abstract type LPImplementation end

struct ConcreteLPImplementation{T} <: LPImplementation
    implementation::T
end

function get_implementation(implementation::LPImplementation)
    return not_implemented(:get_implementation)
end
=#

struct SolverStrategy{Q,L,N}
    qp_solver::Q
    lp_solver::L
    nlp_solver::N
end

function implement(solver::Solver, lp::LP, config=nothing)
    return not_implemented(:implement)
end

function solve(solver::Solver, problem, config=nothing; kwargs...)
    return not_implemented(:solve)
end

function solve(
    solver::Solver,
    lp::LP,
    config=nothing;
    A_eq=nothing,
    b=nothing,
    c=nothing,
)
    return not_implemented(:solve)
end

function solve(
    lp::LP,
    solver::Solver,
    config=nothing;
    A_eq=nothing,
    b=nothing,
    c=nothing,
)
    return solve(solver, lp, config; A_eq=A_eq, b=b, c=c)
end

function _config_value(config, key::Symbol, default)
    config === nothing && return default
    if config isa NamedTuple
        return haskey(config, key) ? getfield(config, key) : default
    elseif config isa AbstractDict
        return get(config, key, default)
    elseif hasproperty(config, key)
        return getproperty(config, key)
    else
        return default
    end
end

function _solver_value(solver::Solver, config, key::Symbol, default)
    value = _config_value(config, key, nothing)
    value !== nothing && return value
    hasproperty(solver, :config) || return default
    return _config_value(getproperty(solver, :config), key, default)
end
