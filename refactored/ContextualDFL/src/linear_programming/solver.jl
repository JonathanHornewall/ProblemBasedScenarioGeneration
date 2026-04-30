abstract type Solver end
abstract type SolverConfig end
abstract type LPImplementation end

struct ConcreteLPImplementation{T} <: LPImplementation
    implementation::T
end

struct SolverStrategy{Q,L,N}
    qp_solver::Q
    lp_solver::L
    nlp_solver::N
end

function implement(solver::Solver, lp::LP, config=nothing)
    return not_implemented(:implement)
end

function get_implementation(implementation::LPImplementation)
    return not_implemented(:get_implementation)
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
    return not_implemented(:solve)
end
