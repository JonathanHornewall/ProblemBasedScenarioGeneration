struct GurobiSolver{C} <: Solver
    config::C
end

struct GurobiImplementation{T} <: LPImplementation
    implementation::T
end

function implement(solver::GurobiSolver, lp::LP, config=nothing)
    return not_implemented(:GurobiSolver)
end

function solve(solver::GurobiSolver, lp::LP, config=nothing; kwargs...)
    return not_implemented(:GurobiSolver)
end
