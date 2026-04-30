struct GLPKSolver{C} <: Solver
    config::C
end

#=
struct GLPKImplementation{T} <: LPImplementation
    implementation::T
end
=#

function implement(solver::GLPKSolver, lp::LP, config=nothing)
    return not_implemented(:GLPKSolver)
end

function solve(solver::GLPKSolver, lp::LP, config=nothing; kwargs...)
    return not_implemented(:GLPKSolver)
end
