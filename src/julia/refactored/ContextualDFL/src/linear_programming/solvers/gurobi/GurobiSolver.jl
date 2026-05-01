struct GurobiSolver{C} <: Solver
    config::C
end

GurobiSolver() = GurobiSolver(nothing)

#=
struct GurobiImplementation{T} <: LPImplementation
    implementation::T
end
=#

function implement(solver::GurobiSolver, lp::LP, config=nothing)
    error("Gurobi backend unavailable unless Gurobi.jl is added to ContextualDFL dependencies.")
end

function solve(solver::GurobiSolver, lp::LP, config=nothing; kwargs...)
    error("Gurobi backend unavailable unless Gurobi.jl is added to ContextualDFL dependencies.")
end
