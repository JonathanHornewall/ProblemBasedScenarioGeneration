function solve(program::StochasticProgram, solver::Solver, scenario; config=nothing)
    return not_implemented(:solve)
end

function solve(program::StochasticProgram, solver::Solver, scenarios::AbstractVector; config=nothing)
    return not_implemented(:solve)
end
