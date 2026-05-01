function solve_rrule(program::StochasticProgram, solver::Solver, scenario; config=nothing)
    return not_implemented(:solve_rrule)
end

function ChainRulesCore.rrule(::typeof(solve), program::StochasticProgram, solver::Solver, scenario; config=nothing)
    return not_implemented(:solve_rrule)
end
