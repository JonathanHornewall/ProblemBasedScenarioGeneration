function solve_rrule(program::StochasticProgram, solver::Solver, scenario; config=nothing)
    return solve(program, solver, scenario; config=config)
end

function ChainRulesCore.rrule(::typeof(solve), program::StochasticProgram, solver::Solver, scenario; config=nothing)
    y = solve(program, solver, scenario; config=config)
    function solve_pullback(_)
        return (NoTangent(), NoTangent(), NoTangent(), NoTangent())
    end
    return y, solve_pullback
end
