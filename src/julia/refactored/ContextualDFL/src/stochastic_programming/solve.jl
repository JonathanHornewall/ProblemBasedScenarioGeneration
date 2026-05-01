function solve(program::StochasticProgram, solver::Solver, scenario; config=nothing)
    scenarios = _scenario_vector(scenario)
    lp = construct_lp(program, scenarios)
    lp_result = solve(solver, lp, config)
    n1 = length(program.c)
    return (
        first_stage_decision=lp_result.primal[1:n1],
        primal=lp_result.primal,
        canonical_primal=lp_result.canonical_primal,
        dual=lp_result.dual,
        cache=lp_result.cache,
        objective_value=lp_result.objective_value,
        lp=lp,
        lp_result=lp_result,
    )
end

function solve(program::StochasticProgram, solver::Solver, scenarios::AbstractVector; config=nothing)
    lp = construct_lp(program, scenarios)
    lp_result = solve(solver, lp, config)
    n1 = length(program.c)
    return (
        first_stage_decision=lp_result.primal[1:n1],
        primal=lp_result.primal,
        canonical_primal=lp_result.canonical_primal,
        dual=lp_result.dual,
        cache=lp_result.cache,
        objective_value=lp_result.objective_value,
        lp=lp,
        lp_result=lp_result,
    )
end
