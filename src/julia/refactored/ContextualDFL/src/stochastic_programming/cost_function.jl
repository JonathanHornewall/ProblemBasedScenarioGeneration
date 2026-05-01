function scenario_wise_cost(program::StochasticProgram, first_stage_decision, scenario; solver=nothing)
    solver = solver === nothing ? GLPKSolver() : solver
    data = _scenario_data(scenario)
    z = vec(first_stage_decision)
    b_eq = data.h_eq - data.T_eq * z
    b_in = data.h_in - data.T_in * z
    lp = LP(data.W_eq, data.W_in, b_eq, b_in, data.q, nothing)
    result = solve(solver, lp)
    return result.objective_value
end

function cost_function(program::StochasticProgram, first_stage_decision, scenarios; solver=nothing)
    scenario_vec = _scenario_vector(scenarios)
    isempty(scenario_vec) && error("At least one realized scenario is required.")
    recourse = sum(scenario_wise_cost(program, first_stage_decision, sc; solver=solver) for sc in scenario_vec) / length(scenario_vec)
    return dot(program.c, first_stage_decision) + recourse
end
