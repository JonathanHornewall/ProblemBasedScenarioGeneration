function scenario_wise_cost(program::StochasticProgram, first_stage_decision, scenario; solver=nothing)
    return not_implemented(:scenario_wise_cost)
end

function cost_function(program::StochasticProgram, first_stage_decision, scenarios; solver=nothing)
    return not_implemented(:cost_function)
end
