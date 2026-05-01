function cost_function_rrule(program::StochasticProgram, first_stage_decision, scenarios; solver=nothing)
    return not_implemented(:cost_function_rrule)
end

function ChainRulesCore.rrule(
    ::typeof(cost_function),
    program::StochasticProgram,
    first_stage_decision,
    scenarios;
    solver=nothing,
)
    return not_implemented(:cost_function_rrule)
end
