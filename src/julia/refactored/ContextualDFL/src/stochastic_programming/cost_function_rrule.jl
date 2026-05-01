function cost_function_rrule(program::StochasticProgram, first_stage_decision, scenarios; solver=nothing)
    return cost_function(program, first_stage_decision, scenarios; solver=solver)
end

function ChainRulesCore.rrule(
    ::typeof(cost_function),
    program::StochasticProgram,
    first_stage_decision,
    scenarios;
    solver=nothing,
)
    y = cost_function(program, first_stage_decision, scenarios; solver=solver)
    function cost_function_pullback(ybar)
        dz = zeros(eltype(first_stage_decision), length(first_stage_decision))
        return (NoTangent(), NoTangent(), ybar .* dz, NoTangent())
    end
    return y, cost_function_pullback
end
