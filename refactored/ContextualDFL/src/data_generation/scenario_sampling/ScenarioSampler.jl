abstract type ScenarioSampler end

function (sampler::ScenarioSampler)(context)
    return not_implemented(:ScenarioSampler)
end

function generate_scenario_set(
    context_data,
    scenarios_per_context::Union{Integer,AbstractVector{<:Integer}},
)
    return not_implemented(:generate_scenario_set)
end

function generate_scenario_set(
    sampler::ScenarioSampler,
    context_data,
    scenarios_per_context::Union{Integer,AbstractVector{<:Integer}},
)
    return not_implemented(:generate_scenario_set)
end
