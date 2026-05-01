abstract type ScenarioSampler end

function (sampler::ScenarioSampler)(context)
    return not_implemented(:ScenarioSampler)
end

function generate_scenario_set(
    context_data,
    scenarios_per_context::Union{Integer,AbstractVector{<:Integer}},
)
    counts = scenarios_per_context isa Integer ? fill(scenarios_per_context, length(context_data)) : scenarios_per_context
    return [[nothing for _ in 1:counts[i]] for i in eachindex(context_data)]
end

function generate_scenario_set(
    sampler::ScenarioSampler,
    context_data,
    scenarios_per_context::Union{Integer,AbstractVector{<:Integer}},
)
    counts = scenarios_per_context isa Integer ? fill(scenarios_per_context, length(context_data)) : scenarios_per_context
    length(counts) == length(context_data) || error("scenarios_per_context length must match context_data length")
    return [[sampler(context_data[i]) for _ in 1:counts[i]] for i in eachindex(context_data)]
end
