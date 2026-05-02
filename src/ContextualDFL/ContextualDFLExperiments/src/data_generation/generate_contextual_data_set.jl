function generate_contextual_data_set(contextual_data, scenario_data)
    length(contextual_data) == length(scenario_data) ||
        throw(DimensionMismatch("contextual_data and scenario_data must have the same length."))

    data_set = ContextualDFL.ContextualDataPoint[]
    for (context, scenarios) in zip(contextual_data, scenario_data)
        context isa AbstractVector ||
            throw(ArgumentError("each context must be an AbstractVector."))

        # Store a single scenario as a one-element collection, matching ContextualDFL's dataset type.
        scenario_collection = if scenarios isa ContextualDFL.ParametricScenario
            [scenarios]
        else
            scenarios isa AbstractVector{<:ContextualDFL.ParametricScenario} ||
                throw(ArgumentError("each scenario entry must be a ParametricScenario or a vector of them."))
            isempty(scenarios) && throw(ArgumentError("scenario collections must not be empty."))
            collect(scenarios)
        end

        push!(
            data_set,
            ContextualDFL.ContextualDataPoint(collect(context), scenario_collection),
        )
    end
    return data_set
end
