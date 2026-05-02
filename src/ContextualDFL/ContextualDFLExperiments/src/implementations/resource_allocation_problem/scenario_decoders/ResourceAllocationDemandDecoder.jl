struct ResourceAllocationDemandDecoder{TBaseScenario} <: ContextualDFL.VectorDecoder
    base_scenario::TBaseScenario
end

ResourceAllocationDemandDecoder(problem::ResourceAllocationProblem) =
    ResourceAllocationDemandDecoder(base_scenario(problem))

function (decoder::ResourceAllocationDemandDecoder)(demand::AbstractVector)
    scenario = decoder.base_scenario
    resource_count = size(scenario.T_eq, 2)
    demand_count = length(scenario.h_eq) - resource_count
    length(demand) == demand_count ||
        throw(DimensionMismatch("demand vector must have length $demand_count."))

    return (
        scenario.W_eq,
        scenario.W_ineq,
        scenario.T_eq,
        scenario.T_ineq,
        vcat(zeros(eltype(demand), resource_count), demand),
        scenario.h_ineq,
        scenario.q,
    )
end

(decoder::ResourceAllocationDemandDecoder)(scenario::ContextualDFL.ParametricScenario) =
    decoder(scenario.h_eq_xi)
