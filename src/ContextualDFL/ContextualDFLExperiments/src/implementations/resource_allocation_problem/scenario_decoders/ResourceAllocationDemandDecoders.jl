struct ResourceAllocationDemandVectorDecoder{TBaseScenario} <: ContextualDFL.VectorDecoder
    base_scenario::TBaseScenario
end

ResourceAllocationDemandVectorDecoder(problem::ResourceAllocationProblem) =
    ResourceAllocationDemandVectorDecoder(base_scenario(problem))

struct ResourceAllocationDemandParametricDecoder{TBaseScenario} <: ContextualDFL.ScenarioDecoder
    base_scenario::TBaseScenario
end

ResourceAllocationDemandParametricDecoder(problem::ResourceAllocationProblem) =
    ResourceAllocationDemandParametricDecoder(base_scenario(problem))

function _resource_allocation_h_eq(scenario, demand::AbstractVector)
    # Generated demand fills the demand rows; the resource-balance rows stay fixed at zero.
    resource_count = size(scenario.T_eq, 2)
    demand_count = length(scenario.h_eq) - resource_count
    length(demand) == demand_count ||
        throw(DimensionMismatch("demand vector must have length $demand_count."))

    return vcat(zeros(eltype(demand), resource_count), demand)
end

function (decoder::ResourceAllocationDemandVectorDecoder)(demand::AbstractVector)
    scenario = decoder.base_scenario

    return (
        scenario.W_eq,
        scenario.W_ineq,
        scenario.T_eq,
        scenario.T_ineq,
        _resource_allocation_h_eq(scenario, demand),
        scenario.h_ineq,
        scenario.q,
    )
end

function (decoder::ResourceAllocationDemandParametricDecoder)(
    scenario_parameters::ContextualDFL.ParametricScenario,
)
    scenario = decoder.base_scenario

    return (
        scenario.W_eq,
        scenario.W_ineq,
        scenario.T_eq,
        scenario.T_ineq,
        _resource_allocation_h_eq(scenario, scenario_parameters.h_eq_xi),
        scenario.h_ineq,
        scenario.q,
    )
end
