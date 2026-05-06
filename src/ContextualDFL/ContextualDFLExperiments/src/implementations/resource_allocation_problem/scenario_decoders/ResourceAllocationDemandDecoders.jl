import ChainRulesCore

# STANDARD for resource allocation h-learning.
# Learns the demand rows of h_eq. Use with output_activation=:identity.
# h is intentionally unconstrained; the equality-form model has slack variables
# so generated h can leave the physical nonnegative demand support.
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

function ChainRulesCore.rrule(
    ::typeof(ContextualDFL.decode_scenario_collection),
    decoder::ResourceAllocationDemandParametricDecoder,
    scenario_parameter_collection::AbstractVector{<:ContextualDFL.ParametricScenario},
)
    output = ContextualDFL.decode_scenario_collection(decoder, scenario_parameter_collection)
    demand_rows = _resource_allocation_demand_rows(decoder)

    function resource_allocation_parametric_decode_pullback(output_tangent)
        dh_eq_array = ContextualDFL._array_cotangent(
            output_tangent,
            5,
            output[5];
            name=:h_eq_array,
        )

        scenario_tangents = map(enumerate(scenario_parameter_collection)) do (k, scenario_parameters)
            ChainRulesCore.Tangent{typeof(scenario_parameters)}(
                W_eq_xi=ChainRulesCore.NoTangent(),
                W_ineq_xi=ChainRulesCore.NoTangent(),
                T_eq_xi=ChainRulesCore.NoTangent(),
                T_ineq_xi=ChainRulesCore.NoTangent(),
                h_eq_xi=ChainRulesCore.ProjectTo(scenario_parameters.h_eq_xi)(
                    view(dh_eq_array, demand_rows, k),
                ),
                h_ineq_xi=ChainRulesCore.NoTangent(),
                q_xi=ChainRulesCore.NoTangent(),
            )
        end

        return (
            ChainRulesCore.NoTangent(),
            ChainRulesCore.NoTangent(),
            scenario_tangents,
        )
    end

    return output, resource_allocation_parametric_decode_pullback
end

function _resource_allocation_demand_rows(decoder::ResourceAllocationDemandParametricDecoder)
    scenario = decoder.base_scenario
    resource_count = size(scenario.T_eq, 2)
    demand_count = length(scenario.h_eq) - resource_count
    return (resource_count + 1):(resource_count + demand_count)
end

function ChainRulesCore.rrule(
    ::typeof(ContextualDFL.decode_scenario_collection),
    decoder::ResourceAllocationDemandVectorDecoder,
    demand_vector::AbstractVector{<:Number};
    nr_scenarios=nothing,
)
    isnothing(nr_scenarios) &&
        throw(ArgumentError(
            "ResourceAllocationDemandVectorDecoder rrule requires explicit nr_scenarios.",
        ))
    nr_scenarios isa Integer && nr_scenarios > 0 ||
        throw(ArgumentError("nr_scenarios must be a positive integer."))

    scenario = decoder.base_scenario
    resource_count = size(scenario.T_eq, 2)
    demand_count = length(scenario.h_eq) - resource_count
    expected_length = demand_count * nr_scenarios
    length(demand_vector) == expected_length ||
        throw(DimensionMismatch(
            "demand_vector has length $(length(demand_vector)); expected " *
            "$(expected_length) for demand_count=$demand_count, " *
            "nr_scenarios=$nr_scenarios.",
        ))

    output = ContextualDFL.decode_scenario_collection(
        decoder,
        demand_vector;
        nr_scenarios=nr_scenarios,
    )
    demand_rows = (resource_count + 1):(resource_count + demand_count)
    project_demand = ChainRulesCore.ProjectTo(demand_vector)

    function resource_allocation_vector_decode_pullback(output_tangent)
        dh_eq_array = ContextualDFL._array_cotangent(
            output_tangent,
            5,
            output[5];
            name=:h_eq_array,
        )
        ddemand_vector = vec(copy(view(dh_eq_array, demand_rows, :)))

        return (
            ChainRulesCore.NoTangent(),
            ChainRulesCore.NoTangent(),
            project_demand(ddemand_vector),
        )
    end

    return output, resource_allocation_vector_decode_pullback
end

function _resource_allocation_demand_rows(decoder::ResourceAllocationDemandVectorDecoder)
    scenario = decoder.base_scenario
    resource_count = size(scenario.T_eq, 2)
    demand_count = length(scenario.h_eq) - resource_count
    return (resource_count + 1):(resource_count + demand_count)
end
