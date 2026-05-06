# Conservative q-learning decoder for the original resource-allocation objective.
# Learns only unmet-demand costs b_j >= epsilon.
struct ResourceAllocationOriginalCostVectorDecoder <: ContextualDFL.VectorDecoder
    base_scenario
    demand_count::Int
    epsilon::Float64
    scale::Float64
end

function ResourceAllocationOriginalCostVectorDecoder(
    problem::ResourceAllocationProblem;
    epsilon=1e-4,
    scale=1.0,
)
    _, demand_count = size(problem.problem_data.service_rate_parameters)
    return ResourceAllocationOriginalCostVectorDecoder(
        base_scenario(problem),
        demand_count,
        Float64(epsilon),
        Float64(scale),
    )
end

function (decoder::ResourceAllocationOriginalCostVectorDecoder)(raw::AbstractVector)
    J = decoder.demand_count
    length(raw) == J ||
        throw(DimensionMismatch("expected $J resource-allocation unmet-demand costs."))

    base = decoder.base_scenario
    b = decoder.epsilon .+ decoder.scale .* _decoder_softplus.(raw)
    q = vcat(b, base.q[(J + 1):end])

    return (
        base.W_eq,
        base.W_ineq,
        base.T_eq,
        base.T_ineq,
        base.h_eq,
        base.h_ineq,
        q,
    )
end

# STANDARD for resource allocation q-learning when we want nontrivial q.
# Learns allocation costs and unmet-demand costs using the exact boundedness
# lower bound a_ij >= -c_i/rho_i and b_j >= 0.
struct ResourceAllocationEconomicCostVectorDecoder <: ContextualDFL.VectorDecoder
    problem::ResourceAllocationProblem
    epsilon::Float64
    allocation_scale::Float64
    unmet_scale::Float64
end

function ResourceAllocationEconomicCostVectorDecoder(
    problem::ResourceAllocationProblem;
    epsilon=1e-4,
    allocation_scale=1.0,
    unmet_scale=1.0,
)
    return ResourceAllocationEconomicCostVectorDecoder(
        problem,
        Float64(epsilon),
        Float64(allocation_scale),
        Float64(unmet_scale),
    )
end

function (decoder::ResourceAllocationEconomicCostVectorDecoder)(raw::AbstractVector)
    data = decoder.problem.problem_data
    I, J = size(data.service_rate_parameters)
    expected_length = J + I * J
    length(raw) == expected_length ||
        throw(DimensionMismatch("expected $expected_length resource-allocation q values."))

    raw_unmet = view(raw, 1:J)
    raw_allocation = view(raw, (J + 1):expected_length)
    b = decoder.epsilon .+ decoder.unmet_scale .* _decoder_softplus.(raw_unmet)
    allocation_lowers = ChainRulesCore.ignore_derivatives() do
        repeat(-(data.first_stage_costs ./ data.yield_parameters); inner=J)
    end
    allocation_costs =
        allocation_lowers .+ decoder.epsilon .+
        decoder.allocation_scale .* _decoder_softplus.(raw_allocation)

    base = base_scenario(decoder.problem)
    q = vcat(b, allocation_costs, base.q[(J + I * J + 1):end])

    return (
        base.W_eq,
        base.W_ineq,
        base.T_eq,
        base.T_ineq,
        base.h_eq,
        base.h_ineq,
        q,
    )
end
