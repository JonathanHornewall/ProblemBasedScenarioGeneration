ShipmentPlanningParametricDecoder(problem::ShipmentPlanningProblem) =
    ContextualDFL.ParametricDecoder()

# STANDARD for shipment-planning h-learning.
# Uses affine unconstrained demand. In the literature-aligned formulation,
# demand rows have a negative demand-slack column, so arbitrary h is feasible.
struct ShipmentPlanningDemandVectorDecoder <: ContextualDFL.VectorDecoder
    base_scenario
    demand_count::Int
    offset::Vector{Float64}
    scale::Vector{Float64}
end

function ShipmentPlanningDemandVectorDecoder(
    problem::ShipmentPlanningProblem;
    offset=problem.demand_intercepts,
    scale=fill(10.0, problem.demand_count),
)
    return ShipmentPlanningDemandVectorDecoder(
        base_scenario(problem),
        problem.demand_count,
        _checked_vector_or_default(offset, problem.demand_intercepts, problem.demand_count, :offset),
        _checked_vector_or_default(scale, fill(10.0, problem.demand_count), problem.demand_count, :scale),
    )
end

function (decoder::ShipmentPlanningDemandVectorDecoder)(raw::AbstractVector)
    J = decoder.demand_count
    length(raw) == J || throw(DimensionMismatch("expected $J shipment demand values."))

    demand = decoder.offset .+ decoder.scale .* raw
    base = decoder.base_scenario
    h_eq = vcat(demand, base.h_eq[(J + 1):end])

    return (
        base.W_eq,
        base.W_ineq,
        base.T_eq,
        base.T_ineq,
        h_eq,
        base.h_ineq,
        base.q,
    )
end

# Conservative shipment h decoder.
# Keeps generated demand nonnegative for physical interpretability.
struct ShipmentPlanningPositiveDemandVectorDecoder <: ContextualDFL.VectorDecoder
    base_scenario
    demand_count::Int
    epsilon::Float64
    scale::Vector{Float64}
end

function ShipmentPlanningPositiveDemandVectorDecoder(
    problem::ShipmentPlanningProblem;
    epsilon=1e-4,
    scale=fill(10.0, problem.demand_count),
)
    return ShipmentPlanningPositiveDemandVectorDecoder(
        base_scenario(problem),
        problem.demand_count,
        Float64(epsilon),
        _checked_vector_or_default(scale, fill(10.0, problem.demand_count), problem.demand_count, :scale),
    )
end

function (decoder::ShipmentPlanningPositiveDemandVectorDecoder)(raw::AbstractVector)
    J = decoder.demand_count
    length(raw) == J || throw(DimensionMismatch("expected $J shipment demand values."))

    demand = decoder.epsilon .+ decoder.scale .* _decoder_softplus.(raw)
    base = decoder.base_scenario
    h_eq = vcat(demand, base.h_eq[(J + 1):end])

    return (
        base.W_eq,
        base.W_ineq,
        base.T_eq,
        base.T_ineq,
        h_eq,
        base.h_ineq,
        base.q,
    )
end

# Conservative shipment q decoder.
# Learns positive shipping costs only. Emergency and slack costs stay fixed.
struct ShipmentPlanningPositiveShippingCostVectorDecoder <: ContextualDFL.VectorDecoder
    problem::ShipmentPlanningProblem
    epsilon::Float64
    scale::Float64
end

function ShipmentPlanningPositiveShippingCostVectorDecoder(
    problem::ShipmentPlanningProblem;
    epsilon=1e-4,
    scale=1.0,
)
    return ShipmentPlanningPositiveShippingCostVectorDecoder(
        problem,
        Float64(epsilon),
        Float64(scale),
    )
end

function (decoder::ShipmentPlanningPositiveShippingCostVectorDecoder)(raw::AbstractVector)
    I = decoder.problem.warehouse_count
    J = decoder.problem.demand_count
    expected_length = I * J
    length(raw) == expected_length ||
        throw(DimensionMismatch("expected $expected_length shipment costs."))

    shipping = decoder.epsilon .+ decoder.scale .* _decoder_softplus.(raw)
    base = base_scenario(decoder.problem)
    q = vcat(base.q[1:I], shipping, base.q[(I + expected_length + 1):end])

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

# STANDARD for shipment-planning q-learning.
# Learns shipping costs only, allowing negative shipping costs down to the
# finite-value lower bound -min(c_i, emergency_i). Emergency and slack costs
# remain fixed.
struct ShipmentPlanningEconomicShippingCostVectorDecoder <: ContextualDFL.VectorDecoder
    problem::ShipmentPlanningProblem
    epsilon::Float64
    scale::Float64
end

function ShipmentPlanningEconomicShippingCostVectorDecoder(
    problem::ShipmentPlanningProblem;
    epsilon=1e-4,
    scale=1.0,
)
    return ShipmentPlanningEconomicShippingCostVectorDecoder(
        problem,
        Float64(epsilon),
        Float64(scale),
    )
end

function (decoder::ShipmentPlanningEconomicShippingCostVectorDecoder)(raw::AbstractVector)
    I = decoder.problem.warehouse_count
    J = decoder.problem.demand_count
    expected_length = I * J
    length(raw) == expected_length ||
        throw(DimensionMismatch("expected $expected_length shipment costs."))

    program = stochastic_program(decoder.problem)
    base = base_scenario(decoder.problem)
    shipping_lowers = ChainRulesCore.ignore_derivatives() do
        repeat(-min.(program.c, base.q[1:I]); outer=J)
    end
    shipping = shipping_lowers .+ decoder.epsilon .+ decoder.scale .* _decoder_softplus.(raw)
    q = vcat(base.q[1:I], shipping, base.q[(I + expected_length + 1):end])

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
