struct ResourceAllocationPhysicalCostVectorDecoder{TBaseScenario,TCosts,TYields,TH} <:
       ContextualDFL.VectorDecoder
    base_scenario::TBaseScenario
    first_stage_costs::TCosts
    yield_parameters::TYields
    fixed_h_eq::TH
end

struct ResourceAllocationFullCostVectorDecoder{TBaseScenario,TCosts,TYields,TH} <:
       ContextualDFL.VectorDecoder
    base_scenario::TBaseScenario
    first_stage_costs::TCosts
    yield_parameters::TYields
    fixed_h_eq::TH
end

struct ResourceAllocationOriginalCostVectorDecoder{TBaseScenario} <:
       ContextualDFL.VectorDecoder
    base_scenario::TBaseScenario
    demand_count::Int
    epsilon::Float64
    scale::Float64
end

struct ResourceAllocationEconomicCostVectorDecoder{TProblem,TH} <:
       ContextualDFL.VectorDecoder
    problem::TProblem
    fixed_h_eq::TH
    epsilon::Float64
    allocation_scale::Float64
    unmet_scale::Float64
end

function ResourceAllocationPhysicalCostVectorDecoder(
    problem::ResourceAllocationProblem;
    fixed_h_eq=nothing,
    fixed_demand=nothing,
)
    return ResourceAllocationPhysicalCostVectorDecoder(
        base_scenario(problem),
        problem.problem_data.first_stage_costs,
        problem.problem_data.yield_parameters,
        _resource_allocation_fixed_h_eq(
            base_scenario(problem),
            fixed_h_eq,
            fixed_demand,
        ),
    )
end

function ResourceAllocationFullCostVectorDecoder(
    problem::ResourceAllocationProblem;
    fixed_h_eq=nothing,
    fixed_demand=nothing,
)
    return ResourceAllocationFullCostVectorDecoder(
        base_scenario(problem),
        problem.problem_data.first_stage_costs,
        problem.problem_data.yield_parameters,
        _resource_allocation_fixed_h_eq(
            base_scenario(problem),
            fixed_h_eq,
            fixed_demand,
        ),
    )
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

function ResourceAllocationEconomicCostVectorDecoder(
    problem::ResourceAllocationProblem;
    fixed_h_eq=nothing,
    fixed_demand=nothing,
    epsilon=1e-4,
    allocation_scale=1.0,
    unmet_scale=1.0,
)
    return ResourceAllocationEconomicCostVectorDecoder(
        problem,
        _resource_allocation_fixed_h_eq(
            base_scenario(problem),
            fixed_h_eq,
            fixed_demand,
        ),
        Float64(epsilon),
        Float64(allocation_scale),
        Float64(unmet_scale),
    )
end

function _resource_allocation_fixed_h_eq(scenario, fixed_h_eq, fixed_demand)
    !(isnothing(fixed_h_eq) || isnothing(fixed_demand)) &&
        throw(ArgumentError("Provide at most one of fixed_h_eq and fixed_demand."))

    if !isnothing(fixed_h_eq)
        length(fixed_h_eq) == length(scenario.h_eq) ||
            throw(DimensionMismatch(
                "fixed_h_eq has length $(length(fixed_h_eq)); expected $(length(scenario.h_eq)).",
            ))
        return collect(Float64, fixed_h_eq)
    elseif !isnothing(fixed_demand)
        return _resource_allocation_h_eq(scenario, fixed_demand)
    end

    return copy(scenario.h_eq)
end

function _resource_allocation_check_positive_scale(value, name::Symbol)
    isfinite(value) && value > 0.0 ||
        throw(ArgumentError("$(name) must be finite and positive."))
    return nothing
end

function _resource_allocation_cost_dimensions(decoder)
    scenario = decoder.base_scenario
    resource_count = length(decoder.first_stage_costs)
    demand_count = length(scenario.h_eq) - resource_count
    allocation_count = resource_count * demand_count
    return resource_count, demand_count, allocation_count
end

function _resource_allocation_check_cost_parameters(decoder)
    all(>=(0), decoder.first_stage_costs) ||
        throw(ArgumentError("Resource-allocation cost decoders require nonnegative first-stage costs."))
    all(>(0), decoder.yield_parameters) ||
        throw(ArgumentError("Resource-allocation cost decoders require positive yield parameters."))
    length(decoder.first_stage_costs) == length(decoder.yield_parameters) ||
        throw(DimensionMismatch("first_stage_costs and yield_parameters must have matching lengths."))
    return nothing
end

function _resource_allocation_cost_lower_bounds(decoder)
    _resource_allocation_check_cost_parameters(decoder)
    _, demand_count, _ = _resource_allocation_cost_dimensions(decoder)
    return repeat(
        .-decoder.first_stage_costs ./ decoder.yield_parameters;
        inner=demand_count,
    )
end

_resource_allocation_softplus(x) =
    ifelse(x > zero(x), x + log1p(exp(-x)), log1p(exp(x)))

function _resource_allocation_softplus_derivative(x)
    exp_x = exp(x)
    return ifelse(x >= zero(x), inv(one(x) + exp(-x)), exp_x / (one(x) + exp_x))
end

function _resource_allocation_physical_cost_width(decoder)
    _, demand_count, allocation_count = _resource_allocation_cost_dimensions(decoder)
    return demand_count + allocation_count
end

function _resource_allocation_full_cost_width(decoder)
    return length(decoder.base_scenario.q)
end

function _resource_allocation_economic_cost_width(decoder)
    resource_count, demand_count = size(decoder.problem.problem_data.service_rate_parameters)
    return demand_count + resource_count * demand_count
end

function (decoder::ResourceAllocationOriginalCostVectorDecoder)(raw::AbstractVector{<:Real})
    demand_count = decoder.demand_count
    length(raw) == demand_count ||
        throw(DimensionMismatch(
            "expected $demand_count resource-allocation unmet-demand costs.",
        ))
    decoder.epsilon >= 0.0 ||
        throw(ArgumentError("epsilon must be nonnegative."))
    _resource_allocation_check_positive_scale(decoder.scale, :scale)

    scenario = decoder.base_scenario
    unmet_costs =
        decoder.epsilon .+ decoder.scale .* _resource_allocation_softplus.(raw)
    q = vcat(unmet_costs, scenario.q[(demand_count + 1):end])

    return (
        scenario.W_eq,
        scenario.W_ineq,
        scenario.T_eq,
        scenario.T_ineq,
        scenario.h_eq,
        scenario.h_ineq,
        q,
    )
end

function (decoder::ResourceAllocationEconomicCostVectorDecoder)(raw::AbstractVector{<:Real})
    data = decoder.problem.problem_data
    resource_count, demand_count = size(data.service_rate_parameters)
    expected_length = demand_count + resource_count * demand_count
    length(raw) == expected_length ||
        throw(DimensionMismatch("expected $expected_length resource-allocation q values."))
    decoder.epsilon >= 0.0 ||
        throw(ArgumentError("epsilon must be nonnegative."))
    _resource_allocation_check_positive_scale(decoder.allocation_scale, :allocation_scale)
    _resource_allocation_check_positive_scale(decoder.unmet_scale, :unmet_scale)

    raw_unmet = view(raw, 1:demand_count)
    raw_allocation = view(raw, (demand_count + 1):expected_length)
    unmet_costs =
        decoder.epsilon .+
        decoder.unmet_scale .* _resource_allocation_softplus.(raw_unmet)
    allocation_lowers =
        repeat(.-data.first_stage_costs ./ data.yield_parameters; inner=demand_count)
    allocation_costs =
        allocation_lowers .+
        decoder.epsilon .+
        decoder.allocation_scale .* _resource_allocation_softplus.(raw_allocation)

    scenario = base_scenario(decoder.problem)
    q = vcat(
        unmet_costs,
        allocation_costs,
        scenario.q[(demand_count + resource_count * demand_count + 1):end],
    )

    return (
        scenario.W_eq,
        scenario.W_ineq,
        scenario.T_eq,
        scenario.T_ineq,
        decoder.fixed_h_eq,
        scenario.h_ineq,
        q,
    )
end

function (decoder::ResourceAllocationPhysicalCostVectorDecoder)(raw_cost::AbstractVector{<:Real})
    scenario = decoder.base_scenario
    resource_count, demand_count, allocation_count =
        _resource_allocation_cost_dimensions(decoder)
    expected_length = demand_count + allocation_count
    length(raw_cost) == expected_length ||
        throw(DimensionMismatch(
            "raw physical cost vector has length $(length(raw_cost)); expected $expected_length.",
        ))

    raw_unmet = view(raw_cost, 1:demand_count)
    raw_allocation = view(raw_cost, (demand_count + 1):expected_length)
    unmet_costs = _resource_allocation_softplus.(raw_unmet)
    allocation_costs =
        _resource_allocation_cost_lower_bounds(decoder) .+
        _resource_allocation_softplus.(raw_allocation)

    q = vcat(
        unmet_costs,
        allocation_costs,
        zeros(eltype(allocation_costs), resource_count + demand_count),
    )

    return (
        scenario.W_eq,
        scenario.W_ineq,
        scenario.T_eq,
        scenario.T_ineq,
        decoder.fixed_h_eq,
        scenario.h_ineq,
        q,
    )
end

function (decoder::ResourceAllocationFullCostVectorDecoder)(raw_cost::AbstractVector{<:Real})
    scenario = decoder.base_scenario
    resource_count, demand_count, allocation_count =
        _resource_allocation_cost_dimensions(decoder)
    expected_length = _resource_allocation_full_cost_width(decoder)
    length(raw_cost) == expected_length ||
        throw(DimensionMismatch(
            "raw full cost vector has length $(length(raw_cost)); expected $expected_length.",
        ))

    unmet_range = 1:demand_count
    allocation_range = (last(unmet_range) + 1):(demand_count + allocation_count)
    resource_slack_range =
        (last(allocation_range) + 1):(demand_count + allocation_count + resource_count)
    demand_slack_range = (last(resource_slack_range) + 1):expected_length

    first_stage_slack_bounds = .-decoder.first_stage_costs ./ decoder.yield_parameters
    q = vcat(
        _resource_allocation_softplus.(view(raw_cost, unmet_range)),
        _resource_allocation_cost_lower_bounds(decoder) .+
        _resource_allocation_softplus.(view(raw_cost, allocation_range)),
        first_stage_slack_bounds .+
        _resource_allocation_softplus.(view(raw_cost, resource_slack_range)),
        _resource_allocation_softplus.(view(raw_cost, demand_slack_range)),
    )

    return (
        scenario.W_eq,
        scenario.W_ineq,
        scenario.T_eq,
        scenario.T_ineq,
        decoder.fixed_h_eq,
        scenario.h_ineq,
        q,
    )
end

function _resource_allocation_physical_raw_cost_pullback(decoder, raw_cost, dq)
    _, demand_count, allocation_count = _resource_allocation_cost_dimensions(decoder)
    expected_length = demand_count + allocation_count
    gradient = zeros(promote_type(eltype(raw_cost), eltype(dq)), expected_length)

    unmet_range = 1:demand_count
    allocation_range = (demand_count + 1):expected_length

    gradient[unmet_range] .=
        view(dq, unmet_range) .*
        _resource_allocation_softplus_derivative.(view(raw_cost, unmet_range))
    gradient[allocation_range] .=
        view(dq, allocation_range) .*
        _resource_allocation_softplus_derivative.(view(raw_cost, allocation_range))

    return gradient
end

function _resource_allocation_full_raw_cost_pullback(decoder, raw_cost, dq)
    expected_length = _resource_allocation_full_cost_width(decoder)
    gradient = zeros(promote_type(eltype(raw_cost), eltype(dq)), expected_length)
    gradient .=
        view(dq, 1:expected_length) .*
        _resource_allocation_softplus_derivative.(view(raw_cost, 1:expected_length))
    return gradient
end

function _resource_allocation_economic_raw_cost_pullback(decoder, raw_cost, dq)
    data = decoder.problem.problem_data
    resource_count, demand_count = size(data.service_rate_parameters)
    expected_length = demand_count + resource_count * demand_count
    gradient = zeros(promote_type(eltype(raw_cost), eltype(dq)), expected_length)

    unmet_range = 1:demand_count
    allocation_range = (demand_count + 1):expected_length

    gradient[unmet_range] .=
        decoder.unmet_scale .*
        view(dq, unmet_range) .*
        _resource_allocation_softplus_derivative.(view(raw_cost, unmet_range))
    gradient[allocation_range] .=
        decoder.allocation_scale .*
        view(dq, allocation_range) .*
        _resource_allocation_softplus_derivative.(view(raw_cost, allocation_range))

    return gradient
end

function ChainRulesCore.rrule(
    ::typeof(ContextualDFL.decode_scenario_collection),
    decoder::ResourceAllocationPhysicalCostVectorDecoder,
    raw_cost_vector::AbstractVector{<:Real};
    nr_scenarios=nothing,
)
    isnothing(nr_scenarios) &&
        throw(ArgumentError(
            "ResourceAllocationPhysicalCostVectorDecoder rrule requires explicit nr_scenarios.",
        ))
    nr_scenarios isa Integer && nr_scenarios > 0 ||
        throw(ArgumentError("nr_scenarios must be a positive integer."))

    scenario_width = _resource_allocation_physical_cost_width(decoder)
    _resource_allocation_check_decoded_cost_length(
        raw_cost_vector,
        scenario_width,
        nr_scenarios,
    )
    output = ContextualDFL.decode_scenario_collection(
        decoder,
        raw_cost_vector;
        nr_scenarios=nr_scenarios,
    )
    project_raw_cost = ChainRulesCore.ProjectTo(raw_cost_vector)

    function physical_cost_decode_pullback(output_tangent)
        dq_array = ContextualDFL._array_cotangent(
            output_tangent,
            7,
            output[7];
            name=:q_array,
        )
        raw_matrix = reshape(raw_cost_vector, scenario_width, Int(nr_scenarios))
        draw_matrix = zeros(
            promote_type(eltype(raw_cost_vector), eltype(dq_array)),
            scenario_width,
            Int(nr_scenarios),
        )

        for scenario_index in 1:Int(nr_scenarios)
            draw_matrix[:, scenario_index] =
                _resource_allocation_physical_raw_cost_pullback(
                    decoder,
                    view(raw_matrix, :, scenario_index),
                    view(dq_array, :, scenario_index),
                )
        end

        return (
            ChainRulesCore.NoTangent(),
            ChainRulesCore.NoTangent(),
            project_raw_cost(vec(draw_matrix)),
        )
    end

    return output, physical_cost_decode_pullback
end

function ChainRulesCore.rrule(
    ::typeof(ContextualDFL.decode_scenario_collection),
    decoder::ResourceAllocationFullCostVectorDecoder,
    raw_cost_vector::AbstractVector{<:Real};
    nr_scenarios=nothing,
)
    isnothing(nr_scenarios) &&
        throw(ArgumentError(
            "ResourceAllocationFullCostVectorDecoder rrule requires explicit nr_scenarios.",
        ))
    nr_scenarios isa Integer && nr_scenarios > 0 ||
        throw(ArgumentError("nr_scenarios must be a positive integer."))

    scenario_width = _resource_allocation_full_cost_width(decoder)
    _resource_allocation_check_decoded_cost_length(
        raw_cost_vector,
        scenario_width,
        nr_scenarios,
    )
    output = ContextualDFL.decode_scenario_collection(
        decoder,
        raw_cost_vector;
        nr_scenarios=nr_scenarios,
    )
    project_raw_cost = ChainRulesCore.ProjectTo(raw_cost_vector)

    function full_cost_decode_pullback(output_tangent)
        dq_array = ContextualDFL._array_cotangent(
            output_tangent,
            7,
            output[7];
            name=:q_array,
        )
        raw_matrix = reshape(raw_cost_vector, scenario_width, Int(nr_scenarios))
        draw_matrix = zeros(
            promote_type(eltype(raw_cost_vector), eltype(dq_array)),
            scenario_width,
            Int(nr_scenarios),
        )

        for scenario_index in 1:Int(nr_scenarios)
            draw_matrix[:, scenario_index] =
                _resource_allocation_full_raw_cost_pullback(
                    decoder,
                    view(raw_matrix, :, scenario_index),
                    view(dq_array, :, scenario_index),
                )
        end

        return (
            ChainRulesCore.NoTangent(),
            ChainRulesCore.NoTangent(),
            project_raw_cost(vec(draw_matrix)),
        )
    end

    return output, full_cost_decode_pullback
end

function ChainRulesCore.rrule(
    ::typeof(ContextualDFL.decode_scenario_collection),
    decoder::ResourceAllocationEconomicCostVectorDecoder,
    raw_cost_vector::AbstractVector{<:Real};
    nr_scenarios=nothing,
)
    isnothing(nr_scenarios) &&
        throw(ArgumentError(
            "ResourceAllocationEconomicCostVectorDecoder rrule requires explicit nr_scenarios.",
        ))
    nr_scenarios isa Integer && nr_scenarios > 0 ||
        throw(ArgumentError("nr_scenarios must be a positive integer."))

    scenario_width = _resource_allocation_economic_cost_width(decoder)
    _resource_allocation_check_decoded_cost_length(
        raw_cost_vector,
        scenario_width,
        nr_scenarios,
    )
    output = ContextualDFL.decode_scenario_collection(
        decoder,
        raw_cost_vector;
        nr_scenarios=nr_scenarios,
    )
    project_raw_cost = ChainRulesCore.ProjectTo(raw_cost_vector)

    function economic_cost_decode_pullback(output_tangent)
        dq_array = ContextualDFL._array_cotangent(
            output_tangent,
            7,
            output[7];
            name=:q_array,
        )
        raw_matrix = reshape(raw_cost_vector, scenario_width, Int(nr_scenarios))
        draw_matrix = zeros(
            promote_type(eltype(raw_cost_vector), eltype(dq_array)),
            scenario_width,
            Int(nr_scenarios),
        )

        for scenario_index in 1:Int(nr_scenarios)
            draw_matrix[:, scenario_index] =
                _resource_allocation_economic_raw_cost_pullback(
                    decoder,
                    view(raw_matrix, :, scenario_index),
                    view(dq_array, :, scenario_index),
                )
        end

        return (
            ChainRulesCore.NoTangent(),
            ChainRulesCore.NoTangent(),
            project_raw_cost(vec(draw_matrix)),
        )
    end

    return output, economic_cost_decode_pullback
end

function _resource_allocation_check_decoded_cost_length(raw_cost_vector, scenario_width, nr_scenarios)
    expected_length = scenario_width * Int(nr_scenarios)
    length(raw_cost_vector) == expected_length ||
        throw(DimensionMismatch(
            "raw_cost_vector has length $(length(raw_cost_vector)); expected " *
            "$expected_length for scenario_width=$scenario_width, nr_scenarios=$nr_scenarios.",
        ))
    return nothing
end
