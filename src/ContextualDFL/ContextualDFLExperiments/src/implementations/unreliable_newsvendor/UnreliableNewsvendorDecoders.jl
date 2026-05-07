import ChainRulesCore

# Smoke-test instance only. Not part of the main benchmark set.
# We do not add h/q-learning decoders here for the deadline experiments.
struct UnreliableNewsvendorParametricDecoder{TBaseScenario} <: ContextualDFL.ScenarioDecoder
    base_scenario::TBaseScenario
end

UnreliableNewsvendorParametricDecoder(problem::UnreliableNewsvendorProblem) =
    UnreliableNewsvendorParametricDecoder(base_scenario(problem))

struct UnreliableNewsvendorParameterVectorDecoder{TBaseScenario} <: ContextualDFL.VectorDecoder
    base_scenario::TBaseScenario
    demand_upper_bound::Float64
    demand_epsilon::Float64
    reliability_epsilon::Float64
end

function UnreliableNewsvendorParameterVectorDecoder(
    problem::UnreliableNewsvendorProblem;
    demand_epsilon=1e-6,
    reliability_epsilon=1e-6,
)
    checked_demand_epsilon = Float64(demand_epsilon)
    checked_reliability_epsilon = Float64(reliability_epsilon)
    0.0 <= checked_demand_epsilon < problem.demand_upper_bound / 2 ||
        throw(ArgumentError("demand_epsilon must be in [0, demand_upper_bound / 2)."))
    0.0 <= checked_reliability_epsilon < 0.5 ||
        throw(ArgumentError("reliability_epsilon must be in [0, 0.5)."))

    return UnreliableNewsvendorParameterVectorDecoder(
        base_scenario(problem),
        problem.demand_upper_bound,
        checked_demand_epsilon,
        checked_reliability_epsilon,
    )
end

function (decoder::UnreliableNewsvendorParametricDecoder)(
    scenario_parameters::ContextualDFL.ParametricScenario,
)
    demand, reliability = _unreliable_newsvendor_parameters(scenario_parameters)
    scenario = decoder.base_scenario

    return (
        scenario.W_eq,
        scenario.W_ineq,
        _unreliable_newsvendor_T_eq(demand, reliability),
        scenario.T_ineq,
        _unreliable_newsvendor_h_eq(demand, reliability),
        scenario.h_ineq,
        scenario.q,
    )
end

function (decoder::UnreliableNewsvendorParameterVectorDecoder)(
    raw::AbstractVector,
)
    length(raw) == 2 ||
        throw(DimensionMismatch("newsvendor vector decoder expects two raw parameters."))

    demand_width = decoder.demand_upper_bound - 2 * decoder.demand_epsilon
    reliability_width = 1.0 - 2 * decoder.reliability_epsilon
    demand = decoder.demand_epsilon + demand_width * _unreliable_newsvendor_sigmoid(raw[1])
    reliability =
        decoder.reliability_epsilon +
        reliability_width * _unreliable_newsvendor_sigmoid(raw[2])
    scenario = decoder.base_scenario

    return (
        scenario.W_eq,
        scenario.W_ineq,
        _unreliable_newsvendor_T_eq(demand, reliability),
        scenario.T_ineq,
        _unreliable_newsvendor_h_eq(demand, reliability),
        scenario.h_ineq,
        scenario.q,
    )
end

function _unreliable_newsvendor_sigmoid(x)
    if x >= zero(x)
        return inv(one(x) + exp(-x))
    end
    exp_x = exp(x)
    return exp_x / (one(x) + exp_x)
end

function _unreliable_newsvendor_parameters(
    scenario_parameters::ContextualDFL.ParametricScenario,
)
    parameters = scenario_parameters.h_eq_xi
    parameters isa AbstractVector ||
        throw(ArgumentError("newsvendor scenario parameters must be a vector [D, U]."))
    length(parameters) == 2 ||
        throw(DimensionMismatch("newsvendor scenario parameters must have length 2."))
    return parameters[1], parameters[2]
end

function _unreliable_newsvendor_T_eq(demand, reliability)
    zero_entry = zero(demand + reliability)
    return reshape([zero_entry, -reliability], 2, 1)
end

function _unreliable_newsvendor_h_eq(demand, reliability)
    zero_entry = zero(demand + reliability)
    return [-demand, zero_entry]
end

function ChainRulesCore.rrule(
    ::typeof(ContextualDFL.decode_scenario_collection),
    decoder::UnreliableNewsvendorParametricDecoder,
    scenario_parameter_collection::AbstractVector{<:ContextualDFL.ParametricScenario},
)
    output = ContextualDFL.decode_scenario_collection(decoder, scenario_parameter_collection)

    function unreliable_newsvendor_decode_pullback(output_tangent)
        dT_eq_array = ContextualDFL._array_cotangent(
            output_tangent,
            3,
            output[3];
            name=:T_eq_array,
        )
        dh_eq_array = ContextualDFL._array_cotangent(
            output_tangent,
            5,
            output[5];
            name=:h_eq_array,
        )

        scenario_tangents = map(enumerate(scenario_parameter_collection)) do (k, scenario_parameters)
            parameter_tangent = [
                -dh_eq_array[1, k],
                -dT_eq_array[2, 1, k],
            ]

            ChainRulesCore.Tangent{typeof(scenario_parameters)}(
                W_eq_xi=ChainRulesCore.NoTangent(),
                W_ineq_xi=ChainRulesCore.NoTangent(),
                T_eq_xi=ChainRulesCore.NoTangent(),
                T_ineq_xi=ChainRulesCore.NoTangent(),
                h_eq_xi=ChainRulesCore.ProjectTo(scenario_parameters.h_eq_xi)(
                    parameter_tangent,
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

    return output, unreliable_newsvendor_decode_pullback
end
