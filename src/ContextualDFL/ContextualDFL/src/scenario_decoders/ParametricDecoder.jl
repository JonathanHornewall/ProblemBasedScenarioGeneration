const SCENARIO_COMPONENTS = (:W_eq, :W_ineq, :T_eq, :T_ineq, :h_eq, :h_ineq, :q)

struct ParametricDecoder{IC,W_EQ,W_INEQ,T_EQ,T_INEQ,H_EQ,H_INEQ,Q} <: ScenarioDecoder
    input_components::IC
    base_W_eq::W_EQ
    base_W_ineq::W_INEQ
    base_T_eq::T_EQ
    base_T_ineq::T_INEQ
    base_h_eq::H_EQ
    base_h_ineq::H_INEQ
    base_q::Q
end

function ParametricDecoder(
    input_components=SCENARIO_COMPONENTS;
    base_W_eq=nothing,
    base_W_ineq=nothing,
    base_T_eq=nothing,
    base_T_ineq=nothing,
    base_h_eq=nothing,
    base_h_ineq=nothing,
    base_q=nothing,
)
    issubset(input_components, SCENARIO_COMPONENTS) ||
        throw(ArgumentError("input_components must be a subset of $SCENARIO_COMPONENTS."))

    return ParametricDecoder(
        input_components,
        base_W_eq,
        base_W_ineq,
        base_T_eq,
        base_T_ineq,
        base_h_eq,
        base_h_ineq,
        base_q,
    )
end

function (decoder::ParametricDecoder)(scenario_parameters::ParametricScenario)
    W_eq = :W_eq in decoder.input_components ? scenario_parameters.W_eq_xi : decoder.base_W_eq
    W_ineq = :W_ineq in decoder.input_components ? scenario_parameters.W_ineq_xi : decoder.base_W_ineq
    T_eq = :T_eq in decoder.input_components ? scenario_parameters.T_eq_xi : decoder.base_T_eq
    T_ineq = :T_ineq in decoder.input_components ? scenario_parameters.T_ineq_xi : decoder.base_T_ineq
    h_eq = :h_eq in decoder.input_components ? scenario_parameters.h_eq_xi : decoder.base_h_eq
    h_ineq = :h_ineq in decoder.input_components ? scenario_parameters.h_ineq_xi : decoder.base_h_ineq
    q = :q in decoder.input_components ? scenario_parameters.q_xi : decoder.base_q

    any(isnothing, (W_eq, W_ineq, T_eq, T_ineq, h_eq, h_ineq, q)) &&
        throw(ArgumentError("All scenario components must be provided."))

    return W_eq, W_ineq, T_eq, T_ineq, h_eq, h_ineq, q
end

function _tuple_cotangent_component(output_tangent, index)
    output_tangent = ChainRulesCore.unthunk(output_tangent)

    if output_tangent isa ChainRulesCore.AbstractZero
        return ChainRulesCore.ZeroTangent()
    elseif output_tangent isa Tuple
        index > length(output_tangent) && return ChainRulesCore.ZeroTangent()
        return ChainRulesCore.unthunk(output_tangent[index])
    elseif output_tangent isa ChainRulesCore.Tangent
        index > length(output_tangent) && return ChainRulesCore.ZeroTangent()
        return ChainRulesCore.unthunk(output_tangent[index])
    end

    throw(
        ArgumentError(
            "Expected tuple-like cotangent for decode_scenario_collection output; got $(typeof(output_tangent)).",
        ),
    )
end

function _maybe_array_component(component, template; name)
    component = ChainRulesCore.unthunk(component)

    if _is_zero_cotangent(component)
        return ChainRulesCore.NoTangent()
    end

    component isa AbstractArray ||
        throw(ArgumentError("Expected array cotangent for $name; got $(typeof(component))."))

    size(component) == size(template) || throw(
        DimensionMismatch(
            "Cotangent for $name has size $(size(component)); expected $(size(template)).",
        ),
    )

    return component
end

function ChainRulesCore.rrule(
    ::typeof(decode_scenario_collection),
    decoder::ParametricDecoder,
    scenario_parameter_collection::AbstractVector{<:ParametricScenario},
)
    output = decode_scenario_collection(decoder, scenario_parameter_collection)
    W_eq_array,
    W_ineq_array,
    T_eq_array,
    T_ineq_array,
    h_eq_array,
    h_ineq_array,
    q_array = output

    function decode_scenario_collection_pullback(output_tangent)
        dW_eq_array = _maybe_array_component(
            _tuple_cotangent_component(output_tangent, 1),
            W_eq_array;
            name=:W_eq,
        )
        dW_ineq_array = _maybe_array_component(
            _tuple_cotangent_component(output_tangent, 2),
            W_ineq_array;
            name=:W_ineq,
        )
        dT_eq_array = _maybe_array_component(
            _tuple_cotangent_component(output_tangent, 3),
            T_eq_array;
            name=:T_eq,
        )
        dT_ineq_array = _maybe_array_component(
            _tuple_cotangent_component(output_tangent, 4),
            T_ineq_array;
            name=:T_ineq,
        )
        dh_eq_array = _maybe_array_component(
            _tuple_cotangent_component(output_tangent, 5),
            h_eq_array;
            name=:h_eq,
        )
        dh_ineq_array = _maybe_array_component(
            _tuple_cotangent_component(output_tangent, 6),
            h_ineq_array;
            name=:h_ineq,
        )
        dq_array = _maybe_array_component(
            _tuple_cotangent_component(output_tangent, 7),
            q_array;
            name=:q,
        )

        if all(
            tangent -> !(tangent isa AbstractArray),
            (dW_eq_array, dW_ineq_array, dT_eq_array, dT_ineq_array, dh_eq_array, dh_ineq_array, dq_array),
        )
            return (
                ChainRulesCore.NoTangent(),
                ChainRulesCore.NoTangent(),
                ChainRulesCore.NoTangent(),
            )
        end

        scenario_parameter_tangents = map(enumerate(scenario_parameter_collection)) do (k, scenario_parameters)
            return ChainRulesCore.Tangent{typeof(scenario_parameters)}(
                W_eq_xi=
                if :W_eq in decoder.input_components && dW_eq_array isa AbstractArray
                    ChainRulesCore.ProjectTo(scenario_parameters.W_eq_xi)(
                        view(dW_eq_array, :, :, k),
                    )
                else
                    ChainRulesCore.NoTangent()
                end,
                W_ineq_xi=
                if :W_ineq in decoder.input_components && dW_ineq_array isa AbstractArray
                    ChainRulesCore.ProjectTo(scenario_parameters.W_ineq_xi)(
                        view(dW_ineq_array, :, :, k),
                    )
                else
                    ChainRulesCore.NoTangent()
                end,
                T_eq_xi=
                if :T_eq in decoder.input_components && dT_eq_array isa AbstractArray
                    ChainRulesCore.ProjectTo(scenario_parameters.T_eq_xi)(
                        view(dT_eq_array, :, :, k),
                    )
                else
                    ChainRulesCore.NoTangent()
                end,
                T_ineq_xi=
                if :T_ineq in decoder.input_components && dT_ineq_array isa AbstractArray
                    ChainRulesCore.ProjectTo(scenario_parameters.T_ineq_xi)(
                        view(dT_ineq_array, :, :, k),
                    )
                else
                    ChainRulesCore.NoTangent()
                end,
                h_eq_xi=
                if :h_eq in decoder.input_components && dh_eq_array isa AbstractArray
                    ChainRulesCore.ProjectTo(scenario_parameters.h_eq_xi)(
                        view(dh_eq_array, :, k),
                    )
                else
                    ChainRulesCore.NoTangent()
                end,
                h_ineq_xi=
                if :h_ineq in decoder.input_components && dh_ineq_array isa AbstractArray
                    ChainRulesCore.ProjectTo(scenario_parameters.h_ineq_xi)(
                        view(dh_ineq_array, :, k),
                    )
                else
                    ChainRulesCore.NoTangent()
                end,
                q_xi=
                if :q in decoder.input_components && dq_array isa AbstractArray
                    ChainRulesCore.ProjectTo(scenario_parameters.q_xi)(
                        view(dq_array, :, k),
                    )
                else
                    ChainRulesCore.NoTangent()
                end,
            )
        end

        return (
            ChainRulesCore.NoTangent(),
            ChainRulesCore.NoTangent(),
            scenario_parameter_tangents,
        )
    end

    return output, decode_scenario_collection_pullback
end
