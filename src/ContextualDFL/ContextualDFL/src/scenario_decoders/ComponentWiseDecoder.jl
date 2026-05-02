const SCENARIO_COMPONENTS = (:W_eq, :W_ineq, :T_eq, :T_ineq, :h_eq, :h_ineq, :q)

struct ComponentWiseDecoder{IC,W_EQ,W_INEQ,T_EQ,T_INEQ,H_EQ,H_INEQ,Q} <: ScenarioDecoder
    input_components::IC
    base_W_eq::W_EQ
    base_W_ineq::W_INEQ
    base_T_eq::T_EQ
    base_T_ineq::T_INEQ
    base_h_eq::H_EQ
    base_h_ineq::H_INEQ
    base_q::Q
end

function ComponentWiseDecoder(
    input_components;
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

    return ComponentWiseDecoder(
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

function (decoder::ComponentWiseDecoder)(scenario_parameters)
    W_eq = :W_eq in decoder.input_components ? scenario_parameters.W_eq : decoder.base_W_eq
    W_ineq = :W_ineq in decoder.input_components ? scenario_parameters.W_ineq : decoder.base_W_ineq
    T_eq = :T_eq in decoder.input_components ? scenario_parameters.T_eq : decoder.base_T_eq
    T_ineq = :T_ineq in decoder.input_components ? scenario_parameters.T_ineq : decoder.base_T_ineq
    h_eq = :h_eq in decoder.input_components ? scenario_parameters.h_eq : decoder.base_h_eq
    h_ineq = :h_ineq in decoder.input_components ? scenario_parameters.h_ineq : decoder.base_h_ineq
    q = :q in decoder.input_components ? scenario_parameters.q : decoder.base_q

    any(isnothing, (W_eq, W_ineq, T_eq, T_ineq, h_eq, h_ineq, q)) &&
        throw(ArgumentError("All scenario components must be provided."))

    return W_eq, W_ineq, T_eq, T_ineq, h_eq, h_ineq, q
end

function ChainRulesCore.rrule(
    ::typeof(decode_scenario_collection),
    decoder::ComponentWiseDecoder,
    scenario_parameter_collection::AbstractVector,
)
    output = decode_scenario_collection(decoder, scenario_parameter_collection)

    function decode_scenario_collection_pullback(output_tangent)
        output_tangent = ChainRulesCore.unthunk(output_tangent)
        dW_eq_array, dW_ineq_array, dT_eq_array, dT_ineq_array, dh_eq_array, dh_ineq_array, dq_array =
            try
                (
                    ChainRulesCore.unthunk(output_tangent[1]),
                    ChainRulesCore.unthunk(output_tangent[2]),
                    ChainRulesCore.unthunk(output_tangent[3]),
                    ChainRulesCore.unthunk(output_tangent[4]),
                    ChainRulesCore.unthunk(output_tangent[5]),
                    ChainRulesCore.unthunk(output_tangent[6]),
                    ChainRulesCore.unthunk(output_tangent[7]),
                )
            catch
                return (
                    ChainRulesCore.NoTangent(),
                    ChainRulesCore.NoTangent(),
                    ChainRulesCore.NoTangent(),
                )
            end

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
            names = propertynames(scenario_parameters)
            values = map(names) do name
                if name === :W_eq && :W_eq in decoder.input_components && dW_eq_array isa AbstractArray
                    view(dW_eq_array, :, :, k)
                elseif name === :W_ineq && :W_ineq in decoder.input_components && dW_ineq_array isa AbstractArray
                    view(dW_ineq_array, :, :, k)
                elseif name === :T_eq && :T_eq in decoder.input_components && dT_eq_array isa AbstractArray
                    view(dT_eq_array, :, :, k)
                elseif name === :T_ineq && :T_ineq in decoder.input_components && dT_ineq_array isa AbstractArray
                    view(dT_ineq_array, :, :, k)
                elseif name === :h_eq && :h_eq in decoder.input_components && dh_eq_array isa AbstractArray
                    view(dh_eq_array, :, k)
                elseif name === :h_ineq && :h_ineq in decoder.input_components && dh_ineq_array isa AbstractArray
                    view(dh_ineq_array, :, k)
                elseif name === :q && :q in decoder.input_components && dq_array isa AbstractArray
                    view(dq_array, :, k)
                else
                    ChainRulesCore.NoTangent()
                end
            end
            NamedTuple{names}(values)
        end

        return (
            ChainRulesCore.NoTangent(),
            ChainRulesCore.NoTangent(),
            scenario_parameter_tangents,
        )
    end

    return output, decode_scenario_collection_pullback
end
