struct DataSetScenarioDecoder{ChangingComponents,S<:DecoderStrategy,B<:BaseScenario} <: ScenarioDecoder
    decoder_strategy::S
    base_scenario::B
    changing_components::ChangingComponents
end

function DataSetScenarioDecoder(base_scenario::BaseScenario, changing_components; decoder_strategy=DecoderStrategy())
    return DataSetScenarioDecoder(decoder_strategy, base_scenario, changing_components)
end

function (decoder::DataSetScenarioDecoder)(xi)
    return BaseScenario(
        _decode_component(decoder, xi, :W_eq),
        _decode_component(decoder, xi, :W_ineq),
        _decode_component(decoder, xi, :T_eq),
        _decode_component(decoder, xi, :T_ineq),
        _decode_component(decoder, xi, :h),
        _decode_component(decoder, xi, :q),
    )
end

function _component_decoder(strategy::DecoderStrategy, component::Symbol)
    field = Symbol(component, :_decoder)
    return getfield(strategy, field)
end

_component_default(base::BaseScenario, component::Symbol) = getfield(base, component)

function _decode_component(decoder::DataSetScenarioDecoder, xi, component::Symbol)
    _component_changes(decoder.changing_components, component) ||
        return _component_default(decoder.base_scenario, component)
    data = _component_payload(xi, component, decoder.changing_components)
    return _component_decoder(decoder.decoder_strategy, component)(data)
end

function _component_changes(changing_components, component::Symbol)
    changing_components isa Symbol && return changing_components === component
    changing_components isa NamedTuple && return component in keys(changing_components)
    changing_components isa AbstractDict && return haskey(changing_components, component)
    changing_components isa Tuple && return component in changing_components
    changing_components isa AbstractVector{Symbol} && return component in changing_components
    return false
end

function _component_payload(xi, component::Symbol, changing_components)
    if changing_components isa NamedTuple && component in keys(changing_components)
        return _select_payload(xi, getfield(changing_components, component))
    elseif changing_components isa AbstractDict && haskey(changing_components, component)
        return _select_payload(xi, changing_components[component])
    end

    for name in _component_field_names(component)
        if _has_property(xi, name)
            return getproperty(xi, name)
        end
    end

    if xi isa Tuple && !(xi isa NamedTuple)
        components = changing_components isa Tuple ? changing_components : Tuple(changing_components)
        idx = findfirst(==(component), components)
        idx === nothing || return xi[idx]
    end

    if xi isa AbstractArray && _single_changing_component(changing_components, component)
        return vec(xi)
    end

    error("Could not extract data for changing scenario component $component.")
end

function _select_payload(xi, selector)
    selector isa Function && return selector(xi)
    selector isa Symbol && return getproperty(xi, selector)
    if selector isa AbstractVector{<:Integer} || selector isa AbstractUnitRange{<:Integer}
        return vec(xi)[selector]
    end
    return xi[selector]
end

function _single_changing_component(changing_components, component::Symbol)
    changing_components isa Symbol && return changing_components === component
    if changing_components isa Tuple || changing_components isa AbstractVector{Symbol}
        return length(changing_components) == 1 && first(changing_components) === component
    end
    return false
end

function _component_field_names(component::Symbol)
    if component === :W_eq
        return (:W_eq, :xi_W_eq, :W, :xi_W)
    elseif component === :W_ineq
        return (:W_ineq, :xi_W_ineq, :xi_W_in, :W_in, :xi_W)
    elseif component === :T_eq
        return (:T_eq, :xi_T_eq, :T, :xi_T)
    elseif component === :T_ineq
        return (:T_ineq, :xi_T_ineq, :xi_T_in, :T_in, :xi_T)
    elseif component === :h
        return (:h, :xi_h, :demand, :xi)
    elseif component === :q
        return (:q, :xi_q)
    else
        return (component,)
    end
end

_has_property(x, name::Symbol) = hasproperty(x, name)
