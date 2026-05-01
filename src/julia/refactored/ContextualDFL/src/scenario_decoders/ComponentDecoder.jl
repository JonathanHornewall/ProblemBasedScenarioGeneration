abstract type ComponentDecoder end

struct DefaultComponentDecoder <: ComponentDecoder
    component_type::Symbol
end

struct EmptyComponentDecoder <: ComponentDecoder
    component_type::Symbol
end

function (decoder::ComponentDecoder)(component_data)
    return not_implemented(:ComponentDecoder)
end

function (decoder::DefaultComponentDecoder)(component_data)
    return component_data
end

function (decoder::EmptyComponentDecoder)(component_data)
    error("No data decoder is configured for scenario component $(decoder.component_type).")
end
