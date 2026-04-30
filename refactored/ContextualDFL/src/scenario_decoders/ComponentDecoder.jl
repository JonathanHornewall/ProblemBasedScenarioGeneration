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
    return not_implemented(:DefaultComponentDecoder)
end

function (decoder::EmptyComponentDecoder)(component_data)
    return not_implemented(:EmptyComponentDecoder)
end
