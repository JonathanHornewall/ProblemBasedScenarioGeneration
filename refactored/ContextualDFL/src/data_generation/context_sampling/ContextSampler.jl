abstract type ContextSampler end

function (sampler::ContextSampler)(args...; kwargs...)
    return not_implemented(:ContextSampler)
end

function generate_context_set(nr_context::Integer)
    return not_implemented(:generate_context_set)
end

function generate_context_set(sampler::ContextSampler, nr_context::Integer)
    return not_implemented(:generate_context_set)
end
