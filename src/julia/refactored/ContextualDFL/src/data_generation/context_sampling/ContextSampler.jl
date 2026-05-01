abstract type ContextSampler end

function (sampler::ContextSampler)(args...; kwargs...)
    return not_implemented(:ContextSampler)
end

function generate_context_set(nr_context::Integer)
    return [randn(3) for _ in 1:nr_context]
end

function generate_context_set(sampler::ContextSampler, nr_context::Integer)
    return [sampler() for _ in 1:nr_context]
end
