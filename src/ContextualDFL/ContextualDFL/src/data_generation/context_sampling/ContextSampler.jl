abstract type ContextSampler end

(sampler::ContextSampler)(args...; kwargs...) =
    error("Context sampling is not defined for $(typeof(sampler)).")

generate_context_set(sampler::ContextSampler, nr_context::Integer; kwargs...) =
    error("Context-set generation has not been implemented yet.")
