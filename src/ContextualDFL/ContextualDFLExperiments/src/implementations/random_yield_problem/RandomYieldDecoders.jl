RandomYieldParametricDecoder(problem::RandomYieldProblem) =
    ContextualDFL.ParametricDecoder()

# STANDARD for random-yield / random-W experiments.
# Fixes W and h to the base scenario and learns a positive q vector.
# Do not learn W directly.
struct RandomYieldPositiveQVectorDecoder <: ContextualDFL.VectorDecoder
    base_scenario
    epsilon::Vector{Float64}
    scale::Vector{Float64}
end

function RandomYieldPositiveQVectorDecoder(
    problem::RandomYieldProblem;
    activity_scale=2.0,
    positive_slack_scale=50.0,
    negative_slack_scale=0.1,
    epsilon=nothing,
    scale=nothing,
)
    base = base_scenario(problem)
    default_scale = vcat(
        fill(Float64(activity_scale), problem.activity_count),
        fill(Float64(positive_slack_scale), problem.product_count),
        fill(Float64(negative_slack_scale), problem.product_count),
    )
    checked_scale = _checked_vector_or_default(scale, default_scale, length(base.q), :scale)
    checked_epsilon =
        _checked_vector_or_default(epsilon, 1e-4 .* checked_scale, length(base.q), :epsilon)
    return RandomYieldPositiveQVectorDecoder(base, checked_epsilon, checked_scale)
end

function (decoder::RandomYieldPositiveQVectorDecoder)(raw::AbstractVector)
    length(raw) == length(decoder.scale) ||
        throw(DimensionMismatch("expected $(length(decoder.scale)) random-yield q values."))

    base = decoder.base_scenario
    q = decoder.epsilon .+ decoder.scale .* _decoder_softplus.(raw)

    return (
        base.W_eq,
        base.W_ineq,
        base.T_eq,
        base.T_ineq,
        base.h_eq,
        base.h_ineq,
        q,
    )
end

# Optional only. Not recommended for the random-W experiment.
# Random-yield's intended learned component is q, not h.
struct RandomYieldHVectorDecoder <: ContextualDFL.VectorDecoder
    base_scenario
    offset::Vector{Float64}
    scale::Vector{Float64}
end

function RandomYieldHVectorDecoder(
    problem::RandomYieldProblem;
    offset=base_scenario(problem).h_eq,
    scale=ones(length(base_scenario(problem).h_eq)),
)
    base = base_scenario(problem)
    return RandomYieldHVectorDecoder(
        base,
        _checked_vector_or_default(offset, base.h_eq, length(base.h_eq), :offset),
        _checked_vector_or_default(scale, ones(length(base.h_eq)), length(base.h_eq), :scale),
    )
end

function (decoder::RandomYieldHVectorDecoder)(raw::AbstractVector)
    length(raw) == length(decoder.scale) ||
        throw(DimensionMismatch("expected $(length(decoder.scale)) random-yield h_eq values."))

    base = decoder.base_scenario
    h = decoder.offset .+ decoder.scale .* raw

    return (
        base.W_eq,
        base.W_ineq,
        base.T_eq,
        base.T_ineq,
        h,
        base.h_ineq,
        base.q,
    )
end
