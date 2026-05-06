transshipment_decoder(problem::ContextualDFL.TransShipmentProblem) =
    ContextualDFL.TransShipmentScenarioDecoder(problem)

# Raw component decoder. Unsafe unless caller guarantees admissible h/q.
struct TransShipmentComponentVectorDecoder{TDecoder} <: ContextualDFL.VectorDecoder
    decoder::TDecoder
    learned_components::Tuple{Vararg{Symbol}}
end

function TransShipmentComponentVectorDecoder(
    decoder::ContextualDFL.TransShipmentScenarioDecoder;
    learned_components=(:q,),
)
    return TransShipmentComponentVectorDecoder(
        decoder,
        _checked_transshipment_learned_components(learned_components),
    )
end

TransShipmentComponentVectorDecoder(
    problem::ContextualDFL.TransShipmentProblem;
    learned_components=(:q,),
) = TransShipmentComponentVectorDecoder(
    transshipment_decoder(problem);
    learned_components=learned_components,
)

TransShipmentComponentVectorDecoder(
    problem::TransShipmentExperimentProblem;
    learned_components=(:q,),
) = TransShipmentComponentVectorDecoder(
    transshipment_decoder(problem);
    learned_components=learned_components,
)

function (decoder::TransShipmentComponentVectorDecoder)(vector::AbstractVector)
    rhs_mean = decoder.decoder.mean_rhs_values
    q_mean = decoder.decoder.mean_objective_values
    components = decoder.learned_components

    scenario = if components == (:h_eq,)
        length(vector) == length(rhs_mean) ||
            throw(DimensionMismatch("expected $(length(rhs_mean)) transshipment h_eq values."))
        (; rhs=vector, q=q_mean)
    elseif components == (:q,)
        length(vector) == length(q_mean) ||
            throw(DimensionMismatch("expected $(length(q_mean)) transshipment q values."))
        (; rhs=rhs_mean, q=vector)
    else
        expected_length = length(rhs_mean) + length(q_mean)
        length(vector) == expected_length ||
            throw(DimensionMismatch("expected $expected_length transshipment h_eq and q values."))
        rhs_range = 1:length(rhs_mean)
        q_range = (length(rhs_mean) + 1):expected_length
        (; rhs=view(vector, rhs_range), q=view(vector, q_range))
    end

    return decoder.decoder(scenario)
end

function _checked_transshipment_learned_components(learned_components)
    raw_components = learned_components isa Symbol ? (learned_components,) : Tuple(learned_components)
    components = map(raw_components) do component
        symbol = Symbol(component)
        symbol in (:h, :rhs) && return :h_eq
        return symbol
    end
    components in ((:h_eq,), (:q,), (:h_eq, :q)) ||
        throw(ArgumentError("learned_components must be (:h_eq,), (:q,), or (:h_eq, :q)."))
    return components
end

# STANDARD for transshipment q-learning.
# Positive q only. This matches the synthetic contextual q generator and avoids
# unbounded single-scenario LPs.
struct TransShipmentPositiveQVectorDecoder <: ContextualDFL.VectorDecoder
    component_decoder::TransShipmentComponentVectorDecoder
    mean_q::Vector{Float64}
    epsilon::Float64
    scale::Float64
end

function TransShipmentPositiveQVectorDecoder(problem; epsilon=1e-4, scale=1.0)
    core_problem = _transshipment_core_problem(problem)
    component_decoder =
        TransShipmentComponentVectorDecoder(core_problem; learned_components=(:q,))
    mean_q = ContextualDFL.transshipment_mean_parameters(core_problem).q
    return TransShipmentPositiveQVectorDecoder(
        component_decoder,
        Vector{Float64}(mean_q),
        Float64(epsilon),
        Float64(scale),
    )
end

function (decoder::TransShipmentPositiveQVectorDecoder)(raw::AbstractVector)
    length(raw) == length(decoder.mean_q) ||
        throw(DimensionMismatch("expected $(length(decoder.mean_q)) transshipment q values."))
    q = decoder.epsilon .+ decoder.mean_q .* _decoder_softplus.(decoder.scale .* raw)
    return decoder.component_decoder(q)
end

# Optional transshipment h-learning decoder.
# Not the first recommended experiment. Positive RHS only.
struct TransShipmentPositiveHVectorDecoder <: ContextualDFL.VectorDecoder
    component_decoder::TransShipmentComponentVectorDecoder
    mean_h::Vector{Float64}
    epsilon::Float64
    scale::Float64
end

function TransShipmentPositiveHVectorDecoder(problem; epsilon=1e-4, scale=1.0)
    core_problem = _transshipment_core_problem(problem)
    component_decoder =
        TransShipmentComponentVectorDecoder(core_problem; learned_components=(:h_eq,))
    mean_h = ContextualDFL.transshipment_mean_parameters(core_problem).rhs
    return TransShipmentPositiveHVectorDecoder(
        component_decoder,
        Vector{Float64}(mean_h),
        Float64(epsilon),
        Float64(scale),
    )
end

function (decoder::TransShipmentPositiveHVectorDecoder)(raw::AbstractVector)
    length(raw) == length(decoder.mean_h) ||
        throw(DimensionMismatch("expected $(length(decoder.mean_h)) transshipment h_eq values."))
    h = decoder.epsilon .+ decoder.mean_h .* _decoder_softplus.(decoder.scale .* raw)
    return decoder.component_decoder(h)
end

# Optional transshipment h+q decoder.
# Use only after q-only works.
struct TransShipmentPositiveHQVectorDecoder <: ContextualDFL.VectorDecoder
    component_decoder::TransShipmentComponentVectorDecoder
    mean_h::Vector{Float64}
    mean_q::Vector{Float64}
    epsilon_h::Float64
    epsilon_q::Float64
    scale_h::Float64
    scale_q::Float64
end

function TransShipmentPositiveHQVectorDecoder(
    problem;
    epsilon_h=1e-4,
    epsilon_q=1e-4,
    scale_h=1.0,
    scale_q=1.0,
)
    core_problem = _transshipment_core_problem(problem)
    component_decoder =
        TransShipmentComponentVectorDecoder(core_problem; learned_components=(:h_eq, :q))
    mean_parameters = ContextualDFL.transshipment_mean_parameters(core_problem)
    return TransShipmentPositiveHQVectorDecoder(
        component_decoder,
        Vector{Float64}(mean_parameters.rhs),
        Vector{Float64}(mean_parameters.q),
        Float64(epsilon_h),
        Float64(epsilon_q),
        Float64(scale_h),
        Float64(scale_q),
    )
end

function (decoder::TransShipmentPositiveHQVectorDecoder)(raw::AbstractVector)
    H = length(decoder.mean_h)
    Q = length(decoder.mean_q)
    expected_length = H + Q
    length(raw) == expected_length ||
        throw(DimensionMismatch("expected $expected_length transshipment h_eq and q values."))

    raw_h = view(raw, 1:H)
    raw_q = view(raw, (H + 1):expected_length)
    h = decoder.epsilon_h .+ decoder.mean_h .* _decoder_softplus.(decoder.scale_h .* raw_h)
    q = decoder.epsilon_q .+ decoder.mean_q .* _decoder_softplus.(decoder.scale_q .* raw_q)
    return decoder.component_decoder(vcat(h, q))
end

_transshipment_core_problem(problem::ContextualDFL.TransShipmentProblem) = problem
_transshipment_core_problem(problem::TransShipmentExperimentProblem) = problem.core_problem
