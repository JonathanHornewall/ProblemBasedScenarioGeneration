import Random

struct TransShipmentExperimentProblem <: ProgramInstance
    core_problem::ContextualDFL.TransShipmentProblem
    variant::Symbol
    context_dim::Int
    sigma_h::Float64
    sigma_q::Float64
    B_h::Matrix{Float64}
    B_q::Matrix{Float64}
end

function TransShipmentExperimentProblem(;
    core_problem=ContextualDFL.TransShipmentProblem(),
    variant=:q_only,
    context_dim=3,
    sigma_h=0.20,
    sigma_q=0.20,
    parameter_seed=1,
    B_h=nothing,
    B_q=nothing,
)
    checked_variant = _checked_transshipment_variant(variant)
    checked_context_dim = _checked_positive_integer(context_dim, :context_dim)
    checked_sigma_h = Float64(sigma_h)
    checked_sigma_q = Float64(sigma_q)
    checked_sigma_h >= 0.0 || throw(ArgumentError("sigma_h must be nonnegative."))
    checked_sigma_q >= 0.0 || throw(ArgumentError("sigma_q must be nonnegative."))

    mean_parameters = ContextualDFL.transshipment_mean_parameters(core_problem)
    rng = Random.MersenneTwister(parameter_seed)
    rhs_count = length(mean_parameters.rhs)
    q_count = length(mean_parameters.q)

    checked_B_h = _checked_matrix_or_default(
        B_h,
        0.08 .* randn(rng, rhs_count, checked_context_dim),
        rhs_count,
        checked_context_dim,
        :B_h,
    )
    checked_B_q = _checked_matrix_or_default(
        B_q,
        0.08 .* randn(rng, q_count, checked_context_dim),
        q_count,
        checked_context_dim,
        :B_q,
    )

    return TransShipmentExperimentProblem(
        core_problem,
        checked_variant,
        checked_context_dim,
        checked_sigma_h,
        checked_sigma_q,
        checked_B_h,
        checked_B_q,
    )
end

stochastic_program(problem::TransShipmentExperimentProblem) =
    ContextualDFL.stochastic_program(problem.core_problem)

base_scenario(problem::TransShipmentExperimentProblem) =
    ContextualDFL.base_scenario(problem.core_problem)

transshipment_decoder(problem::TransShipmentExperimentProblem) =
    ContextualDFL.TransShipmentScenarioDecoder(problem.core_problem)

function _checked_transshipment_variant(variant)
    checked_variant = Symbol(variant)
    checked_variant in (:q_only, :h_only, :h_and_q) ||
        throw(ArgumentError("variant must be one of :q_only, :h_only, or :h_and_q."))
    return checked_variant
end
