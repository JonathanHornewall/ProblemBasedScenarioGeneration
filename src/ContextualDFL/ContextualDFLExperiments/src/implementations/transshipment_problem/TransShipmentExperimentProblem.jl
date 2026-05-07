struct TransShipmentExperimentProblem <: ProgramInstance
    core_problem::ContextualDFL.TransShipmentProblem
    variant::Symbol
    context_dim::Int
    context_noise::Float64
    scenario_noise::Float64
    signal_scale::Float64
    max_multiplier::Float64
end

function TransShipmentExperimentProblem(;
    core_problem=ContextualDFL.TransShipmentProblem(),
    variant=:q_only,
    context_dim=3,
    parameter_seed=1,
    context_noise=0.25,
    scenario_noise=0.10,
    signal_scale=1.0,
    max_multiplier=8.0,
)
    checked_variant = _checked_transshipment_variant(variant)
    checked_context_dim = _checked_positive_integer(context_dim, :context_dim)
    checked_context_dim == 3 ||
        throw(ArgumentError("transshipment benchmark contexts must have length 3."))
    _checked_positive_integer(parameter_seed, :parameter_seed)
    checked_context_noise =
        _checked_nonnegative_finite_float(context_noise, :context_noise)
    checked_scenario_noise =
        _checked_nonnegative_finite_float(scenario_noise, :scenario_noise)
    checked_signal_scale =
        _checked_positive_finite_float(signal_scale, :signal_scale)
    checked_max_multiplier =
        _checked_positive_finite_float(max_multiplier, :max_multiplier)
    checked_max_multiplier > 1.0 ||
        throw(ArgumentError("max_multiplier must be greater than 1.0."))

    return TransShipmentExperimentProblem(
        core_problem,
        checked_variant,
        checked_context_dim,
        checked_context_noise,
        checked_scenario_noise,
        checked_signal_scale,
        checked_max_multiplier,
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

function _checked_nonnegative_finite_float(value, name::Symbol)
    checked_value = Float64(value)
    isfinite(checked_value) && checked_value >= 0.0 ||
        throw(ArgumentError("$(name) must be finite and nonnegative."))
    return checked_value
end

function _checked_positive_finite_float(value, name::Symbol)
    checked_value = Float64(value)
    isfinite(checked_value) && checked_value > 0.0 ||
        throw(ArgumentError("$(name) must be finite and positive."))
    return checked_value
end
