import Random

const TRANSSHIPMENT_CONTEXT_CENTERS = [
    -1.5 -1.0 0.0
     1.5 -1.0 0.5
    -1.0  1.5 -0.5
     1.2  1.2 1.0
]

const TRANSSHIPMENT_H_MULTIPLIERS = [
    1.90 1.75 1.60 0.65 0.70 0.75 0.80
    0.70 0.75 0.80 1.85 1.70 1.60 1.55
    1.45 0.80 1.35 1.45 0.85 1.35 1.25
    0.60 1.55 0.70 0.65 1.60 0.75 1.50
]

const TRANSSHIPMENT_Q_MULTIPLIERS = [
    0.55 0.65 0.75 1.80 1.70 1.55 1.45
    1.75 1.60 1.45 0.60 0.70 0.80 0.90
    1.90 0.75 1.70 0.80 1.55 0.85 1.40
    0.80 1.85 0.75 1.70 0.70 1.55 0.85
]

function generate_benchmark_contexts(
    problem::TransShipmentExperimentProblem;
    n_contexts,
    rng=Random.default_rng(),
)
    context_count = _checked_positive_integer(n_contexts, :n_contexts)
    centers = TRANSSHIPMENT_CONTEXT_CENTERS
    return [
        Vector{Float64}(view(centers, rand(rng, axes(centers, 1)), :)) .+
        problem.context_noise .* randn(rng, problem.context_dim)
        for _ in 1:context_count
    ]
end

function generate_benchmark_scenarios(
    problem::TransShipmentExperimentProblem,
    context;
    n_scenarios,
    rng=Random.default_rng(),
)
    scenario_count = _checked_positive_integer(n_scenarios, :n_scenarios)
    context_vector = _checked_context_vector(context, problem.context_dim)
    return [
        sample_transshipment_experiment_scenario(problem, context_vector; rng=rng)
        for _ in 1:scenario_count
    ]
end

function sample_transshipment_experiment_scenario(
    problem::TransShipmentExperimentProblem,
    context;
    rng=Random.default_rng(),
)
    context_vector = _checked_context_vector(context, problem.context_dim)
    mean_parameters = ContextualDFL.transshipment_mean_parameters(problem.core_problem)
    regime = _nearest_transshipment_context_regime(context_vector)
    shared_shock = problem.scenario_noise * randn(rng)

    rhs = if problem.variant in (:h_only, :h_and_q)
        _contextual_transshipment_values(
            mean_parameters.rhs,
            view(TRANSSHIPMENT_H_MULTIPLIERS, regime, :),
            problem,
            rng,
            shared_shock,
        )
    else
        copy(mean_parameters.rhs)
    end

    q = if problem.variant in (:q_only, :h_and_q)
        _contextual_transshipment_values(
            mean_parameters.q,
            view(TRANSSHIPMENT_Q_MULTIPLIERS, regime, :),
            problem,
            rng,
            shared_shock,
        )
    else
        copy(mean_parameters.q)
    end

    return ContextualDFL.ParametricScenario(; h_eq_xi=rhs, q_xi=q)
end

function _nearest_transshipment_context_regime(context::AbstractVector)
    centers = TRANSSHIPMENT_CONTEXT_CENTERS
    best_regime = first(axes(centers, 1))
    best_distance = Inf

    for regime in axes(centers, 1)
        center = view(centers, regime, :)
        distance = sum(index -> abs2(context[index] - center[index]), eachindex(context))
        if distance < best_distance
            best_regime = regime
            best_distance = distance
        end
    end

    return best_regime
end

function _contextual_transshipment_values(
    mean_values::AbstractVector,
    multipliers::AbstractVector,
    problem::TransShipmentExperimentProblem,
    rng::Random.AbstractRNG,
    shared_shock::Real,
)
    length(mean_values) == length(multipliers) ||
        throw(DimensionMismatch("mean values and multipliers must have matching lengths."))

    idiosyncratic_noise = problem.scenario_noise .* randn(rng, length(mean_values))
    log_values =
        log.(Float64.(mean_values)) .+
        problem.signal_scale .* log.(Float64.(multipliers)) .+
        Float64(shared_shock) .+
        idiosyncratic_noise
    upper = max.(1e-4, problem.max_multiplier .* Float64.(mean_values))
    return clamp.(exp.(log_values), 1e-4, upper)
end
