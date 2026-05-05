import Random

function generate_benchmark_contexts(
    problem::TransShipmentExperimentProblem;
    n_contexts,
    rng=Random.default_rng(),
)
    context_count = _checked_positive_integer(n_contexts, :n_contexts)
    return [randn(rng, problem.context_dim) for _ in 1:context_count]
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

    rhs = if problem.variant in (:h_only, :h_and_q)
        _contextual_positive_values(
            mean_parameters.rhs,
            problem.B_h,
            context_vector,
            problem.sigma_h,
            rng,
        )
    else
        copy(mean_parameters.rhs)
    end

    q = if problem.variant in (:q_only, :h_and_q)
        _contextual_positive_values(
            mean_parameters.q,
            problem.B_q,
            context_vector,
            problem.sigma_q,
            rng,
        )
    else
        copy(mean_parameters.q)
    end

    return ContextualDFL.ParametricScenario(; h_eq_xi=rhs, q_xi=q)
end

function _contextual_positive_values(
    mean_values::AbstractVector,
    slopes::AbstractMatrix,
    context::AbstractVector,
    sigma::Real,
    rng::Random.AbstractRNG,
)
    noise = sigma .* randn(rng, length(mean_values))
    log_values = log.(Float64.(mean_values)) .+ slopes * Float64.(context) .+ noise
    return max.(1e-4, exp.(log_values))
end
