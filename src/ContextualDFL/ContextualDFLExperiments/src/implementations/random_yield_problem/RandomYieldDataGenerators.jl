import Random

function generate_benchmark_contexts(
    problem::RandomYieldProblem;
    n_contexts,
    rng=Random.default_rng(),
)
    context_count = _checked_positive_integer(n_contexts, :n_contexts)
    return [randn(rng, problem.context_dim) for _ in 1:context_count]
end

function generate_benchmark_scenarios(
    problem::RandomYieldProblem,
    context;
    n_scenarios,
    rng=Random.default_rng(),
)
    scenario_count = _checked_positive_integer(n_scenarios, :n_scenarios)
    context_vector = _checked_context_vector(context, problem.context_dim)
    return [
        sample_random_yield_scenario(problem, context_vector; rng=rng)
        for _ in 1:scenario_count
    ]
end

function random_yield_probabilities(problem::RandomYieldProblem, context)
    context_vector = _checked_context_vector(context, problem.context_dim)
    scores = problem.alpha .+ problem.beta * context_vector
    shifted_scores = scores .- maximum(scores)
    weights = exp.(shifted_scores)
    return weights ./ sum(weights)
end

function random_yield_support_scenarios(problem::RandomYieldProblem, context)
    _checked_context_vector(context, problem.context_dim)
    return [_random_yield_scenario(problem, W_eq) for W_eq in problem.W_support]
end

function sample_random_yield_scenario(
    problem::RandomYieldProblem,
    context;
    rng=Random.default_rng(),
)
    probabilities = random_yield_probabilities(problem, context)
    support_index = _sample_probability_index(probabilities, rng)
    return _random_yield_scenario(problem, problem.W_support[support_index])
end

function _random_yield_scenario(problem::RandomYieldProblem, W_eq::AbstractMatrix)
    base = base_scenario(problem)
    return ContextualDFL.ParametricScenario(;
        W_eq_xi=copy(W_eq),
        W_ineq_xi=copy(base.W_ineq),
        T_eq_xi=copy(base.T_eq),
        T_ineq_xi=copy(base.T_ineq),
        h_eq_xi=copy(base.h_eq),
        h_ineq_xi=copy(base.h_ineq),
        q_xi=copy(base.q),
    )
end

function _sample_probability_index(probabilities::AbstractVector, rng::Random.AbstractRNG)
    u = rand(rng)
    cumulative = 0.0
    for index in eachindex(probabilities)
        cumulative += probabilities[index]
        u <= cumulative && return index
    end
    return lastindex(probabilities)
end
