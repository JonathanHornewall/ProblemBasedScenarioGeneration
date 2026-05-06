import Random

function generate_benchmark_contexts(
    problem::UnreliableNewsvendorProblem;
    n_contexts,
    rng=Random.default_rng(),
)
    context_count = _checked_positive_integer(n_contexts, :n_contexts)
    return [
        1.0 .+ 1.0e-6 .* rand(rng, problem.context_dim)
        for _ in 1:context_count
    ]
end

function generate_benchmark_scenarios(
    problem::UnreliableNewsvendorProblem,
    context;
    n_scenarios,
    rng=Random.default_rng(),
)
    scenario_count = _checked_positive_integer(n_scenarios, :n_scenarios)
    _checked_context_vector(context, problem.context_dim)
    return [
        unreliable_newsvendor_scenario(
            problem,
            problem.demand_upper_bound * rand(rng),
            rand(rng),
        ) for _ in 1:scenario_count
    ]
end

function unreliable_newsvendor_scenario(
    problem::UnreliableNewsvendorProblem,
    demand::Real,
    reliability::Real,
)
    checked_demand = Float64(demand)
    checked_reliability = Float64(reliability)

    isfinite(checked_demand) ||
        throw(ArgumentError("demand must be finite."))
    0.0 <= checked_demand <= problem.demand_upper_bound ||
        throw(ArgumentError("demand must be between 0 and $(problem.demand_upper_bound)."))
    isfinite(checked_reliability) ||
        throw(ArgumentError("reliability must be finite."))
    0.0 <= checked_reliability <= 1.0 ||
        throw(ArgumentError("reliability must be between 0 and 1."))

    return ContextualDFL.ParametricScenario(;
        h_eq_xi=[checked_demand, checked_reliability],
    )
end
