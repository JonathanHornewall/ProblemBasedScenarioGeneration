import Random

function generate_benchmark_contexts(
    problem::ShipmentPlanningProblem;
    n_contexts,
    rng=Random.default_rng(),
)
    context_count = _checked_positive_integer(n_contexts, :n_contexts)
    return [abs.(randn(rng, problem.context_dim)) for _ in 1:context_count]
end

function generate_benchmark_scenarios(
    problem::ShipmentPlanningProblem,
    context;
    n_scenarios,
    rng=Random.default_rng(),
)
    scenario_count = _checked_positive_integer(n_scenarios, :n_scenarios)
    context_vector = _checked_context_vector(context, problem.context_dim)
    return [
        _shipment_planning_scenario(
            problem,
            _shipment_planning_demand(problem, context_vector, rng),
        ) for _ in 1:scenario_count
    ]
end

function _shipment_planning_demand(
    problem::ShipmentPlanningProblem,
    context::AbstractVector,
    rng::Random.AbstractRNG,
)
    features = Float64.(context) .^ problem.p
    demand = zeros(Float64, problem.demand_count)
    for j in 1:problem.demand_count
        signal = problem.demand_intercepts[j] +
                 sum(problem.demand_slopes[j, term] * features[term] for term in 1:problem.context_dim)
        demand[j] = max(1e-6, signal + problem.sigma * randn(rng))
    end
    return demand
end

function _shipment_planning_scenario(problem::ShipmentPlanningProblem, demand::AbstractVector)
    length(demand) == problem.demand_count ||
        throw(DimensionMismatch("demand must have length $(problem.demand_count)."))

    base = base_scenario(problem)
    h_eq = copy(base.h_eq)
    h_eq[1:problem.demand_count] = Float64.(demand)

    return ContextualDFL.ParametricScenario(;
        W_eq_xi=copy(base.W_eq),
        W_ineq_xi=copy(base.W_ineq),
        T_eq_xi=copy(base.T_eq),
        T_ineq_xi=copy(base.T_ineq),
        h_eq_xi=h_eq,
        h_ineq_xi=copy(base.h_ineq),
        q_xi=copy(base.q),
    )
end
