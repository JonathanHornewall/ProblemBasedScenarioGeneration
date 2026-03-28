"""
Decision regret loss for problem-based scenario generation.

The loss measures how much worse the decision is when using predicted
scenarios vs. the actual scenario.
"""

# -----------------------------------------------------------------------
# Surrogate first-stage solve
# -----------------------------------------------------------------------

"""
    surrogate_first_stage(slp::TwoStageSLP, mu::Float64; kw...) -> Vector{Float64}

Solve the extensive-form log-barrier LP and return the first-stage decision x₁*.

Steps:
1. Build extensive-form (A_ext, b_ext, c_ext) from `slp`
2. Solve with barrier parameter `mu`
3. Return first n₁ components of the solution
"""
function surrogate_first_stage(slp::TwoStageSLP, mu::Real; kw...)
    n1 = length(slp.c)
    A_ext, b_ext, c_ext = extensive_form(slp)

    # Use HiGHS LP solver (fast, with subgradient rrule for AD)
    x_full, _ = solve_lp(A_ext, b_ext, c_ext)

    return x_full[1:n1]
end

"""
    build_mu_vector(slp::TwoStageSLP, mu::Real) -> Vector

Build the extensive-form barrier parameter vector:
μ for first-stage variables, μ*pₛ for second-stage variables of scenario s.
"""
function build_mu_vector(slp::TwoStageSLP{T}, mu::Real) where {T}
    n1 = length(slp.c)
    n2 = size(slp.scenarios[1].W, 2)
    S  = length(slp.scenarios)
    # Use vcat instead of append! for Zygote compatibility (no mutation)
    return vcat(mu .* ones(T, n1),
                [mu .* slp.p[s] .* ones(T, n2) for s in 1:S]...)
end

# -----------------------------------------------------------------------
# Recourse cost evaluation
# -----------------------------------------------------------------------

"""
    recourse_cost(x1::Vector, sc::Scenario, mu::Real; kw...) -> Float64

Evaluate the second-stage cost for first-stage decision `x1` under scenario `sc`.

Solves: min q'y - μ·Σ log(yᵢ)  s.t.  W·y = h - T·x1,  y > 0
"""
function recourse_cost(x1::AbstractVector, sc::Scenario, mu::Real; kw...)
    b_rec = sc.h - sc.T * x1
    # Use HiGHS LP solver (fast, with subgradient rrule for AD)
    y_opt, _ = solve_lp(sc.W, b_rec, sc.q)
    return dot(sc.q, y_opt)
end

# -----------------------------------------------------------------------
# Total cost of a first-stage decision
# -----------------------------------------------------------------------

"""
    evaluate_cost(x1, slp::TwoStageSLP, mu; kw...) -> Float64

Compute the total cost c₁'x₁ + Σₛ pₛ · Q(x₁, sₛ) for a given first-stage
decision `x1` and scenario set, with barrier regularization `mu`.
"""
function evaluate_cost(x1::AbstractVector, slp::TwoStageSLP, mu::Real; kw...)
    cost_val = dot(slp.c, x1)
    for s in 1:length(slp.scenarios)
        sc = slp.scenarios[s]
        cost_val += slp.p[s] * recourse_cost(x1, sc, mu; kw...)
    end
    return cost_val
end

# -----------------------------------------------------------------------
# Decision regret loss
# -----------------------------------------------------------------------

"""
    decision_regret(prob::ProblemInstance, mu_surr, mu_prim,
                    predicted_scenarios::Vector{Scenario},
                    actual_scenario::Scenario) -> Float64

Compute the decision regret:
1. Solve the surrogate 2SLP with `predicted_scenarios` and barrier μ_surr
   to get first-stage decision x̂₁.
2. Evaluate x̂₁ on the actual scenario with barrier μ_prim.

Returns the cost c₁'x̂₁ + Q(x̂₁, actual_scenario).
"""
function decision_regret(
    prob::ProblemInstance,
    mu_surr::Real,
    mu_prim::Real,
    predicted_scenarios::Vector{<:Scenario},
    actual_scenario::Scenario
)
    A1, b1, c1 = first_stage_data(prob)
    T = Float64
    slp_surr = TwoStageSLP(
        T.(A1), T.(b1), T.(c1),
        [Scenario(T.(sc.W), T.(sc.T), T.(sc.h), T.(sc.q)) for sc in predicted_scenarios]
    )

    # Step 1: Surrogate decision
    x1_hat = surrogate_first_stage(slp_surr, mu_surr)

    # Step 2: Evaluate on actual scenario
    slp_prim = TwoStageSLP(
        T.(A1), T.(b1), T.(c1),
        [Scenario(T.(actual_scenario.W), T.(actual_scenario.T),
                  T.(actual_scenario.h), T.(actual_scenario.q))]
    )
    cost_val = evaluate_cost(x1_hat, slp_prim, mu_prim)

    return cost_val
end

"""
    relative_decision_regret(prob, mu_surr, mu_prim,
                              predicted_scenarios, actual_scenario) -> Float64

Normalized regret: (surrogate_cost - optimal_cost) / |optimal_cost|.
"""
function relative_decision_regret(
    prob::ProblemInstance,
    mu_surr::Real,
    mu_prim::Real,
    predicted_scenarios::Vector{<:Scenario},
    actual_scenario::Scenario
)
    surr_cost   = decision_regret(prob, mu_surr, mu_prim, predicted_scenarios, actual_scenario)
    actual_slp  = TwoStageSLP(Float64.(first_stage_data(prob)[1]),
                               Float64.(first_stage_data(prob)[2]),
                               Float64.(first_stage_data(prob)[3]),
                               [actual_scenario])
    opt_cost    = evaluate_cost(surrogate_first_stage(actual_slp, mu_prim), actual_slp, mu_prim)
    return (surr_cost - opt_cost) / (abs(opt_cost) + 1e-10)
end
