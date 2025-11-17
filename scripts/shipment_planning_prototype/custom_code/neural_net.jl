using ProblemBasedScenarioGeneration
using ChainRulesCore

import ProblemBasedScenarioGeneration: loss

const _ShipmentProblem = ProblemBasedScenarioGeneration.ShipmentPlanningProblem

"""
    surrogate_solution(problem_instance::ShipmentPlanningProblem, reg_param, scenario_collection;
                       solver = LogBarCanLP_standard_solver)

Specialized surrogate solver that works directly with scenario collections produced by the neural net.
"""
function surrogate_solution(problem_instance::_ShipmentProblem,
                            regularization_parameter,
                            scenario_collection,
                            solver = LogBarCanLP_standard_solver)
    A = problem_instance.s1_constraint_matrix
    b = problem_instance.s1_constraint_vector
    c = problem_instance.s1_cost_vector
    W, T, h, q = scenario_collection_realization(problem_instance, scenario_collection)
    surrogate_problem = LogBarCanLP(TwoStageSLP(A, b, c, W, T, h, q), regularization_parameter)
    optimal_decision, _ = solver(surrogate_problem)
    return optimal_decision[1:length(c)]
end

"""
    derivative_surrogate_solution(problem_instance::ShipmentPlanningProblem, ...)

Computes ∂z/∂ξ by differentiating the KKT system with respect to the demand rows of the right-hand side.
"""
function derivative_surrogate_solution(problem_instance::_ShipmentProblem,
                                       regularization_parameter,
                                       scenario_collection,
                                       solver = LogBarCanLP_standard_solver)
    A = problem_instance.s1_constraint_matrix
    b = problem_instance.s1_constraint_vector
    c = problem_instance.s1_cost_vector
    W, T, h, q = scenario_collection_realization(problem_instance, scenario_collection)
    logbar_problem = LogBarCanLP(TwoStageSLP(A, b, c, W, T, h, q), regularization_parameter)
    der_b = diff_opt_b(logbar_problem; solver = solver)

    n₁ = length(c)
    m₂ = size(problem_instance.s2_constraint_matrix, 1)
    n_locations = size(problem_instance.problem_data.shipment_costs, 2)
    S = scenario_collection isa AbstractMatrix ? size(scenario_collection, 2) : 1
    m_total = size(der_b, 2)
    m₁ = m_total - S * m₂

    cols = Int[]
    for s in 1:S
        base = m₁ + (s - 1) * m₂
        append!(cols, (base + 1):(base + n_locations))  # demand rows live at the top
    end

    return Matrix{Float64}(der_b[1:n₁, cols])
end

function ChainRulesCore.rrule(::typeof(surrogate_solution),
                              problem_instance::_ShipmentProblem,
                              regularization_parameter,
                              scenario_collection,
                              solver)
    y = surrogate_solution(problem_instance, regularization_parameter, scenario_collection, solver)

    function pullback(ŷ)
        D_h = derivative_surrogate_solution(problem_instance, regularization_parameter, scenario_collection, solver)
        tangent = reshape(D_h' * ŷ, size(scenario_collection))
        return NoTangent(), NoTangent(), NoTangent(), tangent, NoTangent()
    end

    return y, pullback
end

"""
    primal_problem_cost(problem_instance::ShipmentPlanningProblem, ...)

Evaluates the true cost of a first-stage decision under a given scenario collection.
"""
function primal_problem_cost(problem_instance::_ShipmentProblem,
                             regularization_parameter,
                             scenario_collection,
                             first_stage_decision)
    A = problem_instance.s1_constraint_matrix
    b = problem_instance.s1_constraint_vector
    c = problem_instance.s1_cost_vector
    W, T, h, q = scenario_collection_realization(problem_instance, scenario_collection)
    twoslp = TwoStageSLP(A, b, c, W, T, h, q)
    return s1_cost(twoslp, first_stage_decision, regularization_parameter)
end

function derivative_primal_problem_cost(problem_instance::_ShipmentProblem,
                                        regularization_parameter,
                                        scenario_collection,
                                        first_stage_decision)
    A = problem_instance.s1_constraint_matrix
    b = problem_instance.s1_constraint_vector
    c = problem_instance.s1_cost_vector
    W, T, h, q = scenario_collection_realization(problem_instance, scenario_collection)
    twoslp = TwoStageSLP(A, b, c, W, T, h, q)
    return diff_s1_cost(twoslp, first_stage_decision, regularization_parameter)
end

function ChainRulesCore.rrule(::typeof(primal_problem_cost),
                              problem_instance::_ShipmentProblem,
                              regularization_parameter,
                              scenario_collection,
                              first_stage_decision)
    cost = primal_problem_cost(problem_instance, regularization_parameter, scenario_collection, first_stage_decision)

    function pullback(ȳ)
        cost_derivative = derivative_primal_problem_cost(problem_instance, regularization_parameter, scenario_collection, first_stage_decision)
        return NoTangent(), NoTangent(), NoTangent(), NoTangent(), ȳ * cost_derivative
    end

    return cost, pullback
end

function loss(problem_instance::_ShipmentProblem,
              reg_param_surr,
              reg_param_prim,
              scenario_collection,
              actual_scenario_collection)
    surrogate_decision = surrogate_solution(problem_instance, reg_param_surr, scenario_collection)
    primal_problem_cost(problem_instance, reg_param_prim, actual_scenario_collection, surrogate_decision)
end
