"""
Abstract type representing data related to a specific
"""
abstract type ProblemInstance end

function return_scenario_type(ProblemInstance)
error("Not yet implemented")
end

"""
Method for mapping from a scenario parameter xi, to a scenario realization W, T, h, q specifying the structure of a second stage 
problem. Note that in many cases, xi = W, T, h q
"""
function scenario_realization(ProblemInstance, xi)
    error("You have not yet implemented a scenario realization for your problem instance")
end

"""
Differential of scenario_realization. Required for back-propagation in neural net.
"""
function diff_scenario_realization(ProblemInstance, xi)
    error("You have not yet implemented diff_scenario_realization for your problem instance")
end

function first_stage_parameters(ProblemInstance)
    error("You have not yet specified the first stage parameters of the problem")
end

"""
Construct the given two-stage stochastic program for a given context parameter
"""
function two_slp(ProblemInstance, context_parameter)::
    error("You have not yet implemented diff_scenario_realization for your problem instance")
end

"""
Used as wrapper in neural net
"""
function surrogate_decision(problem_instance::ProblemInstance, context_parameter, W, T, h, q, mu, solver=standard_solver)
    A, b, c = first_stage_parameters(ProblemInstance)
    surrogate_problem = LogBarCanLP(extensive_form_canonical(TwoStageSLP(A, b, c, [W], [T], [h], [q])), mu) 
    optimal_decision, optimal_dual = standard_solver(surrogate_problem)
    return optimal_decision
end

function primal_problem_cost(problem_instance::ProblemInstance, context_parameter, first_stage_decision)
    Main_problem = LogBarCanLP(extensive_form_canonical(two_slp(problem_instance, context_parameter)), mu)
    cost = two_slp_cost(Main_problem, first_stage_decision)
    return cost
end