model = Chain(
    Dense(9, 15, relu),   # hidden layer 1: 9 → 64, ReLU activation
    Dense(15, 21, relu),  # hidden layer 2: 64 → 64, ReLU
    Dense(21, 30, relu)         # output layer:   64 → 30, linear activation
)


"""
    surrogate_solution(problem_instance::ProblemInstanceC2SCanLP,  Ws, Ts, hs, qs, regularization_parameter, solver=LogBarCanLP_standard_solver)
Solves for the first stage decision given a specific scenario W, T, h, q
"""
function surrogate_solution(problem_instance::ResourceAllocationProblem, scenario_parameter, regularization_parameter, solver=LogBarCanLP_standard_solver)
    A, b, c = problem_instance.s1_constraint_matrix, problem_instance.s1_constraint_vector, problem_instance.s1_cost_vector
    W, T, h, q = scenario_realization(problem_instance, scenario_parameter)
    surrogate_problem = LogBarCanLP(TwoStageSLP(A, b, c, [W], [T], [h], [q]), regularization_parameter) 
    optimal_decision, optimal_dual = solver(surrogate_problem)
    return optimal_decision[1:length(c)]
end

"""
    derivative_surrogate_solution(problem_instance::ProblemInstanceC2SCanLP, context_parameter, W, T, h, q, mu, solver=LogBarCanLP_standard_solver)
Derivative of first stage decision for surrogate_problem with respect to scenario parameters
"""
function derivative_surrogate_solution(problem_instance::ProblemInstanceC2SCanLP, scenario_parameter, regularization_parameter, solver=LogBarCanLP_standard_solver)
    A, b, c = problem_instance.s1_constraint_matrix, problem_instance.s1_constraint_vector, problem_instance.s1_cost_vector
    W, T, h, q = scenario_realization(problem_instance, scenario_parameter)
    extensive_form_regularized = LogBarCanLP(TwoStageSLP(A, b, c, [W], [T], [h], [q]), regularization_parameter)
    optimal_solution, optimal_dual = solver(extensive_form_regularized)
    I = size(problem_instance.problem_data.service_rate_parameters, 1)
    return optimal_dual[2 + I: end]
end

"""
Provides the pullback for the surrogate_solution function, allowing for back-propagation through the neural network
Note: The derivative computations can be rewritten without the D_xiY function to improve performance
"""
function ChainRulesCore.rrule(::typeof(surrogate_solution), problem_instance, scenario_parameter, regularization_parameter, solver)
    y = surrogate_solution(problem_instance, scenario_parameter, regularization_parameter)
    
    function pullback(y_hat)
        D_h = derivative_surrogate_solution(problem_instance, scenario_parameter, regularization_parameter, solver)
        D_h_tangent = y_hat * D_h

        return NoTangent(), NoTangent(), D_h_tangent, NoTangent(), NoTangent()  # returning NoTangent for the regularization parameter
    end
    
    return y, pullback
end

"""
    primal_problem_cost(problem_instance::ProblemInstanceC2SCanLP, context_parameter, regularization_parameter, first_stage_decision)
Computes the cost of the primal problem as a function of the context parameter, first-stage decision and regularization_parameter
"""
function primal_problem_cost(problem_instance::ProblemInstanceC2SCanLP, scenario_parameter, regularization_parameter, first_stage_decision)
    A, b, c = problem_instance.s1_constraint_matrix, problem_instance.s1_constraint_vector, problem_instance.s1_cost_vector
    W, T, h, q = scenario_realization(problem_instance, scenario_parameter)
    twoslp = TwoStageSLP(A, b, c, [W], [T], [h], [q])
    main_problem = LogBarCanLP(twoslp, regularization_parameter)
    cost = cost_2s_LogBarCanLP(first_stage_decision, main_problem, regularization_parameter)
    return cost
end

function derivative_primal_problem_cost(problem_instance::ProblemInstanceC2SCanLP, scenario_parameter, regularization_parameter, first_stage_decision)
    A, b, c = problem_instance.s1_constraint_matrix, problem_instance.s1_constraint_vector, problem_instance.s1_cost_vector
    W, T, h, q = scenario_realization(problem_instance, scenario_parameter)
    main_problem = LogBarCanLP(twoslp, regularization_parameter)
    D_x = diff_cost_2s_LogBarCanLP(main_problem, regularization_parameter, first_stage_decision)
    return D_x
end

"""
Provides the pullback for the primal_problem_cost function, allowing for back-propagation through the neural network
"""
function ChainRulesCore.rrule(::typeof(primal_problem_cost), problem_instance::ProblemInstanceC2SCanLP, scenario_parameter, regularization_parameter, 
                            first_stage_decision)

    cost = primal_problem_cost(problem_instance, scenario_parameter, regularization_parameter, first_stage_decision)
    
    function pullback(y_hat)
        cost_derivative = derivative_primal_problem_cost(problem_instance,  scenario_parameter, regularization_parameter, first_stage_decision)
        tangent = y_hat * cost_derivative
        return NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), tangent
    end

    return cost, pullback
end



function loss(problem_instance, regularization_parameter, scenario_parameter, actual_scenario)
    surrogate_decision = surrogate_solution(problem_instance, scenario_parameter, regularization_parameter)
    primal_problem_cost(problem_instance, actual_scenario, regularization_parameter, surrogate_decision)
end