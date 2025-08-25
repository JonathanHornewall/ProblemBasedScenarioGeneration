model = Chain(
    Dense(3, 128, relu),
    Dense(128, 128, relu),
    Dense(128, 128, relu),
    Dense(128, 30, relu)     # linear head
) |> f64

"""
    surrogate_solution(problem_instance::ProblemInstanceC2SCanLP,  Ws, Ts, hs, qs, regularization_parameter, solver=LogBarCanLP_standard_solver)
Solves for the first stage decision given a specific scenario W, T, h, q
"""
function surrogate_solution(problem_instance::ResourceAllocationProblem, scenario_parameter, regularization_parameter, solver=LogBarCanLP_standard_solver)
    A, b, c = problem_instance.s1_constraint_matrix, problem_instance.s1_constraint_vector, problem_instance.s1_cost_vector
    W, T, h, q = scenario_collection_realization(problem_instance, scenario_parameter)
    surrogate_problem = LogBarCanLP(TwoStageSLP(A, b, c, W, T, h, q), regularization_parameter) 
    optimal_decision, optimal_dual = solver(surrogate_problem)
    return optimal_decision[1:length(c)]
end

"""
    derivative_surrogate_solution(problem_instance::ProblemInstanceC2SCanLP, context_parameter, W, T, h, q, mu, solver=LogBarCanLP_standard_solver)
Derivative of first stage decision for surrogate_problem with respect to scenario parameters
"""
function derivative_surrogate_solution(problem_instance::ResourceAllocationProblem, scenario_parameter, regularization_parameter, solver=LogBarCanLP_standard_solver)
    A, b, c = problem_instance.s1_constraint_matrix, problem_instance.s1_constraint_vector, problem_instance.s1_cost_vector
    W, T, h, q = scenario_collection_realization(problem_instance, scenario_parameter)
    extensive_form_regularized = LogBarCanLP(TwoStageSLP(A, b, c, W, T, h, q), regularization_parameter)
    der_b = diff_opt_b(extensive_form_regularized; solver=solver)
    I = size(problem_instance.problem_data.service_rate_parameters, 1)
    return Matrix{Float64}(der_b[1:I , 1 + I: end])
end

"""
Provides the pullback for the surrogate_solution function, allowing for back-propagation through the neural network
Note: The derivative computations can be rewritten without the D_xiY function to improve performance
"""
function ChainRulesCore.rrule(::typeof(surrogate_solution), problem_instance, scenario_parameter, regularization_parameter, solver)
    y = surrogate_solution(problem_instance, scenario_parameter, regularization_parameter, solver)
    
    function pullback(y_hat)
        D_h = derivative_surrogate_solution(problem_instance, scenario_parameter, regularization_parameter, solver)

        D_h_tangent = D_h' * y_hat

        return NoTangent(), NoTangent(), D_h_tangent, NoTangent(), NoTangent()  # returning NoTangent for the regularization parameter
    end
    
    return y, pullback
end

"""
    primal_problem_cost(problem_instance::ProblemInstanceC2SCanLP, context_parameter, regularization_parameter, first_stage_decision)
Computes the cost of the primal problem as a function of the context parameter, first-stage decision and regularization_parameter
"""
function primal_problem_cost(problem_instance::ResourceAllocationProblem, scenario_parameter, regularization_parameter, first_stage_decision)
    A, b, c = problem_instance.s1_constraint_matrix, problem_instance.s1_constraint_vector, problem_instance.s1_cost_vector
    W, T, h, q = scenario_collection_realization(problem_instance, scenario_parameter)
    twoslp = TwoStageSLP(A, b, c, W, T, h, q)
    cost = s1_cost(twoslp, first_stage_decision, regularization_parameter)
    return cost
end

function derivative_primal_problem_cost(problem_instance::ResourceAllocationProblem, scenario_parameter, regularization_parameter, first_stage_decision)
    A, b, c = problem_instance.s1_constraint_matrix, problem_instance.s1_constraint_vector, problem_instance.s1_cost_vector
    W, T, h, q = scenario_collection_realization(problem_instance, scenario_parameter)
    twoslp = TwoStageSLP(A, b, c, W, T, h, q)
    D_x = diff_s1_cost(twoslp, first_stage_decision, regularization_parameter)
    return D_x
end

"""
Provides the pullback for the primal_problem_cost function, allowing for back-propagation through the neural network
"""
function ChainRulesCore.rrule(::typeof(primal_problem_cost), problem_instance::ResourceAllocationProblem, scenario_parameter, regularization_parameter, 
                            first_stage_decision)

    cost = primal_problem_cost(problem_instance, scenario_parameter, regularization_parameter, first_stage_decision)
    
    function pullback(y_hat)
        cost_derivative = derivative_primal_problem_cost(problem_instance,  scenario_parameter, regularization_parameter, first_stage_decision)
        tangent = y_hat * cost_derivative
        return NoTangent(), NoTangent(), NoTangent(), NoTangent(), tangent
    end

    return cost, pullback
end



"""
    loss(problem_instance::ResourceAllocationProblem, reg_param_surr, reg_param_prim, scenario_parameter, actual_scenario)
Computes the loss of the surrogate solution compared to the optimal solution. Used for training the neural network
Example usage: loss(problem_instance, reg_param_surr, reg_param_prim, scenario_parameter, actual_scenario)
"""
function loss(problem_instance::ResourceAllocationProblem, reg_param_surr, reg_param_prim, scenario_parameters, actual_scenarios)
        surrogate_decision = surrogate_solution(problem_instance, scenario_parameters, reg_param_surr)
        primal_problem_cost(problem_instance, actual_scenarios, reg_param_prim, surrogate_decision)
end


#=
"""
    relative_loss(problem_instance::ResourceAllocationProblem, reg_param_surr, reg_param_prim, scenario_parameter, actual_scenario)
Computes the relative loss of the surrogate solution compared to the optimal solution. Mostly used for testing.
"""
function relative_loss(problem_instance::ResourceAllocationProblem, reg_param_surr, reg_param_prim, scenario_parameter, actual_scenario)
    # Handle matrix inputs by extracting individual vectors
    if scenario_parameter isa Matrix
        nr_of_scenarios = size(scenario_parameter, 2)
        total_evaluated_loss = 0.0
        total_optimal_loss = 0.0
        
        for i in 1:nr_of_scenarios
            # Extract individual scenario vectors from the matrix
            scenario = scenario_parameter[:, i]
            actual = actual_scenario[:, i]
            
            # Compute evaluated loss
            evaluated_loss = loss(problem_instance, reg_param_surr, reg_param_prim, scenario, actual)
            total_evaluated_loss += evaluated_loss
            
            # Compute optimal loss
            optimal_loss = loss(problem_instance, reg_param_prim, reg_param_prim, actual, actual)
            total_optimal_loss += optimal_loss
        end
        
        avg_evaluated_loss = total_evaluated_loss / nr_of_scenarios
        avg_optimal_loss = total_optimal_loss / nr_of_scenarios
        
        return (avg_evaluated_loss - avg_optimal_loss) / abs(avg_optimal_loss)
    else
        # Handle vector inputs (backward compatibility)
        evaluated_loss = loss(problem_instance, reg_param_surr, reg_param_prim, scenario_parameter, actual_scenario)
        optimal_loss = loss(problem_instance, reg_param_prim, reg_param_prim, actual_scenario, actual_scenario)
        
        return (evaluated_loss - optimal_loss) / abs(optimal_loss)
    end
end
=#
#=
"""
    loss(problem_instance::ResourceAllocationProblem, reg_param_surr, reg_param_prim, scenario_parameter, actual_scenario)
Computes the loss using the main package's structure but with local functions.
"""
function loss(problem_instance::ResourceAllocationProblem, reg_param_surr, reg_param_prim, scenario_parameter, actual_scenario)
    nr_of_scenarios = size(scenario_parameter, 2)
    total_loss = 0.0
    
    for i in 1:nr_of_scenarios
        # Extract individual scenario vectors from the matrix
        scenario = scenario_parameter[:, i]
        actual = actual_scenario[:, i]
        
        # Generate surrogate solution using the local function that expects vectors
        surrogate_decision = surrogate_solution(problem_instance, scenario, reg_param_surr)
        
        # Evaluate primal problem cost using the local function that expects vectors
        scenario_loss = primal_problem_cost(problem_instance, actual, reg_param_prim, surrogate_decision)
        total_loss += scenario_loss
    end
    
    return total_loss / nr_of_scenarios  # Return average loss across scenarios
end

"""
    relative_loss(problem_instance::ResourceAllocationProblem, reg_param_surr, reg_param_prim, scenario_parameter, actual_scenario)
Computes the relative loss using the local loss function.
"""
function relative_loss(problem_instance::ResourceAllocationProblem, reg_param_surr, reg_param_prim, scenario_parameter, actual_scenario)
    evaluated_loss = loss(problem_instance, reg_param_surr, reg_param_prim, scenario_parameter, actual_scenario)
    optimal_loss = loss(problem_instance, reg_param_prim, reg_param_prim, actual_scenario, actual_scenario)
    
    return (evaluated_loss - optimal_loss) / abs(optimal_loss)
end
=#