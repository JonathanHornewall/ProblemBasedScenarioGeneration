function surrogate_solution(problem_instance, reg_param_surr, scenario_parameters)
    Ws_surrogate, Ts_surrogate, hs_surrogate, qs_surrogate = scenario_collection_realization(problem_instance, scenario_parameters)
    A, b, c = return_first_stage_parameters(problem_instance)
    sur_two_slp = TwoStageSLP(A, b, c, Ws_surrogate, Ts_surrogate, hs_surrogate, qs_surrogate)
    surr_prob = LogBarCanLP(sur_two_slp, reg_param_surr)
    A_e, b_e, c_e, mu_e = surr_prob.linear_program.constraint_matrix, surr_prob.linear_program.constraint_vector, surr_prob.linear_program.cost_vector, surr_prob.regularization_parameters
    surr_solution = LogBarCanLP_standard_solver_primal(A_e, b_e, c_e, mu_e)
    return surr_solution[1:length(c)]
end

"""
    loss(problem_instance, reg_param_surr, reg_param_prim, scenario_parameters, actual_scenario)

Compute the loss for a given problem instance by evaluating the cost of the surrogate solution on the actual scenario.

# Arguments
- `problem_instance`: The problem instance (should be a subtype of `ProblemInstanceC2SCanLP`).
- `reg_param_surr`: Regularization parameter used for the surrogate problem.
- `reg_param_prim`: Regularization parameter used for the primal (actual) problem.
- `scenario_parameters`: Scenario parameters representing the surrogate scenario collection returned by the neural network.
- `actual_scenario`: Actual scenarios parameters associated with the context variable.

# Returns
- The cost of the surrogate solution evaluated on the actual problem.

# Description
This function first computes the surrogate solution by solving the surrogate problem defined by `scenario_parameters` and `reg_param_surr`. 
It then evaluates the cost of this solution on the actual scenario, using `reg_param_prim` as the regularization parameter. 
"""

function loss(problem_instance, reg_param_surr, reg_param_prim, scenario_parameters, actual_scenario)
    # Compute the surrogate solution
    #=
    Ws_surrogate, Ts_surrogate, hs_surrogate, qs_surrogate = scenario_collection_realization(problem_instance, scenario_parameters)
    A, b, c = return_first_stage_parameters(problem_instance)
    sur_two_slp = TwoStageSLP(A, b, c, Ws_surrogate, Ts_surrogate, hs_surrogate, qs_surrogate)
    A_ext, b_ext, c_ext = extensive_form_canonical(sur_two_slp)

    # Create regularization parameter vector for extensive form
    n_1 = length(c)
    n_2 = size(Ws_surrogate, 2)
    S = size(Ws_surrogate, 3)
    ps = ones(S) / S  # Default equiprobable scenarios
    mu_ext = vcat(reg_param_surr * ones(n_1), vcat([reg_param_surr * ps[s] * ones(n_2) for s in 1:S]...))
    surrogate_solution = LogBarCanLP_standard_solver_primal(A_ext, b_ext, c_ext, mu_ext)[1:length(c)]
    =#
    
    surr_solution = surrogate_solution(problem_instance, reg_param_surr, scenario_parameters)
    # Compute the performance 
    Ws_actual, Ts_actual, hs_actual, qs_actual = scenario_collection_realization(problem_instance, actual_scenario)
    A, b, c = return_first_stage_parameters(problem_instance)
    prim_two_slp = TwoStageSLP(A, b, c, Ws_actual, Ts_actual, hs_actual, qs_actual)
    cost = s1_cost(prim_two_slp, surr_solution, reg_param_prim)
    return cost
end


function relative_loss(problem_instance, reg_param_surr, reg_param_prim, scenario_parameters, actual_scenario)
    surr_solution = loss(problem_instance, reg_param_surr, reg_param_prim, scenario_parameters, actual_scenario)
    actual_solution = loss(problem_instance, reg_param_prim, reg_param_prim, actual_scenario, actual_scenario)
    return (surr_solution - actual_solution) / abs(actual_solution)
end

"""
---------------------------------------------------------------------------------------------
ChainRules.jl differentiation rules

These are required for the decision focused learning of the neural network.
---------------------------------------------------------------------------------------------
"""

"""
Provides the pullback for the s1_cost function, allowing for back-propagation through the neural network.
Uses the derivative computation from diff_s1_cost.
"""
function ChainRulesCore.rrule(::typeof(s1_cost), two_slp::TwoStageSLP, s1_decision, regularization_parameter; solver=LogBarCanLP_standard_solver)
    cost_val = s1_cost(two_slp, s1_decision, regularization_parameter; solver=solver)
    
    function pullback(cost_hat)
        # Use the derivative from 2sp_differentials.jl
        cost_derivative = diff_s1_cost(two_slp, s1_decision, regularization_parameter; solver=solver)
        # Ensure correct shape and handle thunks
        ȳ = ChainRulesCore.unthunk(cost_hat)
        tangent = ȳ .* cost_derivative
        return NoTangent(), NoTangent(), tangent, NoTangent()
    end
    
    return cost_val, pullback
end

"""
Provides the pullback for the primal LogBarCanLP solver that takes A, b, c as inputs.
This enables differentiation through the optimal solution computation with respect to the problem parameters.
Uses lazy evaluation (thunks) to only compute derivatives when needed.

Performance improvement: Previously, all three derivatives (diff_A, diff_b, diff_c) were computed 
upfront even when only one was needed. Now, each derivative is wrapped in a thunk and only 
computed when the corresponding tangent is actually accessed during backpropagation.
"""
function ChainRulesCore.rrule(::typeof(LogBarCanLP_standard_solver_primal), constraint_matrix, constraint_vector, cost_vector, mu::Union{Real,AbstractVector}; solver_tolerance=1e-9, feasibility_margin=1e-8)
    # Create a temporary LogBarCanLP instance and solve for the optimal solution
    temp_lp = CanLP(constraint_matrix, constraint_vector, cost_vector)
    temp_instance = LogBarCanLP(temp_lp, mu)
    
    # Solve the problem ONCE and cache the results
    optimal_solution, optimal_dual = LogBarCanLP_standard_solver(temp_instance)
    
    # Define the pullback function with lazy evaluation using thunks
    function LogBarCanLP_standard_solver_primal_pullback(solution_tangent)
        Δx = ChainRulesCore.unthunk(solution_tangent)
        
        # Create thunks for each derivative - they will only be computed if accessed
        # This avoids the expensive matrix operations when derivatives aren't needed
        A_tangent_thunk = @thunk begin
            diff_A = diff_opt_A(temp_instance, optimal_solution, optimal_dual)
            @einsum A_tangent[i,j] := diff_A[k,i,j] * Δx[k]
            A_tangent
        end
        
        b_tangent_thunk = @thunk begin
            diff_b = diff_opt_b(temp_instance, optimal_solution, optimal_dual)
            diff_b' * Δx
        end
        
        c_tangent_thunk = @thunk begin
            diff_c = diff_opt_c(temp_instance, optimal_solution, optimal_dual)
            diff_c' * Δx
        end
        
        return NoTangent(), A_tangent_thunk, b_tangent_thunk, c_tangent_thunk, NoTangent()
    end
    
    return optimal_solution, LogBarCanLP_standard_solver_primal_pullback
end