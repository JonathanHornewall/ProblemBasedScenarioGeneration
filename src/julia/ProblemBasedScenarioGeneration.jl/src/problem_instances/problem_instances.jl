"""
Abstract type representing data related to a specific
"""
abstract type ProblemInstanceC2SCanLP end

"""
    scenario_realization(problem_instance::ProblemInstanceC2SCanLP, xi)
Method for mapping from a scenario parameter xi, to a scenario realization W, T, h, q specifying the structure of a second stage 
problem. Note that in many cases, xi = W, T, h q
"""
function scenario_realization(problem_instance::ProblemInstanceC2SCanLP, scenario_parameter)
    return scenario_parameter
end

"""
    return_first_stage_parameters(problem_instance::ProblemInstanceC2SCanLP)
Getter method for retrieving the first stage parameters of the problem instance
"""
function return_first_stage_parameters(problem_instance::ProblemInstanceC2SCanLP)
    error("You have not yet specified the first stage parameters of the problem")
end

"""
    scenario_collection_realization(instance::ProblemInstanceC2SCanLP, scenario_collection)
Method for mapping from a scenario collection matrix, to a scenario collection realization Ws, Ts, hs, qs.

scenario_collection[:,s] represents the s-th scenario in the collection.

(Ws[:,:,s], Ts[:,:,s], hs[:,s], qs[:,s]) represent the scenario realization for the s-th scenario in the collection.

Note: It has to be differentiable via zygote.

"""
function scenario_collection_realization(instance::ProblemInstanceC2SCanLP, scenario_collection)
    # Use functional approach instead of push! to avoid mutation
    scenario_results = [scenario_realization(instance, scenario) for scenario in eachcol(scenario_collection)]
    W_list = [result[1] for result in scenario_results]
    T_list = [result[2] for result in scenario_results]
    h_list = [result[3] for result in scenario_results]
    q_list = [result[4] for result in scenario_results]
    
    Ws = cat(W_list..., dims=3)
    Ts = cat(T_list..., dims=3)
    
    # Create 2D matrices: h vectors become columns in (m_2, S) matrix, q vectors become columns in (n_2, S) matrix
    hs = hcat(h_list...)
    qs = hcat(q_list...)
    
    return Ws, Ts, hs, qs
end


"""
    construct_neural_network(problem_instance::ProblemInstanceC2SCanLP)
Specifies a neural network architecture for the given problem instance. 

This function should be implemented for each concrete subtype of `ProblemInstanceC2SCanLP` to return a Flux.jl model 
appropriate for the problem's input and output dimensions.
    The input dimension must be equal to the dimension of the context parameter.
    The output dimension must be equal to the dimension of the scenario parameters.
"""

function construct_neural_network(problem_instance::ProblemInstanceC2SCanLP)
    error("You have not yet specified a neural network for your problem instance")
end

#=
"""
_________________________________________________________________
Loss-function related functionalities for problem instances
_________________________________________________________________
"""

"""
    surrogate_solution(problem_instance::ProblemInstanceC2SCanLP,  Ws, Ts, hs, qs, regularization_parameter, solver=LogBarCanLP_standard_solver)
Solves for the first stage decision given a specific scenario W, T, h, q
"""
function surrogate_solution(problem_instance::ProblemInstanceC2SCanLP, Ws, Ts, hs, qs, regularization_parameter, solver=LogBarCanLP_standard_solver)
    A, b, c = return_first_stage_parameters(problem_instance)
    surrogate_problem = LogBarCanLP(TwoStageSLP(A, b, c, Ws, Ts, hs, qs), regularization_parameter) 
    optimal_decision, optimal_dual = solver(surrogate_problem)
    return optimal_decision[1:length(c)]
end

"""
    derivative_surrogate_solution(problem_instance::ProblemInstanceC2SCanLP, context_parameter, W, T, h, q, mu, solver=LogBarCanLP_standard_solver)
Derivative of first stage decision for surrogate_problem with respect to scenario parameters
"""
function derivative_surrogate_solution(problem_instance::ProblemInstanceC2SCanLP, Ws, Ts, hs, qs, regularization_parameter, ps=nothing, solver=LogBarCanLP_standard_solver)
    A, b, c = return_first_stage_parameters(problem_instance)
    surrogate_2slp = TwoStageSLP(A, b, c, Ws, Ts, hs, qs, ps)
    scenario_type = return_scenario_type(problem_instance)
    return D_xiY(surrogate_2slp, regularization_parameter, scenario_type, solver)
end

"""
Provides the pullback for the surrogate_solution function, allowing for back-propagation through the neural network
Note: The derivative computations can be rewritten without the D_xiY function to improve performance
"""
function ChainRulesCore.rrule(::typeof(surrogate_solution), problem_instance, Ws, Ts, hs, qs, regularization_parameter, ps, solver)
    y = surrogate_solution(problem_instance, Ws, Ts, hs, qs, regularization_parameter)
    
    function pullback(y_hat)
        D_Ws, D_Ts, D_hs, D_qs = derivative_surrogate_solution(problem_instance, Ws, Ts, hs, qs, regularization_parameter, ps, solver)


        scenario_type = return_scenario_type(problem_instance)
        has_W, has_T, has_h, has_q = typeof(scenario_type).parameters

        if !has_W
        D_Ws_tangent = NoTangent()
        else
        D_Ws_tangent = [@einsum dW[i,j] := y_hat[k] * D_W[k,i,j] for D_W in D_Ws]
        end

        if !has_T
        D_Ts_tangent = NoTangent()
        else
        D_Ts_tangent = [@einsum dT[i,j] := y_hat[k] * D_T[k,i,j] for D_T in D_Ts]
        end

        if !has_h
        D_hs_tangent = NoTangent()
        else
        D_hs_tangent = [y_hat * D_h for D_h in D_hs]
        end

        if !has_q
        D_qs_tangent = NoTangent()
        else
        D_qs_tangent = [y_hat * D_q for D_q in D_qs]
        end

        return NoTangent(), NoTangent(), D_Ws_tangent, D_Ts_tangent, D_hs_tangent, D_qs_tangent, NoTangent(), NoTangent(), NoTangent()  # returning NoTangent for the regularization parameter
    end
    
    return y, pullback
end

"""
    primal_problem_cost(problem_instance::ProblemInstanceC2SCanLP, context_parameter, regularization_parameter, first_stage_decision)
Computes the cost of the primal problem as a function of the context parameter, first-stage decision and regularization_parameter
"""
function primal_problem_cost(problem_instance::ProblemInstanceC2SCanLP, Ws, Ts, hs, qs, regularization_parameter, first_stage_decision)
    twoslp = TwoStageSLP(return_first_stage_parameters(problem_instance)..., Ws, Ts, hs, qs)
    cost = s1_cost(twoslp, first_stage_decision, regularization_parameter)
    return cost
end

function derivative_primal_problem_cost(problem_instance::ProblemInstanceC2SCanLP, Ws, Ts, hs, qs, regularization_parameter, first_stage_decision)
    A, b, c = return_first_stage_parameters(problem_instance)
    twoslp = TwoStageSLP(A, b, c, Ws, Ts, hs, qs)
    main_problem = LogBarCanLP(twoslp, regularization_parameter)
    D_x = diff_s1_cost(main_problem, first_stage_decision, regularization_parameter)
    return D_x
end

"""
Provides the pullback for the primal_problem_cost function, allowing for back-propagation through the neural network
"""
function ChainRulesCore.rrule(::typeof(primal_problem_cost), problem_instance::ProblemInstanceC2SCanLP, Ws, Ts, hs, qs, 
                            regularization_parameter, first_stage_decision)

    cost = primal_problem_cost(problem_instance, Ws, Ts, hs, qs, regularization_parameter, first_stage_decision)
    
    function pullback(y_hat)
        cost_derivative = derivative_primal_problem_cost(problem_instance,  Ws, Ts, hs, qs, regularization_parameter, first_stage_decision)
        tangent = y_hat * cost_derivative
        return NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), tangent
    end

    return cost, pullback
end
=#
