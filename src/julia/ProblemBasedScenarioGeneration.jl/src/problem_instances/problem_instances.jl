"""
Abstract type representing data related to a specific
"""
abstract type ProblemInstanceC2SCanLP end

"""
    return_scenario_type(problem_instance::ProblemInstanceC2SCanLP)
Getter method for retrieving the "scenario type" of the problem instance, specifying which problem parameters 
with noise.
"""
function return_scenario_type(problem_instance::ProblemInstanceC2SCanLP)
error("You have not specified the scenario type for your problem instance.")
end

"""
    Scenario{R}(W=nothing, T=nothing, h=nothing, q=nothing)
"""
struct Scenario{R<:Real}
    W::Matrix{R}
    T::Matrix{R}
    h::Vector{R}
    q::Vector{R}
end

function Scenario(; W=nothing, T=nothing, h=nothing, q=nothing)
    # Choose R from the first non-nothing input
    arrays = [W, T, h, q]
    R = eltype(first(filter(!isnothing, arrays), Float64[]))  # fallback Float64 if none given

    Wm = isnothing(W) ? Matrix{R}(undef, 0, 0) : W
    Tm = isnothing(T) ? Matrix{R}(undef, 0, 0) : T
    hv = isnothing(h) ? Vector{R}() : h
    qv = isnothing(q) ? Vector{R}() : q

    Scenario{R}(Wm, Tm, hv, qv)
end

#=
struct ScenarioList{R<:Real}
    scenario_list::Vector{Scenario}
    probabilities::Vector{Float64}
end

function ScenarioList(; scenarios::Vector{Scenario}, probabilities::Vector{Float64})
    if length(scenarios) != length(probabilities)
        error("Number of scenarios must match number of probabilities")
    end
    new{eltype(scenarios[1].W)}(scenarios, probabilities)
end
=#

function return_standard_scenario(problem_instance::ProblemInstanceC2SCanLP)
    error("You have not yet specified a standard scenario for your problem instance")
end

function return_standard_scenarios(instance::ProblemInstanceC2SCanLP, number_of_scenarios::Int)
    vectorize: x -> isempty(x) ? Vector{eltype(x)}() : fill(x, number_of_scenarios)

    W, T, h, q = return_standard_scenario(instance)
    return map(vectorize, (W, T, h, q))
end


"""
    scenario_realization(problem_instance::ProblemInstanceC2SCanLP, xi)
Method for mapping from a scenario parameter xi, to a scenario realization W, T, h, q specifying the structure of a second stage 
problem. Note that in many cases, xi = W, T, h q
"""
function scenario_realization(problem_instance::ProblemInstanceC2SCanLP, scenario_parameter)
    return scenario_parameter
end

function return_scenario_parameter_dimension(problem_instance::ProblemInstanceC2SCanLP)
    error("You have not yet specified the scenario parameter dimension for your problem instance")
end

"""
    return_first_stage_parameters(problem_instance::ProblemInstanceC2SCanLP)
Getter method for retrieving the first stage parameters of the problem instance
"""
function return_first_stage_parameters(problem_instance::ProblemInstanceC2SCanLP)
    error("You have not yet specified the first stage parameters of the problem")
end


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
    Ws, Ts, hs, qs = scenarios
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
    main_problem = LogBarCanLP(twoslp, regularization_parameter)
    cost = cost_2s_LogBarCanLP(first_stage_decision, main_problem, regularization_parameter)
    return cost
end

function derivative_primal_problem_cost(problem_instance::ProblemInstanceC2SCanLP, Ws, Ts, hs, qs, regularization_parameter, first_stage_decision)
    A, b, c = return_first_stage_parameters(problem_instance)
    twoslp = TwoStageSLP(A, b, c, Ws, Ts, hs, qs)
    main_problem = LogBarCanLP(twoslp, regularization_parameter)
    D_x = diff_cost_2s_LogBarCanLP(main_problem, regularization_parameter, first_stage_decision)
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

"""
---------------------------------------------------------------------------------
Struct representing a problem instance with manually specified problem data.
---------------------------------------------------------------------------------
"""

struct manual_C2SCanLP{R<:Real} <: ProblemInstanceC2SCanLP
    A::Matrix{R}  # constraint matrix
    b::Vector{R}  # constraint vector
    c::Vector{R}  # cost vector
    W::Matrix{R}  # second stage constraint matrix
    T::Matrix{R}  # coupling matrix
    h::Vector{R}  # second stage decision vector
    q::Vector{R}  # second stage decision vector
    standard_scenario::Scenario{R}  # standard scenario
    scenario_type::ScenarioType  # scenario type
    scenario_parameter_dimension::Int  # dimension of scenario parameters
end

function return_first_stage_parameters(instance::manual_C2SCanLP)
    return instance.A, instance.b, instance.c
end

function return_scenario_type(instance::manual_C2SCanLP)
    return instance.scenario_type
end

function return_standard_scenario(instance::manual_C2SCanLP)
    return instance.standard_scenario
end

function return_scenario_parameter_dimension(instance::manual_C2SCanLP)
    return instance.scenario_parameter_dimension
end