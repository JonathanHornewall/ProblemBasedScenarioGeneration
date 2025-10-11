module RefactoredLoss

using ChainRulesCore
using ChainRulesCore: ZeroTangent, NoTangent
using Zygote
using LinearAlgebra

using ..ProblemBasedScenarioGeneration
const PBSG = ProblemBasedScenarioGeneration

export refactored_loss

"""
    surrogate_solution_copy(problem_instance, reg_param_surr, scenario_collection)

Replica of the existing `surrogate_solution` helper from `neural_net/loss.jl`.
It solves the surrogate two-stage problem defined by `scenario_collection`
and returns the first-stage decision.
"""
function surrogate_solution_copy(problem_instance, reg_param_surr, scenario_collection)
    Ws_surrogate, Ts_surrogate, hs_surrogate, qs_surrogate =
        PBSG.scenario_collection_realization(problem_instance, scenario_collection)
    A, b, c = PBSG.return_first_stage_parameters(problem_instance)
    sur_two_slp = PBSG.TwoStageSLP(A, b, c, Ws_surrogate, Ts_surrogate, hs_surrogate, qs_surrogate)
    surr_prob = PBSG.LogBarCanLP(sur_two_slp, reg_param_surr)
    lp = surr_prob.linear_program
    A_e = lp.constraint_matrix
    b_e = lp.constraint_vector
    c_e = lp.cost_vector
    mu_e = surr_prob.regularization_parameters
    solution = PBSG.LogBarCanLP_standard_solver_primal(A_e, b_e, c_e, mu_e)
    return solution[1:length(c)]
end

"""
    refactored_loss(problem_instance, reg_param_surr, reg_param_prim,
                    scenario_collection, actual_scenario_collection)

Reimplementation of `ProblemBasedScenarioGeneration.loss` with an explicit
reverse-rule for faster differentiation.
"""
function refactored_loss(problem_instance, reg_param_surr, reg_param_prim,
                         scenario_collection, actual_scenario_collection)
    surr_solution = surrogate_solution_copy(problem_instance, reg_param_surr, scenario_collection)
    Ws_actual, Ts_actual, hs_actual, qs_actual =
        PBSG.scenario_collection_realization(problem_instance, actual_scenario_collection)
    A, b, c = PBSG.return_first_stage_parameters(problem_instance)
    prim_two_slp = PBSG.TwoStageSLP(A, b, c, Ws_actual, Ts_actual, hs_actual, qs_actual)
    return PBSG.s1_cost(prim_two_slp, surr_solution, reg_param_prim)
end

# ------------------------------------------------------------------------------
# Custom reverse rule
# ------------------------------------------------------------------------------

# Helper to contract gradient with a vector of third-order tensors
function _contract_tensor_list!(dest, tensors, vec)
    for (i, T) in enumerate(tensors)
        contracted = dropdims(sum(T .* reshape(vec, :, 1, 1), dims=1), dims=1)
        dest[:, :, i] .= contracted
    end
    return dest
end

function _contract_matrix_list!(dest, matrices, vec)
    for (i, M) in enumerate(matrices)
        dest[:, i] .= transpose(M) * vec
    end
    return dest
end

function _derivative_surrogate_solution_h(problem_instance::PBSG.ResourceAllocationProblem,
                                          reg_param_surr,
                                          Ws, Ts, hs, qs)
    A, b, c = PBSG.return_first_stage_parameters(problem_instance)
    surrogate_slp = PBSG.TwoStageSLP(A, b, c, Ws, Ts, hs, qs)
    logbar_problem = PBSG.LogBarCanLP(surrogate_slp, reg_param_surr)
    der_b = PBSG.diff_opt_b(logbar_problem)

    n₁ = length(c)
    I = size(problem_instance.problem_data.service_rate_parameters, 1)
    J = size(problem_instance.problem_data.service_rate_parameters, 2)
    m₂ = size(hs, 1)
    S = size(hs, 2)

    m_total = size(der_b, 2)
    m₁ = m_total - S * m₂

    result = Vector{Matrix{eltype(der_b)}}(undef, S)
    demand_start = I + 1
    demand_stop = I + J
    for s in 1:S
        offset = m₁ + (s - 1) * m₂
        demand_cols = (offset + demand_start):(offset + demand_stop)
        M = zeros(eltype(der_b), n₁, m₂)
        @views M[:, demand_start:demand_stop] .= der_b[1:n₁, demand_cols]
        result[s] = M
    end
    return result
end

function ChainRulesCore.rrule(::typeof(refactored_loss), problem_instance, reg_param_surr,
                              reg_param_prim, scenario_collection, actual_scenario_collection)
    Ws_surrogate, Ts_surrogate, hs_surrogate, qs_surrogate =
        PBSG.scenario_collection_realization(problem_instance, scenario_collection)
    surr_solution = surrogate_solution_copy(problem_instance, reg_param_surr, scenario_collection)

    Ws_actual, Ts_actual, hs_actual, qs_actual =
        PBSG.scenario_collection_realization(problem_instance, actual_scenario_collection)
    A, b, c = PBSG.return_first_stage_parameters(problem_instance)
    prim_two_slp = PBSG.TwoStageSLP(A, b, c, Ws_actual, Ts_actual, hs_actual, qs_actual)
    cost_val = PBSG.s1_cost(prim_two_slp, surr_solution, reg_param_prim)

    function refactored_loss_pullback(ȳ)
        ȳ = ChainRulesCore.unthunk(ȳ)
        ȳ === ZeroTangent() && (ȳ = 0)

        # Gradient with respect to the surrogate decision
        diff_cost = PBSG.diff_s1_cost(prim_two_slp, surr_solution, reg_param_prim)
        surr_cotangent = ȳ .* diff_cost

        # Derivative of the surrogate decision with respect to scenario parameters
        if problem_instance isa PBSG.ResourceAllocationProblem
            D_Ws = Vector{Array{eltype(Ws_surrogate), 3}}()
            D_Ts = Vector{Array{eltype(Ts_surrogate), 3}}()
            D_qs = Vector{Matrix{eltype(qs_surrogate)}}()
            D_hs = _derivative_surrogate_solution_h(problem_instance, reg_param_surr,
                                                    Ws_surrogate, Ts_surrogate, hs_surrogate, qs_surrogate)
        else
            D_Ws, D_Ts, D_hs, D_qs = PBSG.derivative_surrogate_solution(
                problem_instance, reg_param_surr, Ws_surrogate, Ts_surrogate, hs_surrogate, qs_surrogate)
        end

        ΔWs = isempty(D_Ws) ? nothing : fill!(similar(Ws_surrogate), zero(eltype(Ws_surrogate)))
        if ΔWs !== nothing
            _contract_tensor_list!(ΔWs, D_Ws, surr_cotangent)
        end

        ΔTs = isempty(D_Ts) ? nothing : fill!(similar(Ts_surrogate), zero(eltype(Ts_surrogate)))
        if ΔTs !== nothing
            _contract_tensor_list!(ΔTs, D_Ts, surr_cotangent)
        end

        Δhs = isempty(D_hs) ? nothing : fill!(similar(hs_surrogate), zero(eltype(hs_surrogate)))
        if Δhs !== nothing
            _contract_matrix_list!(Δhs, D_hs, surr_cotangent)
        end

        Δqs = isempty(D_qs) ? nothing : fill!(similar(qs_surrogate), zero(eltype(qs_surrogate)))
        if Δqs !== nothing
            _contract_matrix_list!(Δqs, D_qs, surr_cotangent)
        end

        scenario_grad =
            if problem_instance isa PBSG.ResourceAllocationProblem &&
               ΔWs === nothing && ΔTs === nothing && Δqs === nothing && Δhs !== nothing
                I = size(problem_instance.problem_data.service_rate_parameters, 1)
                copy(@view Δhs[(I + 1):end, :])
            else
                function surrogate_backprop(sc)
                    Ws, Ts, hs, qs = PBSG.scenario_collection_realization(problem_instance, sc)
                    total = zero(eltype(sc))
                    if ΔWs !== nothing
                        total += sum(Ws .* ΔWs)
                    end
                    if ΔTs !== nothing
                        total += sum(Ts .* ΔTs)
                    end
                    if Δhs !== nothing
                        total += sum(hs .* Δhs)
                    end
                    if Δqs !== nothing
                        total += sum(qs .* Δqs)
                    end
                    return total
                end

                scenario_grad_raw = Zygote.gradient(surrogate_backprop, scenario_collection)[1]
                scenario_grad_raw === nothing ? ZeroTangent() : scenario_grad_raw
            end

        actual_grad = ZeroTangent()

        return NoTangent(), ZeroTangent(), ZeroTangent(), ZeroTangent(), scenario_grad, actual_grad
    end

    return cost_val, refactored_loss_pullback
end

end # module
