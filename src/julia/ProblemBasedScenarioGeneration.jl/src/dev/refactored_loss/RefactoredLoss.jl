module RefactoredLoss

using ChainRulesCore
using ChainRulesCore: ZeroTangent, NoTangent
using Zygote
using LinearAlgebra

using ..ProblemBasedScenarioGeneration
const PBSG = ProblemBasedScenarioGeneration

export refactored_loss
export der_refactored_loss

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

function der_refactored_loss(problem_instance, reg_param_surr, reg_param_prim,
                             scenario_collection, actual_scenario_collection)
    _, pullback = ChainRulesCore.rrule(refactored_loss,
                                       problem_instance,
                                       reg_param_surr,
                                       reg_param_prim,
                                       scenario_collection,
                                       actual_scenario_collection)
    _, _, _, _, grad, _ = pullback(1.0)
    return grad
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

# Cache of linear basis realizations keyed by object id and scenario size
const _linear_basis_cache = IdDict{Tuple{UInt, Tuple{Int,Int}}, Any}()
const _derivative_cache = IdDict{Tuple{UInt, Tuple{Int,Int,Int,Int}, Float64}, Tuple}()

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

_is_allzero(arr) = all(iszero, arr)

function _store_basis(arr)
    _is_allzero(arr) ? nothing : copy(arr)
end

function _get_surrogate_derivatives(problem_instance, reg_param_surr,
                                    Ws_surrogate, Ts_surrogate,
                                    hs_surrogate, qs_surrogate)
    key = (objectid(problem_instance),
           (size(Ws_surrogate, 1), size(Ws_surrogate, 2),
            size(Ts_surrogate, 1), size(Ts_surrogate, 2)),
           float(reg_param_surr))
    get!(_derivative_cache, key) do
        PBSG.derivative_surrogate_solution(
            problem_instance, reg_param_surr, Ws_surrogate, Ts_surrogate,
            hs_surrogate, qs_surrogate)
    end
end

function _get_linear_basis(problem_instance, scenario_collection)
    key = (objectid(problem_instance), size(scenario_collection))
    get!(_linear_basis_cache, key) do
        n, S = size(scenario_collection)
        total = n * S
        scenario_basis = similar(scenario_collection)
        fill!(scenario_basis, zero(eltype(scenario_basis)))

        Ws_cells = Vector{Any}(undef, total)
        Ts_cells = Vector{Any}(undef, total)
        hs_cells = Vector{Any}(undef, total)
        qs_cells = Vector{Any}(undef, total)

        idx = 1
        one_el = one(eltype(scenario_basis))
        zero_el = zero(eltype(scenario_basis))
        for s in 1:S
            for i in 1:n
                scenario_basis[i, s] = one_el
                WsB, TsB, hsB, qsB = PBSG.scenario_collection_realization(problem_instance, scenario_basis)
                Ws_cells[idx] = _store_basis(WsB)
                Ts_cells[idx] = _store_basis(TsB)
                hs_cells[idx] = _store_basis(hsB)
                qs_cells[idx] = _store_basis(qsB)
                scenario_basis[i, s] = zero_el
                idx += 1
            end
        end
        (; Ws = Ws_cells, Ts = Ts_cells, hs = hs_cells, qs = qs_cells)
    end
end

function _linear_scenario_gradient(problem_instance,
                                   ΔWs, ΔTs, Δhs, Δqs,
                                   scenario_collection,
                                   basis_cache)
    n, S = size(scenario_collection)
    grad = similar(scenario_collection)

    idx = 1
    for s in 1:S
        for i in 1:n
            total = zero(eltype(grad))

            if ΔWs !== nothing
                basis_ws = basis_cache.Ws[idx]
                if basis_ws !== nothing
                    total += sum(ΔWs .* basis_ws)
                end
            end
            if ΔTs !== nothing
                basis_ts = basis_cache.Ts[idx]
                if basis_ts !== nothing
                    total += sum(ΔTs .* basis_ts)
                end
            end
            if Δhs !== nothing
                basis_hs = basis_cache.hs[idx]
                if basis_hs !== nothing
                    total += sum(Δhs .* basis_hs)
                end
            end
            if Δqs !== nothing
                basis_qs = basis_cache.qs[idx]
                if basis_qs !== nothing
                    total += sum(Δqs .* basis_qs)
                end
            end

            grad[i, s] = total
            idx += 1
        end
    end
    return grad
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
        scenario_type = PBSG.return_scenario_type(problem_instance)
        has_W, has_T, has_h, has_q = typeof(scenario_type).parameters

        D_Ws = Array{eltype(Ws_surrogate), 3}[]
        D_Ts = Array{eltype(Ts_surrogate), 3}[]
        D_hs = Matrix{eltype(hs_surrogate)}[]
        D_qs = Matrix{eltype(qs_surrogate)}[]

        if has_W || has_T || has_h || has_q
            D_Ws, D_Ts, D_hs, D_qs = _get_surrogate_derivatives(
                problem_instance, reg_param_surr,
                Ws_surrogate, Ts_surrogate, hs_surrogate, qs_surrogate)
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

        basis_cache = _get_linear_basis(problem_instance, scenario_collection)

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

        scenario_grad = _linear_scenario_gradient(problem_instance,
                                                  ΔWs, ΔTs, Δhs, Δqs,
                                                  scenario_collection,
                                                  basis_cache)

        actual_grad = ZeroTangent()

        return NoTangent(), ZeroTangent(), ZeroTangent(), ZeroTangent(), scenario_grad, actual_grad
    end

    return cost_val, refactored_loss_pullback
end

end # module
