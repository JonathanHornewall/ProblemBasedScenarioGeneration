# Differentiates cost_function with respect to z only. Scenario arrays
# W_*, T_*, h_*, and q_array are treated as constants by this rrule.
function ChainRulesCore.rrule(
    ::typeof(cost_function),
    program::StochasticProgram,
    solver::Solver,
    z,
    W_eq_array,
    W_ineq_array,
    T_eq_array,
    T_ineq_array,
    h_eq_array,
    h_ineq_array,
    q_array;
    μ=0,
    probabilities=nothing,
    return_dual=false,
    kwargs...,
)
    return_dual &&
        throw(ArgumentError("The cost_function rrule is defined for scalar cost output."))

    value, λ_h_eq_array, λ_h_ineq_array = cost_function(
        program,
        solver,
        z,
        W_eq_array,
        W_ineq_array,
        T_eq_array,
        T_ineq_array,
        h_eq_array,
        h_ineq_array,
        q_array,
        ;
        μ=μ,
        probabilities=probabilities,
        return_dual=true,
        kwargs...,
    )

    K = size(W_eq_array, 3)
    first_stage_lp = program.first_stage_lp
    T = _sp_eltype(first_stage_lp.c, T_eq_array, T_ineq_array, λ_h_eq_array, λ_h_ineq_array)

    p_vector = if isnothing(probabilities)
        fill(one(T) / K, K)
    else
        probabilities
    end

    # The second-stage RHS is h - T*z, so equality recourse contributes
    # -T_eq'λ. The stored inequality duals use the LP solver's sign convention.
    dz = T.(first_stage_lp.c)
    for k in 1:K
        dz .+= p_vector[k] .* (
            -transpose(view(T_eq_array, :, :, k)) * view(λ_h_eq_array, :, k) +
            transpose(view(T_ineq_array, :, :, k)) * view(λ_h_ineq_array, :, k)
        )
    end

    function cost_function_pullback(value_tangent)
        value_tangent = ChainRulesCore.unthunk(value_tangent)
        tangent = value_tangent isa Number ? value_tangent : zero(T)

        return (
            ChainRulesCore.NoTangent(),
            ChainRulesCore.NoTangent(),
            ChainRulesCore.NoTangent(),
            tangent .* dz,
            ChainRulesCore.NoTangent(),
            ChainRulesCore.NoTangent(),
            ChainRulesCore.NoTangent(),
            ChainRulesCore.NoTangent(),
            ChainRulesCore.NoTangent(),
            ChainRulesCore.NoTangent(),
            ChainRulesCore.NoTangent(),
        )
    end

    return value, cost_function_pullback
end
