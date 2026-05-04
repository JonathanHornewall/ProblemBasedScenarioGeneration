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
    ρ=0,
    rho=ρ,
    probabilities=nothing,
    return_dual=false,
    kwargs...,
)
    return_dual &&
        throw(ArgumentError("The cost_function rrule is defined for scalar cost output."))

    value, dz = _cost_and_z_gradient(
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
        ρ=rho,
        probabilities=probabilities,
        kwargs...,
    )
    T = eltype(dz)

    function cost_function_pullback(value_tangent)
        value_tangent = ChainRulesCore.unthunk(value_tangent)
        tangent = if _is_zero_cotangent(value_tangent)
            zero(T)
        elseif value_tangent isa Number
            value_tangent
        else
            throw(ArgumentError(
                "Expected scalar cotangent for scalar cost_function output; got $(typeof(value_tangent)).",
            ))
        end

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

function _cost_and_z_gradient(
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
    ρ=0,
    rho=ρ,
    probabilities=nothing,
    kwargs...,
)
    K = _sp_n_scenarios(
        W_eq_array,
        W_ineq_array,
        T_eq_array,
        T_ineq_array,
        h_eq_array,
        h_ineq_array,
        q_array,
    )
    first_stage_lp = program.first_stage_lp
    T = _sp_eltype(
        first_stage_lp.c,
        z,
        W_eq_array,
        W_ineq_array,
        T_eq_array,
        T_ineq_array,
        h_eq_array,
        h_ineq_array,
        q_array,
    )

    p_vector = if isnothing(probabilities)
        fill(one(T) / K, K)
    else
        length(probabilities) == K ||
            throw(DimensionMismatch("probabilities must have one entry per scenario."))
        probabilities
    end
    first_stage_μ = _first_stage_barrier_parameter(first_stage_lp, W_ineq_array, μ)
    first_stage_ρ = _first_stage_quadratic_parameter(first_stage_lp, q_array, rho)
    T = promote_type(T, eltype(p_vector), eltype(first_stage_μ), eltype(first_stage_ρ))

    value =
        sum(first_stage_lp.c .* z) +
        _first_stage_quadratic_value(z, first_stage_ρ) -
        _first_stage_barrier_value(first_stage_lp, z, first_stage_μ)
    dz = T.(first_stage_lp.c)
    _add_first_stage_quadratic_gradient!(dz, z, first_stage_ρ)
    _add_first_stage_barrier_gradient!(dz, first_stage_lp, z, first_stage_μ)

    for k in 1:K
        scenario_μ = _scenario_barrier_parameter(
            size(W_ineq_array, 1),
            K,
            μ,
            k,
            length(first_stage_lp.b_ineq),
            p_vector[k],
        )
        scenario_ρ = _scenario_quadratic_parameter(
            size(q_array, 1),
            K,
            rho,
            k,
            length(first_stage_lp.c),
            p_vector[k],
        )
        result = try
            second_stage_lp = LP(
                view(W_eq_array, :, :, k),
                view(W_ineq_array, :, :, k),
                view(h_eq_array, :, k) - view(T_eq_array, :, :, k) * z,
                view(h_ineq_array, :, k) - view(T_ineq_array, :, :, k) * z,
                view(q_array, :, k),
            )

            solve(solver, second_stage_lp; μ=scenario_μ, ρ=scenario_ρ, kwargs...)
        catch error
            _throw_stochastic_program_failure(
                error,
                :second_stage_cost,
                solver,
                program,
                W_eq_array,
                W_ineq_array,
                T_eq_array,
                T_ineq_array,
                h_eq_array,
                h_ineq_array,
                q_array;
                μ=μ,
                ρ=rho,
                probabilities=probabilities,
                kwargs=(; kwargs...),
                z=z,
                scenario_index=k,
                scenario_μ=scenario_μ,
                scenario_ρ=scenario_ρ,
            )
        end

        value += p_vector[k] * result.objective_value
        dz .+= p_vector[k] .* (
            -transpose(view(T_eq_array, :, :, k)) * result.dual_eq +
            transpose(view(T_ineq_array, :, :, k)) * result.dual_ineq
        )
    end

    return value, dz
end
