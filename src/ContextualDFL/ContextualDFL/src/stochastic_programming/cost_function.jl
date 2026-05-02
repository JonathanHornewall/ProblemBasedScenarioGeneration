function cost_function(
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
    return G(
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
        return_dual=return_dual,
        kwargs...,
    )
end

function G(
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
    K = _sp_n_scenarios(W_eq_array, W_ineq_array, T_eq_array, T_ineq_array, h_eq_array, h_ineq_array, q_array)
    first_stage_lp = program.first_stage_lp
    T = _sp_eltype(
        first_stage_lp.c,
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

    second_stage_value = zero(T)
    λ_h_eq_array = zeros(T, size(h_eq_array))
    λ_h_ineq_array = zeros(T, size(h_ineq_array))

    for k in 1:K
        scenario_μ = _scenario_barrier_parameter(size(W_ineq_array, 1), K, μ, k)
        scenario_value_or_dual = G_hat(
            solver,
            z,
            view(W_eq_array, :, :, k),
            view(W_ineq_array, :, :, k),
            view(T_eq_array, :, :, k),
            view(T_ineq_array, :, :, k),
            view(h_eq_array, :, k),
            view(h_ineq_array, :, k),
            view(q_array, :, k),
            ;
            μ=scenario_μ,
            return_dual=return_dual,
            kwargs...,
        )

        if return_dual
            y, λ_h_eq, λ_h_ineq = scenario_value_or_dual
            scenario_value = sum(view(q_array, :, k) .* y)
            scenario_μ_vector = _barrier_parameter_vector(size(W_ineq_array, 1), scenario_μ)
            if !_is_zero_barrier_parameter(scenario_μ_vector)
                slack =
                    view(h_ineq_array, :, k) - view(T_ineq_array, :, :, k) * z -
                    view(W_ineq_array, :, :, k) * y
                scenario_value -= sum(scenario_μ_vector .* log.(slack))
            end

            second_stage_value += p_vector[k] * scenario_value
            λ_h_eq_array[:, k] = λ_h_eq
            λ_h_ineq_array[:, k] = λ_h_ineq
        else
            second_stage_value += p_vector[k] * scenario_value_or_dual
        end
    end

    value = sum(first_stage_lp.c .* z) + second_stage_value

    return_dual && return value, λ_h_eq_array, λ_h_ineq_array
    return value
end

function G_hat(
    solver::Solver,
    z,
    W_eq,
    W_ineq,
    T_eq,
    T_ineq,
    h_eq,
    h_ineq,
    q;
    μ=0,
    return_dual=false,
    kwargs...,
)
    # Fix z in the second-stage recourse problem:
    # W_eq y = h_eq - T_eq z and W_ineq y <= h_ineq - T_ineq z.
    second_stage_lp = LP(
        W_eq,
        W_ineq,
        h_eq - T_eq * z,
        h_ineq - T_ineq * z,
        q,
    )

    result = solve(solver, second_stage_lp; μ=μ, kwargs...)

    if return_dual
        # The dual variables are returned for differentiability purposes.
        λ_h_eq = result.dual_eq
        λ_h_ineq = result.dual_ineq
        return result.z, λ_h_eq, λ_h_ineq
    end

    return result.objective_value
end

function _scenario_barrier_parameter(n_inequalities, n_scenarios, μ, scenario_index)
    μ isa Number && return μ

    length(μ) == n_inequalities && return μ
    length(μ) == n_inequalities * n_scenarios ||
        throw(DimensionMismatch("μ must have one entry per scenario inequality or per stacked scenario inequality."))

    rows = ((scenario_index - 1) * n_inequalities + 1):(scenario_index * n_inequalities)
    return view(μ, rows)
end
