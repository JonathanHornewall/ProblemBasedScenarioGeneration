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
    ρ=0,
    rho=ρ,
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
        ρ=rho,
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
    ρ=0,
    rho=ρ,
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
    first_stage_μ = _first_stage_barrier_parameter(first_stage_lp, W_ineq_array, μ)
    first_stage_ρ = _first_stage_quadratic_parameter(first_stage_lp, q_array, rho)
    T = promote_type(T, eltype(p_vector), eltype(first_stage_μ), eltype(first_stage_ρ))

    second_stage_value = zero(T)
    λ_h_eq_array = zeros(T, size(h_eq_array))
    λ_h_ineq_array = zeros(T, size(h_ineq_array))

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
        scenario_value_or_dual = try
            G_hat(
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
                ρ=scenario_ρ,
                return_dual=return_dual,
                kwargs...,
            )
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

        if return_dual
            y, λ_h_eq, λ_h_ineq = scenario_value_or_dual
            scenario_value = sum(view(q_array, :, k) .* y)
            scenario_ρ_vector = _quadratic_parameter_vector(length(y), scenario_ρ)
            scenario_value += 0.5 * sum(scenario_ρ_vector .* (y .^ 2))
            scenario_μ_vector = _barrier_parameter_vector(size(W_ineq_array, 1), scenario_μ)
            positive_barrier_indices = findall(!iszero, scenario_μ_vector)
            if !isempty(positive_barrier_indices)
                slack =
                    view(h_ineq_array, :, k) - view(T_ineq_array, :, :, k) * z -
                    view(W_ineq_array, :, :, k) * y
                scenario_value -= sum(
                    scenario_μ_vector[i] * log(slack[i])
                    for i in positive_barrier_indices
                )
            end

            second_stage_value += p_vector[k] * scenario_value
            λ_h_eq_array[:, k] = λ_h_eq
            λ_h_ineq_array[:, k] = λ_h_ineq
        else
            second_stage_value += p_vector[k] * scenario_value_or_dual
        end
    end

    value =
        sum(first_stage_lp.c .* z) +
        _first_stage_quadratic_value(z, first_stage_ρ) -
        _first_stage_barrier_value(first_stage_lp, z, first_stage_μ) +
        second_stage_value

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
    ρ=0,
    rho=ρ,
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

    result = solve(solver, second_stage_lp; μ=μ, ρ=rho, kwargs...)

    if return_dual
        # The dual variables are returned for differentiability purposes.
        λ_h_eq = result.dual_eq
        λ_h_ineq = result.dual_ineq
        return result.z, λ_h_eq, λ_h_ineq
    end

    return result.objective_value
end

function _first_stage_barrier_parameter(first_stage_lp, W_ineq_array, μ)
    n_first_stage_inequalities = length(first_stage_lp.b_ineq)
    μ isa Number && return _barrier_parameter_vector(n_first_stage_inequalities, μ)
    n_first_stage_inequalities == 0 && return view(μ, 1:0)

    n_extensive_inequalities =
        n_first_stage_inequalities + size(W_ineq_array, 1) * size(W_ineq_array, 3)
    length(μ) == n_extensive_inequalities ||
        throw(DimensionMismatch(
            "μ must be a scalar or have one entry per extensive-form inequality when first-stage inequalities are present.",
        ))

    return view(μ, 1:n_first_stage_inequalities)
end

function _first_stage_quadratic_parameter(first_stage_lp, q_array, ρ)
    n_first_stage_variables = length(first_stage_lp.c)
    ρ isa Number && return _quadratic_parameter_vector(n_first_stage_variables, ρ)

    n_extensive_variables = n_first_stage_variables + size(q_array, 1) * size(q_array, 2)
    ρ_vector = _quadratic_parameter_vector(n_extensive_variables, ρ)
    return view(ρ_vector, 1:n_first_stage_variables)
end

function _first_stage_quadratic_value(z, ρ_vector)
    positive_quadratic_indices = findall(!iszero, ρ_vector)
    isempty(positive_quadratic_indices) && return zero(_sp_eltype(z, ρ_vector))

    return 0.5 * sum(ρ_vector[i] * z[i]^2 for i in positive_quadratic_indices)
end

function _add_first_stage_quadratic_gradient!(dz, z, ρ_vector)
    _is_zero_quadratic_parameter(ρ_vector) && return dz
    dz .+= ρ_vector .* z
    return dz
end

function _first_stage_barrier_value(first_stage_lp, z, μ_vector)
    positive_barrier_indices = findall(!iszero, μ_vector)
    isempty(positive_barrier_indices) && return zero(_sp_eltype(first_stage_lp.b_ineq, z, μ_vector))

    slack = first_stage_lp.b_ineq - first_stage_lp.A_ineq * z
    all(i -> slack[i] > zero(slack[i]), positive_barrier_indices) ||
        throw(DomainError(slack, "The first-stage log-barrier cost requires positive inequality slack."))

    return sum(μ_vector[i] * log(slack[i]) for i in positive_barrier_indices)
end

function _add_first_stage_barrier_gradient!(dz, first_stage_lp, z, μ_vector)
    positive_barrier_indices = findall(!iszero, μ_vector)
    isempty(positive_barrier_indices) && return dz

    slack = first_stage_lp.b_ineq - first_stage_lp.A_ineq * z
    all(i -> slack[i] > zero(slack[i]), positive_barrier_indices) ||
        throw(DomainError(slack, "The first-stage log-barrier cost requires positive inequality slack."))

    weights = zeros(promote_type(eltype(slack), eltype(μ_vector)), length(μ_vector))
    for i in positive_barrier_indices
        weights[i] = μ_vector[i] / slack[i]
    end

    dz .+= transpose(first_stage_lp.A_ineq) * weights
    return dz
end

function _scenario_barrier_parameter(
    n_inequalities,
    n_scenarios,
    μ,
    scenario_index,
    n_first_stage_inequalities=0,
    probability=nothing,
)
    μ isa Number && return μ

    if n_first_stage_inequalities > 0
        n_extensive_inequalities =
            n_first_stage_inequalities + n_inequalities * n_scenarios
        if length(μ) == n_extensive_inequalities
            rows = (
                n_first_stage_inequalities + (scenario_index - 1) * n_inequalities + 1
            ):(n_first_stage_inequalities + scenario_index * n_inequalities)
            scenario_μ = view(μ, rows)
            isnothing(probability) && return scenario_μ
            iszero(probability) &&
                throw(ArgumentError("probabilities must be nonzero when μ is an extensive-form vector."))
            return scenario_μ ./ probability
        end
    end

    length(μ) == n_inequalities && return μ
    length(μ) == n_inequalities * n_scenarios ||
        throw(DimensionMismatch("μ must have one entry per scenario inequality or per stacked scenario inequality."))

    rows = ((scenario_index - 1) * n_inequalities + 1):(scenario_index * n_inequalities)
    return view(μ, rows)
end

function _scenario_quadratic_parameter(
    n_variables,
    n_scenarios,
    ρ,
    scenario_index,
    n_first_stage_variables=0,
    probability=nothing,
)
    ρ isa Number && return ρ

    if n_first_stage_variables > 0
        n_extensive_variables = n_first_stage_variables + n_variables * n_scenarios
        if length(ρ) == n_extensive_variables
            cols = (
                n_first_stage_variables + (scenario_index - 1) * n_variables + 1
            ):(n_first_stage_variables + scenario_index * n_variables)
            scenario_ρ = view(ρ, cols)
            isnothing(probability) && return scenario_ρ
            iszero(probability) &&
                throw(ArgumentError("probabilities must be nonzero when ρ is an extensive-form vector."))
            return scenario_ρ ./ probability
        end
    end

    if length(ρ) == n_variables * n_scenarios
        cols = ((scenario_index - 1) * n_variables + 1):(scenario_index * n_variables)
        scenario_ρ = view(ρ, cols)
        isnothing(probability) && return scenario_ρ
        iszero(probability) &&
            throw(ArgumentError("probabilities must be nonzero when ρ is a stacked extensive-form vector."))
        return scenario_ρ ./ probability
    end

    length(ρ) == n_variables ||
        throw(DimensionMismatch("ρ must have one entry per scenario variable or per extensive-form variable."))

    return ρ
end
