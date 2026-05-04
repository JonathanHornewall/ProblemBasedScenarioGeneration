struct SPOPlusLoss{
    TInputScenarioDecoder<:VectorDecoder,
    TReferenceScenarioDecoder<:ScenarioDecoder,
    TSolver<:Solver,
    TProgram<:StochasticProgram,
} <: LossFunction
    input_scenario_decoder::TInputScenarioDecoder
    reference_scenario_decoder::TReferenceScenarioDecoder
    solver::TSolver
    program::TProgram
    nr_scenarios::Int
end

function SPOPlusLoss(
    input_scenario_decoder::VectorDecoder,
    reference_scenario_decoder::ScenarioDecoder,
    solver::Solver,
    program::StochasticProgram;
    nr_scenarios::Integer=1,
)
    nr_scenarios > 0 || throw(ArgumentError("nr_scenarios must be a positive integer."))
    return SPOPlusLoss{
        typeof(input_scenario_decoder),
        typeof(reference_scenario_decoder),
        typeof(solver),
        typeof(program),
    }(
        input_scenario_decoder,
        reference_scenario_decoder,
        solver,
        program,
        Int(nr_scenarios),
    )
end

function (loss::SPOPlusLoss)(
    input_scenario_parameter_collection,
    reference_scenario_parameter_collection,
    mu_in=0,
    mu_ref=mu_in;
    probabilities=nothing,
    nr_scenarios=loss.nr_scenarios,
    validate_fixed_feasible_set=true,
    fixed_feasible_set_atol=0,
    fixed_feasible_set_rtol=0,
    kwargs...,
)
    _check_spo_plus_mu(mu_in, mu_ref)

    input_arrays = decode_scenario_collection(
        loss.input_scenario_decoder,
        input_scenario_parameter_collection;
        nr_scenarios=nr_scenarios,
    )
    reference_arrays = decode_scenario_collection(
        loss.reference_scenario_decoder,
        reference_scenario_parameter_collection,
    )

    return _spo_plus_loss_value(
        loss.program,
        loss.solver,
        input_arrays...,
        reference_arrays...;
        probabilities=probabilities,
        validate_fixed_feasible_set=validate_fixed_feasible_set,
        fixed_feasible_set_atol=fixed_feasible_set_atol,
        fixed_feasible_set_rtol=fixed_feasible_set_rtol,
        kwargs...,
    )
end

function _spo_plus_loss_value(
    program::StochasticProgram,
    solver::Solver,
    input_W_eq_array,
    input_W_ineq_array,
    input_T_eq_array,
    input_T_ineq_array,
    input_h_eq_array,
    input_h_ineq_array,
    input_q_array,
    reference_W_eq_array,
    reference_W_ineq_array,
    reference_T_eq_array,
    reference_T_ineq_array,
    reference_h_eq_array,
    reference_h_ineq_array,
    reference_q_array;
    probabilities=nothing,
    validate_fixed_feasible_set=true,
    fixed_feasible_set_atol=0,
    fixed_feasible_set_rtol=0,
    kwargs...,
)
    value, _, _, _ = _spo_plus_oracle(
        program,
        solver,
        input_W_eq_array,
        input_W_ineq_array,
        input_T_eq_array,
        input_T_ineq_array,
        input_h_eq_array,
        input_h_ineq_array,
        input_q_array,
        reference_W_eq_array,
        reference_W_ineq_array,
        reference_T_eq_array,
        reference_T_ineq_array,
        reference_h_eq_array,
        reference_h_ineq_array,
        reference_q_array;
        probabilities=probabilities,
        validate_fixed_feasible_set=validate_fixed_feasible_set,
        fixed_feasible_set_atol=fixed_feasible_set_atol,
        fixed_feasible_set_rtol=fixed_feasible_set_rtol,
        kwargs...,
    )
    return value
end

function ChainRulesCore.rrule(
    ::typeof(_spo_plus_loss_value),
    program::StochasticProgram,
    solver::Solver,
    input_W_eq_array,
    input_W_ineq_array,
    input_T_eq_array,
    input_T_ineq_array,
    input_h_eq_array,
    input_h_ineq_array,
    input_q_array,
    reference_W_eq_array,
    reference_W_ineq_array,
    reference_T_eq_array,
    reference_T_ineq_array,
    reference_h_eq_array,
    reference_h_ineq_array,
    reference_q_array;
    probabilities=nothing,
    validate_fixed_feasible_set=true,
    fixed_feasible_set_atol=0,
    fixed_feasible_set_rtol=0,
    kwargs...,
)
    value, reference_y, perturbed_y, p_vector = _spo_plus_oracle(
        program,
        solver,
        input_W_eq_array,
        input_W_ineq_array,
        input_T_eq_array,
        input_T_ineq_array,
        input_h_eq_array,
        input_h_ineq_array,
        input_q_array,
        reference_W_eq_array,
        reference_W_ineq_array,
        reference_T_eq_array,
        reference_T_ineq_array,
        reference_h_eq_array,
        reference_h_ineq_array,
        reference_q_array;
        probabilities=probabilities,
        validate_fixed_feasible_set=validate_fixed_feasible_set,
        fixed_feasible_set_atol=fixed_feasible_set_atol,
        fixed_feasible_set_rtol=fixed_feasible_set_rtol,
        kwargs...,
    )

    function spo_plus_loss_pullback(value_tangent)
        tangent = _spo_plus_scalar_tangent(value_tangent)
        dq = similar(
            input_q_array,
            promote_type(eltype(input_q_array), eltype(reference_y), eltype(perturbed_y), eltype(p_vector), typeof(tangent)),
            size(input_q_array),
        )
        for k in axes(input_q_array, 2)
            dq[:, k] .= tangent .* (2 .* p_vector[k]) .* (reference_y[:, k] .- perturbed_y[:, k])
        end

        return (
            ChainRulesCore.NoTangent(),
            ChainRulesCore.NoTangent(),
            ChainRulesCore.NoTangent(),
            ChainRulesCore.NoTangent(),
            ChainRulesCore.NoTangent(),
            ChainRulesCore.NoTangent(),
            ChainRulesCore.NoTangent(),
            ChainRulesCore.NoTangent(),
            ChainRulesCore.NoTangent(),
            dq,
            ChainRulesCore.NoTangent(),
            ChainRulesCore.NoTangent(),
            ChainRulesCore.NoTangent(),
            ChainRulesCore.NoTangent(),
            ChainRulesCore.NoTangent(),
            ChainRulesCore.NoTangent(),
            ChainRulesCore.NoTangent(),
        )
    end

    return value, spo_plus_loss_pullback
end

function _spo_plus_oracle(
    program::StochasticProgram,
    solver::Solver,
    input_W_eq_array,
    input_W_ineq_array,
    input_T_eq_array,
    input_T_ineq_array,
    input_h_eq_array,
    input_h_ineq_array,
    input_q_array,
    reference_W_eq_array,
    reference_W_ineq_array,
    reference_T_eq_array,
    reference_T_ineq_array,
    reference_h_eq_array,
    reference_h_ineq_array,
    reference_q_array;
    probabilities=nothing,
    validate_fixed_feasible_set=true,
    fixed_feasible_set_atol=0,
    fixed_feasible_set_rtol=0,
    kwargs...,
)
    _sp_n_scenarios(
        input_W_eq_array,
        input_W_ineq_array,
        input_T_eq_array,
        input_T_ineq_array,
        input_h_eq_array,
        input_h_ineq_array,
        input_q_array,
    )
    K = _sp_n_scenarios(
        reference_W_eq_array,
        reference_W_ineq_array,
        reference_T_eq_array,
        reference_T_ineq_array,
        reference_h_eq_array,
        reference_h_ineq_array,
        reference_q_array,
    )
    size(input_q_array) == size(reference_q_array) ||
        throw(DimensionMismatch("input and reference q arrays must have the same size."))

    if validate_fixed_feasible_set
        _check_spo_plus_fixed_feasible_set(
            input_W_eq_array,
            input_W_ineq_array,
            input_T_eq_array,
            input_T_ineq_array,
            input_h_eq_array,
            input_h_ineq_array,
            reference_W_eq_array,
            reference_W_ineq_array,
            reference_T_eq_array,
            reference_T_ineq_array,
            reference_h_eq_array,
            reference_h_ineq_array;
            atol=fixed_feasible_set_atol,
            rtol=fixed_feasible_set_rtol,
        )
    end

    p_vector = _spo_plus_probability_vector(reference_q_array, probabilities)
    perturbed_q_array = 2 .* input_q_array .- reference_q_array

    reference_z, reference_y, _, _, _, _ = solve(
        solver,
        program,
        reference_W_eq_array,
        reference_W_ineq_array,
        reference_T_eq_array,
        reference_T_ineq_array,
        reference_h_eq_array,
        reference_h_ineq_array,
        reference_q_array;
        probabilities=probabilities,
        μ=0,
        kwargs...,
    )
    perturbed_z, perturbed_y, _, _, _, _ = solve(
        solver,
        program,
        reference_W_eq_array,
        reference_W_ineq_array,
        reference_T_eq_array,
        reference_T_ineq_array,
        reference_h_eq_array,
        reference_h_ineq_array,
        perturbed_q_array;
        probabilities=probabilities,
        μ=0,
        kwargs...,
    )

    value =
        _spo_plus_linear_objective(program, reference_z, reference_y, perturbed_q_array; probabilities=probabilities) -
        _spo_plus_linear_objective(program, perturbed_z, perturbed_y, perturbed_q_array; probabilities=probabilities)

    return value, reference_y, perturbed_y, p_vector
end

function _spo_plus_linear_objective(
    program::StochasticProgram,
    z,
    y,
    q_array;
    probabilities=nothing,
)
    K = size(q_array, 2)
    p_vector = _spo_plus_probability_vector(q_array, probabilities)
    value = sum(program.first_stage_lp.c .* z)
    for k in 1:K
        value += p_vector[k] * sum(view(q_array, :, k) .* view(y, :, k))
    end
    return value
end

function _spo_plus_probability_vector(q_array, probabilities)
    K = size(q_array, 2)
    T = eltype(q_array)
    if isnothing(probabilities)
        return fill(one(T) / K, K)
    end

    length(probabilities) == K ||
        throw(DimensionMismatch("probabilities must have one entry per scenario."))
    return probabilities
end

function _check_spo_plus_mu(mu_in, mu_ref)
    _is_zero_barrier_parameter(mu_in) && _is_zero_barrier_parameter(mu_ref) && return nothing
    throw(ArgumentError(
        "SPOPlusLoss implements the standard nonsmoothed SPO+ surrogate; pass mu_in=0 and mu_ref=0.",
    ))
end

function _check_spo_plus_fixed_feasible_set(
    input_W_eq_array,
    input_W_ineq_array,
    input_T_eq_array,
    input_T_ineq_array,
    input_h_eq_array,
    input_h_ineq_array,
    reference_W_eq_array,
    reference_W_ineq_array,
    reference_T_eq_array,
    reference_T_ineq_array,
    reference_h_eq_array,
    reference_h_ineq_array;
    atol,
    rtol,
)
    _check_spo_plus_same_array(:W_eq, input_W_eq_array, reference_W_eq_array; atol=atol, rtol=rtol)
    _check_spo_plus_same_array(:W_ineq, input_W_ineq_array, reference_W_ineq_array; atol=atol, rtol=rtol)
    _check_spo_plus_same_array(:T_eq, input_T_eq_array, reference_T_eq_array; atol=atol, rtol=rtol)
    _check_spo_plus_same_array(:T_ineq, input_T_ineq_array, reference_T_ineq_array; atol=atol, rtol=rtol)
    _check_spo_plus_same_array(:h_eq, input_h_eq_array, reference_h_eq_array; atol=atol, rtol=rtol)
    _check_spo_plus_same_array(:h_ineq, input_h_ineq_array, reference_h_ineq_array; atol=atol, rtol=rtol)
    return nothing
end

function _check_spo_plus_same_array(name, input_array, reference_array; atol, rtol)
    size(input_array) == size(reference_array) ||
        throw(DimensionMismatch("SPOPlusLoss requires matching $(name) arrays."))
    isapprox(input_array, reference_array; atol=atol, rtol=rtol) ||
        throw(ArgumentError(
            "SPOPlusLoss supports objective-vector predictions only; predicted $(name) must match the reference $(name).",
        ))
    return nothing
end

function _spo_plus_scalar_tangent(value_tangent)
    value_tangent = ChainRulesCore.unthunk(value_tangent)
    _is_zero_cotangent(value_tangent) && return 0
    value_tangent isa Number && return value_tangent
    throw(ArgumentError(
        "Expected scalar cotangent for scalar SPOPlusLoss output; got $(typeof(value_tangent)).",
    ))
end
