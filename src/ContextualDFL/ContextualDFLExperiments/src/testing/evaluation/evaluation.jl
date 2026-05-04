"""
Compute optimal first-stage decisions by solving stochastic programs.

This function:
- USES scenarios to OPTIMIZE z
- returns z*(x)

This is NOT used for evaluating learned policies. If `evaluation_batches` is
greater than 1, the optimal z is still solved once on the full scenario
collection; batches are only used afterward to evaluate that fixed z for Monte
Carlo uncertainty reporting.
"""
function solve_dataset_to_optimality(
    contextual_data_set,
    program,
    parametric_decoder,
    solver;
    mu=0,
    rho=0,
    evaluation_batches=nothing,
    splits=nothing,
    evaluate_mode=:batched,
    kwargs...,
)
    batch_count =
        _evaluation_batch_count(evaluation_batches, splits, evaluate_mode; default=1)
    _check_batch_probability_kwargs(batch_count, kwargs)

    results = NamedTuple[]
    for data_point in contextual_data_set
        # Decode the full scenario collection and solve once for z*(x).
        W_eq, W_ineq, T_eq, T_ineq, h_eq, h_ineq, q =
            ContextualDFL.decode_scenario_collection(
                parametric_decoder,
                data_point.scenario_parameters,
            )

        z, y, λ_b_eq, λ_b_ineq, λ_h_eq, λ_h_ineq = ContextualDFL.solve(
            solver,
            program,
            W_eq,
            W_ineq,
            T_eq,
            T_ineq,
            h_eq,
            h_ineq,
            q;
            μ=mu,
            ρ=rho,
            kwargs...,
        )

        # cost_function must evaluate recourse for this fixed z; it must not
        # modify or re-optimize the first-stage decision.
        objective_value = ContextualDFL.cost_function(
            program,
            solver,
            z,
            W_eq,
            W_ineq,
            T_eq,
            T_ineq,
            h_eq,
            h_ineq,
            q;
            μ=mu,
            ρ=rho,
            kwargs...,
        )
        batch_objective_values = _evaluate_fixed_decision_batches(
            z,
            data_point,
            program,
            parametric_decoder,
            solver;
            mu=mu,
            rho=rho,
            evaluation_batches=batch_count,
            kwargs...,
        )

        push!(
            results,
            (;
                evaluation_batch_count=batch_count,
                batch_objective_values=batch_objective_values,
                objective_value=objective_value,
                z=z,
                y=y,
                λ_b_eq=λ_b_eq,
                λ_b_ineq=λ_b_ineq,
                λ_h_eq=λ_h_eq,
                λ_h_ineq=λ_h_ineq,
            ),
        )
    end
    return results
end

"""
Evaluate a fixed policy.

This function:
- computes one decision z(x) for each context
- DOES NOT optimize z
- ONLY evaluates z over scenarios
"""
function evaluate_policy(
    decision_set::AbstractMatrix,
    contextual_data_set,
    program,
    parametric_decoder,
    solver;
    mu=0,
    rho=0,
    evaluation_batches=nothing,
    splits=nothing,
    evaluate_mode=:batched,
    kwargs...,
)
    batch_count =
        _evaluation_batch_count(evaluation_batches, splits, evaluate_mode; default=1)
    values, _ = _evaluate_decision_set(
        decision_set,
        contextual_data_set,
        program,
        parametric_decoder,
        solver;
        mu=mu,
        rho=rho,
        evaluation_batches=batch_count,
        kwargs...,
    )
    return values
end

function _evaluate_decision_set(
    decision_set::AbstractMatrix,
    contextual_data_set,
    program,
    parametric_decoder,
    solver;
    mu=0,
    rho=0,
    evaluation_batches=1,
    kwargs...,
)
    @assert size(decision_set, 2) == length(contextual_data_set) "Each context must map to exactly one decision z"
    size(decision_set, 2) == length(contextual_data_set) ||
        throw(DimensionMismatch("decision_set must have one column per data point."))

    batch_count = _checked_evaluation_batch_count(evaluation_batches)
    _check_batch_probability_kwargs(batch_count, kwargs)
    values = []
    batch_values_by_sample = Vector{Vector{Float64}}()

    for data_index in eachindex(contextual_data_set)
        data_point = contextual_data_set[data_index]
        z = view(decision_set, :, data_index)
        value, batch_values = _evaluate_decision_on_data_point(
            z,
            data_point,
            program,
            parametric_decoder,
            solver;
            mu=mu,
            rho=rho,
            evaluation_batches=batch_count,
            kwargs...,
        )

        push!(values, value)
        push!(batch_values_by_sample, batch_values)
    end
    return values, batch_values_by_sample
end

"""
Evaluate a *fixed first-stage decision z* on a collection of scenarios.

IMPORTANT:
- z must NOT depend on scenarios
- scenarios are only used for Monte Carlo estimation of expected cost
- evaluation batches are used ONLY for variance estimation, not optimization
"""
function _evaluate_decision_on_data_point(
    z,
    data_point,
    program,
    parametric_decoder,
    solver;
    mu=0,
    rho=0,
    evaluation_batches=1,
    kwargs...,
)
    batch_values = _evaluate_fixed_decision_batches(
        z,
        data_point,
        program,
        parametric_decoder,
        solver;
        mu=mu,
        rho=rho,
        evaluation_batches=evaluation_batches,
        kwargs...,
    )

    return summary_mean(batch_values), batch_values
end

function _evaluate_fixed_decision_batches(
    z,
    data_point,
    program,
    parametric_decoder,
    solver;
    mu=0,
    rho=0,
    evaluation_batches=1,
    kwargs...,
)
    batch_count = _checked_evaluation_batch_count(evaluation_batches)
    batch_values = Float64[]

    # NOTE:
    # Splitting scenarios into batches is used for Monte Carlo variance
    # estimation. It does NOT change the decision z and does NOT correspond to
    # multiple optimization problems.
    for scenario_range in _scenario_batch_ranges(data_point, batch_count)
        # Decode realized scenarios, then score the fixed decision z on them.
        W_eq, W_ineq, T_eq, T_ineq, h_eq, h_ineq, q =
            ContextualDFL.decode_scenario_collection(
                parametric_decoder,
                view(data_point.scenario_parameters, scenario_range),
            )

        # cost_function must evaluate recourse only; it must NOT modify z.
        push!(
            batch_values,
            ContextualDFL.cost_function(
                program,
                solver,
                z,
                W_eq,
                W_ineq,
                T_eq,
                T_ineq,
                h_eq,
                h_ineq,
                q;
                μ=mu,
                ρ=rho,
                kwargs...,
            ),
        )
    end

    return batch_values
end

function _evaluation_batch_count(
    evaluation_batches,
    legacy_splits,
    evaluate_mode;
    default=1,
)
    mode = _checked_evaluate_mode(evaluate_mode)
    mode == :mean_only && return 1

    requested = if !isnothing(evaluation_batches)
        evaluation_batches
    elseif !isnothing(legacy_splits)
        legacy_splits
    else
        default
    end
    return _checked_evaluation_batch_count(requested)
end

function _checked_evaluate_mode(evaluate_mode)
    mode = Symbol(evaluate_mode)
    mode in (:mean_only, :batched) ||
        throw(ArgumentError("evaluate_mode must be :mean_only or :batched, got $evaluate_mode."))
    return mode
end

function _checked_evaluation_batch_count(evaluation_batches)
    evaluation_batches isa Integer ||
        throw(ArgumentError(
            "evaluation_batches must be a positive integer, got $(typeof(evaluation_batches)).",
        ))

    batch_count = Int(evaluation_batches)
    batch_count > 0 ||
        throw(ArgumentError("evaluation_batches must be positive, got $batch_count."))
    return batch_count
end

_checked_split_count(splits) = _checked_evaluation_batch_count(splits)

function _scenario_batch_ranges(data_point, batch_count::Integer)
    scenario_count = length(data_point.scenario_parameters)
    scenario_count > 0 || throw(ArgumentError("scenario collections must not be empty."))
    scenario_count % batch_count == 0 ||
        throw(ArgumentError(
            "scenario count $scenario_count is not divisible by evaluation_batches=$batch_count.",
        ))

    batch_size = scenario_count ÷ batch_count
    return [
        ((batch_index - 1) * batch_size + 1):(batch_index * batch_size)
        for batch_index in 1:batch_count
    ]
end

_scenario_split_ranges(data_point, split_count::Integer) =
    _scenario_batch_ranges(data_point, split_count)

function _check_batch_probability_kwargs(batch_count::Integer, kwargs)
    batch_count == 1 && return nothing
    :probabilities in keys((; kwargs...)) || return nothing

    throw(ArgumentError(
        "evaluation_batches > 1 does not currently support explicit probabilities; omit probabilities or use evaluate_mode=:mean_only.",
    ))
end

_check_split_probability_kwargs(split_count::Integer, kwargs) =
    _check_batch_probability_kwargs(split_count, kwargs)

function summarize_values(values; prefix)
    numeric_values = Float64.(collect(values))
    count = length(numeric_values)
    prefix = Symbol(prefix)

    summary = if count == 0
        (;
            count=0,
            mean=NaN,
            median=NaN,
            std=NaN,
            min=NaN,
            max=NaN,
            p95=NaN,
        )
    else
        (;
            count=count,
            mean=summary_mean(numeric_values),
            median=summary_median(numeric_values),
            std=summary_std(numeric_values),
            min=minimum(numeric_values),
            max=maximum(numeric_values),
            p95=percentile_95(numeric_values),
        )
    end

    return prefix_named_tuple(prefix, summary)
end

function summarize_regret(policy_values, optimal_values; prefix)
    length(policy_values) == length(optimal_values) ||
        throw(DimensionMismatch("policy_values and optimal_values must have the same length."))

    regrets = Float64.(policy_values) .- Float64.(optimal_values)
    relative_regrets = [
        regret / max(abs(Float64(optimal_value)), eps(Float64)) for
        (regret, optimal_value) in zip(regrets, optimal_values)
    ]

    return merge(
        summarize_values(regrets; prefix=Symbol(prefix, :_regret)),
        summarize_values(relative_regrets; prefix=Symbol(prefix, :_relative_regret)),
    )
end

"""
Compare a fixed policy against precomputed optimal results.

IMPORTANT:
- optimal_results MUST be computed using the SAME scenario realizations as
  contextual_data_set; otherwise regret estimates are biased
- policy evaluation computes one z(x), then scores that fixed z on scenarios
- `evaluate_mode=:mean_only` evaluates one full-scenario mean
- `evaluate_mode=:batched` reports Monte Carlo batch means for uncertainty
"""
function evaluate_policy_against_optimum(
    policy_or_decision_set,
    contextual_data_set,
    program,
    parametric_decoder,
    solver;
    optimal_results,
    split_name=:test,
    mu=0,
    rho=0,
    evaluation_batches=nothing,
    splits=nothing,
    evaluate_mode=:mean_only,
    kwargs...,
)
    @assert length(optimal_results) == length(contextual_data_set) "optimal_results must have one entry per data point"
    length(optimal_results) == length(contextual_data_set) ||
        throw(DimensionMismatch("optimal_results must have one entry per data point."))

    batch_count = _evaluation_batch_count_for_optimal_results(
        optimal_results,
        evaluation_batches,
        splits,
        evaluate_mode,
    )
    policy_values = nothing
    policy_batch_values = nothing
    policy_eval_seconds = @elapsed begin
        decision_set =
            _decision_set_for_evaluation(policy_or_decision_set, contextual_data_set)
        policy_values, policy_batch_values = _evaluate_decision_set(
            decision_set,
            contextual_data_set,
            program,
            parametric_decoder,
            solver;
            mu=mu,
            rho=rho,
            evaluation_batches=batch_count,
            kwargs...,
        )
    end

    raw_optimal_batch_values = [
        _optimal_result_batch_values(result) for result in optimal_results
    ]
    optimal_batch_values = batch_count == 1 ?
        [[summary_mean(values)] for values in raw_optimal_batch_values] :
        raw_optimal_batch_values
    optimal_values = [summary_mean(values) for values in optimal_batch_values]
    split_name = Symbol(split_name)
    regrets = Float64.(policy_values) .- Float64.(optimal_values)
    relative_regrets = [
        regret / max(abs(Float64(optimal_value)), eps(Float64)) for
        (regret, optimal_value) in zip(regrets, optimal_values)
    ]
    policy_uncertainty = _batch_uncertainties(policy_batch_values)
    optimal_uncertainty = _batch_uncertainties(optimal_batch_values)

    # Evaluation protocol:
    #
    # for each context x:
    #     z = policy(x)
    #     sample omega_1,...,omega_N
    #     compute cost(z, omega_i) for all i
    #     average -> estimate expected cost
    #
    # regret = policy_value - optimal_value
    metrics = merge(
        summarize_values(policy_values; prefix=Symbol(split_name, :_policy_value)),
        summarize_values(optimal_values; prefix=Symbol(split_name, :_optimal_value)),
        summarize_regret(policy_values, optimal_values; prefix=split_name),
        prefix_named_tuple(
            split_name,
            (;
                sample_count=length(contextual_data_set),
                evaluation_batch_count=batch_count,
                evaluate_mode=Symbol(evaluate_mode),
                policy_eval_seconds=policy_eval_seconds,
                policy_value_batch_std_mean=summary_mean([
                    item.std for item in policy_uncertainty
                ]),
                policy_value_batch_stderr_mean=summary_mean([
                    item.stderr for item in policy_uncertainty
                ]),
                optimal_value_batch_std_mean=summary_mean([
                    item.std for item in optimal_uncertainty
                ]),
                optimal_value_batch_stderr_mean=summary_mean([
                    item.stderr for item in optimal_uncertainty
                ]),
            ),
        ),
    )

    per_sample = [
        (;
            sample_index=index,
            policy_value=Float64(policy_values[index]),
            optimal_value=Float64(optimal_values[index]),
            regret=regrets[index],
            relative_regret=relative_regrets[index],
            policy_batch_values=policy_batch_values[index],
            optimal_batch_values=optimal_batch_values[index],
            policy_value_std=policy_uncertainty[index].std,
            policy_value_stderr=policy_uncertainty[index].stderr,
            optimal_value_std=optimal_uncertainty[index].std,
            optimal_value_stderr=optimal_uncertainty[index].stderr,
        ) for index in eachindex(policy_values)
    ]

    return (;
        metrics=metrics,
        per_sample=per_sample,
        optimal_results=optimal_results,
    )
end

_decision_set_for_evaluation(decision_set::AbstractMatrix, contextual_data_set) = decision_set

function _decision_set_for_evaluation(policy::Policy, contextual_data_set)
    return generate_decision_set(policy, contextual_data_set)
end

function _evaluation_batch_count_for_optimal_results(
    optimal_results,
    requested_evaluation_batches,
    requested_splits,
    evaluate_mode,
)
    mode = _checked_evaluate_mode(evaluate_mode)
    requested_batch_count = if !isnothing(requested_evaluation_batches)
        _checked_evaluation_batch_count(requested_evaluation_batches)
    elseif !isnothing(requested_splits)
        _checked_evaluation_batch_count(requested_splits)
    else
        nothing
    end

    mode == :mean_only && return 1

    isempty(optimal_results) && return isnothing(requested_batch_count) ? 1 : requested_batch_count

    batch_counts = [
        length(_optimal_result_batch_values(result))
        for result in optimal_results
    ]
    all(>(0), batch_counts) ||
        throw(ArgumentError("optimal_results must contain at least one value per sample."))
    all(==(first(batch_counts)), batch_counts) ||
        throw(ArgumentError("optimal_results contain mixed evaluation batch counts."))

    inferred_batch_count = first(batch_counts)
    if !isnothing(requested_batch_count) && requested_batch_count != inferred_batch_count
        throw(ArgumentError(
            "requested evaluation_batches=$requested_batch_count does not match optimal_results evaluation batch count=$inferred_batch_count.",
        ))
    end

    return inferred_batch_count
end

function _split_count_for_optimal_results(optimal_results, requested_splits)
    return _evaluation_batch_count_for_optimal_results(
        optimal_results,
        nothing,
        requested_splits,
        :batched,
    )
end

function _optimal_result_batch_values(result)
    if hasproperty(result, :batch_objective_values)
        values = Float64.(collect(result.batch_objective_values))
        isempty(values) &&
            throw(ArgumentError("optimal_results must contain at least one value per sample."))
        return values
    elseif hasproperty(result, :objective_values)
        values = Float64.(collect(result.objective_values))
        isempty(values) &&
            throw(ArgumentError("optimal_results must contain at least one value per sample."))
        return values
    end

    return [Float64(result.objective_value)]
end

_optimal_result_split_values(result) = _optimal_result_batch_values(result)

function _batch_uncertainty(batch_values)
    count = length(batch_values)
    std = summary_std(Float64.(collect(batch_values)))
    stderr = count == 0 ? NaN : std / sqrt(count)
    return (; std=std, stderr=stderr)
end

function _batch_uncertainties(batch_values_by_sample)
    return [_batch_uncertainty(batch_values) for batch_values in batch_values_by_sample]
end

function percentile_95(values::AbstractVector{<:Real})
    isempty(values) && return NaN
    sorted = sort!(collect(Float64.(values)))
    index = clamp(ceil(Int, 0.95 * length(sorted)), 1, length(sorted))
    return sorted[index]
end

function summary_mean(values::AbstractVector{<:Real})
    isempty(values) && return NaN
    return sum(values) / length(values)
end

function summary_median(values::AbstractVector{<:Real})
    isempty(values) && return NaN

    sorted = sort!(collect(Float64.(values)))
    count = length(sorted)
    midpoint = count ÷ 2

    if isodd(count)
        return sorted[midpoint + 1]
    end

    return (sorted[midpoint] + sorted[midpoint + 1]) / 2
end

function summary_std(values::AbstractVector{<:Real})
    count = length(values)
    count == 0 && return NaN
    count == 1 && return 0.0

    mean_value = summary_mean(values)
    return sqrt(sum((value - mean_value)^2 for value in values) / (count - 1))
end

function prefix_named_tuple(prefix::Symbol, values::NamedTuple)
    pairs = Pair{Symbol,Any}[]
    for key in keys(values)
        push!(pairs, Symbol(prefix, :_, key) => getproperty(values, key))
    end
    return NamedTuple{Tuple(first.(pairs))}(Tuple(last.(pairs)))
end
