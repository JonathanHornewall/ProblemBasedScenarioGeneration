function solve_dataset_to_optimality(
    contextual_data_set,
    program,
    parametric_decoder,
    solver;
    mu=0,
    splits=1,
    kwargs...,
)
    split_count = _checked_split_count(splits)
    _check_split_probability_kwargs(split_count, kwargs)

    results = NamedTuple[]
    for data_point in contextual_data_set
        objective_values = Float64[]
        split_results = NamedTuple[]

        for scenario_range in _scenario_split_ranges(data_point, split_count)
            # Decode this data point's scenario parameters into stochastic-program arrays.
            W_eq, W_ineq, T_eq, T_ineq, h_eq, h_ineq, q =
                ContextualDFL.decode_scenario_collection(
                    parametric_decoder,
                    view(data_point.scenario_parameters, scenario_range),
                )

            # Solve the full stochastic program for the optimal first-stage decision.
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
                kwargs...,
            )

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
                kwargs...,
            )

            push!(objective_values, objective_value)
            push!(
                split_results,
                (;
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

        objective_value = summary_mean(objective_values)

        push!(
            results,
            (;
                split_count=split_count,
                objective_values=objective_values,
                objective_value=objective_value,
                split_results=split_results,
                z=_single_or_split_values(result.z for result in split_results),
                y=_single_or_split_values(result.y for result in split_results),
                λ_b_eq=_single_or_split_values(
                    result.λ_b_eq for result in split_results
                ),
                λ_b_ineq=_single_or_split_values(
                    result.λ_b_ineq for result in split_results
                ),
                λ_h_eq=_single_or_split_values(
                    result.λ_h_eq for result in split_results
                ),
                λ_h_ineq=_single_or_split_values(
                    result.λ_h_ineq for result in split_results
                ),
            ),
        )
    end
    return results
end

function evaluate_policy(
    decision_set::AbstractMatrix,
    contextual_data_set,
    program,
    parametric_decoder,
    solver;
    mu=0,
    splits=1,
    kwargs...,
)
    values, _ = _evaluate_decision_set(
        decision_set,
        contextual_data_set,
        program,
        parametric_decoder,
        solver;
        mu=mu,
        splits=splits,
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
    splits=1,
    kwargs...,
)
    size(decision_set, 2) == length(contextual_data_set) ||
        throw(DimensionMismatch("decision_set must have one column per data point."))

    split_count = _checked_split_count(splits)
    _check_split_probability_kwargs(split_count, kwargs)
    values = []
    split_values_by_sample = Vector{Vector{Float64}}()

    for data_index in eachindex(contextual_data_set)
        data_point = contextual_data_set[data_index]
        z = view(decision_set, :, data_index)
        value, split_values = _evaluate_decision_on_data_point(
            z,
            data_point,
            program,
            parametric_decoder,
            solver;
            mu=mu,
            splits=split_count,
            kwargs...,
        )

        push!(values, value)
        push!(split_values_by_sample, split_values)
    end
    return values, split_values_by_sample
end

function _evaluate_decision_on_data_point(
    z,
    data_point,
    program,
    parametric_decoder,
    solver;
    mu=0,
    splits=1,
    kwargs...,
)
    split_values = Float64[]

    for scenario_range in _scenario_split_ranges(data_point, splits)
        # Decode the realized scenarios, then score the fixed decision z on them.
        W_eq, W_ineq, T_eq, T_ineq, h_eq, h_ineq, q =
            ContextualDFL.decode_scenario_collection(
                parametric_decoder,
                view(data_point.scenario_parameters, scenario_range),
            )

        push!(
            split_values,
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
                kwargs...,
            ),
        )
    end

    return summary_mean(split_values), split_values
end

function _checked_split_count(splits)
    splits isa Integer ||
        throw(ArgumentError("splits must be a positive integer, got $(typeof(splits))."))

    split_count = Int(splits)
    split_count > 0 || throw(ArgumentError("splits must be positive, got $split_count."))
    return split_count
end

function _scenario_split_ranges(data_point, split_count::Integer)
    scenario_count = length(data_point.scenario_parameters)
    scenario_count > 0 || throw(ArgumentError("scenario collections must not be empty."))
    scenario_count % split_count == 0 ||
        throw(ArgumentError(
            "scenario count $scenario_count is not divisible by splits=$split_count.",
        ))

    split_size = scenario_count ÷ split_count
    return [
        ((split_index - 1) * split_size + 1):(split_index * split_size)
        for split_index in 1:split_count
    ]
end

function _single_or_split_values(values)
    collected = collect(values)
    return length(collected) == 1 ? only(collected) : collected
end

function _check_split_probability_kwargs(split_count::Integer, kwargs)
    split_count == 1 && return nothing
    :probabilities in keys((; kwargs...)) || return nothing

    throw(ArgumentError(
        "splits > 1 does not currently support explicit probabilities; omit probabilities or evaluate one unsplit stochastic program.",
    ))
end

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

function evaluate_policy_against_optimum(
    policy_or_decision_set,
    contextual_data_set,
    program,
    parametric_decoder,
    solver;
    optimal_results,
    split_name=:test,
    mu=0,
    splits=nothing,
    kwargs...,
)
    length(optimal_results) == length(contextual_data_set) ||
        throw(DimensionMismatch("optimal_results must have one entry per data point."))

    split_count = _split_count_for_optimal_results(optimal_results, splits)
    policy_values = nothing
    policy_split_values = nothing
    policy_eval_seconds = @elapsed begin
        decision_set =
            _decision_set_for_evaluation(policy_or_decision_set, contextual_data_set)
        policy_values, policy_split_values = _evaluate_decision_set(
            decision_set,
            contextual_data_set,
            program,
            parametric_decoder,
            solver;
            mu=mu,
            splits=split_count,
            kwargs...,
        )
    end

    optimal_split_values = [_optimal_result_split_values(result) for result in optimal_results]
    optimal_values = [summary_mean(values) for values in optimal_split_values]
    split_name = Symbol(split_name)
    regrets = Float64.(policy_values) .- Float64.(optimal_values)
    relative_regrets = [
        regret / max(abs(Float64(optimal_value)), eps(Float64)) for
        (regret, optimal_value) in zip(regrets, optimal_values)
    ]

    metrics = merge(
        summarize_values(policy_values; prefix=Symbol(split_name, :_policy_value)),
        summarize_values(optimal_values; prefix=Symbol(split_name, :_optimal_value)),
        summarize_regret(policy_values, optimal_values; prefix=split_name),
        prefix_named_tuple(
            split_name,
            (;
                sample_count=length(contextual_data_set),
                split_count=split_count,
                policy_eval_seconds=policy_eval_seconds,
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
            policy_split_values=policy_split_values[index],
            optimal_split_values=optimal_split_values[index],
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

function _split_count_for_optimal_results(optimal_results, requested_splits)
    requested_split_count =
        isnothing(requested_splits) ? nothing : _checked_split_count(requested_splits)

    isempty(optimal_results) && return isnothing(requested_split_count) ? 1 : requested_split_count

    split_counts = [
        hasproperty(result, :objective_values) ? length(result.objective_values) : 1
        for result in optimal_results
    ]
    all(>(0), split_counts) ||
        throw(ArgumentError("optimal_results must contain at least one value per sample."))
    all(==(first(split_counts)), split_counts) ||
        throw(ArgumentError("optimal_results contain mixed split counts."))

    inferred_split_count = first(split_counts)
    if !isnothing(requested_split_count) && requested_split_count != inferred_split_count
        throw(ArgumentError(
            "requested splits=$requested_split_count does not match optimal_results split count=$inferred_split_count.",
        ))
    end

    return inferred_split_count
end

function _optimal_result_split_values(result)
    if hasproperty(result, :objective_values)
        values = Float64.(collect(result.objective_values))
        isempty(values) &&
            throw(ArgumentError("optimal_results must contain at least one value per sample."))
        return values
    end

    return [Float64(result.objective_value)]
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
