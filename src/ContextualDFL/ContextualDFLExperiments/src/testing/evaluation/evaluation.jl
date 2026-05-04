"""
Compute benchmark optima for a contextual data set.

`evaluation_batches` is the number of independent scenario collections stored
contiguously in each data point. For each context, every collection is solved to
optimality separately and the returned objective value is their average.
"""
function solve_dataset_to_optimality(
    contextual_data_set,
    program,
    parametric_decoder,
    solver;
    mu=0,
    rho=0,
    evaluation_batches=1,
    kwargs...,
)
    batch_count = _checked_evaluation_batches(evaluation_batches)
    batch_count > 1 && :probabilities in keys((; kwargs...)) &&
        throw(ArgumentError("evaluation_batches > 1 expects equally weighted scenario collections; omit explicit probabilities."))

    results = NamedTuple[]
    for data_point in contextual_data_set
        objective_values = Float64[]

        for scenario_range in _scenario_collection_ranges(data_point, batch_count)
            W_eq, W_ineq, T_eq, T_ineq, h_eq, h_ineq, q =
                ContextualDFL.decode_scenario_collection(
                    parametric_decoder,
                    view(data_point.scenario_parameters, scenario_range),
                )

            solution = ContextualDFL.solve(
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
            z = solution[1]

            push!(
                objective_values,
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

        push!(
            results,
            (;
                evaluation_batches=batch_count,
                objective_values=objective_values,
                objective_value=summary_mean(objective_values),
            ),
        )
    end
    return results
end

"""
Evaluate a fixed policy or decision matrix on scenario collections.

The policy supplies one first-stage decision per context. That fixed decision is
scored on every scenario collection for the same context, and the returned value
is the mean over collections.
"""
function evaluate_policy(
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
    values, _ = _evaluate_decision_set(
        decision_set,
        contextual_data_set,
        program,
        parametric_decoder,
        solver;
        mu=mu,
        rho=rho,
        evaluation_batches=evaluation_batches,
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

    batch_count = _checked_evaluation_batches(evaluation_batches)
    batch_count > 1 && :probabilities in keys((; kwargs...)) &&
        throw(ArgumentError("evaluation_batches > 1 expects equally weighted scenario collections; omit explicit probabilities."))

    values = Float64[]
    values_by_collection = Vector{Vector{Float64}}()

    for data_index in eachindex(contextual_data_set)
        collection_values = _evaluate_decision_on_collections(
            view(decision_set, :, data_index),
            contextual_data_set[data_index],
            program,
            parametric_decoder,
            solver;
            mu=mu,
            rho=rho,
            evaluation_batches=batch_count,
            kwargs...,
        )

        push!(values, summary_mean(collection_values))
        push!(values_by_collection, collection_values)
    end
    return values, values_by_collection
end

function _evaluate_decision_on_collections(
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
    values = Float64[]
    for scenario_range in _scenario_collection_ranges(data_point, evaluation_batches)
        W_eq, W_ineq, T_eq, T_ineq, h_eq, h_ineq, q =
            ContextualDFL.decode_scenario_collection(
                parametric_decoder,
                view(data_point.scenario_parameters, scenario_range),
            )

        push!(
            values,
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
    return values
end

function _checked_evaluation_batches(evaluation_batches)
    isnothing(evaluation_batches) && return 1
    evaluation_batches isa Integer ||
        throw(ArgumentError(
            "evaluation_batches must be a positive integer, got $(typeof(evaluation_batches)).",
        ))

    batch_count = Int(evaluation_batches)
    batch_count > 0 ||
        throw(ArgumentError("evaluation_batches must be positive, got $batch_count."))
    return batch_count
end

function _scenario_collection_ranges(data_point, batch_count::Integer)
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
Compare a fixed policy against precomputed benchmark optima.

`optimal_results` must come from `solve_dataset_to_optimality` on the same
scenario realizations. Policy and optimal values are compared collection by
collection, then averaged per context.
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
    kwargs...,
)
    @assert length(optimal_results) == length(contextual_data_set) "optimal_results must have one entry per data point"
    length(optimal_results) == length(contextual_data_set) ||
        throw(DimensionMismatch("optimal_results must have one entry per data point."))

    optimal_values_by_collection = [
        _optimal_objective_values(result) for result in optimal_results
    ]
    collection_counts = length.(optimal_values_by_collection)
    batch_count = if isempty(collection_counts)
        1
    else
        all(>(0), collection_counts) ||
            throw(ArgumentError("optimal_results must contain at least one objective value per sample."))
        all(==(first(collection_counts)), collection_counts) ||
            throw(ArgumentError("optimal_results contain mixed evaluation batch counts."))
        first(collection_counts)
    end

    policy_values = Float64[]
    policy_values_by_collection = Vector{Vector{Float64}}()
    policy_eval_seconds = @elapsed begin
        decision_set =
            _decision_set_for_evaluation(policy_or_decision_set, contextual_data_set)
        policy_values, policy_values_by_collection = _evaluate_decision_set(
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

    optimal_values = [summary_mean(values) for values in optimal_values_by_collection]
    gap_values_by_collection = [
        Float64.(policy_values_by_collection[index]) .-
        Float64.(optimal_values_by_collection[index])
        for index in eachindex(policy_values)
    ]
    regrets = [summary_mean(values) for values in gap_values_by_collection]
    relative_regrets = [
        regret / max(abs(Float64(optimal_value)), eps(Float64)) for
        (regret, optimal_value) in zip(regrets, optimal_values)
    ]
    gap_uncertainty = [_uncertainty(values) for values in gap_values_by_collection]

    split_name = Symbol(split_name)
    metrics = merge(
        summarize_values(policy_values; prefix=Symbol(split_name, :_policy_value)),
        summarize_values(optimal_values; prefix=Symbol(split_name, :_optimal_value)),
        summarize_regret(policy_values, optimal_values; prefix=split_name),
        prefix_named_tuple(
            split_name,
            (;
                sample_count=length(contextual_data_set),
                evaluation_batches=batch_count,
                policy_eval_seconds=policy_eval_seconds,
                gap_std_mean=summary_mean(Float64[item.std for item in gap_uncertainty]),
                gap_stderr_mean=summary_mean(Float64[item.stderr for item in gap_uncertainty]),
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
            policy_collection_values=policy_values_by_collection[index],
            optimal_collection_values=optimal_values_by_collection[index],
            gap_values=gap_values_by_collection[index],
            gap_std=gap_uncertainty[index].std,
            gap_stderr=gap_uncertainty[index].stderr,
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

function _optimal_objective_values(result)
    if hasproperty(result, :objective_values)
        values = Float64.(collect(result.objective_values))
        isempty(values) &&
            throw(ArgumentError("optimal_results must contain at least one objective value per sample."))
        return values
    elseif hasproperty(result, :batch_objective_values)
        throw(ArgumentError(
            "optimal_results contain batch_objective_values from the old evaluation protocol; regenerate them with solve_dataset_to_optimality.",
        ))
    elseif hasproperty(result, :objective_value)
        return [Float64(result.objective_value)]
    end

    throw(ArgumentError("optimal_results entries must contain objective_values."))
end

function _uncertainty(values)
    count = length(values)
    std = summary_std(Float64.(collect(values)))
    stderr = count == 0 ? NaN : std / sqrt(count)
    return (; std=std, stderr=stderr)
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
