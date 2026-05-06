"""
Compute benchmark optima for a contextual data set.

`evaluation_batches` is the number of independent scenario collections stored
contiguously in each data point. For each context, every collection is solved to
optimality separately and the returned objective value is their average. This is
an average batch oracle, not the optimum of the union of all scenarios in the
data point.
"""
function solve_dataset_to_optimality(
    contextual_data_set,
    program,
    parametric_decoder,
    solver;
    mu=0,
    rho=0,
    evaluation_batches=1,
    progress_io=nothing,
    progress_label="",
    kwargs...,
)
    batch_count = _checked_evaluation_batches(evaluation_batches)
    batch_count > 1 && :probabilities in keys((; kwargs...)) &&
        throw(ArgumentError("evaluation_batches > 1 expects equally weighted scenario collections; omit explicit probabilities."))

    results = NamedTuple[]
    total_contexts = length(contextual_data_set)
    total_batches = total_contexts * batch_count
    completed_batches = 0
    progress_start = time()
    progress_prefix = isempty(String(progress_label)) ?
        "optimality" :
        "optimality[$(String(progress_label))]"

    for (data_point_index, data_point) in enumerate(contextual_data_set)
        objective_values = Float64[]

        for (batch_index, scenario_range) in
            enumerate(_scenario_collection_ranges(data_point, batch_count))
            progress_io !== nothing && _print_batch_progress(
                progress_io,
                progress_prefix,
                "start";
                data_point_index=data_point_index,
                total_contexts=total_contexts,
                batch_index=batch_index,
                batch_count=batch_count,
                completed_batches=completed_batches,
                total_batches=total_batches,
            )

            objective_value = Ref{Float64}()
            batch_seconds = @elapsed begin
                W_eq, W_ineq, T_eq, T_ineq, h_eq, h_ineq, q =
                    ContextualDFL.decode_scenario_collection(
                        parametric_decoder,
                        view(data_point.scenario_parameters, scenario_range),
                    )

                _, _, _, solve_result = ContextualDFL._solve_stochastic_extensive(
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
                objective_value[] = _checked_solve_result_objective(
                    solve_result;
                    data_point_index=data_point_index,
                    batch_index=batch_index,
                )
            end

            push!(
                objective_values,
                objective_value[],
            )
            completed_batches += 1
            progress_io !== nothing && _print_batch_progress(
                progress_io,
                progress_prefix,
                "finish";
                data_point_index=data_point_index,
                total_contexts=total_contexts,
                batch_index=batch_index,
                batch_count=batch_count,
                completed_batches=completed_batches,
                total_batches=total_batches,
                batch_seconds=batch_seconds,
                average_seconds=(time() - progress_start) / completed_batches,
            )
        end

        push!(
            results,
            _checked_optimality_result(
                (;
                    evaluation_batches=batch_count,
                    objective_values=objective_values,
                    objective_value=summary_mean(objective_values),
                );
                source="solve_dataset_to_optimality context $data_point_index",
            ),
        )
    end
    return results
end

function _checked_solve_result_objective(
    solve_result;
    data_point_index,
    batch_index,
)
    hasproperty(solve_result, :status) ||
        throw(ArgumentError("optimal solve result is missing solver status."))
    status_name = string(solve_result.status)
    status_name in ("OPTIMAL", "LOCALLY_SOLVED") ||
        throw(ArgumentError(
            "optimal solve failed for context $(Int(data_point_index)), batch $(Int(batch_index)): status=$status_name.",
        ))
    if hasproperty(solve_result, :metadata) &&
       hasproperty(solve_result.metadata, :primal_status)
        primal_status_name = string(solve_result.metadata.primal_status)
        primal_status_name in ("FEASIBLE_POINT", "NEARLY_FEASIBLE_POINT") ||
            throw(ArgumentError(
                "optimal solve failed for context $(Int(data_point_index)), batch $(Int(batch_index)): primal_status=$primal_status_name.",
            ))
    end

    hasproperty(solve_result, :z) ||
        throw(ArgumentError("optimal solve result is missing primal solution."))
    all(isfinite, solve_result.z) ||
        throw(DomainError(
            solve_result.z,
            "optimal solve returned a non-finite primal solution for context $(Int(data_point_index)), batch $(Int(batch_index)).",
        ))

    hasproperty(solve_result, :objective_value) ||
        throw(ArgumentError("optimal solve result is missing objective_value."))
    objective_value = Float64(solve_result.objective_value)
    isfinite(objective_value) ||
        throw(DomainError(
            solve_result.objective_value,
            "optimal solve returned a non-finite objective for context $(Int(data_point_index)), batch $(Int(batch_index)).",
        ))

    return objective_value
end

function _print_batch_progress(
    io,
    prefix::AbstractString,
    event::AbstractString;
    data_point_index::Integer,
    total_contexts::Integer,
    batch_index::Integer,
    batch_count::Integer,
    completed_batches::Integer,
    total_batches::Integer,
    batch_seconds=nothing,
    average_seconds=nothing,
)
    fields = [
        prefix,
        event,
        "context=$(Int(data_point_index))/$(Int(total_contexts))",
        "batch=$(Int(batch_index))/$(Int(batch_count))",
        "completed=$(Int(completed_batches))/$(Int(total_batches))",
    ]
    if batch_seconds !== nothing
        push!(fields, "batch_seconds=$(round(Float64(batch_seconds); digits=3))")
    end
    if average_seconds !== nothing
        remaining_batches = Int(total_batches) - Int(completed_batches)
        eta_seconds = Float64(average_seconds) * remaining_batches
        push!(fields, "average_seconds=$(round(Float64(average_seconds); digits=3))")
        push!(fields, "eta_seconds=$(round(eta_seconds; digits=3))")
    end
    println(io, join(fields, " "))
    flush(io)
    return nothing
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

        value = ContextualDFL.cost_function(
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
        isfinite(Float64(value)) ||
            throw(DomainError(value, "policy evaluation returned a non-finite value."))
        push!(values, value)
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
    negative_gap_tolerance=1e-5,
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
    _assert_no_significant_negative_gaps(
        gap_values_by_collection,
        policy_values_by_collection,
        optimal_values_by_collection;
        tolerance=Float64(negative_gap_tolerance),
        split_name=split_name,
    )
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
        _checked_objective_values!(values; source="optimal_results")
        _check_stored_objective_mean(result, values; source="optimal_results")
        _check_stored_evaluation_batches(result, values; source="optimal_results")
        return values
    elseif hasproperty(result, :batch_objective_values)
        throw(ArgumentError(
            "optimal_results contain batch_objective_values from the old evaluation protocol; regenerate them with solve_dataset_to_optimality.",
        ))
    elseif hasproperty(result, :objective_value)
        values = [Float64(result.objective_value)]
        _checked_objective_values!(values; source="optimal_results")
        _check_stored_evaluation_batches(result, values; source="optimal_results")
        return values
    end

    throw(ArgumentError("optimal_results entries must contain objective_values."))
end

function _checked_optimality_result(result; source)
    values = _optimal_objective_values(result)
    _check_stored_objective_mean(result, values; source=source)
    _check_stored_evaluation_batches(result, values; source=source)
    return result
end

function _checked_objective_values!(values; source)
    isempty(values) &&
        throw(ArgumentError("$source must contain at least one objective value."))
    all(isfinite, values) ||
        throw(DomainError(values, "$source contains non-finite objective values."))
    return values
end

function _check_stored_objective_mean(result, values; source)
    hasproperty(result, :objective_value) || return nothing
    objective_value = Float64(result.objective_value)
    isfinite(objective_value) ||
        throw(DomainError(result.objective_value, "$source has a non-finite objective_value."))
    mean_value = summary_mean(values)
    isapprox(objective_value, mean_value; rtol=1e-10, atol=1e-10) ||
        throw(ArgumentError(
            "$source objective_value=$objective_value does not equal mean(objective_values)=$mean_value.",
        ))
    return nothing
end

function _check_stored_evaluation_batches(result, values; source)
    hasproperty(result, :evaluation_batches) || return nothing
    batch_count = Int(result.evaluation_batches)
    batch_count == length(values) ||
        throw(ArgumentError(
            "$source declares evaluation_batches=$batch_count but has $(length(values)) objective_values.",
        ))
    return nothing
end

function _assert_no_significant_negative_gaps(
    gap_values_by_collection,
    policy_values_by_collection,
    optimal_values_by_collection;
    tolerance,
    split_name,
)
    tolerance >= 0.0 ||
        throw(ArgumentError("negative_gap_tolerance must be nonnegative."))
    for sample_index in eachindex(gap_values_by_collection)
        gaps = gap_values_by_collection[sample_index]
        for batch_index in eachindex(gaps)
            gap = Float64(gaps[batch_index])
            if gap < -tolerance
                throw(ArgumentError(
                    "negative policy-optimum gap for split $(Symbol(split_name)), " *
                    "sample $sample_index, batch $batch_index: " *
                    "policy_value=$(policy_values_by_collection[sample_index][batch_index]), " *
                    "optimal_value=$(optimal_values_by_collection[sample_index][batch_index]), " *
                    "gap=$gap.",
                ))
            end
        end
    end
    return nothing
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
