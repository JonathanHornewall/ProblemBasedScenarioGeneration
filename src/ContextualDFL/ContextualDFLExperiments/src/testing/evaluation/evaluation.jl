function solve_dataset_to_optimality(
    contextual_data_set,
    program,
    parametric_decoder,
    solver;
    mu=0,
    kwargs...,
)
    results = NamedTuple[]
    for data_point in contextual_data_set
        # Decode this data point's scenario parameters into stochastic-program arrays.
        W_eq, W_ineq, T_eq, T_ineq, h_eq, h_ineq, q =
            ContextualDFL.decode_scenario_collection(
                parametric_decoder,
                data_point.scenario_parameters,
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

        push!(
            results,
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
    return results
end

function evaluate_policy(
    decision_set::AbstractMatrix,
    contextual_data_set,
    program,
    parametric_decoder,
    solver;
    mu=0,
    kwargs...,
)
    size(decision_set, 2) == length(contextual_data_set) ||
        throw(DimensionMismatch("decision_set must have one column per data point."))

    values = []
    for data_index in eachindex(contextual_data_set)
        data_point = contextual_data_set[data_index]
        z = view(decision_set, :, data_index)

        # Decode the realized scenarios, then score the fixed decision z on them.
        W_eq, W_ineq, T_eq, T_ineq, h_eq, h_ineq, q =
            ContextualDFL.decode_scenario_collection(
                parametric_decoder,
                data_point.scenario_parameters,
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
                kwargs...,
            ),
        )
    end
    return values
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
    optimal_results=nothing,
    split_name=:test,
    mu=0,
    kwargs...,
)
    optimal_solve_seconds = 0.0
    actual_optimal_results = optimal_results

    if isnothing(actual_optimal_results)
        optimal_solve_seconds = @elapsed begin
            actual_optimal_results = solve_dataset_to_optimality(
                contextual_data_set,
                program,
                parametric_decoder,
                solver;
                mu=mu,
                kwargs...,
            )
        end
    end

    length(actual_optimal_results) == length(contextual_data_set) ||
        throw(DimensionMismatch("optimal_results must have one entry per data point."))

    policy_values = nothing
    policy_eval_seconds = @elapsed begin
        policy_values = evaluate_policy(
            policy_or_decision_set,
            contextual_data_set,
            program,
            parametric_decoder,
            solver;
            mu=mu,
            kwargs...,
        )
    end

    optimal_values = [result.objective_value for result in actual_optimal_results]
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
                optimal_solve_seconds=optimal_solve_seconds,
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
        ) for index in eachindex(policy_values)
    ]

    return (;
        metrics=metrics,
        per_sample=per_sample,
        optimal_results=actual_optimal_results,
    )
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
