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
