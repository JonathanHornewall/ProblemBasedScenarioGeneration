abstract type Policy end

infer(policy::Policy, context) =
    error("Policy inference is not defined for $(typeof(policy)).")

function generate_decision_set(policy::Policy, contextual_data_set)
    isempty(contextual_data_set) &&
        throw(ArgumentError("contextual_data_set must not be empty."))

    decisions = [infer(policy, data_point.context) for data_point in contextual_data_set]
    return reduce(hcat, decisions)
end

function evaluate_policy(policy::Policy, contextual_data_set, program, parametric_decoder, solver; kwargs...)
    decision_set = generate_decision_set(policy, contextual_data_set)
    return evaluate_policy(decision_set, contextual_data_set, program, parametric_decoder, solver; kwargs...)
end
