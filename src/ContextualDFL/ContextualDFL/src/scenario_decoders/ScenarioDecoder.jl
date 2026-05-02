abstract type ScenarioDecoder end

(decoder::ScenarioDecoder)(scenario_parameter) =
    error("Scenario decoding is not defined for $(typeof(decoder)).")

function decode_scenario_collection(
    decoder::ScenarioDecoder,
    scenario_parameter_collection::AbstractVector,
)
    return _decode_scenario_collection(decoder, scenario_parameter_collection)
end

function _decode_scenario_collection(
    decoder::ScenarioDecoder,
    scenario_parameter_collection::AbstractVector,
)
    K = length(scenario_parameter_collection)
    K > 0 || throw(ArgumentError("scenario_parameter_collection must not be empty."))

    decoded_scenarios = map(decoder, scenario_parameter_collection)
    W_eq, W_ineq, T_eq, T_ineq, h_eq, h_ineq, q = first(decoded_scenarios)

    W_eq isa AbstractMatrix || throw(ArgumentError("W_eq must be a matrix."))
    W_ineq isa AbstractMatrix || throw(ArgumentError("W_ineq must be a matrix."))
    T_eq isa AbstractMatrix || throw(ArgumentError("T_eq must be a matrix."))
    T_ineq isa AbstractMatrix || throw(ArgumentError("T_ineq must be a matrix."))
    h_eq isa AbstractVector || throw(ArgumentError("h_eq must be a vector."))
    h_ineq isa AbstractVector || throw(ArgumentError("h_ineq must be a vector."))
    q isa AbstractVector || throw(ArgumentError("q must be a vector."))

    W_eq_array = _stack_scenario_matrices(map(scenario -> scenario[1], decoded_scenarios))
    W_ineq_array = _stack_scenario_matrices(map(scenario -> scenario[2], decoded_scenarios))
    T_eq_array = _stack_scenario_matrices(map(scenario -> scenario[3], decoded_scenarios))
    T_ineq_array = _stack_scenario_matrices(map(scenario -> scenario[4], decoded_scenarios))
    h_eq_array = _stack_scenario_vectors(map(scenario -> scenario[5], decoded_scenarios))
    h_ineq_array = _stack_scenario_vectors(map(scenario -> scenario[6], decoded_scenarios))
    q_array = _stack_scenario_vectors(map(scenario -> scenario[7], decoded_scenarios))

    return W_eq_array,
        W_ineq_array,
        T_eq_array,
        T_ineq_array,
        h_eq_array,
        h_ineq_array,
        q_array
end

function _stack_scenario_matrices(matrices)
    matrix = first(matrices)
    length(matrices) == 1 &&
        return reshape(matrix, size(matrix, 1), size(matrix, 2), 1)
    return cat(matrices...; dims=3)
end

function _stack_scenario_vectors(vectors)
    vector = first(vectors)
    length(vectors) == 1 &&
        return reshape(vector, length(vector), 1)
    return reduce(hcat, vectors)
end
