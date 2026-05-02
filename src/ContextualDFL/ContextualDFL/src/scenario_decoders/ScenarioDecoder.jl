abstract type ScenarioDecoder end

(decoder::ScenarioDecoder)(ξ) =
    error("Scenario decoding is not defined for $(typeof(decoder)).")

function decode_scenario_collection(
    decoder::ScenarioDecoder,
    scenario_parameter_collection::AbstractVector,
)
    K = length(scenario_parameter_collection)
    K > 0 || throw(ArgumentError("scenario_parameter_collection must not be empty."))

    scenario = decoder(first(scenario_parameter_collection))
    W_eq, W_ineq, T_eq, T_ineq, h_eq, h_ineq, q = scenario

    W_eq isa AbstractMatrix || throw(ArgumentError("W_eq must be a matrix."))
    W_ineq isa AbstractMatrix || throw(ArgumentError("W_ineq must be a matrix."))
    T_eq isa AbstractMatrix || throw(ArgumentError("T_eq must be a matrix."))
    T_ineq isa AbstractMatrix || throw(ArgumentError("T_ineq must be a matrix."))
    h_eq isa AbstractVector || throw(ArgumentError("h_eq must be a vector."))
    h_ineq isa AbstractVector || throw(ArgumentError("h_ineq must be a vector."))
    q isa AbstractVector || throw(ArgumentError("q must be a vector."))

    W_eq_array = Array{eltype(W_eq)}(undef, size(W_eq, 1), size(W_eq, 2), K)
    W_ineq_array = Array{eltype(W_ineq)}(undef, size(W_ineq, 1), size(W_ineq, 2), K)
    T_eq_array = Array{eltype(T_eq)}(undef, size(T_eq, 1), size(T_eq, 2), K)
    T_ineq_array = Array{eltype(T_ineq)}(undef, size(T_ineq, 1), size(T_ineq, 2), K)
    h_eq_array = Matrix{eltype(h_eq)}(undef, length(h_eq), K)
    h_ineq_array = Matrix{eltype(h_ineq)}(undef, length(h_ineq), K)
    q_array = Matrix{eltype(q)}(undef, length(q), K)

    copyto!(view(W_eq_array, :, :, 1), W_eq)
    copyto!(view(W_ineq_array, :, :, 1), W_ineq)
    copyto!(view(T_eq_array, :, :, 1), T_eq)
    copyto!(view(T_ineq_array, :, :, 1), T_ineq)
    copyto!(view(h_eq_array, :, 1), h_eq)
    copyto!(view(h_ineq_array, :, 1), h_ineq)
    copyto!(view(q_array, :, 1), q)

    @inbounds for k in 2:K
        scenario = decoder(scenario_parameter_collection[k])
        W_eq, W_ineq, T_eq, T_ineq, h_eq, h_ineq, q = scenario

        copyto!(view(W_eq_array, :, :, k), W_eq)
        copyto!(view(W_ineq_array, :, :, k), W_ineq)
        copyto!(view(T_eq_array, :, :, k), T_eq)
        copyto!(view(T_ineq_array, :, :, k), T_ineq)
        copyto!(view(h_eq_array, :, k), h_eq)
        copyto!(view(h_ineq_array, :, k), h_ineq)
        copyto!(view(q_array, :, k), q)
    end

    return W_eq_array,
        W_ineq_array,
        T_eq_array,
        T_ineq_array,
        h_eq_array,
        h_ineq_array,
        q_array
end
