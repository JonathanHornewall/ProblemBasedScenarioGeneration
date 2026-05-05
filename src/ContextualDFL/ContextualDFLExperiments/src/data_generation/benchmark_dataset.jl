import Random

function generate_benchmark_contexts(problem; n_contexts, rng=Random.default_rng())
    error("Benchmark context generation is not defined for $(typeof(problem)).")
end

function generate_benchmark_scenarios(
    problem,
    context;
    n_scenarios,
    rng=Random.default_rng(),
)
    error("Benchmark scenario generation is not defined for $(typeof(problem)).")
end

function generate_benchmark_dataset(
    problem;
    n_contexts,
    scenarios_per_context,
    seed=1,
    rng=Random.MersenneTwister(seed),
)
    context_count = _checked_positive_integer(n_contexts, :n_contexts)
    scenario_count = _checked_positive_integer(scenarios_per_context, :scenarios_per_context)

    contexts = generate_benchmark_contexts(problem; n_contexts=context_count, rng=rng)
    scenario_collections = [
        generate_benchmark_scenarios(problem, context; n_scenarios=scenario_count, rng=rng)
        for context in contexts
    ]

    return generate_contextual_data_set(contexts, scenario_collections)
end

function _checked_positive_integer(value, name::Symbol)
    value isa Integer ||
        throw(ArgumentError("$(name) must be a positive integer, got $(typeof(value))."))

    checked_value = Int(value)
    checked_value > 0 ||
        throw(ArgumentError("$(name) must be positive, got $checked_value."))
    return checked_value
end

function _checked_context_vector(context, expected_length)
    context isa AbstractVector ||
        throw(ArgumentError("context must be an AbstractVector."))
    length(context) == expected_length ||
        throw(DimensionMismatch("context must have length $expected_length."))
    return Vector{Float64}(context)
end

function _checked_vector_or_default(value, default, expected_length, name::Symbol)
    vector = isnothing(value) ? default : value
    checked_vector = Vector{Float64}(vector)
    length(checked_vector) == expected_length ||
        throw(DimensionMismatch("$(name) must have length $expected_length."))
    return checked_vector
end

function _checked_matrix_or_default(value, default, expected_rows, expected_cols, name::Symbol)
    matrix = isnothing(value) ? default : value
    checked_matrix = Matrix{Float64}(matrix)
    size(checked_matrix) == (expected_rows, expected_cols) ||
        throw(DimensionMismatch("$(name) must have size ($(expected_rows), $(expected_cols))."))
    return checked_matrix
end
