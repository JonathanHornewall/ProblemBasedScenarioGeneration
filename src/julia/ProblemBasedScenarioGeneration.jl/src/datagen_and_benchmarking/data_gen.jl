#-------- Generate test data functionality

"""
    generate_scenario_collection(problem_instance::ProblemInstanceC2SCanLP,
                                 context,
                                 collection_size::Int,
                                 params)

Construct a matrix of scenarios by repeated calls to `generate_scenario`.

This function invokes `generate_scenario(problem_instance, context; params...)`
`collection_size` times and stacks the results column-wise. If the scenario
generator returns a `Vector`, `1×N`, or `N×1`, the values are flattened to a
vector to ensure each column has consistent shape.

Arguments
- `problem_instance::ProblemInstanceC2SCanLP`: Problem instance used by the generator.
- `context`: Context/features provided to `generate_scenario`.
- `collection_size::Int`: Number of scenario columns to generate.
- `params`: Additional keyword arguments forwarded to `generate_scenario`
  (e.g., a `NamedTuple` like `(; sigma=5.0, p=2, L=3)`).

Returns
- `Matrix{T}` with size `(scenario_dim, collection_size)`, where `scenario_dim`
  equals `length(generate_scenario(problem_instance, context; params...))`.
  Column `j` is the `j`‑th generated scenario.

Notes
- The element type `T` matches the scenario generator's output element type.
- Uses `similar` to preallocate and `.=` for in-place column assignment.

Example
    S = generate_scenario_collection(inst, x, 10, (; sigma=5.0, p=2, L=3))
    size(S)  # (scenario_dim, 10)
"""
function generate_scenario_collection(
                                    problem_instance::ProblemInstanceC2SCanLP, 
                                    context,
                                    collection_size::Int,
                                    params,
                                    )
    scenario_1 = vec(generate_scenario(problem_instance, context; params...))
    scenario_size = length(scenario_1)
    scenario_collection = similar(scenario_1, (scenario_size, collection_size))
    scenario_collection[:, 1] .= scenario_1
    for i in 2:collection_size
        scenario_collection[:, i] .= vec(generate_scenario(problem_instance, context; params...))
    end
    return scenario_collection
end
