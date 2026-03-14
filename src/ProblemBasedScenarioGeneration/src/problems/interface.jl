"""
Abstract interface for two-stage stochastic LP problem instances.
"""

"""
    ProblemInstance

Abstract base type for two-stage stochastic LP problem instances.
Subtypes must implement:
- `first_stage_data(prob) -> (A, b, c)`
- `scenario_realization(prob, param) -> Scenario`
- `generate_dataset(prob, n) -> Vector{Tuple{Vector{Float64}, Scenario}}`
- `noise_pattern(prob) -> NoisePattern`
"""
abstract type ProblemInstance end

"""
    NoisePattern

Encodes which second-stage parameters vary across scenarios.
"""
@enum NoisePattern begin
    H_ONLY   # only h (RHS) varies
    Q_ONLY   # only q (cost) varies
    W_ONLY   # only W (constraint matrix) varies
    WH       # W and h vary
    WQ       # W and q vary
    WHQ      # W, h, and q all vary
end

"""
    first_stage_data(prob::ProblemInstance) -> (A::Matrix, b::Vector, c::Vector)

Return the first-stage constraint matrix, RHS, and cost vector.
"""
function first_stage_data(prob::ProblemInstance)
    error("first_stage_data not implemented for $(typeof(prob))")
end

"""
    scenario_realization(prob::ProblemInstance, param::Vector) -> Scenario

Map a scenario parameter vector to a `Scenario` struct.
"""
function scenario_realization(prob::ProblemInstance, param::AbstractVector)
    error("scenario_realization not implemented for $(typeof(prob))")
end

"""
    generate_dataset(prob::ProblemInstance, n::Int) -> Vector{Tuple{Vector{Float64}, Scenario}}

Generate `n` (context, scenario) pairs for training/evaluation.
"""
function generate_dataset(prob::ProblemInstance, n::Int)
    error("generate_dataset not implemented for $(typeof(prob))")
end

"""
    noise_pattern(prob::ProblemInstance) -> NoisePattern

Return the noise pattern for this problem instance.
"""
function noise_pattern(prob::ProblemInstance)
    error("noise_pattern not implemented for $(typeof(prob))")
end
