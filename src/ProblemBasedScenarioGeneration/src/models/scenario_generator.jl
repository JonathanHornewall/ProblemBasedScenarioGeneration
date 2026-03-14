"""
Neural network factory for scenario generators.

All scenario generators map context vectors x ∈ ℝ^d to scenario
parameter vectors ξ ∈ ℝ^p (positive outputs via softplus).
"""

using Flux

"""
    build_generator(input_dim, output_dim;
                    hidden_dim=128, n_layers=3, activation=relu) -> Flux.Chain

Build a feed-forward scenario generator network:
    Dense(input→hidden, activation) × n_layers → Dense(hidden→output, softplus)

The softplus output ensures all predictions are strictly positive,
suitable for demand/cost scenarios.
"""
function build_generator(
    input_dim::Int,
    output_dim::Int;
    hidden_dim::Int = 128,
    n_layers::Int   = 3,
    activation      = relu
)
    layers = Any[]

    # Input layer
    push!(layers, Dense(input_dim => hidden_dim, activation))

    # Hidden layers
    for _ in 2:n_layers
        push!(layers, Dense(hidden_dim => hidden_dim, activation))
    end

    # Output layer with softplus activation (strictly positive outputs)
    push!(layers, Dense(hidden_dim => output_dim, softplus))

    return Flux.Chain(layers...)
end

"""
    build_generator(prob::ProblemInstance; nr_of_scenarios=1, kw...) -> Flux.Chain

Build a generator appropriate for the given problem instance.
Infers input/output dimensions from the problem.
"""
function build_generator(prob::ProblemInstance; nr_of_scenarios::Int=1, kw...)
    _, _, c1 = first_stage_data(prob)
    # Infer context dim from first dataset sample or use a default
    sc_dim = _scenario_param_dim(prob)
    input_dim = _context_dim(prob)
    output_dim = sc_dim * nr_of_scenarios
    return build_generator(input_dim, output_dim; kw...)
end

# Helper to infer context dimension
_context_dim(::ResourceAllocationProblem) = 3
_context_dim(::ShipmentPlanningProblem)   = 3
_context_dim(::UnreliableNewsvendorProblem) = 1

# Helper to infer scenario parameter dimension
_scenario_param_dim(prob::ResourceAllocationProblem) = length(prob.qw)  # J
_scenario_param_dim(prob::ShipmentPlanningProblem)   = size(prob.shipment_costs, 2)  # n_loc
_scenario_param_dim(::UnreliableNewsvendorProblem)   = 2  # [D, U]
