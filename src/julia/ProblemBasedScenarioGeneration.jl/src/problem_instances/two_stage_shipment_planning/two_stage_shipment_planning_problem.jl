"""
    ShipmentPlanningProblemData(first_stage_costs, recourse_penalties; context_dimension=3)

Container for the structural parameters of the two-stage shipment planning problem from Homem-de-Mello et al. (2024).
- `first_stage_costs`: planned shipment cost for each market (length M).
- `recourse_penalties`: penalty for unmet demand or emergency shipping for each market (length M).
- `context_dimension`: dimensionality of the contextual feature vector used by the learning model (defaults to 3, mirroring the
  resource allocation prototype).

The data are intentionally lightweight placeholders because the published article does not report a canonical parameter set.
They can be replaced by real estimates once available.
"""
struct ShipmentPlanningProblemData
    first_stage_costs::Vector{Float64}
    recourse_penalties::Vector{Float64}
    context_dimension::Int

    function ShipmentPlanningProblemData(first_stage_costs::Vector{Float64},
                                         recourse_penalties::Vector{Float64};
                                         context_dimension::Int = 3)
        length(first_stage_costs) == length(recourse_penalties) ||
            error("First stage and recourse costs must have the same length")
        context_dimension > 0 || error("The contextual feature dimension must be positive")
        new(first_stage_costs, recourse_penalties, context_dimension)
    end
end

"""
    ShipmentPlanningProblem(problem_data::ShipmentPlanningProblemData)

Concrete implementation of the shipment planning problem.  The first-stage decision plans shipments for each market.  After demand
is realized, emergency re-shipments (or lost-sales penalties) are paid in the second stage.
"""
struct ShipmentPlanningProblem <: ProblemInstanceC2SCanLP
    problem_data::ShipmentPlanningProblemData
    s1_constraint_matrix::Matrix{Float64}
    s1_constraint_vector::Vector{Float64}
    s1_cost_vector::Vector{Float64}
    s2_constraint_matrix::Matrix{Float64}
    s2_coupling_matrix::Matrix{Float64}
    s2_cost_vector::Vector{Float64}
end

function ShipmentPlanningProblem(problem_data::ShipmentPlanningProblemData)
    markets = length(problem_data.first_stage_costs)

    # First-stage: no explicit equality constraints are enforced in this prototype, matching the resource allocation setup.
    A = zeros(1, markets)
    b = [0.0]
    c = copy(problem_data.first_stage_costs)

    # Second-stage: one slack variable per market capturing unmet demand / emergency shipping.
    W = Matrix{Float64}(I, markets, markets)
    T = Matrix{Float64}(I, markets, markets)
    q = copy(problem_data.recourse_penalties)

    return ShipmentPlanningProblem(problem_data, A, b, c, W, T, q)
end

"""
    scenario_realization(instance::ShipmentPlanningProblem, scenario_parameter)

Maps a demand vector to the canonical LP representation `(W, T, h, q)` used by the differentiation stack.
`scenario_parameter` is a vector of length equal to the number of markets.
"""
function scenario_realization(instance::ShipmentPlanningProblem, scenario_parameter)
    length(scenario_parameter) == length(instance.problem_data.recourse_penalties) ||
        error("Scenario parameter dimension does not match the number of markets")

    W = instance.s2_constraint_matrix
    T = instance.s2_coupling_matrix
    q = instance.s2_cost_vector
    h = copy(scenario_parameter)

    return W, T, h, q
end

function return_scenario_type(::ShipmentPlanningProblem)
    return ScenarioType(:H)
end

function return_first_stage_parameters(instance::ShipmentPlanningProblem)
    return instance.s1_constraint_matrix, instance.s1_constraint_vector, instance.s1_cost_vector
end

"""
    construct_neural_network(instance::ShipmentPlanningProblem; nr_of_scenarios = 1)

Uses the same feed-forward architecture adopted for the resource allocation benchmark while adapting the input and output
dimensions to the shipment planning context.
"""
function construct_neural_network(instance::ShipmentPlanningProblem; nr_of_scenarios::Int = 1)
    scenario_dim = length(instance.problem_data.recourse_penalties)
    input_dim = instance.problem_data.context_dimension
    output_dim = scenario_dim * nr_of_scenarios

    return Chain(
        Dense(input_dim, 128, relu),
        Dense(128, 128, relu),
        Dense(128, 128, relu),
        Dense(128, output_dim, relu),
        x -> reshape(x, scenario_dim, nr_of_scenarios)
    ) |> f64
end
