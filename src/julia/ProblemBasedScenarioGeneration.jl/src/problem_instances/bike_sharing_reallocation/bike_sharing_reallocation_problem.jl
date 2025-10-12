"""
    BikeSharingReallocationProblemData(first_stage_costs, emergency_relocation_costs, lost_demand_penalties;
                                       context_dimension=4)

Data container for the bike sharing reallocation problem proposed in Homem-de-Mello et al. (2024).
- `first_stage_costs`: pre-event repositioning cost per station (length S).
- `emergency_relocation_costs`: cost of reactive rebalancing during operations (length S).
- `lost_demand_penalties`: penalty assigned to unmet user demand at each station (length S).
- `context_dimension`: number of contextual covariates captured by the learning model (defaults to 4 to reflect typical
  [weather, weekday, event, peak] descriptors).

The values can be freely replaced by calibrated estimates when available.  The placeholders simply preserve the intended structure.
"""
struct BikeSharingReallocationProblemData
    first_stage_costs::Vector{Float64}
    emergency_relocation_costs::Vector{Float64}
    lost_demand_penalties::Vector{Float64}
    context_dimension::Int

    function BikeSharingReallocationProblemData(first_stage_costs::Vector{Float64},
                                                emergency_relocation_costs::Vector{Float64},
                                                lost_demand_penalties::Vector{Float64};
                                                context_dimension::Int = 4)
        length(first_stage_costs) == length(emergency_relocation_costs) == length(lost_demand_penalties) ||
            error("All cost vectors must share the same length")
        context_dimension > 0 || error("The contextual feature dimension must be positive")
        new(first_stage_costs, emergency_relocation_costs, lost_demand_penalties, context_dimension)
    end
end

"""
    BikeSharingReallocationProblem(problem_data::BikeSharingReallocationProblemData)

Concrete problem instance representing the two-stage bike sharing reallocation model.
The first-stage decision places bikes before demand materializes.  Second-stage decisions capture on-the-fly rebalancing and
lost demand penalties.
"""
struct BikeSharingReallocationProblem <: ProblemInstanceC2SCanLP
    problem_data::BikeSharingReallocationProblemData
    s1_constraint_matrix::Matrix{Float64}
    s1_constraint_vector::Vector{Float64}
    s1_cost_vector::Vector{Float64}
    s2_constraint_matrix::Matrix{Float64}
    s2_coupling_matrix::Matrix{Float64}
    s2_cost_vector::Vector{Float64}
end

function BikeSharingReallocationProblem(problem_data::BikeSharingReallocationProblemData)
    stations = length(problem_data.first_stage_costs)

    A = zeros(1, stations)
    b = [0.0]
    c = copy(problem_data.first_stage_costs)

    # Second-stage variables are arranged as [emergency_relocation; lost_demand] – both non-negative.
    W = hcat(Matrix{Float64}(I, stations, stations), Matrix{Float64}(I, stations, stations))
    T = Matrix{Float64}(I, stations, stations)
    q = vcat(problem_data.emergency_relocation_costs, problem_data.lost_demand_penalties)

    return BikeSharingReallocationProblem(problem_data, A, b, c, W, T, q)
end

"""
    scenario_realization(instance::BikeSharingReallocationProblem, scenario_parameter)

Transforms a demand vector into the canonical LP representation `(W, T, h, q)` used downstream.
`scenario_parameter` must have length equal to the number of stations.
"""
function scenario_realization(instance::BikeSharingReallocationProblem, scenario_parameter)
    length(scenario_parameter) == length(instance.problem_data.first_stage_costs) ||
        error("Scenario parameter dimension does not match the number of stations")

    W = instance.s2_constraint_matrix
    T = instance.s2_coupling_matrix
    q = instance.s2_cost_vector
    h = copy(scenario_parameter)

    return W, T, h, q
end

function return_scenario_type(::BikeSharingReallocationProblem)
    return ScenarioType(:H)
end

function return_first_stage_parameters(instance::BikeSharingReallocationProblem)
    return instance.s1_constraint_matrix, instance.s1_constraint_vector, instance.s1_cost_vector
end

"""
    construct_neural_network(instance::BikeSharingReallocationProblem; nr_of_scenarios = 1)

Reuses the resource allocation network topology while adapting the input/output dimensions to the bike-sharing setting.
"""
function construct_neural_network(instance::BikeSharingReallocationProblem; nr_of_scenarios::Int = 1)
    stations = length(instance.problem_data.first_stage_costs)
    input_dim = instance.problem_data.context_dimension
    output_dim = stations * nr_of_scenarios

    return Chain(
        Dense(input_dim, 128, relu),
        Dense(128, 128, relu),
        Dense(128, 128, relu),
        Dense(128, output_dim, relu),
        x -> reshape(x, stations, nr_of_scenarios)
    ) |> f64
end
