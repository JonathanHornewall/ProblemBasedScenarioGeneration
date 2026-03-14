"""
    ShipmentPlanningProblemData(production_costs, emergency_costs, shipment_costs;
                                context_dimension=3)

Container for the structural parameters of the two-stage shipment planning problem described in
Homem-de-Mello et al. (2024) and Bertsimas & Kallus (2020).
- `production_costs`: first-stage unit production/storage costs (length |I|, denoted `c` in the paper).
- `emergency_costs`: second-stage emergency production costs (length |I|, parameter `r`).
- `shipment_costs`: matrix of per-unit shipping costs `sᵢⱼ` from warehouse `i` to location `j` (|I|×|J|).
- `context_dimension`: number of covariates fed to the learning model.
"""
struct ShipmentPlanningProblemData
    production_costs::Vector{Float64}
    emergency_costs::Vector{Float64}
    shipment_costs::Matrix{Float64}
    context_dimension::Int

    function ShipmentPlanningProblemData(production_costs::Vector{Float64},
                                         emergency_costs::Vector{Float64},
                                         shipment_costs::Matrix{Float64};
                                         context_dimension::Int = 3)
        n_warehouses, _ = size(shipment_costs)
        length(production_costs) == n_warehouses ||
            error("Production costs must have one entry per warehouse")
        length(emergency_costs) == n_warehouses ||
            error("Emergency costs must have one entry per warehouse")
        context_dimension > 0 || error("The contextual feature dimension must be positive")
        new(production_costs, emergency_costs, shipment_costs, context_dimension)
    end
end

"""
    ShipmentPlanningProblem(problem_data::ShipmentPlanningProblemData)

Concrete instantiation of the shipment planning problem. The first stage decides warehouse production
quantities. After demand is realized the second stage can trigger emergency production and route shipments
to satisfy each location's demand.
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
    n_warehouses, n_locations = size(problem_data.shipment_costs)

    # First-stage: non-negativity only (encoded in canonical LP), so we keep an empty constraint row.
    A = zeros(1, n_warehouses)
    b = [0.0]
    c = copy(problem_data.production_costs)

    n_yw = n_warehouses                       # emergency production yᵂ
    n_ship = n_warehouses * n_locations       # shipment flows yˢ
    n_dslack = n_locations                    # slack for demand >= constraints
    n_sslack = n_warehouses                   # slack for supply <= constraints
    n_second_stage = n_yw + n_ship + n_dslack + n_sslack

    m_demand = n_locations
    m_supply = n_warehouses
    m_second_stage = m_demand + m_supply

    W = zeros(Float64, m_second_stage, n_second_stage)
    T = zeros(Float64, m_second_stage, n_warehouses)
    q = zeros(Float64, n_second_stage)

    # Cost vector: emergency production followed by shipments.
    q[1:n_yw] .= problem_data.emergency_costs
    q[(n_yw + 1):(n_yw + n_ship)] .= vec(problem_data.shipment_costs)

    shipment_index(i, j) = n_yw + (j - 1) * n_warehouses + i

    # Demand rows: ∑ᵢ yˢᵢⱼ - d_slackⱼ = ξⱼ
    dslack_offset = n_yw + n_ship
    for j in 1:n_locations
        for i in 1:n_warehouses
            W[j, shipment_index(i, j)] = 1.0
        end
        W[j, dslack_offset + j] = -1.0
    end

    # Supply rows: -yᵂᵢ + ∑ⱼ yˢᵢⱼ + s_slackᵢ - zᵢ = 0  ⇒  ∑ⱼ yˢᵢⱼ + s_slackᵢ = zᵢ + yᵂᵢ
    sslack_offset = dslack_offset + n_dslack
    for i in 1:n_warehouses
        row = m_demand + i
        W[row, i] = -1.0  # emergency production term
        for j in 1:n_locations
            W[row, shipment_index(i, j)] = 1.0
        end
        W[row, sslack_offset + i] = 1.0
        T[row, i] = -1.0
    end

    return ShipmentPlanningProblem(problem_data, A, b, c, W, T, q)
end

"""
    scenario_realization(instance::ShipmentPlanningProblem, scenario_parameter)

Maps a demand vector to `(W, T, h, q)` in canonical LP form. `scenario_parameter` must have length equal
to the number of customer locations.
"""
function scenario_realization(instance::ShipmentPlanningProblem, scenario_parameter)
    n_locations = size(instance.problem_data.shipment_costs, 2)
    n_warehouses = length(instance.problem_data.production_costs)

    length(scenario_parameter) == n_locations ||
        error("Scenario parameter dimension does not match the number of locations")

    W = instance.s2_constraint_matrix
    T = instance.s2_coupling_matrix
    q = instance.s2_cost_vector
    h = zeros(Float64, n_locations + n_warehouses)
    h[1:n_locations] .= scenario_parameter

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

Creates a simple feed-forward network that maps contextual features to a scenario collection with
`nr_of_scenarios` columns.
"""
function construct_neural_network(instance::ShipmentPlanningProblem; nr_of_scenarios::Int = 1)
    scenario_dim = size(instance.problem_data.shipment_costs, 2)
    input_dim = instance.problem_data.context_dimension
    output_dim = scenario_dim * nr_of_scenarios

    reshape_out = x -> begin
        if ndims(x) == 1
            reshape(x, scenario_dim, nr_of_scenarios)
        else
            batch = size(x, 2)
            reshaped = reshape(x, scenario_dim, nr_of_scenarios, batch)
            reshape(reshaped, scenario_dim, nr_of_scenarios * batch)
        end
    end

    return Chain(
        Dense(input_dim, 128, gelu),
        Dense(128, 128, gelu),
        Dense(128, 128, gelu),
        Dense(128, output_dim, gelu),
        reshape_out
    ) |> f64
end
