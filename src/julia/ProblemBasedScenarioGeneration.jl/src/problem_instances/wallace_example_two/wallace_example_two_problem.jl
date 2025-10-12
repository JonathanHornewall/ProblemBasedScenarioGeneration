"""
    WallaceExampleTwoProblemData(; building_unit_cost=2.0, early_equipment_unit_cost=2.0,
                                  late_equipment_unit_cost=2.2, minimum_capacity=1.0,
                                  context_dimension=3)

Data container for Example 2 in Wallace (2000), *Decision Making Under Uncertainty: Is Sensitivity
Analysis of Any Use?*. The example models a two-stage capacity expansion decision where the selling
price is revealed after the first-stage commitments.

- `building_unit_cost`: cost per unit of installed building capacity (`c` in the paper).
- `early_equipment_unit_cost`: cost per unit of equipment installed before the price is revealed (`z`).
- `late_equipment_unit_cost`: cost per unit of equipment installed after the price is revealed (`y`).
- `minimum_capacity`: mandatory baseline capacity (Example 2 uses `1.0`).
- `context_dimension`: dimensionality of the contextual feature vector ingested by the learning model.
"""
struct WallaceExampleTwoProblemData
    building_unit_cost::Float64
    early_equipment_unit_cost::Float64
    late_equipment_unit_cost::Float64
    minimum_capacity::Float64
    context_dimension::Int

    function WallaceExampleTwoProblemData(; building_unit_cost::Real = 2.0,
                                          early_equipment_unit_cost::Real = 2.0,
                                          late_equipment_unit_cost::Real = 2.2,
                                          minimum_capacity::Real = 1.0,
                                          context_dimension::Int = 3)
        building_unit_cost > 0 || error("Building cost must be positive")
        early_equipment_unit_cost > 0 || error("Early equipment cost must be positive")
        late_equipment_unit_cost > 0 || error("Late equipment cost must be positive")
        minimum_capacity >= 0 || error("Minimum capacity must be non-negative")
        context_dimension > 0 || error("Context dimension must be positive")
        new(Float64(building_unit_cost),
            Float64(early_equipment_unit_cost),
            Float64(late_equipment_unit_cost),
            Float64(minimum_capacity),
            context_dimension)
    end
end

"""
    WallaceExampleTwoProblem(problem_data::WallaceExampleTwoProblemData)

Canonical-form representation of Wallace's Example 2.  First-stage decisions determine building
capacity (`c`) and pre-installed equipment (`z`).  After the selling price is observed a second-stage
recourse problem installs additional equipment (`y`) and decides the production level (`x`).
"""
struct WallaceExampleTwoProblem <: ProblemInstanceC2SCanLP
    problem_data::WallaceExampleTwoProblemData
    s1_constraint_matrix::Matrix{Float64}
    s1_constraint_vector::Vector{Float64}
    s1_cost_vector::Vector{Float64}
    s2_constraint_matrix::Matrix{Float64}
    s2_coupling_matrix::Matrix{Float64}
    s2_rhs_vector::Vector{Float64}
    s2_cost_template::Vector{Float64}
end

function WallaceExampleTwoProblem(problem_data::WallaceExampleTwoProblemData)
    # First-stage variables: [c, z, s], where s is the slack enforcing c ≥ minimum_capacity.
    A = reshape([1.0, 0.0, -1.0], 1, 3)
    b = [problem_data.minimum_capacity]
    c = [problem_data.building_unit_cost,
         problem_data.early_equipment_unit_cost,
         0.0]  # slack carries no cost

    # Second-stage variables: [x, y, s_cap, s_bld].
    # Constraints (canonical equality form):
    #   (i) x + s_cap - y = z
    #   (ii) y + s_bld   = c - z
    W = [1.0 -1.0 1.0 0.0;
         0.0  1.0 0.0 1.0]
    T = [0.0 -1.0 0.0;
        -1.0  1.0 0.0]
    h = zeros(2)

    # Scenario-independent part of the second-stage cost vector.
    q_template = [0.0,
                  problem_data.late_equipment_unit_cost,
                  0.0,
                  0.0]

    return WallaceExampleTwoProblem(problem_data, A, b, c, W, T, h, q_template)
end

"""
    scenario_realization(instance::WallaceExampleTwoProblem, scenario_parameter)

Maps a price realization `p` to the canonical tuple `(W, T, h, q)`.  The selling price directly affects
the revenue term in the second-stage objective, appearing as the coefficient of the production level.
"""
function scenario_realization(instance::WallaceExampleTwoProblem, scenario_parameter)
    p = if scenario_parameter isa Number
        Float64(scenario_parameter)
    else
        length(scenario_parameter) == 1 ||
            error("Scenario parameter dimension must be 1, got $(length(scenario_parameter))")
        Float64(scenario_parameter[1])
    end

    W = instance.s2_constraint_matrix
    T = instance.s2_coupling_matrix
    h = copy(instance.s2_rhs_vector)

    q = copy(instance.s2_cost_template)
    q[1] = -p  # revenue term: maximizing px corresponds to minimizing -px

    return W, T, h, q
end

function return_scenario_type(::WallaceExampleTwoProblem)
    return ScenarioType(:Q)
end

function return_first_stage_parameters(instance::WallaceExampleTwoProblem)
    return instance.s1_constraint_matrix, instance.s1_constraint_vector, instance.s1_cost_vector
end

"""
    construct_neural_network(instance::WallaceExampleTwoProblem; nr_of_scenarios = 1)

Reuses the resource allocation architecture (three hidden layers of width 128 and ReLU activations),
adjusting the input/output dimensions to the Example 2 context.
"""
function construct_neural_network(instance::WallaceExampleTwoProblem; nr_of_scenarios::Int = 1)
    scenario_dim = nr_of_scenarios
    input_dim = instance.problem_data.context_dimension

    return Chain(
        Dense(input_dim, 128, relu),
        Dense(128, 128, relu),
        Dense(128, 128, relu),
        Dense(128, scenario_dim, relu),
        x -> reshape(x, 1, nr_of_scenarios)
    ) |> f64
end
