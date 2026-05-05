import LinearAlgebra
import Random

struct ShipmentPlanningProblem <: ProgramInstance
    warehouse_count::Int
    demand_count::Int
    context_dim::Int
    p::Float64
    sigma::Float64
    demand_intercepts::Vector{Float64}
    demand_slopes::Matrix{Float64}
    stochastic_program::ContextualDFL.StochasticProgram
    base_scenario::NamedTuple
end

function ShipmentPlanningProblem(;
    I=5,
    J=12,
    context_dim=3,
    p=2.0,
    sigma=5.0,
    parameter_seed=1,
    production_costs=nothing,
    emergency_costs=nothing,
    shipment_costs=nothing,
    shortage_costs=nothing,
    unused_costs=nothing,
    demand_intercepts=nothing,
    demand_slopes=nothing,
)
    warehouse_count = _checked_positive_integer(I, :I)
    demand_count = _checked_positive_integer(J, :J)
    checked_context_dim = _checked_positive_integer(context_dim, :context_dim)
    checked_sigma = Float64(sigma)
    checked_sigma >= 0.0 || throw(ArgumentError("sigma must be nonnegative."))

    rng = Random.MersenneTwister(parameter_seed)
    c_z = _checked_vector_or_default(
        production_costs,
        fill(1.0, warehouse_count),
        warehouse_count,
        :production_costs,
    )
    q_emergency = _checked_vector_or_default(
        emergency_costs,
        fill(5.0, warehouse_count),
        warehouse_count,
        :emergency_costs,
    )
    q_ship = _checked_matrix_or_default(
        shipment_costs,
        _sample_shipment_costs(rng, warehouse_count, demand_count),
        warehouse_count,
        demand_count,
        :shipment_costs,
    )
    q_shortage = _checked_vector_or_default(
        shortage_costs,
        fill(20.0, demand_count),
        demand_count,
        :shortage_costs,
    )
    q_unused = _checked_vector_or_default(
        unused_costs,
        fill(0.01, warehouse_count),
        warehouse_count,
        :unused_costs,
    )
    intercepts = _checked_vector_or_default(
        demand_intercepts,
        50.0 .+ 50.0 .* rand(rng, demand_count),
        demand_count,
        :demand_intercepts,
    )
    slopes = _checked_matrix_or_default(
        demand_slopes,
        5.0 .+ 10.0 .* rand(rng, demand_count, checked_context_dim),
        demand_count,
        checked_context_dim,
        :demand_slopes,
    )

    program, scenario = _shipment_planning_program_and_scenario(
        warehouse_count,
        demand_count,
        c_z,
        q_emergency,
        q_ship,
        q_shortage,
        q_unused,
    )

    return ShipmentPlanningProblem(
        warehouse_count,
        demand_count,
        checked_context_dim,
        Float64(p),
        checked_sigma,
        intercepts,
        slopes,
        program,
        scenario,
    )
end

stochastic_program(problem::ShipmentPlanningProblem) = problem.stochastic_program

base_scenario(problem::ShipmentPlanningProblem) = problem.base_scenario

function _shipment_planning_program_and_scenario(
    warehouse_count,
    demand_count,
    c_z,
    q_emergency,
    q_ship,
    q_shortage,
    q_unused,
)
    recourse_count =
        warehouse_count + warehouse_count * demand_count + demand_count + warehouse_count
    equality_count = demand_count + warehouse_count

    emergency_index(i) = i
    shipment_index(i, j) = warehouse_count + (j - 1) * warehouse_count + i
    shortage_index(j) = warehouse_count + warehouse_count * demand_count + j
    unused_index(i) = warehouse_count + warehouse_count * demand_count + demand_count + i

    W_eq = zeros(Float64, equality_count, recourse_count)
    T_eq = zeros(Float64, equality_count, warehouse_count)
    h_eq = zeros(Float64, equality_count)

    for j in 1:demand_count
        for i in 1:warehouse_count
            W_eq[j, shipment_index(i, j)] = 1.0
        end
        W_eq[j, shortage_index(j)] = 1.0
    end

    for i in 1:warehouse_count
        row = demand_count + i
        W_eq[row, emergency_index(i)] = -1.0
        for j in 1:demand_count
            W_eq[row, shipment_index(i, j)] = 1.0
        end
        W_eq[row, unused_index(i)] = 1.0
        T_eq[row, i] = -1.0
    end

    q = vcat(q_emergency, vec(q_ship), q_shortage, q_unused)

    program = ContextualDFL.StochasticProgram(
        A_eq=zeros(Float64, 0, warehouse_count),
        A_ineq=-Matrix{Float64}(LinearAlgebra.I, warehouse_count, warehouse_count),
        b_eq=Float64[],
        b_ineq=zeros(Float64, warehouse_count),
        c=c_z,
    )

    scenario = (;
        W_eq=W_eq,
        W_ineq=-Matrix{Float64}(LinearAlgebra.I, recourse_count, recourse_count),
        T_eq=T_eq,
        T_ineq=zeros(Float64, recourse_count, warehouse_count),
        h_eq=h_eq,
        h_ineq=zeros(Float64, recourse_count),
        q=q,
    )

    return program, scenario
end

function _sample_shipment_costs(rng::Random.AbstractRNG, warehouse_count, demand_count)
    warehouse_locations = rand(rng, warehouse_count, 2)
    demand_locations = rand(rng, demand_count, 2)
    costs = zeros(Float64, warehouse_count, demand_count)
    for i in 1:warehouse_count
        for j in 1:demand_count
            dx = warehouse_locations[i, 1] - demand_locations[j, 1]
            dy = warehouse_locations[i, 2] - demand_locations[j, 2]
            costs[i, j] = 1.0 + 3.0 * sqrt(dx^2 + dy^2) + 0.1 * rand(rng)
        end
    end
    return costs
end
