import LinearAlgebra
import Random

const SHIPMENT_PLANNING_PRODUCTION_COST = 5.0
const SHIPMENT_PLANNING_EMERGENCY_COST = 100.0

const SHIPMENT_PLANNING_DISTANCE_MATRIX = transpose([
    0.15     1.3124   1.85     1.3124;
    0.50026  0.93408  1.7874   1.6039;
    0.93408  0.50026  1.6039   1.7874;
    1.3124   0.15     1.3124   1.85;
    1.6039   0.50026  0.93408  1.7874;
    1.7874   0.93408  0.50026  1.6039;
    1.85     1.3124   0.15     1.3124;
    1.7874   1.6039   0.50026  0.93408;
    1.6039   1.7874   0.93408  0.50026;
    1.3124   1.85     1.3124   0.15;
    0.93408  1.7874   1.6039   0.50026;
    0.50026  1.6039   1.7874   0.93408
])

const SHIPMENT_PLANNING_SHIPMENT_COSTS =
    10.0 .* SHIPMENT_PLANNING_DISTANCE_MATRIX

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
    I=nothing,
    J=nothing,
    context_dim=3,
    p=1.0,
    sigma=5.0,
    parameter_seed=1,
    production_costs=nothing,
    emergency_costs=nothing,
    shipment_costs=nothing,
    demand_slack_costs=nothing,
    supply_slack_costs=nothing,
    demand_intercepts=nothing,
    demand_slopes=nothing,
)
    q_ship = isnothing(shipment_costs) ?
             copy(SHIPMENT_PLANNING_SHIPMENT_COSTS) :
             Matrix{Float64}(shipment_costs)
    warehouse_count, demand_count = size(q_ship)

    if !isnothing(I)
        checked_I = _checked_positive_integer(I, :I)
        checked_I == warehouse_count ||
            throw(DimensionMismatch("I must match the shipment-cost row count $(warehouse_count)."))
    end
    if !isnothing(J)
        checked_J = _checked_positive_integer(J, :J)
        checked_J == demand_count ||
            throw(DimensionMismatch("J must match the shipment-cost column count $(demand_count)."))
    end

    checked_context_dim = _checked_positive_integer(context_dim, :context_dim)
    checked_sigma = Float64(sigma)
    checked_sigma >= 0.0 || throw(ArgumentError("sigma must be nonnegative."))

    rng = Random.MersenneTwister(parameter_seed)
    c_z = _checked_vector_or_default(
        production_costs,
        fill(SHIPMENT_PLANNING_PRODUCTION_COST, warehouse_count),
        warehouse_count,
        :production_costs,
    )
    q_emergency = _checked_vector_or_default(
        emergency_costs,
        fill(SHIPMENT_PLANNING_EMERGENCY_COST, warehouse_count),
        warehouse_count,
        :emergency_costs,
    )
    q_demand_slack = _checked_vector_or_default(
        demand_slack_costs,
        zeros(Float64, demand_count),
        demand_count,
        :demand_slack_costs,
    )
    q_supply_slack = _checked_vector_or_default(
        supply_slack_costs,
        zeros(Float64, warehouse_count),
        warehouse_count,
        :supply_slack_costs,
    )
    intercepts = _checked_vector_or_default(
        demand_intercepts,
        50.0 .+ 5.0 .* randn(rng, demand_count),
        demand_count,
        :demand_intercepts,
    )
    slopes = _checked_matrix_or_default(
        demand_slopes,
        _shipment_planning_default_demand_slopes(rng, demand_count, checked_context_dim),
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
        q_demand_slack,
        q_supply_slack,
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
    q_demand_slack,
    q_supply_slack,
)
    recourse_count =
        warehouse_count + warehouse_count * demand_count + demand_count + warehouse_count
    equality_count = demand_count + warehouse_count

    emergency_index(i) = i
    shipment_index(i, j) = warehouse_count + (j - 1) * warehouse_count + i
    demand_slack_index(j) = warehouse_count + warehouse_count * demand_count + j
    supply_slack_index(i) = warehouse_count + warehouse_count * demand_count + demand_count + i

    W_eq = zeros(Float64, equality_count, recourse_count)
    T_eq = zeros(Float64, equality_count, warehouse_count)
    h_eq = zeros(Float64, equality_count)

    for j in 1:demand_count
        for i in 1:warehouse_count
            W_eq[j, shipment_index(i, j)] = 1.0
        end
        W_eq[j, demand_slack_index(j)] = -1.0
    end

    for i in 1:warehouse_count
        row = demand_count + i
        W_eq[row, emergency_index(i)] = -1.0
        for j in 1:demand_count
            W_eq[row, shipment_index(i, j)] = 1.0
        end
        W_eq[row, supply_slack_index(i)] = 1.0
        T_eq[row, i] = -1.0
    end

    q = vcat(q_emergency, vec(q_ship), q_demand_slack, q_supply_slack)

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

function _shipment_planning_default_demand_slopes(
    rng::Random.AbstractRNG,
    demand_count,
    context_dim,
)
    if context_dim == 3
        B1 = 10.0 .+ 8.0 .* rand(rng, demand_count) .- 4.0
        B2 = 5.0 .+ 8.0 .* rand(rng, demand_count) .- 4.0
        B3 = 2.0 .+ 8.0 .* rand(rng, demand_count) .- 4.0
        return hcat(B1, B2, B3)
    end
    return 5.0 .+ 10.0 .* rand(rng, demand_count, context_dim)
end
