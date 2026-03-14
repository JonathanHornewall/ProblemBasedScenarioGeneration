"""
Shipment planning two-stage stochastic LP.

First stage: decide production quantities at I=12 warehouses.
Second stage: emergency production + shipment to meet demand at J=4 locations.
Noise pattern: H_ONLY (only the demand RHS h varies).
"""

# -----------------------------------------------------------------------
# Default parameters
# -----------------------------------------------------------------------

const _SP_production_costs = fill(5.0, 12)
const _SP_emergency_costs  = fill(100.0, 12)

# Distance matrix (12 warehouses × 4 demand locations)
const _SP_distances = [
    1.0 2.0 3.0 4.0;
    2.0 1.0 2.0 3.0;
    3.0 2.0 1.0 2.0;
    4.0 3.0 2.0 1.0;
    2.0 1.0 2.0 3.0;
    3.0 2.0 1.0 2.0;
    4.0 3.0 2.0 1.0;
    1.0 2.0 3.0 4.0;
    2.0 3.0 4.0 1.0;
    3.0 4.0 1.0 2.0;
    4.0 1.0 2.0 3.0;
    1.0 2.0 3.0 2.0
]

const _SP_shipment_costs = 10.0 .* _SP_distances  # 12×4

# -----------------------------------------------------------------------
# Problem struct
# -----------------------------------------------------------------------

"""
    ShipmentPlanningProblem

Two-stage shipment planning (I=12 warehouses, J=4 locations, context dim=3).
"""
struct ShipmentPlanningProblem <: ProblemInstance
    production_costs::Vector{Float64}
    emergency_costs::Vector{Float64}
    shipment_costs::Matrix{Float64}
    context_dim::Int

    A1::Matrix{Float64}
    b1::Vector{Float64}
    c1::Vector{Float64}
    W::Matrix{Float64}
    T::Matrix{Float64}
    q::Vector{Float64}
end

"""
    ShipmentPlanningProblem()

Default constructor with hardcoded parameters.
"""
function ShipmentPlanningProblem()
    prod_c  = _SP_production_costs
    emerg_c = _SP_emergency_costs
    ship_c  = _SP_shipment_costs

    n_wh  = length(prod_c)          # 12 warehouses
    n_loc = size(ship_c, 2)          # 4 demand locations
    n_yw  = n_wh                     # emergency production vars
    n_ship = n_wh * n_loc            # shipment flow vars
    n_dslack = n_loc                 # demand slack
    n_sslack = n_wh                  # supply slack
    n2    = n_yw + n_ship + n_dslack + n_sslack

    m_dem = n_loc
    m_sup = n_wh
    m2    = m_dem + m_sup

    # First stage
    A1 = zeros(1, n_wh)
    b1 = [0.0]
    c1 = copy(prod_c)

    W = zeros(m2, n2)
    T = zeros(m2, n_wh)
    q = zeros(n2)

    # Costs
    q[1:n_yw] .= emerg_c
    q[(n_yw+1):(n_yw+n_ship)] .= vec(ship_c)

    shipment_idx(i, j) = n_yw + (j-1)*n_wh + i

    # Demand constraints: Σ_i y^s_{ij} - d_slack_j = ξ_j
    for j in 1:n_loc
        for i in 1:n_wh
            W[j, shipment_idx(i, j)] = 1.0
        end
        W[j, n_yw + n_ship + j] = -1.0
    end

    # Supply constraints: -y^w_i + Σ_j y^s_{ij} + s_slack_i = z_i
    for i in 1:n_wh
        row = m_dem + i
        W[row, i] = -1.0  # emergency production
        for j in 1:n_loc
            W[row, shipment_idx(i, j)] = 1.0
        end
        W[row, n_yw + n_ship + n_dslack + i] = 1.0  # supply slack
        T[row, i] = -1.0  # coupling to first stage
    end

    return ShipmentPlanningProblem(prod_c, emerg_c, ship_c, 3, A1, b1, c1, W, T, q)
end

# -----------------------------------------------------------------------
# ProblemInstance interface
# -----------------------------------------------------------------------

function first_stage_data(prob::ShipmentPlanningProblem)
    return prob.A1, prob.b1, prob.c1
end

noise_pattern(::ShipmentPlanningProblem) = H_ONLY

"""
    scenario_realization(prob::ShipmentPlanningProblem, param::Vector) -> Scenario

`param` is a demand vector of length J (number of locations).
"""
function scenario_realization(prob::ShipmentPlanningProblem, param::AbstractVector)
    n_loc = size(prob.shipment_costs, 2)
    n_wh  = length(prob.production_costs)
    length(param) == n_loc ||
        error("param must have length n_loc=$n_loc, got $(length(param))")
    h = zeros(n_loc + n_wh)
    h[1:n_loc] .= param
    return Scenario(prob.W, prob.T, h, prob.q)
end

"""
    generate_dataset(prob::ShipmentPlanningProblem, n; sigma=2.0, p_exp=1, L=3)
"""
function generate_dataset(prob::ShipmentPlanningProblem, n::Int;
    sigma::Float64=2.0, p_exp::Int=1, L::Int=3)

    n_loc = size(prob.shipment_costs, 2)

    A_coef = 10.0 .+ 2.0 .* randn(n_loc)
    B_coef = hcat(
        2.0 .+ rand(Uniform(-1.0, 1.0), n_loc),
        1.0 .+ rand(Uniform(-1.0, 1.0), n_loc),
        0.5 .+ rand(Uniform(-0.5, 0.5), n_loc)
    )  # n_loc × L

    X = abs.(randn(prob.context_dim, n))  # context_dim × n

    dataset = Vector{Tuple{Vector{Float64}, Scenario{Float64}}}(undef, n)
    for i in 1:n
        x = X[:, i]
        demand = [A_coef[j] + sum(B_coef[j, l] * x[l]^p_exp for l in 1:L) + randn() * sigma
                  for j in 1:n_loc]
        demand = max.(demand, 0.0)
        sc = scenario_realization(prob, demand)
        dataset[i] = (x, sc)
    end
    return dataset
end
