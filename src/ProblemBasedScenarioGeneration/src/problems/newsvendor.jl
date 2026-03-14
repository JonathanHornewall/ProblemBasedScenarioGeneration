"""
Unreliable newsvendor two-stage stochastic LP.

First stage: order quantity z (1D).
Second stage: handle overage/underage with unreliable supply.
Noise pattern: WH (W matrix and h vector vary across scenarios).
"""

# -----------------------------------------------------------------------
# Problem struct
# -----------------------------------------------------------------------

"""
    UnreliableNewsvendorProblem

Two-stage unreliable newsvendor with uncertain demand D and reliability U.

Parameters:
- p: selling price
- c: purchase cost
- π: underage penalty (lost sales)
- η: overage penalty (holding cost)

First stage: z ∈ ℝ₊ (order quantity), cost c.
Second stage variables: [y₁ (overage), y₂ (underage), y₃ (slack)]
Second stage structure: W (2×3), h = [-D; 0], T = [0; -U]
Cost: q = [p+η, π, c-p]
"""
struct UnreliableNewsvendorProblem <: ProblemInstance
    p::Float64   # selling price
    c::Float64   # purchase cost
    pi::Float64  # underage penalty
    eta::Float64 # overage penalty

    A1::Matrix{Float64}
    b1::Vector{Float64}
    c1::Vector{Float64}
    W::Matrix{Float64}
    q::Vector{Float64}
end

"""
    UnreliableNewsvendorProblem(; p=5.0, c=1.0, pi=10.0, eta=0.5)

Default constructor.
"""
function UnreliableNewsvendorProblem(;
    p::Float64   = 5.0,
    c::Float64   = 1.0,
    pi::Float64  = 10.0,
    eta::Float64 = 0.5
)
    A1 = reshape([0.0], 1, 1)
    b1 = [0.0]
    c1 = [0.0]   # first-stage cost is zero in this formulation (c paid in second stage)

    # W matrix (fixed)
    W  = [1.0 -1.0 -1.0; 0.0 0.0 1.0]
    q  = [p + eta, pi, c - p]

    return UnreliableNewsvendorProblem(p, c, pi, eta, A1, b1, c1, W, q)
end

# -----------------------------------------------------------------------
# ProblemInstance interface
# -----------------------------------------------------------------------

function first_stage_data(prob::UnreliableNewsvendorProblem)
    return prob.A1, prob.b1, prob.c1
end

noise_pattern(::UnreliableNewsvendorProblem) = WH

"""
    scenario_realization(prob::UnreliableNewsvendorProblem, param::Vector) -> Scenario

`param` = [D, U] where D = demand ∈ [0,1] and U = reliability ∈ [0,1].
"""
function scenario_realization(prob::UnreliableNewsvendorProblem, param::AbstractVector)
    length(param) == 2 || error("param must have length 2 [D, U], got $(length(param))")
    D, U = param
    T = reshape([0.0, -U], 2, 1)
    h = [-D, 0.0]
    return Scenario(prob.W, T, h, prob.q)
end

"""
    generate_dataset(prob::UnreliableNewsvendorProblem, n)

Generate `n` (context, scenario) pairs. Context is a 1D scalar (D itself),
scenario has random (D, U) ∈ [0,1]².
"""
function generate_dataset(prob::UnreliableNewsvendorProblem, n::Int)
    dataset = Vector{Tuple{Vector{Float64}, Scenario{Float64}}}(undef, n)
    for i in 1:n
        D = rand()
        U = rand()
        param = [D, U]
        sc = scenario_realization(prob, param)
        # Context = [D] (the demand observation)
        dataset[i] = ([D], sc)
    end
    return dataset
end
