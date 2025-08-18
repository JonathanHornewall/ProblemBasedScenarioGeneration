

"""
    ResourceAllocationProblemData(service_rate_parameters, first_stage_costs, second_stage_costs, yield_parameters)
Constructs a data structure for the resource allocation problem, containing service rate parameters, first and second stage costs, and yield parameters.
- `service_rate_parameters`: Matrix of service rates (I×J)
- `first_stage_costs`: Vector of first stage costs (length I)
- `second_stage_costs`: Vector of second stage costs (length J)
- `yield_parameters`: Vector of yield parameters (length I)
"""
struct ResourceAllocationProblemData
    service_rate_parameters::Matrix{Float64}  # I×J matrix of service rates
    first_stage_costs::Vector{Float64}  # Cost vector for the first stage
    second_stage_costs::Vector{Float64}  # Cost vector for the second stage
    yield_parameters::Vector{Float64}  # Yield parameters for the second stage
    
    function ResourceAllocationProblemData(service_rate_parameters::Matrix{Float64},
                                first_stage_costs::Vector{Float64},
                                second_stage_costs::Vector{Float64},
                                yield_parameters::Vector{Float64})
        I, J = size(service_rate_parameters)
        length(first_stage_costs) == I || error("First stage costs must match the number of clients")
        length(second_stage_costs) == J || error("Second stage costs must match the number of resources")
        length(yield_parameters) == I || error("Yield parameters must match the number of clients")
        new(service_rate_parameters, first_stage_costs, second_stage_costs, yield_parameters)
    end
end

"""
    ResourceAllocationProblem(problem_data::ResourceAllocationProblemData)
Struct for an instance of the resource allocation problem. 
"""
struct ResourceAllocationProblem
    problem_data::ResourceAllocationProblemData
    s1_constraint_matrix::Matrix{Float64}  # First stage constraint matrix
    s1_constraint_vector::Vector{Float64}  # First stage constraint vector
    s1_cost_vector::Vector{Float64}  # First stage cost vector
    s2_constraint_matrix::Matrix{Float64}  # Second stage constraint matrix
    s2_coupling_matrix::Matrix{Float64}  # Second stage coupling matrix
    s2_cost_vector::Vector{Float64}  # Second stage cost vector
end

"""
    ResourceAllocationProblem(problem_data::ResourceAllocationProblemData)
Constructor for the ResourceAllocationProblem leveraging ResourceAllocationProblemData
"""
function ResourceAllocationProblem(problem_data::ResourceAllocationProblemData)
    μᵢⱼ = problem_data.service_rate_parameters
    cz = problem_data.first_stage_costs
    qw = problem_data.second_stage_costs
    ρᵢ = problem_data.yield_parameters


    I, J = size(μᵢⱼ)

    # First stage data
    A = zeros(1, length(cz))
    b = [0.0]
    c = cz

    #define W
    W = zeros(I+J, J + I*J + I + J)

    for i in 1:I
        for j in 1:J
            W[i,J + J*(i-1) +j] = 1
        end
        W[i, J + I*J + i] = 1
    end 

    # What's going on here? Seems like not enough indices are being filled.
    for j in 1:J
        W[I+j,j] = 1
        for i in 1:I
            W[I+j,J + J*(i-1) +j] = μᵢⱼ[i,j]
        end
        W[I+j, J + I*J + I + j] = -1
    end 

    #define T
    T = zeros(I+J,I)
    for i in 1:I
        T[i,i] = -ρᵢ[i]
    end

    # define q
    q = zeros(J + I*J + I + J)
    q[1:J] .= qw[:]

    return ResourceAllocationProblem(problem_data, A, b, c, W, T, q)
end

"""
    scenario_realization(instance::ResourceAllocationProblem, scenario_parameter)
Generates scenario data for an instance of the resource allocation problem based on a "scenario parameter".
"""
function scenario_realization(instance::ResourceAllocationProblem, scenario_parameter)
    W, T, q = instance.s2_constraint_matrix, instance.s2_coupling_matrix, instance.s2_cost_vector
    # scenario_parameter is a vector of length J (number of clients), representing demand
    I = size(T, 2)  # Number of resources (rows in T)
    
    # scenario_parameter represents demand for J clients
    # We need to create the right-hand side vector h
    # The first I elements are zeros (resource constraints)
    # The next J elements are the demand values
    h = vcat(zeros(I), scenario_parameter)
    
    return W, T, h, q
end

#=
struct ResourceAllocationProblemData
    # ---------------- dimensions & fixed costs ----------------
    I::Int
    J::Int
    L::Int                      # <–– NEW (covariate dimension)

    c::Vector{Float64}          # |I|
    q::Vector{Float64}          # |J|
    ρ::Vector{Float64}          # |I|
    μ::Matrix{Float64}          # I×J

    # ---------------- canonical‑form blocks -------------------
    W::Matrix{Float64}
    T::Matrix{Float64}
    qvec::Vector{Float64}

    # ---------------- demand‑generator parameters -------------  NEW
    A::Vector{Float64}          # |J|          intercepts  a_j
    B::Matrix{Float64}          # J×L          slopes      b_{j,l}
    σ::Union{Float64,Vector{Float64}}      # noise scale σ (scalar or |J|)
    p::Float64                  # non‑linearity degree
    Σ::Matrix{Float64}          # L×L  SPD     cov(x)
end

function ResourceAllocationProblemData(cz, qw, ρᵢ, μᵢⱼ; σ = 5.0, p = 1.0,
                        rng = MersenneTwister(2025))

    I, J = length(ρᵢ), length(qw)
    L = 3                                # ← set L to 3 like the paper

    # ----------- generate A,B once and freeze -----------------------------
    A = 50 .+ 5 .* rand(rng, Normal(), J)
    B₁ = 10 .+ rand(rng, Uniform(-4,4), J)
    B₂ =  5 .+ rand(rng, Uniform(-4,4), J)
    B₃ =  2 .+ rand(rng, Uniform(-4,4), J)
    B  = hcat(B₁,B₂,B₃)                  # J×L with L = 3

    Σ  = Symmetric(rand(rng, Beta(2,2), L, L))     # random SPD

    # ----------- build W,T,qvec exactly as before -------------------------
    prob_tmp = ResAllocProb(I,J,L, cz,qw,ρᵢ,μᵢⱼ, A,B,σ,p,Σ, zeros(1,1), zeros(1,1), zeros(1))  # placeholders
    W, T, qvec = build_canonical(prob_tmp)    # your previous routine
    return ResourceAllocationProblemData(I,J,L, cz,qw,ρᵢ,μᵢⱼ,
                        A,B,σ,p,Σ,
                        W,T,qvec)
end


struct ResourceAllocationProblem <: ProblemInstanceC2SCanLP
    problem_data::ResourceAllocationProblemData
    data_set::Vector{Tuple{Vector{Float64}, Tuple{Matrix{Float64}, Matrix{Float64}, Vector{Float64}, Vector{Float64}}}}
end

function return_scenario_type(problem_instance::ResourceAllocationProblem)
    # Only the second stage demand is stochastic
    return ScenarioType(:H)
end

function return_first_stage_parameters(problem_instance::ResourceAllocationProblem)
    problem_data = problem_instance.prblem_data
    c = problem_data.c
    return zeros(1, length(c)), zeros(1), c  # No first stage constraints, so A = 0, b = 0
end

function return_standard_scenario(problem_instance::ResourceAllocationProblem)
    data = problem_instance.data
    W, T, h, q = data.W, data.T, Float64[], data.qvec  # h is empty, since it depends on the scenario
    return W, T, h, q
end

function scenario_realization(problem_instance::ResourceAllocationProblem, scenario_parameters)
    Ws, Ts, hs, qs = return_standard_scenario(problem_instance)
    hs = [[zeros(problem_instance.data.I); param] for param in scenario_parameters]
    return Ws, Ts, hs, qs
end
=#