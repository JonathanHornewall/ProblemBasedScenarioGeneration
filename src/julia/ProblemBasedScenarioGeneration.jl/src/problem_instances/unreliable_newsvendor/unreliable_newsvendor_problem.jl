
struct UnreliableNewsvendorProblemData
    p::Float64  # I×J matrix of service rates
    c::Float64  # Cost vector for the first stage
    π::Float64  # Cost vector for the second stage
    η::Float64  # Yield parameters for the second stage
    
    function UnreliableNewsvendorProblemData(p::Float64,
                                c::Float64,
                                π::Float64,
                                η::Float64)
        new(p, c, π, η)
    end
end

"""
    UnreliableNewsvendorProblem(problem_data::UnreliableNewsvendorProblemData)
Struct for an instance of the Unreliable Newsvendor problem. 
"""
struct UnreliableNewsvendorProblem <: ProblemInstanceC2SCanLP
    problem_data::UnreliableNewsvendorProblemData
    s1_constraint_matrix::Matrix{Float64}  # First stage constraint matrix
    s1_constraint_vector::Vector{Float64}  # First stage constraint vector
    s1_cost_vector::Vector{Float64}  # First stage cost vector
    s2_constraint_matrix::Matrix{Float64}  # Second stage constraint matrix
    s2_cost_vector::Vector{Float64}  # Second stage cost vector
end

"""
    UnreliableNewsvendorProblem(problem_data::UnreliableNewsvendorProblemData)
Constructor for the UnreliableNewsvendorProblem leveraging UnreliableNewsvendorProblemData
"""
function UnreliableNewsvendorProblem(problem_data::UnreliableNewsvendorProblemData)

    p = problem_data.p
    c = problem_data.c
    π = problem_data.π
    η = problem_data.η
    
    A = reshape([0.0], 1, 1)
    b = [0.0]
    W = [1.0 -1.0 -1.0; 0.0 0.0 1.0]
    q = [p + η, π, c - p]

    return UnreliableNewsvendorProblem(problem_data, A, b, [0.0], W, q)
end

"""
    scenario_realization(instance::UnreliableNewsvendorProblem, scenario_parameter)
Generates scenario data for an instance of the Unreliable Newsvendor problem based on a "scenario parameter".
"""
function scenario_realization(instance::UnreliableNewsvendorProblem, scenario_parameter)

    p = instance.problem_data.p
    c = instance.problem_data.c
    π = instance.problem_data.π
    η = instance.problem_data.η

    W, q = instance.s2_constraint_matrix, instance.s2_cost_vector
    T = [0.0, -scenario_parameter[2]]
    h = [-scenario_parameter[1], 0.0]
    
    return W, T, h, q
end

"""
    return_first_stage_parameters(instance::UnreliableNewsvendorProblem)
Getter method for retrieving the first stage parameters of the problem instance
"""
function return_first_stage_parameters(instance::UnreliableNewsvendorProblem)
    return instance.s1_constraint_matrix, instance.s1_constraint_vector, instance.s1_cost_vector
end

"""
    construct_neural_network(instance::UnreliableNewsvendorProblem)
Specifies a neural network architecture for the Unreliable Newsvendor problem.
    """


softplus(x) = log(1 + exp(x))

function construct_neural_network(instance::UnreliableNewsvendorProblem)
    return Chain(
        Dense(1, 2, sigmoid)     # to output values between 0 and 1
    ) |> f64
end
