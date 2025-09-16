

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
struct ResourceAllocationProblem <: ProblemInstanceC2SCanLP
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

"""
    return_first_stage_parameters(instance::ResourceAllocationProblem)
Getter method for retrieving the first stage parameters of the problem instance
"""
function return_first_stage_parameters(instance::ResourceAllocationProblem)
    return instance.s1_constraint_matrix, instance.s1_constraint_vector, instance.s1_cost_vector
end

"""
    construct_neural_network(instance::ResourceAllocationProblem)
Specifies a neural network architecture for the resource allocation problem.
"""
function construct_neural_network(instance::ResourceAllocationProblem; nr_of_scenarios = 1)
    scenario_dim = size(instance.problem_data.service_rate_parameters, 2)
    output_dim = scenario_dim * nr_of_scenarios
    return Chain(
        Dense(3, 128, relu),
        Dense(128, 128, relu),
        Dense(128, 128, relu),
        Dense(128, output_dim, relu),     # linear head
        x -> reshape(x, scenario_dim, nr_of_scenarios)  # reshape output to (#scenarios vars) × scenarios matrix
    ) |> f64
end
