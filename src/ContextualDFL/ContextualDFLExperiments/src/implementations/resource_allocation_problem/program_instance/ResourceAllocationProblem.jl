import LinearAlgebra

struct ResourceAllocationProblemData
    service_rate_parameters::Matrix{Float64}
    first_stage_costs::Vector{Float64}
    second_stage_costs::Vector{Float64}
    yield_parameters::Vector{Float64}

    function ResourceAllocationProblemData(
        service_rate_parameters::AbstractMatrix,
        first_stage_costs::AbstractVector,
        second_stage_costs::AbstractVector,
        yield_parameters::AbstractVector,
    )
        service_rates = Matrix{Float64}(service_rate_parameters)
        first_costs = Vector{Float64}(first_stage_costs)
        second_costs = Vector{Float64}(second_stage_costs)
        yields = Vector{Float64}(yield_parameters)

        resource_count, demand_count = size(service_rates)
        length(first_costs) == resource_count ||
            throw(DimensionMismatch("first-stage costs must match resource count."))
        length(second_costs) == demand_count ||
            throw(DimensionMismatch("second-stage costs must match demand count."))
        length(yields) == resource_count ||
            throw(DimensionMismatch("yield parameters must match resource count."))

        return new(service_rates, first_costs, second_costs, yields)
    end
end

function default_resource_allocation_problem_data()
    return ResourceAllocationProblemData(
        RESOURCE_ALLOCATION_SERVICE_RATE_PARAMETERS,
        vec(Float64.(RESOURCE_ALLOCATION_FIRST_STAGE_COSTS)),
        vec(Float64.(RESOURCE_ALLOCATION_SECOND_STAGE_COSTS)),
        vec(Float64.(RESOURCE_ALLOCATION_YIELD_PARAMETERS)),
    )
end

struct ResourceAllocationProblem <: ProgramInstance
    problem_data::ResourceAllocationProblemData
    stochastic_program::ContextualDFL.StochasticProgram
    base_scenario::NamedTuple
end

function ResourceAllocationProblem(
    problem_data::ResourceAllocationProblemData=default_resource_allocation_problem_data(),
)
    service_rates = problem_data.service_rate_parameters
    first_costs = problem_data.first_stage_costs
    second_costs = problem_data.second_stage_costs
    yields = problem_data.yield_parameters

    resource_count, demand_count = size(service_rates)
    recourse_variables =
        demand_count + resource_count * demand_count + resource_count + demand_count
    recourse_rows = resource_count + demand_count

    W_eq = zeros(Float64, recourse_rows, recourse_variables)
    for resource_index in 1:resource_count
        for demand_index in 1:demand_count
            allocation_index = demand_count + demand_count * (resource_index - 1) + demand_index
            W_eq[resource_index, allocation_index] = 1.0
        end
        W_eq[resource_index, demand_count + resource_count * demand_count + resource_index] = 1.0
    end

    for demand_index in 1:demand_count
        row = resource_count + demand_index
        W_eq[row, demand_index] = 1.0
        for resource_index in 1:resource_count
            allocation_index = demand_count + demand_count * (resource_index - 1) + demand_index
            W_eq[row, allocation_index] = service_rates[resource_index, demand_index]
        end
        slack_index = demand_count + resource_count * demand_count + resource_count + demand_index
        W_eq[row, slack_index] = -1.0
    end

    T_eq = zeros(Float64, recourse_rows, resource_count)
    for resource_index in 1:resource_count
        T_eq[resource_index, resource_index] = -yields[resource_index]
    end

    q = zeros(Float64, recourse_variables)
    q[1:demand_count] .= second_costs

    program = ContextualDFL.StochasticProgram(
        A_eq=zeros(Float64, 0, resource_count),
        A_ineq=-Matrix{Float64}(LinearAlgebra.I, resource_count, resource_count),
        b_eq=Float64[],
        b_ineq=zeros(Float64, resource_count),
        c=first_costs,
    )

    scenario = (;
        W_eq=W_eq,
        W_ineq=-Matrix{Float64}(LinearAlgebra.I, recourse_variables, recourse_variables),
        T_eq=T_eq,
        T_ineq=zeros(Float64, recourse_variables, resource_count),
        h_eq=zeros(Float64, recourse_rows),
        h_ineq=zeros(Float64, recourse_variables),
        q=q,
    )

    return ResourceAllocationProblem(problem_data, program, scenario)
end

stochastic_program(problem::ResourceAllocationProblem) = problem.stochastic_program

base_scenario(problem::ResourceAllocationProblem) = problem.base_scenario
