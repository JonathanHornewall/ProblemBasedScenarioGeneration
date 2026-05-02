using ContextualDFL
using Distributions
using Flux
using LinearAlgebra
using Random
using Statistics

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
            throw(ArgumentError("first-stage costs must have length $resource_count"))
        length(second_costs) == demand_count ||
            throw(ArgumentError("second-stage costs must have length $demand_count"))
        length(yields) == resource_count ||
            throw(ArgumentError("yield parameters must have length $resource_count"))

        return new(service_rates, first_costs, second_costs, yields)
    end
end

struct ResourceAllocationProblem
    problem_data::ResourceAllocationProblemData
    s1_constraint_matrix::Matrix{Float64}
    s1_constraint_vector::Vector{Float64}
    s1_cost_vector::Vector{Float64}
    s2_constraint_matrix::Matrix{Float64}
    s2_coupling_matrix::Matrix{Float64}
    s2_cost_vector::Vector{Float64}
end

struct ResourceAllocationDemandDecoder <: ContextualDFL.VectorDecoder
    problem::ResourceAllocationProblem
end

struct ResourceAllocationMSEScenLoss <: ContextualDFL.LossFunction end

function (decoder::ResourceAllocationDemandDecoder)(demand::AbstractVector)
    scenario = scenario_from_demand(decoder.problem, demand)
    return _scenario_tuple(scenario)
end

function (::ResourceAllocationDemandDecoder)(
    scenario_parameter::ContextualDFL.ParametricScenario,
)
    return _scenario_tuple(scenario_parameter)
end

function (::ResourceAllocationMSEScenLoss)(
    program::ContextualDFL.StochasticProgram,
    predicted_demand,
    reference_demand,
    mu;
    kwargs...,
)
    prediction = _demand_matrix(predicted_demand)
    target = _demand_matrix(reference_demand)
    size(prediction) == size(target) ||
        throw(DimensionMismatch("predicted and reference demand matrices have different sizes."))
    return Statistics.mean(abs2, prediction .- target)
end

_demand_matrix(value::AbstractMatrix) = value
_demand_matrix(value::AbstractVector) = reshape(value, :, 1)

function _demand_matrix(value::AbstractVector{<:ContextualDFL.ParametricScenario})
    return reduce(hcat, demand_from_scenario_parameter.(value))
end

function _scenario_tuple(scenario::ContextualDFL.ParametricScenario)
    return (
        scenario.W_eq_xi,
        scenario.W_ineq_xi,
        scenario.T_eq_xi,
        scenario.T_ineq_xi,
        scenario.h_eq_xi,
        scenario.h_ineq_xi,
        scenario.q_xi,
    )
end

struct ConstantSchedule{T}
    value::T
end

(schedule::ConstantSchedule)(args...; kwargs...) = schedule.value

function ResourceAllocationProblem(problem_data::ResourceAllocationProblemData)
    service_rates = problem_data.service_rate_parameters
    first_costs = problem_data.first_stage_costs
    second_costs = problem_data.second_stage_costs
    yields = problem_data.yield_parameters

    resource_count, demand_count = size(service_rates)

    first_stage_a = -Matrix{Float64}(I, length(first_costs), length(first_costs))
    first_stage_b = zeros(length(first_costs))
    first_stage_c = copy(first_costs)

    recourse_variables = demand_count + resource_count * demand_count + resource_count + demand_count
    recourse_rows = resource_count + demand_count
    recourse_w = zeros(recourse_rows, recourse_variables)

    for resource_index in 1:resource_count
        for demand_index in 1:demand_count
            recourse_w[resource_index, demand_count + demand_count * (resource_index - 1) + demand_index] = 1.0
        end
        recourse_w[resource_index, demand_count + resource_count * demand_count + resource_index] = 1.0
    end

    for demand_index in 1:demand_count
        row = resource_count + demand_index
        recourse_w[row, demand_index] = 1.0
        for resource_index in 1:resource_count
            recourse_w[row, demand_count + demand_count * (resource_index - 1) + demand_index] =
                service_rates[resource_index, demand_index]
        end
        recourse_w[row, demand_count + resource_count * demand_count + resource_count + demand_index] = -1.0
    end

    recourse_t = zeros(recourse_rows, resource_count)
    for resource_index in 1:resource_count
        recourse_t[resource_index, resource_index] = -yields[resource_index]
    end

    recourse_q = zeros(recourse_variables)
    recourse_q[1:demand_count] .= second_costs
    recourse_w = [
        recourse_w
        -Matrix{Float64}(I, recourse_variables, recourse_variables)
    ]
    recourse_t = [
        recourse_t
        zeros(Float64, recourse_variables, resource_count)
    ]

    return ResourceAllocationProblem(
        problem_data,
        first_stage_a,
        first_stage_b,
        first_stage_c,
        recourse_w,
        recourse_t,
        recourse_q,
    )
end

function resource_count(problem::ResourceAllocationProblem)
    return size(problem.problem_data.service_rate_parameters, 1)
end

function demand_count(problem::ResourceAllocationProblem)
    return size(problem.problem_data.service_rate_parameters, 2)
end

function scenario_h(problem::ResourceAllocationProblem, demand::AbstractVector)
    return vcat(
        zeros(eltype(demand), resource_count(problem)),
        demand,
        zeros(eltype(demand), length(problem.s2_cost_vector)),
    )
end

function scenario_from_demand(problem::ResourceAllocationProblem, demand::AbstractVector)
    recourse_variable_count = length(problem.s2_cost_vector)
    first_stage_variable_count = length(problem.s1_cost_vector)
    return ContextualDFL.ParametricScenario(
        W_eq_xi=zeros(Float64, 0, recourse_variable_count),
        W_ineq_xi=problem.s2_constraint_matrix,
        T_eq_xi=zeros(Float64, 0, first_stage_variable_count),
        T_ineq_xi=problem.s2_coupling_matrix,
        h_eq_xi=Float64[],
        h_ineq_xi=scenario_h(problem, demand),
        q_xi=problem.s2_cost_vector,
    )
end

function demand_from_scenario_parameter(scenario::ContextualDFL.ParametricScenario)
    resource_count = size(scenario.T_ineq_xi, 2)
    demand_count = demand_count_from_scenario(scenario, resource_count)
    return scenario.h_ineq_xi[(resource_count + 1):(resource_count + demand_count)]
end

function demand_count_from_scenario(
    scenario::ContextualDFL.ParametricScenario,
    resource_count::Integer,
)
    recourse_variable_count = length(scenario.q_xi)
    demand_count, remainder = divrem(
        recourse_variable_count - resource_count,
        resource_count + 2,
    )
    remainder == 0 && demand_count > 0 ||
        throw(ArgumentError("could not infer resource-allocation demand count"))
    return demand_count
end

function stochastic_program(problem::ResourceAllocationProblem)
    return ContextualDFL.StochasticProgram(
        A_eq=zeros(0, length(problem.s1_cost_vector)),
        A_ineq=problem.s1_constraint_matrix,
        b_eq=Float64[],
        b_ineq=problem.s1_constraint_vector,
        c=problem.s1_cost_vector,
    )
end

scenario_decoder(problem::ResourceAllocationProblem) = ResourceAllocationDemandDecoder(problem)

function resource_parameter_path()
    return normpath(
        joinpath(
            @__DIR__,
            "..",
            "..",
            "..",
            "ProblemBasedScenarioGeneration",
            "src",
            "problem_instances",
            "resource_allocation",
            "parameters.jl",
        ),
    )
end

function parse_number_vector(text)
    matches = eachmatch(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?", text)
    return [parse(Float64, match.match) for match in matches]
end

function parse_assignment_vector(text, variable_name)
    pattern = Regex("(?m)^\\s*" * variable_name * "\\s*=\\s*\\[([^\\]]+)\\]")
    match_result = match(pattern, text)
    match_result === nothing &&
        throw(ArgumentError("could not parse vector assignment for $variable_name"))
    return parse_number_vector(match_result.captures[1])
end

function parse_vector_after_comment(text, comment_text)
    comment_position = findfirst(comment_text, text)
    comment_position === nothing &&
        throw(ArgumentError("could not find comment '$comment_text'"))
    tail = text[last(comment_position):end]
    match_result = match(r"(?m)^\s*[^\s=#]+\s*=\s*\[([^\]]+)\]", tail)
    match_result === nothing &&
        throw(ArgumentError("could not parse vector after comment '$comment_text'"))
    return parse_number_vector(match_result.captures[1])
end

function parse_service_rate_matrix(text)
    service_position = findfirst("# service rate parameters", text)
    service_position === nothing &&
        throw(ArgumentError("could not find service-rate section"))
    tail = text[last(service_position):end]
    dims_match = match(r"zeros\(Float64,\s*(\d+)\s*,\s*(\d+)\s*\)", tail)
    dims_match === nothing &&
        throw(ArgumentError("could not parse service-rate dimensions"))

    row_count = parse(Int, dims_match.captures[1])
    column_count = parse(Int, dims_match.captures[2])
    service_rates = zeros(Float64, row_count, column_count)

    row_pattern = r"(?m)^\s*[^\s\[]+\[(\d+),:\]\s*=\s*\[([^\]]+)\]"
    for row_match in eachmatch(row_pattern, tail)
        row_index = parse(Int, row_match.captures[1])
        row_values = parse_number_vector(row_match.captures[2])
        length(row_values) == column_count ||
            throw(ArgumentError("service-rate row $row_index has $(length(row_values)) values"))
        service_rates[row_index, :] .= row_values
    end

    all(any(value -> !iszero(value), service_rates[row, :]) for row in 1:row_count) ||
        throw(ArgumentError("one or more service-rate rows were not parsed"))

    return service_rates
end

function parse_resource_allocation_parameters(path=resource_parameter_path())
    text = read(path, String)
    active_text = split(text, "#="; limit=2)[1]
    first_costs = parse_assignment_vector(active_text, "cz")
    second_costs = parse_assignment_vector(active_text, "qw")
    yields = parse_vector_after_comment(active_text, "# yield parameters")
    service_rates = parse_service_rate_matrix(active_text)
    return ResourceAllocationProblemData(service_rates, first_costs, second_costs, yields)
end

function generated_resource_allocation_parameters(; seed=8675309, resource_count=20, demand_count=30)
    rng = MersenneTwister(seed)
    service_rates = rand(rng, Uniform(1.5, 2.5), resource_count, demand_count)
    service_rates[rand(rng, resource_count, demand_count) .< 0.35] .= 0.0
    first_costs = rand(rng, Uniform(0.7, 1.3), resource_count)
    second_costs = rand(rng, Uniform(1.9, 2.4), demand_count)
    yields = rand(rng, Uniform(0.9, 1.0), resource_count)
    return ResourceAllocationProblemData(service_rates, first_costs, second_costs, yields)
end

function default_resource_allocation_parameters()
    path = resource_parameter_path()
    if isfile(path)
        return parse_resource_allocation_parameters(path)
    end
    return generated_resource_allocation_parameters()
end

function generate_random_correlation_matrix(rng::AbstractRNG, dimension::Int)
    beta_parameter = 2.0
    partial_correlation = zeros(Float64, dimension, dimension)
    correlation = Matrix{Float64}(I, dimension, dimension)

    for k in 1:(dimension - 1)
        for i in (k + 1):dimension
            partial_correlation[k, i] = (rand(rng, Beta(beta_parameter, beta_parameter)) - 0.5) * 2.0
            rho = partial_correlation[k, i]
            for j in (k - 1):-1:1
                rho =
                    rho *
                    sqrt((1 - partial_correlation[j, i]^2) * (1 - partial_correlation[j, k]^2)) +
                    partial_correlation[j, i] * partial_correlation[j, k]
            end
            correlation[k, i] = rho
            correlation[i, k] = rho
        end
    end

    permutation = randperm(rng, dimension)
    return correlation[permutation, permutation]
end

function sample_demand_coefficients(rng::AbstractRNG, demand_count::Int, context_terms::Int)
    intercept = 50 .+ 5 .* randn(rng, demand_count)
    slopes = zeros(Float64, demand_count, context_terms)

    base_slopes = [10.0, 5.0, 2.0]
    for term in 1:context_terms
        center = term <= length(base_slopes) ? base_slopes[term] : base_slopes[end]
        slopes[:, term] .= center .+ rand(rng, Uniform(-4, 4), demand_count)
    end

    return intercept, slopes
end

function generate_resource_allocation_data(
    problem::ResourceAllocationProblem;
    n_samples,
    validation_fraction,
    test_fraction,
    sigma,
    demand_power,
    context_terms,
    rng,
)
    context_dimension = 3
    context_terms <= context_dimension ||
        throw(ArgumentError("context_terms must be <= $context_dimension"))

    correlation = generate_random_correlation_matrix(rng, context_dimension)
    distribution = MvNormal(zeros(context_dimension), Symmetric(correlation + 1e-8I))
    contexts = abs.(rand(rng, distribution, n_samples))

    demands = zeros(Float64, demand_count(problem), n_samples)
    intercept, slopes = sample_demand_coefficients(rng, demand_count(problem), context_terms)

    for demand_index in 1:demand_count(problem)
        for sample_index in 1:n_samples
            signal = intercept[demand_index]
            for term in 1:context_terms
                signal += slopes[demand_index, term] * contexts[term, sample_index]^demand_power
            end
            demands[demand_index, sample_index] = signal + rand(rng, Normal(0, sigma))
        end
    end

    return contexts, max.(demands, 0.0)
end

function dataset_from_indices(problem::ResourceAllocationProblem, contexts, demands, indices)
    return [
        ContextualDFL.ContextualDataPoint(
            Vector{Float64}(contexts[:, sample_index]),
            [scenario_from_demand(problem, demands[:, sample_index])],
        ) for sample_index in indices
    ]
end

function split_resource_allocation_data(
    problem::ResourceAllocationProblem,
    contexts,
    demands;
    validation_fraction,
    test_fraction,
    rng,
)
    sample_count = size(contexts, 2)
    indices = randperm(rng, sample_count)
    test_count = floor(Int, test_fraction * sample_count)
    validation_count = floor(Int, validation_fraction * sample_count)
    train_count = sample_count - validation_count - test_count
    train_count > 0 || throw(ArgumentError("split leaves no training samples"))

    train_indices = indices[1:train_count]
    validation_indices = indices[(train_count + 1):(train_count + validation_count)]
    test_indices = indices[(train_count + validation_count + 1):end]

    return (;
        train=dataset_from_indices(problem, contexts, demands, train_indices),
        validation=dataset_from_indices(problem, contexts, demands, validation_indices),
        test=dataset_from_indices(problem, contexts, demands, test_indices),
    )
end

function build_neural_net(input_dimension, output_dimension; hidden_size, depth, dropout)
    layers = Any[Dense(input_dimension => hidden_size, relu)]

    for _ in 2:depth
        dropout > 0 && push!(layers, Dropout(dropout))
        push!(layers, Dense(hidden_size => hidden_size, relu))
    end

    dropout > 0 && push!(layers, Dropout(dropout))
    push!(layers, Dense(hidden_size => output_dimension))
    push!(layers, x -> Flux.softplus.(x))

    return Chain(layers...) |> f64
end

function build_solver(config)
    if config.solver == :highs
        return ContextualDFL.Solver(ContextualDFL.IpoptSolver(), ContextualDFL.HiGHSSolver())
    end
    throw(ArgumentError("unsupported solver $(config.solver)"))
end

function build_loss(config, decoder, solver, program)
    if config.loss == :mse_scen
        return ResourceAllocationMSEScenLoss()
    elseif config.loss == :dfl_scen
        nr_scenarios = config isa NamedTuple && :nr_scenarios in keys(config) ?
            Int(config.nr_scenarios) :
            1
        return ContextualDFL.DflScenLoss(
            decoder,
            decoder,
            solver,
            program;
            nr_scenarios=nr_scenarios,
        )
    end
    throw(ArgumentError("unsupported loss $(config.loss)"))
end

function resource_allocation_training_objects(config)
    rng = MersenneTwister(config.seed)
    problem = ResourceAllocationProblem(default_resource_allocation_parameters())
    contexts, demands = generate_resource_allocation_data(
        problem;
        n_samples=config.n_samples,
        validation_fraction=config.validation_fraction,
        test_fraction=config.test_fraction,
        sigma=config.sigma,
        demand_power=config.demand_power,
        context_terms=config.context_terms,
        rng=rng,
    )
    splits = split_resource_allocation_data(
        problem,
        contexts,
        demands;
        validation_fraction=config.validation_fraction,
        test_fraction=config.test_fraction,
        rng=rng,
    )

    decoder = scenario_decoder(problem)
    program = stochastic_program(problem)
    neural_net = build_neural_net(
        length(first(splits.train).context),
        demand_count(problem);
        hidden_size=config.hidden_size,
        depth=config.depth,
        dropout=config.dropout,
    )
    solver = build_solver(config)
    generator = ContextualDFL.DFLScenarioGenerator(
        NNScenarioDecoder=decoder,
        Solver=solver,
        neural_net=neural_net,
        program=program,
        learned_components=(:h,),
    )

    return (;
        problem=problem,
        program=program,
        scenario_decoder=decoder,
        solver=solver,
        loss=build_loss(config, decoder, solver, program),
        scenario_generator=generator,
        data=splits,
        schedules=(;
            mu=ConstantSchedule(config.mu),
            rho=ConstantSchedule(config.rho),
            batch_size=ConstantSchedule(config.batch_size),
            step_size=ConstantSchedule(config.learning_rate),
        ),
    )
end
