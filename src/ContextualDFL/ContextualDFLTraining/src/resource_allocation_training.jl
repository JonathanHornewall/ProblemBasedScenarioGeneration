using ContextualDFL
using ContextualDFLExperiments
using Flux
using Random

function resource_allocation_demand_from_scenario(
    scenario::ContextualDFL.ParametricScenario,
)
    isempty(scenario.h_eq_xi) &&
        throw(ArgumentError("expected a resource-allocation demand vector in h_eq_xi"))
    return scenario.h_eq_xi
end

struct ConstantSchedule{T}
    value::T
end

(schedule::ConstantSchedule)(args...; kwargs...) = schedule.value

function resource_allocation_demand_count(problem)
    return size(problem.problem_data.service_rate_parameters, 2)
end

function resource_allocation_training_data(problem, config, rng::AbstractRNG)
    context_generator = ContextualDFLExperiments.ResourceAllocationContextDataGenerator(
        rng=rng,
    )
    scenario_generator =
        ContextualDFLExperiments.ResourceAllocationScenarioDataGenerator(
            problem;
            sigma=config.sigma,
            p=config.demand_power,
            L=config.context_terms,
            rng=rng,
        )

    sample_count = Int(config.n_samples)
    sample_count > 0 || throw(ArgumentError("n_samples must be positive."))

    contexts = [Vector{Float64}(context_generator()) for _ in 1:sample_count]
    scenarios = [scenario_generator(context) for context in contexts]
    return contexts, scenarios
end

function split_resource_allocation_data(
    contexts::AbstractVector,
    scenarios::AbstractVector;
    validation_fraction,
    test_fraction,
    rng::AbstractRNG,
)
    length(contexts) == length(scenarios) ||
        throw(DimensionMismatch("contexts and scenarios must have the same length."))
    0 <= validation_fraction < 1 ||
        throw(ArgumentError("validation_fraction must be in [0, 1)."))
    0 <= test_fraction < 1 ||
        throw(ArgumentError("test_fraction must be in [0, 1)."))
    validation_fraction + test_fraction < 1 ||
        throw(ArgumentError("validation_fraction + test_fraction must be less than 1."))

    sample_count = length(contexts)
    indices = randperm(rng, sample_count)
    test_count = floor(Int, test_fraction * sample_count)
    validation_count = floor(Int, validation_fraction * sample_count)
    train_count = sample_count - validation_count - test_count
    train_count > 0 || throw(ArgumentError("split leaves no training samples."))

    train_indices = indices[1:train_count]
    validation_indices = indices[(train_count + 1):(train_count + validation_count)]
    test_indices = indices[(train_count + validation_count + 1):end]

    return (;
        train=ContextualDFLExperiments.generate_contextual_data_set(
            contexts[train_indices],
            scenarios[train_indices],
        ),
        validation=ContextualDFLExperiments.generate_contextual_data_set(
            contexts[validation_indices],
            scenarios[validation_indices],
        ),
        test=ContextualDFLExperiments.generate_contextual_data_set(
            contexts[test_indices],
            scenarios[test_indices],
        ),
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

function build_loss(config, vector_decoder, reference_decoder, solver, program)
    if config.loss == :dfl_scen
        nr_scenarios = config isa NamedTuple && :nr_scenarios in keys(config) ?
            Int(config.nr_scenarios) :
            1
        return ContextualDFL.DflScenLoss(
            vector_decoder,
            reference_decoder,
            solver,
            program;
            nr_scenarios=nr_scenarios,
        )
    end
    throw(ArgumentError("unsupported loss $(config.loss); use :dfl_scen"))
end

function resource_allocation_training_objects(config)
    rng = MersenneTwister(config.seed)
    problem_data = ContextualDFLExperiments.default_resource_allocation_problem_data()
    problem = ContextualDFLExperiments.ResourceAllocationProblem(problem_data)
    contexts, scenarios = resource_allocation_training_data(problem, config, rng)
    splits = split_resource_allocation_data(
        contexts,
        scenarios;
        validation_fraction=config.validation_fraction,
        test_fraction=config.test_fraction,
        rng=rng,
    )

    vector_decoder = ContextualDFLExperiments.ResourceAllocationDemandVectorDecoder(problem)
    reference_decoder =
        ContextualDFLExperiments.ResourceAllocationDemandParametricDecoder(problem)
    program = ContextualDFLExperiments.stochastic_program(problem)
    neural_net = build_neural_net(
        length(first(splits.train).context),
        resource_allocation_demand_count(problem);
        hidden_size=config.hidden_size,
        depth=config.depth,
        dropout=config.dropout,
    )
    solver = build_solver(config)
    generator = ContextualDFL.ScenarioGenerator(
        neural_net=neural_net,
        scenario_decoder=vector_decoder,
    )

    return (;
        problem=problem,
        program=program,
        scenario_decoder=vector_decoder,
        reference_scenario_decoder=reference_decoder,
        solver=solver,
        loss=build_loss(config, vector_decoder, reference_decoder, solver, program),
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
