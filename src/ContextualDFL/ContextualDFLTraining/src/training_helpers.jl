using ContextualDFL
using Flux
using Random

struct ConstantSchedule{T}
    value::T
end

(schedule::ConstantSchedule)(args...; kwargs...) = schedule.value

function build_neural_net(input_dimension, output_dimension; hidden_size, depth, dropout)
    input_dimension > 0 ||
        throw(ArgumentError("input_dimension must be positive."))
    output_dimension > 0 ||
        throw(ArgumentError("output_dimension must be positive."))
    hidden_size > 0 ||
        throw(ArgumentError("hidden_size must be positive."))
    depth > 0 ||
        throw(ArgumentError("depth must be positive."))

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
    solver_name = Symbol(config_value(config, :solver, :highs))
    if solver_name == :highs
        return ContextualDFL.Solver(ContextualDFL.IpoptSolver(), ContextualDFL.HiGHSSolver())
    end
    throw(ArgumentError("unsupported solver $(solver_name)"))
end

function build_loss(config, vector_decoder, reference_decoder, solver, program)
    loss_name = Symbol(config_value(config, :loss, :dfl_scen))
    if loss_name in (:dfl_scen, :spo_plus)
        nr_scenarios = config_value(config, :nr_scenarios, nothing)
        loss_type = loss_name == :dfl_scen ? ContextualDFL.DflScenLoss : ContextualDFL.SPOPlusLoss
        return loss_type(
            vector_decoder,
            reference_decoder,
            solver,
            program;
            nr_scenarios=isnothing(nr_scenarios) ? 1 : Int(nr_scenarios),
        )
    end
    throw(ArgumentError("unsupported loss $(loss_name); use :dfl_scen or :spo_plus"))
end

function split_contextual_dataset(
    dataset::AbstractVector;
    validation_fraction,
    test_fraction,
    rng::AbstractRNG,
)
    0 <= validation_fraction < 1 ||
        throw(ArgumentError("validation_fraction must be in [0, 1)."))
    0 <= test_fraction < 1 ||
        throw(ArgumentError("test_fraction must be in [0, 1)."))
    validation_fraction + test_fraction < 1 ||
        throw(ArgumentError("validation_fraction + test_fraction must be less than 1."))

    sample_count = length(dataset)
    sample_count > 0 || throw(ArgumentError("dataset must not be empty."))

    indices = randperm(rng, sample_count)
    test_count = floor(Int, test_fraction * sample_count)
    validation_count = floor(Int, validation_fraction * sample_count)
    train_count = sample_count - validation_count - test_count
    train_count > 0 || throw(ArgumentError("split leaves no training samples."))

    train_indices = indices[1:train_count]
    validation_indices = indices[(train_count + 1):(train_count + validation_count)]
    test_indices = indices[(train_count + validation_count + 1):end]

    return (;
        train=dataset[train_indices],
        validation=dataset[validation_indices],
        test=dataset[test_indices],
    )
end
