module StepTrainNeural

using CSV
using DataFrames
using Dates
using Flux
using LinearAlgebra
using Random
using Serialization
using Statistics

using ..Config: ExperimentConfig, seed_rng!
using ..Artifacts: ensure_step_directories, mark_step_complete, write_json_file

push!(Base.LOAD_PATH, normpath(joinpath(@__DIR__, "..", "..", "src")))
import FullBenchmark

const REPO_ROOT = normpath(joinpath(@__DIR__, "..", "..", ".."))
const PROBLEM_PKG_PATH = normpath(joinpath(REPO_ROOT, "src", "julia"))

push!(Base.LOAD_PATH, PROBLEM_PKG_PATH)
using ProblemBasedScenarioGeneration

include(joinpath(REPO_ROOT, "scripts", "resource_allocation_prototype", "parameters.jl"))

export execute_train_neural

const TEST_MODE_ENV_KEY = "FULL_BENCHMARK_TEST_MODE"

is_test_mode() = get(ENV, TEST_MODE_ENV_KEY, "0") == "1"

function execute_train_neural(config::ExperimentConfig, ctx::NamedTuple)
    output_dir = ctx.output_dir
    ensure_step_directories(output_dir, :train_neural)

    training_dir = joinpath(output_dir, "artifacts", "training")
    pairs_path = joinpath(training_dir, "training_pairs.jls")
    if !isfile(pairs_path)
        error("Training pairs not found at $(pairs_path). Generate training data first or supply input artifacts.")
    end

    models_dir = joinpath(output_dir, "artifacts", "models", "neural")
    mkpath(models_dir)

    if is_test_mode()
        timestamp = string(Dates.now())
        placeholder = Dict(
            "status" => "test_mode",
            "timestamp" => timestamp,
            "seed" => config.neural_seed
        )
        Serialization.serialize(joinpath(models_dir, "neural_model_final.jls"), placeholder)
        CSV.write(joinpath(models_dir, "neural_training_history.csv"),
                  DataFrame(phase = Int[], epoch = Int[], relative_loss = Float64[]))
        write_json_file(joinpath(models_dir, "neural_training_log.json"),
                        Dict("status" => "test_mode",
                             "timestamp" => timestamp,
                             "seed" => config.neural_seed))
        mark_step_complete(:train_neural, output_dir)
        return nothing
    end

    pairs = Serialization.deserialize(pairs_path)::Vector{FullBenchmark.Datasets.TrainingPair}
    dataset = build_training_dataset(pairs)

    Random.seed!(config.neural_seed)
    train_rng = seed_rng!(:neural_training, config.neural_seed)

    problem_data = ResourceAllocationProblemData(μᵢⱼ,
                                                 vec(cz),
                                                 vec(qw),
                                                 vec(ρᵢ))
    problem_instance = ResourceAllocationProblem(problem_data)
    model = construct_neural_network(problem_instance; nr_of_scenarios = 1)

    annealing = extend_schedule(config.annealing_schedule)
    epochs = extend_schedule(config.epoch_schedule; pad_value = 30)
    step_sizes = extend_schedule(config.step_size_schedule; pad_value = 1e-4)
    batch_sizes = extend_schedule(config.batch_size_schedule; pad_value = 25)

    num_phases = maximum(map(length, (annealing, epochs, step_sizes, batch_sizes)))
    annealing = extend_to_length(annealing, num_phases)
    epochs = extend_to_length(epochs, num_phases)
    step_sizes = extend_to_length(step_sizes, num_phases)
    batch_sizes = extend_to_length(batch_sizes, num_phases)

    history = run_annealing_training!(model,
                                      dataset,
                                      train_rng,
                                      problem_instance,
                                      annealing,
                                      epochs,
                                      step_sizes,
                                      batch_sizes,
                                      config.surrogate_parameter)

    model_path = joinpath(models_dir, "neural_model_final.jls")
    ProblemBasedScenarioGeneration.save_trained_model(model, model_path)

    history_df = DataFrame(history)
    CSV.write(joinpath(models_dir, "neural_training_history.csv"), history_df)

    final_entry = isempty(history) ? nothing : history[end]

    log_payload = Dict(
        "timestamp" => string(Dates.now()),
        "seed" => config.neural_seed,
        "dataset_size" => length(dataset),
        "annealing_schedule" => annealing,
        "epoch_schedule" => epochs,
        "step_size_schedule" => step_sizes,
        "batch_size_schedule" => batch_sizes,
        "surrogate_parameter" => config.surrogate_parameter,
        "final_relative_loss" => final_entry === nothing ? missing : final_entry.relative_loss
    )
    write_json_file(joinpath(models_dir, "neural_training_log.json"), log_payload)

    mark_step_complete(:train_neural, output_dir)
    return nothing
end

function build_training_dataset(pairs::Vector{FullBenchmark.Datasets.TrainingPair})
    data = Vector{Tuple{Matrix{Float64}, Matrix{Float64}}}()
    for pair in pairs
        for s in 1:size(pair.scenarios, 2)
            x = reshape(copy(pair.covariate), :, 1)
            ξ = reshape(copy(pair.scenarios[:, s]), :, 1)
            push!(data, (x, ξ))
        end
    end
    return data
end

extend_schedule(values; pad_value = nothing) = copy(values)

function extend_to_length(values::Vector, target::Int)
    if length(values) >= target
        return values
    end
    pad_value = isempty(values) ? zero(eltype(values)) : values[end]
    result = copy(values)
    while length(result) < target
        push!(result, pad_value)
    end
    return result
end

function run_annealing_training!(model,
                                 dataset,
                                 rng,
                                 problem_instance,
                                 annealing_schedule,
                                 epoch_schedule,
                                 step_sizes,
                                 batch_sizes,
                                 surrogate_parameter)
    history = Vector{NamedTuple}()
    N = length(dataset)

    for phase in 1:length(annealing_schedule)
        reg_param_surr = annealing_schedule[phase]
        epochs = epoch_schedule[phase]
        step_size = step_sizes[phase]
        batch_size = batch_sizes[phase]

        loss_fn = (pred, actual) -> ProblemBasedScenarioGeneration.loss(problem_instance,
                                                                         reg_param_surr,
                                                                         surrogate_parameter,
                                                                         pred,
                                                                         actual)
        rel_fn = (pred, actual) -> ProblemBasedScenarioGeneration.relative_loss(problem_instance,
                                                                                reg_param_surr,
                                                                                surrogate_parameter,
                                                                                pred,
                                                                                actual)

        opt = Flux.Adam(step_size)
        state = Flux.setup(opt, model)

        order = collect(1:N)
        for epoch in 1:epochs
            Random.shuffle!(rng, order)
            for batch in Iterators.partition(order, batch_size)
                Xb = hcat((dataset[i][1] for i in batch)...)
                Ξb = hcat((dataset[i][2] for i in batch)...)
                gs = Flux.gradient(model) do m
                    minibatch_loss(m, Xb, Ξb, loss_fn)
                end
                gmodel = gs isa Tuple ? gs[1] : gs
                Flux.update!(state, model, gmodel)
            end
            rel_loss_value = mean_relative_loss(model, dataset, rel_fn)
            push!(history, (phase = phase,
                            epoch = epoch,
                            annealing = reg_param_surr,
                            step_size = step_size,
                            batch_size = batch_size,
                            relative_loss = rel_loss_value))
        end
        GC.gc()
    end

    return history
end

function minibatch_loss(model, Xb, Ξb, loss_fn)
    cols = size(Xb, 2)
    return mean(loss_fn(model(Xb[:, i:i]), Ξb[:, i:i]) for i in 1:cols)
end

function mean_relative_loss(model, dataset, rel_fn)
    total = 0.0
    for (x, ξ) in dataset
        total += rel_fn(model(x), ξ)
    end
    return total / length(dataset)
end

end # module
