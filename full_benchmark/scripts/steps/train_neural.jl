module StepTrainNeural

using Serialization
using Dates
using CSV
using DataFrames

include("../util/config.jl")
include("../util/artifacts.jl")

using .Config: ExperimentConfig
using .Artifacts: ensure_step_directories, mark_step_complete, write_json_file

export execute_train_neural

function execute_train_neural(config::ExperimentConfig, ctx::NamedTuple)
    output_dir = ctx.output_dir
    ensure_step_directories(output_dir, :train_neural)

    models_dir = joinpath(output_dir, "artifacts", "models", "neural")
    mkpath(models_dir)

    placeholder = Dict(
        "status" => "pending implementation",
        "timestamp" => string(Dates.now()),
        "seed" => config.neural_seed,
        "annealing" => config.annealing_schedule,
        "epochs" => config.epoch_schedule,
        "step_sizes" => config.step_size_schedule,
        "batch_sizes" => config.batch_size_schedule,
        "surrogate_parameter" => config.surrogate_parameter
    )

    Serialization.serialize(joinpath(models_dir, "neural_model_final.jls"), placeholder)

    history_path = joinpath(models_dir, "neural_training_history.csv")
    CSV.write(history_path, DataFrame(epoch = Int[], loss = Float64[]))

    write_json_file(joinpath(models_dir, "neural_training_log.json"), placeholder)

    mark_step_complete(:train_neural, output_dir)
    return nothing
end

end # module
