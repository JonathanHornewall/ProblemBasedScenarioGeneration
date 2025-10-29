module StepTrainBaselines

using Serialization
using Dates

include("../util/config.jl")
include("../util/artifacts.jl")

using .Config: ExperimentConfig
using .Artifacts: ensure_step_directories, mark_step_complete, write_json_file

export execute_train_baselines

function execute_train_baselines(config::ExperimentConfig, ctx::NamedTuple)
    output_dir = ctx.output_dir
    ensure_step_directories(output_dir, :train_baselines)

    training_dir = joinpath(output_dir, "artifacts", "training")
    pairs_path = joinpath(training_dir, "training_pairs.jls")
    if !isfile(pairs_path)
        error("Training pairs not found at $(pairs_path). Generate training data first or supply input artifacts.")
    end

    models_dir = joinpath(output_dir, "artifacts", "models", "baselines")
    mkpath(models_dir)

    placeholder = Dict("status" => "pending implementation",
                        "timestamp" => string(Dates.now()))
    Serialization.serialize(joinpath(models_dir, "ls_model.jls"), placeholder)
    Serialization.serialize(joinpath(models_dir, "er_saa_model.jls"), placeholder)
    Serialization.serialize(joinpath(models_dir, "cart_model.jls"), placeholder)
    Serialization.serialize(joinpath(models_dir, "knn_model.jls"), placeholder)
    Serialization.serialize(joinpath(models_dir, "nm_model.jls"), placeholder)

    report_path = joinpath(models_dir, "baseline_training_report.json")
    write_json_file(report_path, Dict(
        "status" => "placeholder",
        "message" => "Baseline training not yet integrated",
        "timestamp" => string(Dates.now())
    ))

    mark_step_complete(:train_baselines, output_dir)
    return nothing
end

end # module
