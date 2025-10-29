#!/usr/bin/env julia

include("util/config.jl")
include("util/artifacts.jl")

include("steps/generate_testing_data.jl")
include("steps/compute_saa_baselines.jl")
include("steps/generate_training_data.jl")
include("steps/train_baselines.jl")
include("steps/train_neural.jl")
include("steps/run_benchmark.jl")

using .Config
using .Artifacts
using .StepGenerateTestingData
using .StepComputeSAABaselines
using .StepGenerateTrainingData
using .StepTrainBaselines
using .StepTrainNeural
using .StepRunBenchmark

const STEP_FUNCTIONS = Dict(
    :generate_testing_data => StepGenerateTestingData.execute_generate_testing_data,
    :compute_saa_baselines => StepComputeSAABaselines.execute_compute_saa_baselines,
    :generate_training_data => StepGenerateTrainingData.execute_generate_training_data,
    :train_baselines => StepTrainBaselines.execute_train_baselines,
    :train_neural => StepTrainNeural.execute_train_neural,
    :run_benchmark => StepRunBenchmark.execute_run_benchmark
)

function forced_steps(flag::Symbol)
    if flag == :full_training
        return Set([:generate_training_data, :train_baselines, :train_neural, :run_benchmark])
    elseif flag == :full_testing
        return Set([:generate_testing_data, :compute_saa_baselines])
    elseif flag == :method_training
        return Set([:train_baselines, :run_benchmark])
    elseif flag == :neural_training
        return Set([:train_neural, :run_benchmark])
    else
        return Set{Symbol}()
    end
end

function execute_step(step::Symbol, config::Config.ExperimentConfig, ctx::NamedTuple; forced::Bool)
    fn = STEP_FUNCTIONS[step]
    println("[FullBenchmark] => $(step) -- $(forced ? "running (forced)" : "running")")
    fn(config, ctx)
end

function main(args::Vector{String})
    config = Config.load_config(args)
    mkpath(config.output_dir)
    ctx = (; output_dir = config.output_dir)

    priority = Config.resolve_flag_priority(config)
    forced = forced_steps(priority)

    for step in Config.StepOrder
        if step in forced
            execute_step(step, config, ctx; forced=true)
            continue
        end

        if config.input_dir !== nothing && Artifacts.artifacts_present(config.input_dir, step)
            println("[FullBenchmark] => $(step) -- copying artifacts from input directory")
            success = Artifacts.copy_artifacts(config.input_dir, config.output_dir, step)
            success && println("[FullBenchmark] => $(step) -- artifacts copied")
            if !success
                println("[FullBenchmark] => $(step) -- copy failed, executing step")
                execute_step(step, config, ctx; forced=false)
            end
        elseif Artifacts.artifacts_present(config.output_dir, step)
            println("[FullBenchmark] => $(step) -- artifacts already present, skipping")
        else
            execute_step(step, config, ctx; forced=false)
        end
    end

    println("[FullBenchmark] Pipeline complete. Artifacts available at $(config.output_dir)")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
