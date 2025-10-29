module StepGenerateTrainingData

using Random
using CSV
using DataFrames
using DataFrames: Not
using Serialization
using Dates

using ..Config: ExperimentConfig, seed_rng!
using ..Artifacts: ensure_step_directories, mark_step_complete, write_json_file

push!(Base.LOAD_PATH, normpath(joinpath(@__DIR__, "..", "..", "src")))
import FullBenchmark

const DataGen = FullBenchmark.DataGeneration

export execute_generate_training_data

function execute_generate_training_data(config::ExperimentConfig, ctx::NamedTuple)
    output_dir = ctx.output_dir
    ensure_step_directories(output_dir, :generate_training_data)

    testing_dir = joinpath(output_dir, "artifacts", "testing")
    params_path = joinpath(testing_dir, "problem_parameters.jls")
    isfile(params_path) || error("Testing parameters not found at $(params_path). Run testing-data step or provide input artifacts.")
    params = Serialization.deserialize(params_path)

    training_dir = joinpath(output_dir, "artifacts", "training")
    mkpath(training_dir)

    cov_rng = seed_rng!(:training_covariates, config.training_covariate_seed)
    scenario_rng = seed_rng!(:training_scenarios, config.training_scenario_seed)

    covariates = DataGen.sample_covariates(cov_rng, config.training_size, params)
    cov_ids = collect(1:config.training_size)

    cov_set = DataGen.build_covariate_set(covariates; ids = cov_ids)

    pairs = Vector{FullBenchmark.Datasets.TrainingPair}(undef, length(cov_ids))
    scenario_seeds = Vector{UInt32}(undef, length(cov_ids))
    for idx in eachindex(cov_ids)
        seed = rand(scenario_rng, UInt32)
        scenario_seeds[idx] = seed
        local_rng = MersenneTwister(seed)
        batch = DataGen.generate_scenario_batch(local_rng,
                                                cov_ids[idx],
                                                view(covariates, idx, :),
                                                params,
                                                config.training_scenarios_per_context)
        pairs[idx] = FullBenchmark.Datasets.TrainingPair(batch.covariate_id,
                                                          batch.covariate,
                                                          batch.scenarios)
    end

    Serialization.serialize(joinpath(training_dir, "training_pairs.jls"), pairs)

    cov_df = DataFrame(covariates, :auto)
    rename!(cov_df, Symbol.("x" .* string.(1:size(cov_df, 2))))
    cov_df.id = cov_ids
    select!(cov_df, :id, Not(:id))
    CSV.write(joinpath(training_dir, "training_covariates.csv"), cov_df)

    log_payload = Dict(
        "covariate_seed" => config.training_covariate_seed,
        "scenario_seed" => config.training_scenario_seed,
        "scenario_seeds" => scenario_seeds,
        "training_size" => config.training_size,
        "scenarios_per_context" => config.training_scenarios_per_context,
        "timestamp" => string(Dates.now())
    )
    write_json_file(joinpath(training_dir, "data_generation_log.json"), log_payload)

    mark_step_complete(:generate_training_data, output_dir)
    return nothing
end

end # module
