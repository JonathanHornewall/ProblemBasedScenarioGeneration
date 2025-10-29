module StepGenerateTestingData

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
const Datasets = FullBenchmark.Datasets

export execute_generate_testing_data

function execute_generate_testing_data(config::ExperimentConfig, ctx::NamedTuple)
    output_dir = ctx.output_dir
    ensure_step_directories(output_dir, :generate_testing_data)

    cov_rng = seed_rng!(:testing_covariates, config.testing_covariate_seed)
    scenario_master_rng = seed_rng!(:testing_scenarios, config.testing_scenario_seed)

    params = DataGen.sample_problem_parameters(cov_rng,
                                               config.num_products,
                                               config.feature_dim;
                                               σ = config.sigma,
                                               ω = config.omega,
                                               p = config.testing_param_p)

    total_covariates = max(config.testing_covariates, 100)
    full_covariates = DataGen.sample_covariates(cov_rng, total_covariates, params)
    covariate_ids = collect(1:total_covariates)
    test_count = config.testing_covariates
    test_covariates = full_covariates[1:test_count, :]
    test_ids = covariate_ids[1:test_count]

    scenario_batches = Vector{Datasets.ScenarioBatch}()
    batch_seed_log = Vector{UInt32}()

    for (cov_idx, cov_id) in enumerate(test_ids)
        cov_vec = view(test_covariates, cov_idx, :)
        for _ in 1:config.testing_collections_per_covariate
            local_seed = rand(scenario_master_rng, UInt32)
            push!(batch_seed_log, local_seed)
            local_rng = MersenneTwister(local_seed)
            batch = DataGen.generate_scenario_batch(local_rng,
                                                    cov_id,
                                                    cov_vec,
                                                    params,
                                                    config.testing_scenarios_per_collection)
            push!(scenario_batches, batch)
        end
    end

    scenario_tensor = DataGen.assemble_testing_tensor(scenario_batches,
                                                       config.testing_collections_per_covariate,
                                                       config.testing_scenarios_per_collection,
                                                       config.num_products)

    testing_dir = joinpath(output_dir, "artifacts", "testing")
    mkpath(testing_dir)

    test_df = DataFrame(test_covariates, :auto)
    rename!(test_df, Symbol.("x" .* string.(1:size(test_df, 2))))
    test_df.id = test_ids
    select!(test_df, :id, Not(:id))
    CSV.write(joinpath(testing_dir, "test_covariates.csv"), test_df)

    full_df = DataFrame(full_covariates, :auto)
    rename!(full_df, Symbol.("x" .* string.(1:size(full_df, 2))))
    full_df.id = covariate_ids
    select!(full_df, :id, Not(:id))
    CSV.write(joinpath(testing_dir, "full_context_pool.csv"), full_df)

    Serialization.serialize(joinpath(testing_dir, "test_scenarios.jls"), scenario_tensor)
    Serialization.serialize(joinpath(testing_dir, "problem_parameters.jls"), params)

    seed_log = Dict(
        "covariate_seed" => config.testing_covariate_seed,
        "scenario_seed" => config.testing_scenario_seed,
        "batch_seeds" => batch_seed_log,
        "timestamp" => string(Dates.now()),
        "testing_covariates" => config.testing_covariates,
        "collections_per_covariate" => config.testing_collections_per_covariate,
        "scenarios_per_collection" => config.testing_scenarios_per_collection,
        "problem_parameters" => Dict(
            "phi" => collect(params.ϕ),
            "zeta" => [collect(params.ζ[i, :]) for i in 1:size(params.ζ, 1)],
            "sigma" => params.σ,
            "p" => params.p,
            "correlation_matrix" => [collect(params.Σ[i, :]) for i in 1:size(params.Σ, 1)]
        )
    )
    write_json_file(joinpath(testing_dir, "testing_seeds.json"), seed_log)

    mark_step_complete(:generate_testing_data, output_dir)
    return nothing
end

end # module
