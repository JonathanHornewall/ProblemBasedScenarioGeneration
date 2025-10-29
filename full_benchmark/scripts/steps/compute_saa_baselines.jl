module StepComputeSAABaselines

using CSV
using DataFrames
using Serialization
using Statistics
using LinearAlgebra
using Dates

using ..Config: ExperimentConfig
using ..Artifacts: ensure_step_directories, mark_step_complete

const REPO_ROOT = normpath(joinpath(@__DIR__, "..", "..", ".."))
const SHIM_PATH = normpath(joinpath(REPO_ROOT, "full_benchmark", "shims"))
if !(SHIM_PATH in Base.LOAD_PATH)
    pushfirst!(Base.LOAD_PATH, SHIM_PATH)
end

include(joinpath(REPO_ROOT, "tito", "resource_allocation", "imports.jl"))
include(joinpath(REPO_ROOT, "tito", "resource_allocation", "policy.jl"))
include(joinpath(REPO_ROOT, "tito", "resource_allocation", "optimalityGap.jl"))
include(joinpath(REPO_ROOT, "scripts", "resource_allocation_prototype", "parameters.jl"))

export execute_compute_saa_baselines

function execute_compute_saa_baselines(config::ExperimentConfig, ctx::NamedTuple)
    output_dir = ctx.output_dir
    ensure_step_directories(output_dir, :compute_saa_baselines)
    testing_dir = joinpath(output_dir, "artifacts", "testing")

    scenario_path = joinpath(testing_dir, "test_scenarios.jls")
    covariate_path = joinpath(testing_dir, "test_covariates.csv")
    isfile(scenario_path) || error("Missing testing scenarios artifact at $(scenario_path)")
    isfile(covariate_path) || error("Missing testing covariates artifact at $(covariate_path)")

    scenario_tensor = Serialization.deserialize(scenario_path)
    covariate_df = CSV.read(covariate_path, DataFrame)
    sort!(covariate_df, :id)

    n_covariates = size(scenario_tensor, 1)
    collections_per_cov = size(scenario_tensor, 2)
    scenarios_per_collection = size(scenario_tensor, 3)
    J = size(scenario_tensor, 4)
    I = size(μᵢⱼ, 1)

    runs = DataFrame()
    cz_vec = vec(cz[1:I])
    qw_vec = vec(qw[1:J])
    ρ_vec = vec(ρᵢ[1:I])

    run_rows = Vector{NamedTuple}(undef, n_covariates * collections_per_cov)
    optima_rows = NamedTuple[]

    idx = 1
    for cov_idx in 1:n_covariates
        cov_id = covariate_df.id[cov_idx]
        cov_vector = [covariate_df[cov_idx, Symbol("x" * string(j))] for j in 1:config.feature_dim]
        collection_costs = Float64[]

        for collection_idx in 1:collections_per_cov
            sample_block = scenario_tensor[cov_idx, collection_idx, :, :]
            y_matrix = permutedims(sample_block, (2, 1))
            cost = fullInfoSAA(scenarios_per_collection,
                               J,
                               I,
                               y_matrix,
                               cz_vec,
                               qw_vec,
                               ρ_vec,
                               μᵢⱼ)
            push!(collection_costs, cost)
            attrs = (; covariate_id = cov_id,
                      run_id = collection_idx,
                      objective_value = cost,
                      covariate = cov_vector)
            run_rows[idx] = attrs
            idx += 1
        end

        avg_cost = mean(collection_costs)
        push!(optima_rows, (; covariate_id = cov_id,
                               optimal_cost = avg_cost,
                               run_count = length(collection_costs),
                               covariate_row(cov_vector)...))
    end

    runs_rows = NamedTuple[]
    for row in run_rows
        cov_vector = row.covariate
        push!(runs_rows, (; covariate_id = row.covariate_id,
                             run_id = row.run_id,
                             objective_value = row.objective_value,
                             covariate_row(cov_vector)...))
    end

    runs = DataFrame(runs_rows)
    optima = DataFrame(optima_rows)

    runs_path = joinpath(testing_dir, "saa_runs.csv")
    optima_path = joinpath(testing_dir, "saa_optima.csv")
    CSV.write(runs_path, runs)
    CSV.write(optima_path, optima)

    mark_step_complete(:compute_saa_baselines, output_dir)
    return nothing
end

function covariate_row(vec::AbstractVector)
    return (; (Symbol("x" * string(i)) => vec[i] for i in eachindex(vec))...)
end

end # module
