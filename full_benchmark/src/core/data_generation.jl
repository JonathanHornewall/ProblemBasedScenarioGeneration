module DataGeneration

using Random
using LinearAlgebra
using Statistics
using Distributions

include("../types/datasets.jl")

import ..Datasets: ScenarioBatch, TrainingPair, CovariateSet

export ProblemParameters,
       sample_problem_parameters,
       sample_covariates,
       build_covariate_set,
       generate_scenario_batch,
       generate_training_pairs,
       assemble_testing_tensor

"""
    struct ProblemParameters

Container for the structural parameters of the resource-allocation demand model.
"""
struct ProblemParameters
    ϕ::Vector{Float64}
    ζ::Matrix{Float64}
    Σ::Matrix{Float64}
    σ::Float64
    p::Float64
end

"""
    sample_problem_parameters(rng, J, L; σ, ω, p)

Replicates Tito et al.'s parameter-generation routine.
"""
function sample_problem_parameters(rng::AbstractRNG,
                                   J::Int,
                                   L::Int;
                                   σ::Real = 5.0,
                                   ω::Real = 1.0,
                                   p::Real = 2.0)
    ϕⱼ = 50 .+ 5 .* rand(rng, Normal(0, 1), J)
    ζⱼ₁ = 10 .+ rand(rng, Uniform(-4, 4), J)
    ζⱼ₂ = 5 .+ rand(rng, Uniform(-4, 4), J)
    ζⱼ₃ = 2 .+ rand(rng, Uniform(-4, 4), J)
    ζ = hcat(ζⱼ₁, ζⱼ₂, ζⱼ₃)
    Σ = generate_random_corr_matrix(rng, L)
    return ProblemParameters(ϕⱼ, ζ, Σ, float(σ), float(p))
end

"""
    generate_random_corr_matrix(rng, dim)

Draws a random correlation matrix following the routine used in `tito/resource_allocation/data.jl`
while avoiding console output.
"""
function generate_random_corr_matrix(rng::AbstractRNG, dim::Int)
    betaparam = 2.0
    partCorr = zeros(Float64, dim, dim)
    corrMat = Matrix{Float64}(I, dim, dim)
    for k in 1:dim-1
        for i in k+1:dim
            partCorr[k, i] = (rand(rng, Beta(betaparam, betaparam)) - 0.5) * 2.0
            p = partCorr[k, i]
            for j in (k-1):-1:1
                p = p * sqrt((1 - partCorr[j, i]^2) * (1 - partCorr[j, k]^2)) + partCorr[j, i] * partCorr[j, k]
            end
            corrMat[k, i] = p
            corrMat[i, k] = p
        end
    end
    permut = randperm(rng, dim)
    return corrMat[permut, permut]
end

"""
    sample_covariates(rng, count, params)

Samples `count` covariate vectors, each of length `size(params.Σ, 1)`, from an
absolute multivariate normal distribution.
"""
function sample_covariates(rng::AbstractRNG, count::Int, params::ProblemParameters)
    L = size(params.Σ, 1)
    μ = zeros(L)
    X = rand(rng, MvNormal(μ, params.Σ), count)
    return abs.(permutedims(X)) # contexts × L
end

"""
    build_covariate_set(xs; ids=nothing)

Wrap a matrix of covariates in a `CovariateSet`.
"""
function build_covariate_set(xs::AbstractMatrix{<:Real}; ids::Union{Nothing,AbstractVector}=nothing)
    n, L = size(xs)
    ids === nothing && (ids = collect(1:n))
    return CovariateSet(collect(ids), Array{Float64}(xs))
end

"""
    generate_scenario_batch(rng, covariate, params, scenario_count)

Generates `scenario_count` scenarios for a single covariate vector.
"""
function generate_scenario_batch(rng::AbstractRNG,
                                 covariate_id::Int,
                                 covariate::AbstractVector{<:Real},
                                 params::ProblemParameters,
                                 scenario_count::Int)
    J = length(params.ϕ)
    scenarios = Array{Float64}(undef, J, scenario_count)
    cov_vec = collect(covariate)
    for s in 1:scenario_count
        for j in 1:J
            ζⱼ = view(params.ζ, j, :)
            mean_val = params.ϕ[j] + sum(ζⱼ[l] * (cov_vec[l])^params.p for l in eachindex(cov_vec))
            scenarios[j, s] = mean_val + rand(rng, Normal(0, params.σ))
        end
    end
    return ScenarioBatch(covariate_id, cov_vec, scenarios)
end

"""
    assemble_testing_tensor(batches, collections_per_cov, scenarios_per_collection, J)

Stack scenario batches into a 4-D tensor with shape `(N_covariates, collections, scenarios, J)`.
"""
function assemble_testing_tensor(batches::Vector{ScenarioBatch},
                                 collections_per_cov::Int,
                                 scenarios_per_collection::Int,
                                 J::Int)
    n_covs = length(batches) ÷ collections_per_cov
    data_tensor = Array{Float64}(undef, n_covs, collections_per_cov, scenarios_per_collection, J)
    for (idx, batch) in enumerate(batches)
        cov_idx = div(idx - 1, collections_per_cov) + 1
        col_idx = mod(idx - 1, collections_per_cov) + 1
        if size(batch.scenarios, 2) != scenarios_per_collection
            throw(ArgumentError("Scenario batch size mismatch: expected $scenarios_per_collection, got $(size(batch.scenarios, 2))"))
        end
        if size(batch.scenarios, 1) != J
            throw(ArgumentError("Scenario batch dimension mismatch: expected $J products, got $(size(batch.scenarios, 1))"))
        end
        data_tensor[cov_idx, col_idx, :, :] = permutedims(batch.scenarios, (2, 1))
    end
    return data_tensor
end

"""
    generate_training_pairs(rng, covariate_set, params, scenarios_per_context)

Creates `TrainingPair` entries for the provided covariates.
"""
function generate_training_pairs(rng::AbstractRNG,
                                 covariate_set::CovariateSet,
                                 params::ProblemParameters,
                                 scenarios_per_context::Int)
    pairs = Vector{TrainingPair}(undef, length(covariate_set.ids))
    for (idx, cov_id) in enumerate(covariate_set.ids)
        batch = generate_scenario_batch(rng, cov_id, view(covariate_set.covariates, idx, :), params, scenarios_per_context)
        pairs[idx] = TrainingPair(cov_id, batch.covariate, batch.scenarios)
    end
    return pairs
end

end # module
