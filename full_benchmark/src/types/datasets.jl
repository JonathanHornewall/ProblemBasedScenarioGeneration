module Datasets

export CovariateSet, ScenarioBatch, TrainingPair

"""
    CovariateSet(ids, covariates)

Stores covariate identifiers alongside the matrix of covariate vectors.
The matrix is expected to have shape `(N, L)` where `N` is the number
of covariates and `L` the feature dimension.
"""
struct CovariateSet
    ids::Vector{Int}
    covariates::Matrix{Float64}
end

"""
    ScenarioBatch(covariate_id, covariate, scenarios)

Represents the scenario draws associated with a single covariate.
`scenarios` is a matrix of size `(J, S)` with `J` products and `S`
scenario samples.
"""
struct ScenarioBatch
    covariate_id::Int
    covariate::Vector{Float64}
    scenarios::Matrix{Float64}
end

"""
    TrainingPair(covariate_id, covariate, scenarios)

A thin alias of `ScenarioBatch` used for readability when working with
training data.
"""
struct TrainingPair
    covariate_id::Int
    covariate::Vector{Float64}
    scenarios::Matrix{Float64}
end

end # module
