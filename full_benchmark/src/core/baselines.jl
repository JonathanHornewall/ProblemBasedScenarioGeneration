module Baselines

using Random
using LinearAlgebra
using Statistics
using Logging

import ..Datasets

export ensure_tito_loaded,
       assemble_training_matrices,
       compute_residuals,
       default_knn_k,
       cart_available

const TITO_ROOT = normpath(joinpath(@__DIR__, "..", "..", "..", "tito", "resource_allocation"))
const _TITO_LOADED = Ref(false)
const CART_AVAILABLE = Ref(true)
const CART_ERROR = Ref{Union{Nothing,Tuple{Any,Any}}}(nothing)

"""
    ensure_tito_loaded()

Bring Tito et al.'s baseline implementations into scope. Subsequent calls are no-ops.
"""
function ensure_tito_loaded()
    _TITO_LOADED[] && return

    shim_path = normpath(joinpath(@__DIR__, "..", "..", "shims"))
    if !(shim_path in Base.LOAD_PATH)
        pushfirst!(Base.LOAD_PATH, shim_path)
    end

    include(joinpath(TITO_ROOT, "imports.jl"))
    include(joinpath(TITO_ROOT, "data.jl"))
    include(joinpath(TITO_ROOT, "LS.jl"))
    include(joinpath(TITO_ROOT, "residuals.jl"))
    include(joinpath(TITO_ROOT, "optimalityGap.jl"))
    include(joinpath(TITO_ROOT, "policy.jl"))
    include(joinpath(TITO_ROOT, "NM_para.jl"))

    try
        include(joinpath(TITO_ROOT, "CART.jl"))
        CART_AVAILABLE[] = true
        CART_ERROR[] = nothing
    catch err
        CART_AVAILABLE[] = false
        bt = catch_backtrace()
        CART_ERROR[] = (err, bt)
        @warn "CART baseline unavailable; falling back to linear surrogates" exception = (err, bt)
    end

    include(joinpath(TITO_ROOT, "kNN.jl"))

    _TITO_LOADED[] = true
    return nothing
end

"""
    assemble_training_matrices(pairs)

Flatten a collection of `TrainingPair` entries into the `(T × L)` feature matrix
and `(J × T)` demand matrix expected by Tito et al.'s baselines.
"""
function assemble_training_matrices(pairs::Vector{Datasets.TrainingPair})
    isempty(pairs) && error("No training data supplied for baselines.")

    feature_dim = length(pairs[1].covariate)
    J, scenario_count = size(pairs[1].scenarios)
    total_samples = length(pairs) * scenario_count

    X = Array{Float64}(undef, total_samples, feature_dim)
    Y = Array{Float64}(undef, J, total_samples)

    sample_idx = 0
    for pair in pairs
        size(pair.scenarios, 1) == J || error("Inconsistent product dimension in training data.")
        size(pair.scenarios, 2) == scenario_count || error("Inconsistent scenario count in training data.")
        for s in 1:scenario_count
            sample_idx += 1
            X[sample_idx, :] = pair.covariate
            Y[:, sample_idx] = pair.scenarios[:, s]
        end
    end

    return X, Y
end

"""
    cart_available()

Returns whether CART-based baselines were successfully loaded.
"""
function cart_available()
    ensure_tito_loaded()
    return CART_AVAILABLE[]
end

"""
    compute_residuals(theta, train_x, train_y)

Helper for deriving the in-sample residual matrix used by the ER-SAA baseline.
"""
function compute_residuals(theta::AbstractMatrix,
                           train_x::AbstractMatrix,
                           train_y::AbstractMatrix)
    ensure_tito_loaded()
    T = size(train_x, 1)
    J = size(train_y, 1)
    L = size(train_x, 2)
    forecast = forecast_cesgado(theta, train_x, L, J, T)
    return residuos(forecast, train_y)
end

"""
    default_knn_k(sample_count)

Replicates the heuristic used in Tito et al.'s experiments for choosing the neighbour
count while ensuring `k ≥ 1`.
"""
function default_knn_k(sample_count::Integer)
    sample_count < 1 && error("Sample count must be positive.")
    sample_count == 1 && return 1
    heuristic = Int(round(5 * (sample_count ^ 0.4)))
    k = clamp(heuristic, 1, sample_count - 1)
    return k
end

end # module
