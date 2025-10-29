module StepTrainBaselines

using Dates
using LinearAlgebra
using Optim
using PyCall
using Random
using Serialization
using Statistics

include("../util/config.jl")
include("../util/artifacts.jl")

using .Config: ExperimentConfig, seed_rng!
using .Artifacts: ensure_step_directories, mark_step_complete, write_json_file

push!(Base.LOAD_PATH, normpath(joinpath(@__DIR__, "..", "..", "src")))
import FullBenchmark

const Baselines = FullBenchmark.Baselines

const REPO_ROOT = normpath(joinpath(@__DIR__, "..", "..", ".."))
include(joinpath(REPO_ROOT, "scripts", "resource_allocation_prototype", "parameters.jl"))

const CART_PATH_HELPER = py"""
def dump_tree_artifact(obj, path):
    try:
        import joblib
        joblib.dump(obj, path)
        return "joblib"
    except ImportError:
        import pickle
        with open(path, "wb") as handle:
            pickle.dump(obj, handle)
        return "pickle"
"""

export execute_train_baselines

function execute_train_baselines(config::ExperimentConfig, ctx::NamedTuple)
    output_dir = ctx.output_dir
    ensure_step_directories(output_dir, :train_baselines)

    training_dir = joinpath(output_dir, "artifacts", "training")
    pairs_path = joinpath(training_dir, "training_pairs.jls")
    if !isfile(pairs_path)
        error("Training pairs not found at $(pairs_path). Generate training data first or supply input artifacts.")
    end

    Baselines.ensure_tito_loaded()

    pairs = Serialization.deserialize(pairs_path)::Vector{FullBenchmark.Datasets.TrainingPair}
    train_x, train_y = Baselines.assemble_training_matrices(pairs)

    T = size(train_x, 1)
    L = size(train_x, 2)
    J = size(train_y, 1)
    I = size(μᵢⱼ, 1)
    Baselines.L = L

    models_dir = joinpath(output_dir, "artifacts", "models", "baselines")
    mkpath(models_dir)

    ls_theta = Baselines.LS(train_y, train_x, J, L)
    residuals = Baselines.compute_residuals(ls_theta, train_x, train_y)

    er_model = Dict(
        "theta" => ls_theta,
        "residuals" => residuals,
        "samples" => T,
        "feature_dim" => L,
        "product_count" => J
    )

    knn_k = Baselines.default_knn_k(T)
    knn_model = Dict(
        "features" => train_x,
        "responses" => train_y,
        "k" => knn_k,
        "feature_dim" => L,
        "product_count" => J
    )

    cart_model, cart_metadata = train_cart_model(train_x, train_y, ls_theta, models_dir, config)

    nm_result = Optim.optimize(
        θ -> Baselines.heuristicAD_par(θ,
                                       T,
                                       train_y,
                                       train_x,
                                       J,
                                       I,
                                       vec(cz),
                                       vec(qw),
                                       vec(ρᵢ),
                                       μᵢⱼ),
        ls_theta,
        NelderMead();
        g_tol = 0.0,
        f_tol = 1e-4
    )

    nm_theta = Optim.minimizer(nm_result)
    nm_model = Dict(
        "theta" => nm_theta,
        "feature_dim" => L,
        "product_count" => J,
        "iterations" => Optim.iterations(nm_result),
        "converged" => Optim.converged(nm_result),
        "minimum" => Optim.minimum(nm_result)
    )

    ls_model = Dict(
        "theta" => ls_theta,
        "feature_dim" => L,
        "product_count" => J,
        "samples" => T
    )

    Serialization.serialize(joinpath(models_dir, "ls_model.jls"), ls_model)
    Serialization.serialize(joinpath(models_dir, "er_saa_model.jls"), er_model)
    Serialization.serialize(joinpath(models_dir, "cart_model.jls"), cart_model)
    Serialization.serialize(joinpath(models_dir, "knn_model.jls"), knn_model)
    Serialization.serialize(joinpath(models_dir, "nm_model.jls"), nm_model)

    write_json_file(joinpath(models_dir, "baseline_training_report.json"),
                    Dict(
                        "timestamp" => string(Dates.now()),
                        "training_samples" => T,
                        "feature_dim" => L,
                        "product_count" => J,
                        "knn" => Dict("k" => knn_k),
                        "cart" => cart_metadata,
                        "nelder_mead" => Dict(
                            "iterations" => Optim.iterations(nm_result),
                            "converged" => Optim.converged(nm_result),
                            "objective" => Optim.minimum(nm_result)
                        ),
                        "artifacts" => Dict(
                            "ls" => "ls_model.jls",
                            "er_saa" => "er_saa_model.jls",
                            "cart" => "cart_model.jls",
                            "knn" => "knn_model.jls",
                            "nelder_mead" => "nm_model.jls"
                        )
                    ))

    mark_step_complete(:train_baselines, output_dir)
    return nothing
end

function train_cart_model(train_x::AbstractMatrix,
                          train_y::AbstractMatrix,
                          ls_theta::AbstractMatrix,
                          models_dir::AbstractString,
                          config::ExperimentConfig)
    # prepare data for sklearn
    X = train_x
    y = permutedims(train_y, (2, 1)) # samples × products

    split = py"train_test_split"(X, y; test_size=0.2, random_state=config.training_covariate_seed)
    X_train = Array(split[1])
    X_test = Array(split[2])
    y_train = Array(split[3])
    y_test = Array(split[4])

    tree = py"getRegressor"(X_train, y_train, X_test, y_test)

    tree_path = joinpath(models_dir, "cart_tree.joblib")
    serialization_backend = String(py"dump_tree_artifact"(tree, tree_path))

    cart_model = Dict(
        "tree_path" => tree_path,
        "theta_init" => ls_theta,
        "x_train" => X_train,
        "y_train" => y_train,
        "feature_dim" => size(train_x, 2),
        "product_count" => size(train_y, 1),
        "min_samples_leaf" => 25,
        "serialization_backend" => serialization_backend
    )

    metadata = Dict(
        "tree_path" => tree_path,
        "backend" => serialization_backend,
        "train_size" => size(X_train, 1),
        "test_size" => size(X_test, 1),
        "min_samples_leaf" => 25
    )

    return cart_model, metadata
end

end # module
