module StepRunBenchmark

using CSV
using DataFrames
using Dates
using Serialization
using Statistics
using StatsPlots
using Optim
using PyCall
using CategoricalArrays
using Logging

include("../util/config.jl")
include("../util/artifacts.jl")

using .Config: ExperimentConfig
using .Artifacts: ensure_step_directories, mark_step_complete, write_json_file

push!(Base.LOAD_PATH, normpath(joinpath(@__DIR__, "..", "..", "src")))
import FullBenchmark

const REPO_ROOT = normpath(joinpath(@__DIR__, "..", "..", ".."))
const PROBLEM_PKG_PATH = normpath(joinpath(REPO_ROOT, "src", "julia"))

push!(Base.LOAD_PATH, PROBLEM_PKG_PATH)
using ProblemBasedScenarioGeneration

include(joinpath(REPO_ROOT, "scripts", "resource_allocation_prototype", "parameters.jl"))

const ENV["GKSwstype"] = "100"

export execute_run_benchmark

function execute_run_benchmark(config::ExperimentConfig, ctx::NamedTuple)
    output_dir = ctx.output_dir
    ensure_step_directories(output_dir, :run_benchmark)

    testing_dir = joinpath(output_dir, "artifacts", "testing")
    models_dir = joinpath(output_dir, "artifacts", "models")

    covariates_path = joinpath(testing_dir, "test_covariates.csv")
    scenarios_path = joinpath(testing_dir, "test_scenarios.jls")
    optima_path = joinpath(testing_dir, "saa_optima.csv")

    isfile(covariates_path) || error("Missing testing covariates at $(covariates_path)")
    isfile(scenarios_path) || error("Missing testing scenarios at $(scenarios_path)")
    isfile(optima_path) || error("Missing SAA optima at $(optima_path)")

    covariates_df = CSV.read(covariates_path, DataFrame)
    sort!(covariates_df, :id)

    scenario_tensor = Serialization.deserialize(scenarios_path)
    optima_df = CSV.read(optima_path, DataFrame)
    sort!(optima_df, :covariate_id)

    baselines_dir = joinpath(models_dir, "baselines")
    neural_dir = joinpath(models_dir, "neural")

    ls_model = load_model(joinpath(baselines_dir, "ls_model.jls"))
    er_model = load_model(joinpath(baselines_dir, "er_saa_model.jls"))
    cart_model = load_model(joinpath(baselines_dir, "cart_model.jls"))
    knn_model = load_model(joinpath(baselines_dir, "knn_model.jls"))
    nm_model = load_model(joinpath(baselines_dir, "nm_model.jls"))

    neural_model_path = joinpath(neural_dir, "neural_model_final.jls")
    isfile(neural_model_path) || error("Neural model not found at $(neural_model_path)")
    neural_model = ProblemBasedScenarioGeneration.load_trained_model(neural_model_path)

    Baselines = FullBenchmark.Baselines
    Baselines.ensure_tito_loaded()
    Baselines.L = config.feature_dim

    cart_tree = load_cart_tree(cart_model)

    problem_data = ProblemBasedScenarioGeneration.ResourceAllocationProblemData(μᵢⱼ,
                                                                                vec(cz),
                                                                                vec(qw),
                                                                                vec(ρᵢ))
    problem_instance = ProblemBasedScenarioGeneration.ResourceAllocationProblem(problem_data)

    methods = ["LS", "ER-SAA", "KNN", "CART", "M5+AD", "Nelder-Mead AD", "Neural"]
    rows = NamedTuple[]

    for row in eachrow(covariates_df)
        cov_id = row.id
        cov_vector = [row[Symbol("x$(i)")] for i in 1:config.feature_dim]
        optimum = optima_df[optima_df.covariate_id .== cov_id, :optimal_cost][1]
    method_costs = compute_method_costs(cov_vector,
                                        scenario_tensor,
                                        cov_id,
                                        Baselines,
                                        ls_model,
                                        er_model,
                                        cart_model,
                                        knn_model,
                                        nm_model,
                                        cart_tree,
                                        neural_model,
                                        problem_instance,
                                        config)
        for (method, cost) in method_costs
            gap = optimum == 0.0 ? 0.0 : (cost / optimum - 1.0) * 100
            push!(rows, (covariate_id = cov_id,
                         method = method,
                         evaluated_cost = cost,
                         gap_percent = gap))
        end
    end

    results_df = DataFrame(rows)
    results_dir = joinpath(output_dir, "artifacts", "results")
    mkpath(results_dir)
    CSV.write(joinpath(results_dir, "benchmark_gaps.csv"), results_df)

    summary = build_summary(results_df, methods)
    write_json_file(joinpath(results_dir, "benchmark_summary.json"), summary)

    plot_boxplot(results_df, methods, joinpath(results_dir, "gap_boxplot.png"))

    mark_step_complete(:run_benchmark, output_dir)
    return nothing
end

load_model(path) = Serialization.deserialize(path)

function compute_method_costs(cov_vector,
                              scenario_tensor,
                              cov_id,
                              Baselines,
                              ls_model,
                              er_model,
                              cart_model,
                              knn_model,
                              nm_model,
                              cart_tree,
                              neural_model,
                              problem_instance,
                              config)
    J = config.num_products
    I = size(μᵢⱼ, 1)

    ls_z = ls_decision(Baselines, ls_model, cov_vector)
    er_z = er_saa_decision(Baselines, er_model, cov_vector)
    knn_z = knn_decision(Baselines, knn_model, cov_vector)
    cart_z, m5_z = cart_decisions(Baselines, cart_model, cart_tree, cov_vector)
    nm_z = nm_decision(Baselines, nm_model, cov_vector)
    neural_z = neural_decision(neural_model, cov_vector, problem_instance, config.surrogate_parameter)

    ls_cost = evaluate_cost(ls_z, scenario_tensor, cov_id, J, I)
    er_cost = evaluate_cost(er_z, scenario_tensor, cov_id, J, I)
    knn_cost = evaluate_cost(knn_z, scenario_tensor, cov_id, J, I)
    cart_cost = evaluate_cost(cart_z, scenario_tensor, cov_id, J, I)
    m5_cost = evaluate_cost(m5_z, scenario_tensor, cov_id, J, I)
    nm_cost = evaluate_cost(nm_z, scenario_tensor, cov_id, J, I)
    neural_cost = evaluate_cost(neural_z, scenario_tensor, cov_id, J, I)

    return [
        "LS" => ls_cost,
        "ER-SAA" => er_cost,
        "KNN" => knn_cost,
        "CART" => cart_cost,
        "M5+AD" => m5_cost,
        "Nelder-Mead AD" => nm_cost,
        "Neural" => neural_cost
    ]
end

function ls_decision(Baselines, model, cov_vector)
    theta = model["theta"]
    y_hat = Baselines.pointPred(cov_vector, theta, size(theta, 1))
    return Baselines.kannanOpt(length(y_hat), size(μᵢⱼ, 1), y_hat, vec(cz), vec(qw), vec(ρᵢ), μᵢⱼ)
end

function er_saa_decision(Baselines, model, cov_vector)
    theta = model["theta"]
    residuals = model["residuals"]
    T = size(residuals, 2)
    y_hat = Baselines.pointPred(cov_vector, theta, size(theta, 1))
    y_matrix = repeat(reshape(y_hat, :, 1), 1, T) .+ residuals
    return Baselines.SAA_kannan(T, size(theta, 1), size(μᵢⱼ, 1), y_matrix, vec(cz), vec(qw), vec(ρᵢ), μᵢⱼ)
end

function knn_decision(Baselines, model, cov_vector)
    Xs = model["features"]
    Ys = model["responses"]
    k = model["k"]
    Xfresh = reshape(cov_vector, 1, :)
    y_knn = Baselines.generateDemandKNN(Xs, Ys, k, Xfresh)
    return Baselines.SAA_kannan(size(y_knn, 2), size(Ys, 1), size(μᵢⱼ, 1), y_knn, vec(cz), vec(qw), vec(ρᵢ), μᵢⱼ)
end

function cart_decisions(Baselines, model, tree, cov_vector)
    X_train = model["x_train"]
    y_train = model["y_train"]
    theta_init = model["theta_init"]

    getLeav = py"getLeav"(tree, cov_vector, X_train, y_train)

    cart_prediction = vec(Array(getLeav[3]))
    cart_z = Baselines.kannanOpt(length(cart_prediction), size(μᵢⱼ, 1), cart_prediction, vec(cz), vec(qw), vec(ρᵢ), μᵢⱼ)

    x_leaf = Array(getLeav[1])
    y_leaf = Array(getLeav[2])

    samples = size(x_leaf, 1)
    X_leaf = reshape(x_leaf, samples, size(x_leaf, 3))
    Y_leaf = reshape(y_leaf, samples, size(y_leaf, 3))'

    result = try
        Optim.optimize(θ -> Baselines.heuristicAD_par(θ,
                                                      samples,
                                                      Y_leaf,
                                                      X_leaf,
                                                      size(Y_leaf, 1),
                                                      size(μᵢⱼ, 1),
                                                      vec(cz),
                                                      vec(qw),
                                                      vec(ρᵢ),
                                                      μᵢⱼ),
                        theta_init,
                        Optim.NelderMead();
                        g_tol = 0.0,
                        f_tol = 1e-4)
    catch err
        @warn "M5+AD optimisation failed, falling back to CART coefficients" exception = (err, catch_backtrace())
        nothing
    end

    theta_m5 = result === nothing || !Optim.converged(result) ? theta_init : Optim.minimizer(result)
    y_hat = Baselines.pointPred(cov_vector, theta_m5, size(theta_m5, 1))
    m5_z = Baselines.kannanOpt(length(y_hat), size(μᵢⱼ, 1), y_hat, vec(cz), vec(qw), vec(ρᵢ), μᵢⱼ)

    return cart_z, m5_z
end

function nm_decision(Baselines, model, cov_vector)
    theta = model["theta"]
    y_hat = Baselines.pointPred(cov_vector, theta, size(theta, 1))
    return Baselines.kannanOpt(length(y_hat), size(μᵢⱼ, 1), y_hat, vec(cz), vec(qw), vec(ρᵢ), μᵢⱼ)
end

function neural_decision(model, cov_vector, problem_instance, surrogate_parameter)
    x = reshape(cov_vector, :, 1)
    ξ_hat = model(x)
    decision = ProblemBasedScenarioGeneration.surrogate_solution(problem_instance, surrogate_parameter, ξ_hat)
    return vec(decision)
end

function evaluate_cost(decision, scenario_tensor, cov_id, J, I)
    collections = size(scenario_tensor, 2)
    total = 0.0
    for c in 1:collections
        block = scenario_tensor[cov_id, c, :, :]
        scenario_matrix = permutedims(block, (2, 1))
        total += FullBenchmark.Baselines.solution_cost(decision,
                                                       size(scenario_matrix, 2),
                                                       J,
                                                       I,
                                                       scenario_matrix,
                                                       vec(cz),
                                                       vec(qw),
                                                       vec(ρᵢ),
                                                       μᵢⱼ)
    end
    return total / collections
end

function load_cart_tree(model)
    backend = model["serialization_backend"]
    path = model["tree_path"]
    if backend == "joblib"
        joblib = pyimport("joblib")
        return joblib.load(path)
    else
        pickle = pyimport("pickle")
        builtins = pyimport("builtins")
        io = builtins.open(path, "rb")
        tree = pickle.load(io)
        io.close()
        return tree
    end
end

function build_summary(results_df, methods)
    summary = Dict{String,Any}()
    for method in methods
        subset = results_df[results_df.method .== method, :]
        isempty(subset) && continue
        gaps = subset.gap_percent
        summary[method] = Dict(
            "mean_gap" => mean(gaps),
            "std_gap" => length(gaps) > 1 ? std(gaps) : 0.0,
            "min_gap" => minimum(gaps),
            "max_gap" => maximum(gaps),
            "median_gap" => median(gaps),
            "p25" => percentile(gaps, 25.0),
            "p75" => percentile(gaps, 75.0)
        )
    end
    return summary
end

function percentile(values, p)
    isempty(values) && return NaN
    return Statistics.quantile(values, p / 100)
end

function plot_boxplot(results_df, methods, path)
    ordered = filter(row -> row.method in methods, results_df)
    ordered.method = categorical(ordered.method, levels = methods)
    plt = @df ordered boxplot(:method, :gap_percent;
                              legend = false,
                              ylabel = "Percentage gap",
                              xlabel = "",
                              color = :darkblue,
                              grid = false)
    savefig(plt, path)
end

end # module
