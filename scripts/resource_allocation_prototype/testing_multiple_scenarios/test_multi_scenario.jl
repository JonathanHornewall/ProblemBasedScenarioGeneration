import Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()

using Random
using Statistics
using Dates
using Printf
using LinearAlgebra
using DataFrames
using CSV
using Flux
using ChainRulesCore
using Plots
using Serialization
using PrettyTables
using ProblemBasedScenarioGeneration
using ProblemBasedScenarioGeneration: ResourceAllocationProblemData, ResourceAllocationProblem, 
    dataGeneration, construct_neural_network, scenario_realization, scenario_collection_realization,
    TwoStageSLP, CanLP, optimal_value, s1_cost, diff_opt_b, diff_s1_cost, LogBarCanLP, LogBarCanLP_standard_solver
import ProblemBasedScenarioGeneration: loss, relative_loss, surrogate_solution, save_trained_model, load_trained_model

include(joinpath(@__DIR__, "..", "parameters.jl"))
include(joinpath(@__DIR__, "..", "custom_code", "neural_net.jl"))
include(joinpath(@__DIR__, "..", "tests_SAA", "test_function_SAA.jl"))

cz, qw, ρᵢ = vec(cz), vec(qw), vec(ρᵢ)

const DEFAULT_NTRAIN = 100
const DEFAULT_NTEST = 30
const DEFAULT_SIGMA = 5.0
const DEFAULT_P = 2
const DEFAULT_L = 3
const DEFAULT_N_XI_PER_X = 100
const DEFAULT_BATCHSIZE = 1
const DEFAULT_STEP_SIZE = 1e-3
const DEFAULT_EPOCHS = 10
const DEFAULT_NOISE_SCALE = 0.05
const DEFAULT_REPLICATES = 3
const DEFAULT_SCENARIO_COUNTS = (1, 2, 3)
const DEFAULT_BASE_SEED = 42

const ANNEALING_PARAMS = [1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.08, 0.06, 0.04, 0.02, 0.01]
const ANNEALING_STEP_SIZES = fill(DEFAULT_STEP_SIZE, length(ANNEALING_PARAMS) + 1)

const CACHE_ROOT = joinpath(@__DIR__, "cached_values")
const SAA_CACHE_DIR = joinpath(CACHE_ROOT, "saa_preprocessing")
const DATA_CACHE_DIR = joinpath(CACHE_ROOT, "datasets")
const TRAIN_DATA_CACHE_DIR = joinpath(DATA_CACHE_DIR, "training")
const TEST_DATA_CACHE_DIR = joinpath(DATA_CACHE_DIR, "testing")

function ensure_cache_directories()
    for dir in (SAA_CACHE_DIR, TRAIN_DATA_CACHE_DIR, TEST_DATA_CACHE_DIR)
        mkpath(dir)
    end
end

function sanitize_float(x::Real)
    return replace(@sprintf("%0.2f", x), "." => "_")
end

function training_dataset_cache_path(seed::Int, trial_index::Int;
        Ntrain::Int, Ntest::Int, N_xi_per_x::Int, σ::Float64, p::Int, L::Int)
    ensure_cache_directories()
    return joinpath(TRAIN_DATA_CACHE_DIR, @sprintf(
        "train_seed_%d_trial_%02d_Ntrain_%d_Ntest_%d_Nxi_%d_sigma_%s_p_%d_L_%d.jls",
        seed, trial_index, Ntrain, Ntest, N_xi_per_x, sanitize_float(σ), p, L))
end

function testing_dataset_cache_path(seed::Int, trial_index::Int;
        Ntest::Int, N_xi_per_x::Int, σ::Float64, p::Int, L::Int)
    ensure_cache_directories()
    return joinpath(TEST_DATA_CACHE_DIR, @sprintf(
        "test_seed_%d_trial_%02d_Ntest_%d_Nxi_%d_sigma_%s_p_%d_L_%d.jls",
        seed, trial_index, Ntest, N_xi_per_x, sanitize_float(σ), p, L))
end

function legacy_dataset_cache_path(seed::Int, trial_index::Int;
        Ntrain::Int, Ntest::Int, N_xi_per_x::Int, σ::Float64, p::Int, L::Int)
    return joinpath(DATA_CACHE_DIR, @sprintf(
        "dataset_seed_%d_trial_%02d_Ntrain_%d_Ntest_%d_Nxi_%d_sigma_%s_p_%d_L_%d.jls",
        seed, trial_index, Ntrain, Ntest, N_xi_per_x, sanitize_float(σ), p, L))
end

function saa_cache_path(seed::Int, trial_index::Int; Ntest::Int, N_xi_per_x::Int, σ::Float64, p::Int, L::Int)
    ensure_cache_directories()
    return joinpath(SAA_CACHE_DIR, @sprintf(
        "saa_test_seed_%d_trial_%02d_Ntest_%d_Nxi_%d_sigma_%s_p_%d_L_%d.jls",
        seed, trial_index, Ntest, N_xi_per_x, sanitize_float(σ), p, L))
end

function find_legacy_saa_cache(seed::Int, trial_index::Int; Ntest::Int, N_xi_per_x::Int, σ::Float64, p::Int, L::Int)
    !isdir(SAA_CACHE_DIR) && return nothing
    target = @sprintf("saa_seed_%d_", seed)
    suffix = @sprintf("_trial_%02d_Ntest_%d_Nxi_%d_sigma_%s_p_%d_L_%d_jls",
        trial_index, Ntest, N_xi_per_x, sanitize_float(σ), p, L)
    for filename in readdir(SAA_CACHE_DIR)
        if startswith(filename, target) && endswith(filename, suffix)
            return joinpath(SAA_CACHE_DIR, filename)
        end
    end
    return nothing
end


default_output_root() = joinpath(@__DIR__, "results", "run_" * Dates.format(now(), "yyyymmdd_HHMMSS"))

function parse_cli_args(args::Vector{String})
    options = Dict{String, String}()
    for arg in args
        if startswith(arg, "--")
            trimmed = arg[3:end]
            parts = split(trimmed, "=", limit=2)
            if length(parts) == 2
                options[parts[1]] = parts[2]
            else
                options[parts[1]] = "true"
            end
        end
    end
    return options
end

function parse_int_option(options::Dict{String, String}, key::String, default::Int)
    return haskey(options, key) ? parse(Int, options[key]) : default
end

function parse_float_option(options::Dict{String, String}, key::String, default::Float64)
    return haskey(options, key) ? parse(Float64, options[key]) : default
end

function parse_bool_option(options::Dict{String, String}, key::String, default::Bool)
    return haskey(options, key) ? (lowercase(options[key]) == "true") : default
end

function parse_int_tuple(options::Dict{String, String}, key::String, default::Tuple{Vararg{Int}})
    if !haskey(options, key)
        return default
    end
    raw = options[key]
    values = split(raw, ",")
    return Tuple(parse.(Int, values))
end

function perturb_training_data(data::Dict, noise_scale::Float64)
    isempty(data) && return Dict{keytype(data), valtype(data)}()
    perturbed = Dict{keytype(data), valtype(data)}()
    for (x, ξ) in data
        scale = max(mean(abs.(ξ)), 1.0)
        noise = noise_scale * scale .* randn(Float64, size(ξ)...) 
        perturbed[x] = ξ .+ noise
    end
    return perturbed
end

function dictionary_like(data)
    if data isa Dict
        return data
    end
    return Dict(data)
end

function load_or_generate_training_dataset(problem_instance, seed::Int, trial_index::Int;
        Ntrain::Int = DEFAULT_NTRAIN, Ntest::Int = DEFAULT_NTEST, N_xi_per_x::Int = DEFAULT_N_XI_PER_X,
        σ::Float64 = DEFAULT_SIGMA, p::Int = DEFAULT_P, L::Int = DEFAULT_L)

    cache_file = training_dataset_cache_path(seed, trial_index; Ntrain = Ntrain, Ntest = Ntest,
        N_xi_per_x = N_xi_per_x, σ = σ, p = p, L = L)

    if isfile(cache_file)
        println("Loading cached training dataset from $(cache_file)")
        training_data = Serialization.deserialize(cache_file)
        return dictionary_like(training_data)
    end

    legacy_file = legacy_dataset_cache_path(seed, trial_index; Ntrain = Ntrain, Ntest = Ntest,
        N_xi_per_x = N_xi_per_x, σ = σ, p = p, L = L)
    if isfile(legacy_file)
        println("Converting legacy dataset cache from $(legacy_file)")
        legacy_training, legacy_testing = Serialization.deserialize(legacy_file)
        training_data = dictionary_like(legacy_training)
        Serialization.serialize(cache_file, training_data)
        candidate_test_cache = testing_dataset_cache_path(seed, trial_index; Ntest = Ntest,
            N_xi_per_x = N_xi_per_x, σ = σ, p = p, L = L)
        if !isfile(candidate_test_cache)
            Serialization.serialize(candidate_test_cache, dictionary_like(legacy_testing))
            println("Saved converted testing dataset cache to $(candidate_test_cache)")
        end
        return training_data
    end

    println("Generating training dataset for trial $(trial_index) with seed $(seed)")
    Random.seed!(seed)
    training_data_raw, _ = dataGeneration(problem_instance, Ntrain, Ntest, N_xi_per_x, σ, p, L)
    training_data = dictionary_like(training_data_raw)

    Serialization.serialize(cache_file, training_data)
    println("Saved training dataset cache to $(cache_file)")
    return training_data
end

function load_or_generate_testing_dataset(problem_instance, seed::Int, trial_index::Int;
        Ntrain::Int = DEFAULT_NTRAIN, Ntest::Int = DEFAULT_NTEST, N_xi_per_x::Int = DEFAULT_N_XI_PER_X,
        σ::Float64 = DEFAULT_SIGMA, p::Int = DEFAULT_P, L::Int = DEFAULT_L)

    cache_file = testing_dataset_cache_path(seed, trial_index; Ntest = Ntest,
        N_xi_per_x = N_xi_per_x, σ = σ, p = p, L = L)

    if isfile(cache_file)
        println("Loading cached testing dataset from $(cache_file)")
        testing_data = Serialization.deserialize(cache_file)
        return dictionary_like(testing_data)
    end

    legacy_file = legacy_dataset_cache_path(seed, trial_index; Ntrain = Ntrain, Ntest = Ntest,
        N_xi_per_x = N_xi_per_x, σ = σ, p = p, L = L)
    if isfile(legacy_file)
        println("Converting legacy dataset cache from $(legacy_file)")
        legacy_training, legacy_testing = Serialization.deserialize(legacy_file)
        testing_data = dictionary_like(legacy_testing)
        Serialization.serialize(cache_file, testing_data)
        candidate_train_cache = training_dataset_cache_path(seed, trial_index; Ntrain = Ntrain, Ntest = Ntest,
            N_xi_per_x = N_xi_per_x, σ = σ, p = p, L = L)
        if !isfile(candidate_train_cache)
            Serialization.serialize(candidate_train_cache, dictionary_like(legacy_training))
            println("Saved converted training dataset cache to $(candidate_train_cache)")
        end
        return testing_data
    end

    println("Generating testing dataset for trial $(trial_index) with seed $(seed)")
    Random.seed!(seed)
    _, testing_data_raw = dataGeneration(problem_instance, Ntrain, Ntest, N_xi_per_x, σ, p, L)
    testing_data = dictionary_like(testing_data_raw)

    Serialization.serialize(cache_file, testing_data)
    println("Saved testing dataset cache to $(cache_file)")
    return testing_data
end

function extract_dataset_arrays(dataset::Dict)
    xs = collect(keys(dataset))
    xis = collect(values(dataset))
    return xs, xis
end

function batch_loss(loss_fn, model, Xb, Ξb)
    nbatch = size(Xb, 2)
    preds = model(Xb)
    return (1 / nbatch) * sum(loss_fn(preds[:, i:i], Ξb[:, i:i]) for i in 1:nbatch)
end

function batch_relative_loss(relative_loss_fn, model, Xb, Ξb)
    nbatch = size(Xb, 2)
    preds = model(Xb)
    return (1 / nbatch) * sum(relative_loss_fn(preds[:, i:i], Ξb[:, i:i]) for i in 1:nbatch)
end

function compute_opt_cost(two_slp, A, b, c, Ws_list, Ts_list, hs_list, qs_list)
    can_lp = CanLP(two_slp)
    margins = (1e-7, 5e-7, 1e-6, 5e-6, 1e-5)
    for margin in margins
        solver = instance -> solve_canonical_lp(instance; feasibility_margin = margin)
        println(@sprintf("    [Benchmark] Attempting canonical LP with feasibility_margin=%.1e", margin))
        try
            return optimal_value(can_lp, solver; feasibility_margin = margin)
        catch err
            if err isa ErrorException && (occursin("Infeasible: max |Ax - b|", err.msg) || occursin("Decision is not feasible", err.msg))
                println(@sprintf("    [Benchmark] Canonical LP failed (margin=%.1e): %s", margin, err.msg))
                continue
            else
                rethrow(err)
            end
        end
    end
    error("All canonical LP attempts failed: GLPK reported infeasibility beyond permitted tolerances")
end

function train_stage!(model, dataset::Dict, problem_instance, reg_param_surr, reg_param_prim;
        step_size::Float64, batchsize::Int, epochs::Int, stage_label::String)

    xs, xis = extract_dataset_arrays(dataset)
    N = length(xs)
    N == 0 && return Float64[]

    opt = Adam(step_size)
    epoch_history = Float64[]
    total_batches = ceil(Int, N / batchsize)

    for epoch in 1:epochs
        println(@sprintf("    [%s] Epoch %d/%d (reg_param_surr=%.3f, reg_param_prim=%.3f)",
                stage_label, epoch, epochs, reg_param_surr, reg_param_prim))
        state = Flux.setup(opt, model)
        batch_losses = Float64[]
        batch_counter = 0
        loss_fn(ξ_output, ξ_actual) = loss(problem_instance, reg_param_surr, reg_param_prim, ξ_output, ξ_actual)
        relative_loss_fn(ξ_output, ξ_actual) = relative_loss(problem_instance, reg_param_surr, reg_param_prim, ξ_output, ξ_actual)

        for idxs in Iterators.partition(1:N, batchsize)
            batch_counter += 1
            Xb = hcat(xs[idxs]...)
            Ξb = hcat(xis[idxs]...)
            gs = Flux.gradient(model) do m
                batch_loss(loss_fn, m, Xb, Ξb)
            end
            gmodel = gs isa Tuple ? gs[1] : gs
            Flux.update!(state, model, gmodel)

            batch_rel = batch_relative_loss(relative_loss_fn, model, Xb, Ξb)
            push!(batch_losses, batch_rel)

            if epoch == 1
                println(@sprintf("      [%s] Processed batch %d/%d (rel-loss = %.4f)", stage_label, batch_counter, total_batches, batch_rel))
            elseif batch_counter == total_batches
                println(@sprintf("      [%s] Completed batch %d/%d", stage_label, batch_counter, total_batches))
            end
        end

        avg_epoch_loss = isempty(batch_losses) ? 0.0 : mean(batch_losses)
        push!(epoch_history, avg_epoch_loss)
        println(@sprintf("    [%s] Epoch %d average relative loss = %.4f", stage_label, epoch, avg_epoch_loss))
        GC.gc()
    end

    return epoch_history
end

function plot_training_history(history::Vector{Float64}, output_path::String; stage_markers=nothing)
    if isempty(history)
        println("No training history to plot for $(output_path)")
        return
    end
    epochs = 1:length(history)
    plt = plot(epochs, history;
        xlabel="Epoch",
        ylabel="Average relative loss",
        title="Training progression",
        legend=false,
        linewidth=2,
        marker=:circle)
    if stage_markers !== nothing
        ymax = maximum(history)
        for (label, pos) in stage_markers
            vline!(plt, [pos]; color=:gray, linestyle=:dash, linewidth=1)
            annotate!(plt, pos, ymax, Plots.text(label, 8, :left))
        end
    end
    savefig(plt, output_path)
    println("Saved training plot to $(output_path)")
end

function plot_performance_summary(clean_eval, noisy_eval, output_dir::String; title_suffix::String="")
    ensure_directory(output_dir)
    metrics = [(:mean_gap, "Mean Gap"), (:mean_relative_gap, "Mean Relative Gap"), (:mean_opt_cost, "Mean Optimal Cost")]
    labels = ["clean", "perturbed"]
    for (field, label) in metrics
        values = [getfield(clean_eval, field), getfield(noisy_eval, field)]
        plt = bar(labels, values;
            xlabel="Model",
            ylabel=label,
            title=isempty(title_suffix) ? label : string(label, " - ", title_suffix),
            legend=false)
        filename = joinpath(output_dir, string(Symbol(field), "_comparison.png"))
        savefig(plt, filename)
        println("Saved performance plot to $(filename)")
    end
end

function preprocess_testing_data(problem_instance, dataset_testing::Dict, N_xi_per_x::Int;
        cache_path::Union{Nothing,String}=nothing, metadata=nothing, legacy_seed::Union{Nothing,Int}=nothing,
        trial_index::Union{Nothing,Int}=nothing, Ntest::Union{Nothing,Int}=nothing, σ::Float64=DEFAULT_SIGMA,
        p::Int=DEFAULT_P, L::Int=DEFAULT_L)

    if cache_path !== nothing && isfile(cache_path)
        println("Loading cached SAA preprocessing from $(cache_path)")
        payload = Serialization.deserialize(cache_path)
        return payload["data"]
    elseif cache_path !== nothing && legacy_seed !== nothing && trial_index !== nothing && Ntest !== nothing
        legacy_path = find_legacy_saa_cache(legacy_seed, trial_index; Ntest = Ntest, N_xi_per_x = N_xi_per_x, σ = σ, p = p, L = L)
        if legacy_path !== nothing && isfile(legacy_path)
            println("Loading legacy cached SAA preprocessing from $(legacy_path)")
            payload = Serialization.deserialize(legacy_path)
            # Re-save under new cache path for future runs
            if cache_path !== nothing
                ensure_directory(dirname(cache_path))
                Serialization.serialize(cache_path, payload)
                println("Mirrored legacy cache to $(cache_path)")
            end
            return payload["data"]
        end
    end

    println("Preprocessing testing dataset for benchmarking...")
    A, b, c = problem_instance.s1_constraint_matrix, problem_instance.s1_constraint_vector, problem_instance.s1_cost_vector
    preprocessed = NamedTuple{(:x_mat, :scenarios), Tuple{Matrix{Float64}, Vector{Tuple{Any, Float64}}}}[]

    samples = collect(dataset_testing)
    total_samples = length(samples)
    sample_start = time()

    for (sample_idx, (x, ξ_tensor)) in enumerate(samples)
        println(@sprintf("  [Benchmark] Sample %d/%d with %d scenario blocks", sample_idx, total_samples, size(ξ_tensor, 1)))
        x_mat = reshape(x, :, 1)
        scenario_evals = Tuple{Any, Float64}[]
        scenario_count = size(ξ_tensor, 1)
        sample_limit = min(N_xi_per_x, size(ξ_tensor, 2))

        for m in 1:scenario_count
            if sample_idx == 1
                println(@sprintf("    [Benchmark] Building scenario %d/%d for first sample", m, scenario_count))
            end
            Ws_list, Ts_list, hs_list, qs_list = Any[], Any[], Any[], Any[]
            for k in 1:sample_limit
                scenario_vec = vec(ξ_tensor[m, k, :])
                W, T, h, q = scenario_realization(problem_instance, scenario_vec)
                push!(Ws_list, W)
                push!(Ts_list, T)
                push!(hs_list, h)
                push!(qs_list, q)
            end

            Ws = cat(Ws_list...; dims=3)
            Ts = cat(Ts_list...; dims=3)
            hs = hcat(hs_list...)
            qs = hcat(qs_list...)

            two_slp = TwoStageSLP(A, b, c, Ws, Ts, hs, qs)
            opt_cost = compute_opt_cost(two_slp, A, b, c, Ws_list, Ts_list, hs_list, qs_list)
            push!(scenario_evals, (two_slp, opt_cost))
        end

        push!(preprocessed, (x_mat = x_mat, scenarios = scenario_evals))
    end

    println(@sprintf("Finished preprocessing %d samples in %.2f seconds", total_samples, time() - sample_start))

    if cache_path !== nothing
        ensure_directory(dirname(cache_path))
        payload = Dict(
            "metadata" => metadata,
            "data" => preprocessed
        )
        Serialization.serialize(cache_path, payload)
        println("Saved SAA preprocessing cache to $(cache_path)")
    end

    return preprocessed
end

function compute_average_training_loss(model, dataset::Dict, problem_instance, reg_param_surr, reg_param_prim)
    losses = Float64[]
    for (x, ξ) in dataset
        x_mat = reshape(x, :, 1)
        ξ_mat = reshape(ξ, :, 1)
        pred = model(x_mat)
        push!(losses, loss(problem_instance, reg_param_surr, reg_param_prim, pred, ξ_mat))
    end
    return mean(losses)
end

function evaluate_model(problem_instance, model, preprocessed_testing, reg_param_surr, reg_param_ref; label::String="")
    gaps = Float64[]
    rel_gaps = Float64[]
    abs_gaps = Float64[]
    opt_costs = Float64[]
    model_costs = Float64[]
    scenario_rows = NamedTuple{(:sample_index, :scenario_index, :x1, :x2, :x3, :model_cost, :saa_cost, :absolute_gap, :relative_gap),
        Tuple{Int, Int, Float64, Float64, Float64, Float64, Float64, Float64, Float64}}[]
    sample_rows = NamedTuple{(:sample_index, :x1, :x2, :x3, :mean_relative_gap, :std_relative_gap, :mean_model_cost, :mean_saa_cost, :mean_absolute_gap, :worst_relative_gap),
        Tuple{Int, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64}}[]

    total_samples = length(preprocessed_testing)
    display_label = isempty(label) ? "model" : label
    println(@sprintf("Evaluating %s on test set (%d samples)", display_label, total_samples))
    eval_start = time()

    for (sample_idx, sample) in enumerate(preprocessed_testing)
        if sample_idx == 1 || sample_idx == total_samples || sample_idx % max(1, Int(floor(total_samples / 5))) == 0
            println(@sprintf("  [%s] Sample %d/%d", display_label, sample_idx, total_samples))
        end
        x_mat = sample.x_mat
        ξ_hat = model(x_mat)
        surrogate_decision = surrogate_solution(problem_instance, reg_param_surr, ξ_hat)
        x_vec = vec(x_mat)

        sample_rel_gaps = Float64[]
        sample_abs_gaps = Float64[]
        sample_model_costs = Float64[]
        sample_opt_costs = Float64[]

        scenario_total = length(sample.scenarios)
        for (scenario_idx, (two_slp, opt_cost)) in enumerate(sample.scenarios)
            if sample_idx == 1 && (scenario_idx == 1 || scenario_idx == scenario_total || scenario_idx % max(1, Int(floor(scenario_total / 5))) == 0)
                println(@sprintf("    [%s] Scenario %d/%d (sample %d)", display_label, scenario_idx, scenario_total, sample_idx))
            end
            evaluated_cost = s1_cost(two_slp, surrogate_decision, reg_param_ref)
            gap = evaluated_cost - opt_cost
            rel_gap = gap / max(abs(opt_cost), eps())

            push!(gaps, gap)
            push!(rel_gaps, rel_gap)
            abs_gap = abs(gap)
            push!(abs_gaps, abs_gap)
            push!(opt_costs, opt_cost)
            push!(model_costs, evaluated_cost)

            push!(sample_rel_gaps, rel_gap)
            push!(sample_abs_gaps, abs_gap)
            push!(sample_model_costs, evaluated_cost)
            push!(sample_opt_costs, opt_cost)

            push!(scenario_rows, (sample_idx, scenario_idx, x_vec[1], x_vec[2], x_vec[3], evaluated_cost, opt_cost, gap, rel_gap))
        end

        if !isempty(sample_rel_gaps)
            mean_rel = mean(sample_rel_gaps)
            std_rel = length(sample_rel_gaps) > 1 ? std(sample_rel_gaps; corrected = false) : 0.0
            mean_model_cost = mean(sample_model_costs)
            mean_opt_cost = mean(sample_opt_costs)
            mean_abs_gap = mean(sample_abs_gaps)
            worst_rel = maximum(sample_rel_gaps)
            push!(sample_rows, (sample_idx, x_vec[1], x_vec[2], x_vec[3], mean_rel, std_rel, mean_model_cost, mean_opt_cost, mean_abs_gap, worst_rel))
            if sample_idx == 1 || sample_idx == total_samples || sample_idx % max(1, Int(floor(total_samples / 5))) == 0
                println(@sprintf("  [%s] Sample %d average relative gap = %.6f (std = %.6f)",
                    display_label, sample_idx, mean_rel, std_rel))
            end
        end
    end

    println(@sprintf("Completed evaluation for %s in %.2f seconds", display_label, time() - eval_start))

    scenario_df = DataFrame(scenario_rows)
    sample_df = DataFrame(sample_rows)

    overall_mean_gap = isempty(gaps) ? NaN : mean(gaps)
    overall_mean_rel = isempty(rel_gaps) ? NaN : mean(rel_gaps)
    overall_mean_opt = isempty(opt_costs) ? NaN : mean(opt_costs)
    overall_mean_model = isempty(model_costs) ? NaN : mean(model_costs)
    overall_mean_abs = isempty(abs_gaps) ? NaN : mean(abs_gaps)
    overall_std_rel = length(rel_gaps) > 1 ? std(rel_gaps; corrected = false) : 0.0
    worst_sample_rel = isempty(sample_rows) ? NaN : maximum(getfield.(sample_rows, :worst_relative_gap))

    return (; mean_gap = overall_mean_gap,
            mean_relative_gap = overall_mean_rel,
            mean_opt_cost = overall_mean_opt,
            mean_model_cost = overall_mean_model,
            mean_absolute_gap = overall_mean_abs,
            overall_relative_gap_std = overall_std_rel,
            worst_sample_relative_gap = worst_sample_rel,
            scenario_details = scenario_df,
            sample_summary = sample_df)
end

function train_with_annealing!(model, dataset::Dict, problem_instance;
        step_sizes::AbstractVector{<:Real} = ANNEALING_STEP_SIZES,
        batchsize::Int = DEFAULT_BATCHSIZE, default_epochs::Int = DEFAULT_EPOCHS,
        save_dir::String, resume::Bool = false)
    reg_param_ref = 0.0
    total_stages = length(ANNEALING_PARAMS) + 1
    epoch_list = fill(default_epochs, total_stages)
    epoch_list[1] = 20

    step_schedule = Float64.(collect(step_sizes))
    if length(step_schedule) != total_stages
        error("Expected $(total_stages) step sizes (including final stage); got $(length(step_schedule))")
    end

    stage_specs = Vector{NamedTuple{(:index, :label, :reg_param_surr, :reg_param_prim, :epochs, :step_size, :model_path, :history_path),
        Tuple{Int, String, Float64, Float64, Int, Float64, String, String}}}()
    for (idx, reg_param_surr) in enumerate(ANNEALING_PARAMS)
        stage_label = @sprintf("stage_%02d", idx)
        model_path = joinpath(save_dir, @sprintf("stage_%02d_model.jls", idx))
        history_path = joinpath(save_dir, "$(stage_label)_history.jls")
        push!(stage_specs, (; index = idx,
            label = stage_label,
            reg_param_surr = reg_param_surr,
            reg_param_prim = reg_param_surr,
            epochs = epoch_list[idx],
            step_size = step_schedule[idx],
            model_path,
            history_path))
    end

    final_index = total_stages
    final_label = @sprintf("stage_%02d_final", final_index)
    final_model_path = joinpath(save_dir, @sprintf("stage_%02d_model.jls", final_index))
    final_history_path = joinpath(save_dir, "$(final_label)_history.jls")
    push!(stage_specs, (; index = final_index,
        label = final_label,
        reg_param_surr = ANNEALING_PARAMS[end],
        reg_param_prim = 0.0,
        epochs = epoch_list[end],
        step_size = step_schedule[end],
        model_path = final_model_path,
        history_path = final_history_path))

    training_history = Float64[]
    stage_markers = Tuple{String, Int}[]
    training_log = NamedTuple{(:stage_label, :stage_index, :epoch, :reg_param_surr, :reg_param_prim, :step_size, :relative_loss),
        Tuple{String, Int, Int, Float64, Float64, Float64, Float64}}[]

    for spec in stage_specs
        model_path = spec.model_path
        history_path = spec.history_path
        if resume && isfile(model_path)
            println(@sprintf("  [%s] Checkpoint detected; loading weights from %s", spec.label, model_path))
            loaded_model = load_trained_model(model_path)
            model = loaded_model

            stage_history = Float64[]
            if isfile(history_path)
                try
                    stage_history = Serialization.deserialize(history_path)
                catch err
                    println(@sprintf("  [%s] Warning: failed to load history from %s (%s); continuing without history",
                        spec.label, history_path, err))
                    stage_history = Float64[]
                end
            else
                println(@sprintf("  [%s] No history file found at %s; continuing with empty history", spec.label, history_path))
            end

            append!(training_history, stage_history)
            push!(stage_markers, (spec.label, length(training_history)))
            if !isempty(stage_history)
                for (epoch_idx, epoch_loss) in enumerate(stage_history)
                    push!(training_log, (spec.label, spec.index, epoch_idx, spec.reg_param_surr,
                        spec.reg_param_prim, spec.step_size, epoch_loss))
                end
            end
            continue
        end

        println(@sprintf("  Starting %s with reg_param_surr=%.3f, reg_param_prim=%.3f, step_size=%.3e (%d epochs)",
            spec.label, spec.reg_param_surr, spec.reg_param_prim, spec.step_size, spec.epochs))
        stage_history = train_stage!(model, dataset, problem_instance, spec.reg_param_surr, spec.reg_param_prim;
            step_size = spec.step_size, batchsize = batchsize, epochs = spec.epochs, stage_label = spec.label)
        append!(training_history, stage_history)
        push!(stage_markers, (spec.label, length(training_history)))

        for (epoch_idx, epoch_loss) in enumerate(stage_history)
            push!(training_log, (spec.label, spec.index, epoch_idx, spec.reg_param_surr,
                spec.reg_param_prim, spec.step_size, epoch_loss))
        end

        try
            Serialization.serialize(history_path, stage_history)
            println(@sprintf("  Saved stage history to %s", history_path))
        catch err
            println(@sprintf("  [%s] Warning: failed to write history to %s (%s)", spec.label, history_path, err))
        end

        save_trained_model(model, model_path)
        println("  Saved model checkpoint to $(model_path)")
    end

    final_spec = stage_specs[end]
    return (; reg_param_prim = final_spec.reg_param_prim,
            reg_param_ref = reg_param_ref,
            final_reg_param_surr = final_spec.reg_param_surr,
            training_history = training_history,
            stage_markers = stage_markers,
            training_history_log = training_log,
            step_size_schedule = step_schedule,
            model = model)
end

function create_problem_instance()
    problem_data = ResourceAllocationProblemData(μᵢⱼ, cz, qw, ρᵢ)
    return ResourceAllocationProblem(problem_data)
end

function ensure_directory(path::String)
    mkpath(path)
    return path
end

function ensure_table_directories(base_dir::String, variant::String)
    variant_dir = ensure_directory(joinpath(base_dir, variant))
    data_dir = ensure_directory(joinpath(variant_dir, "data"))
    formatted_dir = ensure_directory(joinpath(variant_dir, "formatted"))
    return (variant_dir, data_dir, formatted_dir)
end

function save_dataframe_markdown(df::DataFrame, path::String; title::Union{Nothing,String}=nothing, max_rows::Union{Nothing,Int}=nothing)
    display_df = df
    truncated = false
    if max_rows !== nothing && nrow(df) > max_rows
        display_df = first(df, max_rows)
        truncated = true
    end
    open(path, "w") do io
        if title !== nothing
            println(io, title)
            println(io)
        end
        pretty_table(io, display_df; backend = :markdown)
        if truncated
            println(io)
            println(io, "(showing first $(max_rows) of $(nrow(df)) rows)")
        end
    end
end

function run_single_trial(problem_instance, scenario_count::Int, trial_index::Int;
        training_data::Dict, testing_data::Dict, preprocessed_testing,
        Ntrain::Int = DEFAULT_NTRAIN, Ntest::Int = DEFAULT_NTEST, N_xi_per_x::Int = DEFAULT_N_XI_PER_X,
        σ::Float64 = DEFAULT_SIGMA, p::Int = DEFAULT_P, L::Int = DEFAULT_L, noise_scale::Float64 = DEFAULT_NOISE_SCALE,
        step_size::Float64 = DEFAULT_STEP_SIZE, batchsize::Int = DEFAULT_BATCHSIZE, default_epochs::Int = DEFAULT_EPOCHS,
        output_dir::String, train_data_seed::Int, test_data_seed::Int, training_seed::Int,
        resume::Bool = false)

    Random.seed!(training_seed)

    perturbed_training_data = perturb_training_data(training_data, noise_scale)

    model_clean = construct_neural_network(problem_instance; nr_of_scenarios = scenario_count)
    model_noisy = construct_neural_network(problem_instance; nr_of_scenarios = scenario_count)

    trial_dir = ensure_directory(joinpath(output_dir, @sprintf("scenarios_%d", scenario_count), @sprintf("trial_%02d", trial_index)))
    clean_dir = ensure_directory(joinpath(trial_dir, "clean"))
    noisy_dir = ensure_directory(joinpath(trial_dir, "perturbed"))
    plots_dir = ensure_directory(joinpath(trial_dir, "plots"))
    clean_plot_dir = ensure_directory(joinpath(plots_dir, "clean"))
    noisy_plot_dir = ensure_directory(joinpath(plots_dir, "perturbed"))
    comparison_plot_dir = ensure_directory(joinpath(plots_dir, "comparison"))
    tables_dir = ensure_directory(joinpath(trial_dir, "tables"))
    _, clean_table_data_dir, clean_table_formatted_dir = ensure_table_directories(tables_dir, "clean")
    _, pert_table_data_dir, pert_table_formatted_dir = ensure_table_directories(tables_dir, "perturbed")

    step_schedule = Float64.(step_size .* (ANNEALING_STEP_SIZES ./ DEFAULT_STEP_SIZE))

    println(@sprintf("Starting trial %02d with %d scenarios (clean model)", trial_index, scenario_count))
    clean_reg_params = train_with_annealing!(model_clean, training_data, problem_instance;
        step_sizes = step_schedule, batchsize = batchsize, default_epochs = default_epochs,
        save_dir = clean_dir, resume = resume)
    model_clean = clean_reg_params.model
    step_schedule = clean_reg_params.step_size_schedule
    plot_training_history(clean_reg_params.training_history, joinpath(clean_plot_dir, "training_loss.png");
        stage_markers = clean_reg_params.stage_markers)

    println(@sprintf("Starting trial %02d with %d scenarios (perturbed model)", trial_index, scenario_count))
    noisy_reg_params = train_with_annealing!(model_noisy, perturbed_training_data, problem_instance;
        step_sizes = step_schedule, batchsize = batchsize, default_epochs = default_epochs,
        save_dir = noisy_dir, resume = resume)
    model_noisy = noisy_reg_params.model
    plot_training_history(noisy_reg_params.training_history, joinpath(noisy_plot_dir, "training_loss.png");
        stage_markers = noisy_reg_params.stage_markers)

    clean_loss = compute_average_training_loss(model_clean, training_data, problem_instance,
        clean_reg_params.final_reg_param_surr, clean_reg_params.reg_param_prim)

    noisy_loss = compute_average_training_loss(model_noisy, perturbed_training_data, problem_instance,
        noisy_reg_params.final_reg_param_surr, noisy_reg_params.reg_param_prim)

    clean_eval = evaluate_model(problem_instance, model_clean, preprocessed_testing,
        clean_reg_params.final_reg_param_surr, clean_reg_params.reg_param_ref; label = "clean model")
    noisy_eval = evaluate_model(problem_instance, model_noisy, preprocessed_testing,
        noisy_reg_params.final_reg_param_surr, noisy_reg_params.reg_param_ref; label = "perturbed model")

    # Persist tables for clean model
    clean_training_df = DataFrame(clean_reg_params.training_history_log)
    CSV.write(joinpath(clean_table_data_dir, "training_history.csv"), clean_training_df)
    save_dataframe_markdown(clean_training_df, joinpath(clean_table_formatted_dir, "training_history.md");
        title = "Training History", max_rows = 100)

    CSV.write(joinpath(clean_table_data_dir, "sample_summary.csv"), clean_eval.sample_summary)
    save_dataframe_markdown(clean_eval.sample_summary, joinpath(clean_table_formatted_dir, "sample_summary.md");
        title = "Per-sample Summary", max_rows = 50)

    CSV.write(joinpath(clean_table_data_dir, "scenario_details.csv"), clean_eval.scenario_details)
    save_dataframe_markdown(clean_eval.scenario_details, joinpath(clean_table_formatted_dir, "scenario_details.md");
        title = "Scenario-level Details", max_rows = 50)

    # Persist tables for perturbed model
    noisy_training_df = DataFrame(noisy_reg_params.training_history_log)
    CSV.write(joinpath(pert_table_data_dir, "training_history.csv"), noisy_training_df)
    save_dataframe_markdown(noisy_training_df, joinpath(pert_table_formatted_dir, "training_history.md");
        title = "Training History", max_rows = 100)

    CSV.write(joinpath(pert_table_data_dir, "sample_summary.csv"), noisy_eval.sample_summary)
    save_dataframe_markdown(noisy_eval.sample_summary, joinpath(pert_table_formatted_dir, "sample_summary.md");
        title = "Per-sample Summary", max_rows = 50)

    CSV.write(joinpath(pert_table_data_dir, "scenario_details.csv"), noisy_eval.scenario_details)
    save_dataframe_markdown(noisy_eval.scenario_details, joinpath(pert_table_formatted_dir, "scenario_details.md");
        title = "Scenario-level Details", max_rows = 50)

    plot_performance_summary(clean_eval, noisy_eval, comparison_plot_dir;
        title_suffix = @sprintf("scenarios=%d trial=%02d", scenario_count, trial_index))

    results_df = DataFrame(
        scenario_count = [scenario_count, scenario_count],
        trial = [trial_index, trial_index],
        data_variant = ["clean", "perturbed"],
        training_loss = [clean_loss, noisy_loss],
        mean_gap = [clean_eval.mean_gap, noisy_eval.mean_gap],
        mean_relative_gap = [clean_eval.mean_relative_gap, noisy_eval.mean_relative_gap],
        mean_opt_cost = [clean_eval.mean_opt_cost, noisy_eval.mean_opt_cost],
        mean_model_cost = [clean_eval.mean_model_cost, noisy_eval.mean_model_cost],
        mean_absolute_gap = [clean_eval.mean_absolute_gap, noisy_eval.mean_absolute_gap],
        overall_relative_gap_std = [clean_eval.overall_relative_gap_std, noisy_eval.overall_relative_gap_std],
        worst_relative_gap = [clean_eval.worst_sample_relative_gap, noisy_eval.worst_sample_relative_gap],
        noise_scale = [0.0, noise_scale]
    )

    summary_path = joinpath(trial_dir, "results.csv")
    CSV.write(summary_path, results_df)

    config_info = Dict(
        "scenario_count" => scenario_count,
        "trial" => trial_index,
        "train_data_seed" => train_data_seed,
        "test_data_seed" => test_data_seed,
        "training_seed" => training_seed,
        "noise_scale" => noise_scale,
        "Ntraining" => Ntrain,
        "Ntesting" => Ntest,
        "N_xi_per_x" => N_xi_per_x,
        "sigma" => σ,
        "p" => p,
        "L" => L,
        "step_size" => step_size,
        "annealing_step_sizes" => collect(step_schedule),
        "batchsize" => batchsize,
        "default_epochs" => default_epochs,
        "annealing_params" => collect(ANNEALING_PARAMS)
    )

    config_path = joinpath(trial_dir, "config.txt")
    open(config_path, "w") do io
        for key in sort(collect(keys(config_info)))
            println(io, "$(key)=$(config_info[key])")
        end
    end

    return results_df
end

function run_experiment(; scenario_counts = DEFAULT_SCENARIO_COUNTS, replicates::Int = DEFAULT_REPLICATES,
        Ntrain::Int = DEFAULT_NTRAIN, Ntest::Int = DEFAULT_NTEST, N_xi_per_x::Int = DEFAULT_N_XI_PER_X,
        σ::Float64 = DEFAULT_SIGMA, p::Int = DEFAULT_P, L::Int = DEFAULT_L, noise_scale::Float64 = DEFAULT_NOISE_SCALE,
        step_size::Float64 = DEFAULT_STEP_SIZE, batchsize::Int = DEFAULT_BATCHSIZE, default_epochs::Int = DEFAULT_EPOCHS,
        output_root::String = default_output_root(), base_seed::Int = DEFAULT_BASE_SEED,
        train_seed_base::Int = DEFAULT_BASE_SEED, test_seed_base::Int = DEFAULT_BASE_SEED,
        training_seed_base::Int = DEFAULT_BASE_SEED, resume::Bool = false)

    problem_instance = create_problem_instance()
    ensure_directory(output_root)
    ensure_cache_directories()
    aggregated = DataFrame()

    scenario_order = collect(scenario_counts)
    dataset_reference_scenario = first(scenario_order)

    for trial_index in 1:replicates
        train_data_seed = train_seed_base + dataset_reference_scenario * 100 + trial_index
        test_data_seed = test_seed_base + dataset_reference_scenario * 100 + trial_index
        trial_results_paths = Dict(scenario_count => joinpath(output_root,
            @sprintf("scenarios_%d", scenario_count), @sprintf("trial_%02d", trial_index), "results.csv")
            for scenario_count in scenario_order)

        pending_scenarios = [sc for sc in scenario_order if !(resume && isfile(trial_results_paths[sc]))]

        if isempty(pending_scenarios)
            if resume
                println(@sprintf("Trial %02d already completed for all scenario counts; skipping (resume mode)", trial_index))
                for scenario_count in scenario_order
                    existing_df = DataFrame(CSV.File(trial_results_paths[scenario_count]))
                    aggregated = vcat(aggregated, existing_df)
                end
            end
            continue
        end

        training_data = load_or_generate_training_dataset(problem_instance, train_data_seed, trial_index;
            Ntrain = Ntrain, Ntest = Ntest, N_xi_per_x = N_xi_per_x, σ = σ, p = p, L = L)
        testing_data = load_or_generate_testing_dataset(problem_instance, test_data_seed, trial_index;
            Ntrain = Ntrain, Ntest = Ntest, N_xi_per_x = N_xi_per_x, σ = σ, p = p, L = L)

        saa_cache = saa_cache_path(test_data_seed, trial_index; Ntest = Ntest, N_xi_per_x = N_xi_per_x, σ = σ, p = p, L = L)
        cache_metadata = Dict(
            "train_data_seed" => train_data_seed,
            "test_data_seed" => test_data_seed,
            "trial" => trial_index,
            "Ntrain" => Ntrain,
            "Ntest" => Ntest,
            "N_xi_per_x" => N_xi_per_x,
            "sigma" => σ,
            "p" => p,
            "L" => L
        )

        preprocessed_testing = preprocess_testing_data(problem_instance, testing_data, N_xi_per_x;
            cache_path = saa_cache, metadata = cache_metadata, legacy_seed = test_data_seed, trial_index = trial_index,
            Ntest = Ntest, σ = σ, p = p, L = L)

        for scenario_count in scenario_order
            results_path = trial_results_paths[scenario_count]
            if resume && isfile(results_path)
                println(@sprintf("Skipping scenario %d trial %02d (already completed)", scenario_count, trial_index))
                existing_df = DataFrame(CSV.File(results_path))
                aggregated = vcat(aggregated, existing_df)
                continue
            end
            training_seed = training_seed_base + scenario_count * 100 + trial_index
            results_df = run_single_trial(problem_instance, scenario_count, trial_index;
                training_data = training_data, testing_data = testing_data, preprocessed_testing = preprocessed_testing,
                Ntrain = Ntrain, Ntest = Ntest, N_xi_per_x = N_xi_per_x, σ = σ, p = p, L = L,
                noise_scale = noise_scale, step_size = step_size, batchsize = batchsize,
                default_epochs = default_epochs, output_dir = output_root,
                train_data_seed = train_data_seed, test_data_seed = test_data_seed,
                training_seed = training_seed, resume = resume)
            aggregated = vcat(aggregated, results_df)
        end
    end

    summary_path = joinpath(output_root, "aggregated_results.csv")
    CSV.write(summary_path, aggregated)
    return aggregated
end

function main()
    options = parse_cli_args(ARGS)
    scenario_counts = parse_int_tuple(options, "scenarios", DEFAULT_SCENARIO_COUNTS)
    replicates = parse_int_option(options, "replicates", DEFAULT_REPLICATES)
    Ntrain = parse_int_option(options, "train", DEFAULT_NTRAIN)
    Ntest = parse_int_option(options, "test", DEFAULT_NTEST)
    N_xi_per_x = parse_int_option(options, "xi_per_x", DEFAULT_N_XI_PER_X)
    σ = parse_float_option(options, "sigma", DEFAULT_SIGMA)
    p = parse_int_option(options, "p", DEFAULT_P)
    L = parse_int_option(options, "L", DEFAULT_L)
    noise_scale = parse_float_option(options, "noise", DEFAULT_NOISE_SCALE)
    step_size = parse_float_option(options, "step", DEFAULT_STEP_SIZE)
    batchsize = parse_int_option(options, "batchsize", DEFAULT_BATCHSIZE)
    default_epochs = parse_int_option(options, "epochs", DEFAULT_EPOCHS)
    base_seed = parse_int_option(options, "seed", DEFAULT_BASE_SEED)
    train_seed_base = parse_int_option(options, "train_seed", base_seed)
    test_seed_base = parse_int_option(options, "test_seed", base_seed)
    training_seed_base = parse_int_option(options, "training_seed", base_seed)
    resume = parse_bool_option(options, "resume", false)
    output_dir = get(options, "output", default_output_root())

    aggregated = run_experiment(; scenario_counts = scenario_counts, replicates = replicates,
        Ntrain = Ntrain, Ntest = Ntest, N_xi_per_x = N_xi_per_x, σ = σ, p = p, L = L,
        noise_scale = noise_scale, step_size = step_size, batchsize = batchsize,
        default_epochs = default_epochs, output_root = output_dir, base_seed = base_seed,
        train_seed_base = train_seed_base, test_seed_base = test_seed_base,
        training_seed_base = training_seed_base, resume = resume)

    println("Aggregated results saved to ", joinpath(output_dir, "aggregated_results.csv"))
    println("Summary statistics by variant:")
    println(combine(groupby(aggregated, [:scenario_count, :data_variant]),
        :training_loss => mean => :training_loss_mean,
        :mean_gap => mean => :mean_gap_mean,
        :mean_relative_gap => mean => :mean_relative_gap_mean,
        :mean_opt_cost => mean => :mean_saa_cost_mean,
        :mean_model_cost => mean => :mean_model_cost_mean,
        :mean_absolute_gap => mean => :mean_absolute_gap_mean,
        :overall_relative_gap_std => mean => :overall_relative_gap_std_mean,
        :worst_relative_gap => mean => :worst_relative_gap_mean,
        :worst_relative_gap => maximum => :worst_relative_gap_max))
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
