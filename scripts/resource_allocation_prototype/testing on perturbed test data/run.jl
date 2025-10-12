import Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()

using LinearAlgebra
using Random
using Statistics
using Printf
using Serialization
using CSV
using DataFrames
using ProblemBasedScenarioGeneration
using ProblemBasedScenarioGeneration: ResourceAllocationProblemData, ResourceAllocationProblem,
    dataGeneration, scenario_realization, scenario_collection_realization,
    LogBarCanLP, LogBarCanLP_standard_solver, TwoStageSLP,
    CanLP, optimal_value, s1_cost, load_trained_model

using ChainRulesCore
import ProblemBasedScenarioGeneration: surrogate_solution, derivative_surrogate_solution, primal_problem_cost

include(joinpath(@__DIR__, "..", "custom_code", "neural_net.jl"))

const RUN_ROOT = joinpath(@__DIR__, "..", "testing_multiple_scenarios", "results", "run_20251008_134918")

struct TrialInfo
    scenario_count::Int
    trial_index::Int
    config::Dict{String,String}
    model_path::String
    noise_scale::Float64
    reg_param_final::Float64
end

struct DatasetPack
    contexts_clean::Vector{Vector{Float64}}
    contexts_perturbed::Vector{Vector{Float64}}
    samples_clean::Vector{Matrix{Float64}}
    samples_perturbed::Vector{Matrix{Float64}}
end

struct EvaluationMetrics
    mean_gap::Float64
    mean_relative_gap::Float64
    mean_absolute_gap::Float64
    mean_opt_cost::Float64
    mean_model_cost::Float64
    gaps::Vector{Float64}
    relative_gaps::Vector{Float64}
end

function parse_cli_args(args::Vector{String})
    options = Dict{String,String}()
    for arg in args
        startswith(arg, "--") || continue
        raw = arg[3:end]
        parts = split(raw, "=", limit=2)
        if length(parts) == 2
            options[parts[1]] = parts[2]
        else
            options[parts[1]] = "true"
        end
    end
    options
end

function read_config(path::String)
    lines = readlines(path)
    dict = Dict{String,String}()
    for line in lines
        isempty(strip(line)) && continue
        parts = split(line, "=", limit=2)
        length(parts) == 2 || continue
        dict[strip(parts[1])] = strip(parts[2])
    end
    dict
end

function parse_array(str::String)
    replaced = replace(str, "[" => "[", "]" => "]")
    Meta.parse(replaced) |> eval
end

function select_best_trial(scenario_count::Int)
    scenario_dir = joinpath(RUN_ROOT, @sprintf("scenarios_%d", scenario_count))
    trials = filter(name -> occursin("trial_", name), readdir(scenario_dir))
    best_trial = nothing
    best_metric = Inf
    best_config = Dict{String,String}()
    best_model_path = ""
    noise_scale = 0.0
    reg_param_final = 0.01

    for trial_name in trials
        trial_dir = joinpath(scenario_dir, trial_name)
        results_path = joinpath(trial_dir, "results.csv")
        isfile(results_path) || continue
        df = DataFrame(CSV.File(results_path))
        subset = df[df.data_variant .== "clean", :]
        isempty(subset) && continue
        row = subset[argmin(subset.mean_relative_gap), :]
        metric = row.mean_relative_gap
        if metric < best_metric
            best_metric = metric
            best_trial = parse(Int, match(r"\d+", trial_name).match)
            config_path = joinpath(trial_dir, "config.txt")
            best_config = read_config(config_path)
            noise_scale = parse(Float64, best_config["noise_scale"])
            params = parse_array(best_config["annealing_params"])
            reg_param_final = params[end]
            clean_dir = joinpath(trial_dir, "clean")
            stage_files = filter(f -> endswith(f, "_model.jls"), readdir(clean_dir))
            if isempty(stage_files)
                error("No model checkpoints found in $(clean_dir)")
            end
            sorted = sort(stage_files; by = f -> parse(Int, match(r"\d+", f).match))
            best_model_path = joinpath(clean_dir, sorted[end])
        end
    end

    best_trial === nothing && error("Failed to select best trial for scenario count $(scenario_count)")

    return TrialInfo(scenario_count, best_trial, best_config, best_model_path, noise_scale, reg_param_final)
end

function create_problem_instance()
    include(joinpath(@__DIR__, "..", "parameters.jl"))
    cz_vec, qw_vec, ρ_vec = vec(cz), vec(qw), vec(ρᵢ)
    problem_data = ResourceAllocationProblemData(μᵢⱼ, cz_vec, qw_vec, ρ_vec)
    ResourceAllocationProblem(problem_data)
end

function compute_means_for_context(x::Vector{Float64}, A::Vector{Float64}, B::Matrix{Float64}, p::Int, L::Int)
    means = similar(A)
    for j in 1:length(A)
        total = A[j]
        for ℓ in 1:L
            total += B[j, ℓ] * (x[ℓ])^p
        end
        means[j] = total
    end
    means
end

function generate_demand_samples(means::Vector{Float64}, noise::Matrix{Float64}, σ::Float64)
    repeat(means, 1, size(noise, 2)) .+ σ .* noise
end

function regenerate_training_data(problem_instance::ResourceAllocationProblem, info::TrialInfo, scenario_samples::Int, perturb_scale::Float64, perturb_rng::AbstractRNG, noise_rng::AbstractRNG)
    cfg = info.config
    Ntrain = parse(Int, cfg["Ntraining"])
    N_xi_per_x = parse(Int, cfg["N_xi_per_x"])
    σ = parse(Float64, cfg["sigma"])
    p = parse(Int, cfg["p"])
    L = parse(Int, cfg["L"])
    collections = 1
    train_seed = parse(Int, cfg["train_data_seed"])

    Random.seed!(train_seed)
    training_data, _, A, B = dataGeneration(problem_instance, Ntrain, 0, N_xi_per_x, σ, p, L, collections)

    pairs = collect(training_data)
    sort!(pairs, by = x -> tuple(x[1]...))
    contexts = [copy(first(pair)) for pair in pairs]
    J = length(last(first(pairs)))

    noise_mats = [randn(noise_rng, J, scenario_samples) for _ in contexts]
    means = [compute_means_for_context(x, A, B, p, L) for x in contexts]
    samples_clean = [generate_demand_samples(means[i], noise_mats[i], σ) for i in eachindex(contexts)]

    perturbed_contexts = [x .+ perturb_scale .* randn(perturb_rng, length(x)) for x in contexts]
    pert_means = [compute_means_for_context(xp, A, B, p, L) for xp in perturbed_contexts]
    samples_pert = [generate_demand_samples(pert_means[i], noise_mats[i], σ) for i in eachindex(perturbed_contexts)]

    DatasetPack(contexts, perturbed_contexts, samples_clean, samples_pert)
end

function regenerate_testing_data(problem_instance::ResourceAllocationProblem, info::TrialInfo, scenario_samples::Int, perturb_scale::Float64, perturb_rng::AbstractRNG)
    cfg = info.config
    Ntest = parse(Int, cfg["Ntesting"])
    N_xi_per_x = parse(Int, cfg["N_xi_per_x"])
    σ = parse(Float64, cfg["sigma"])
    p = parse(Int, cfg["p"])
    L = parse(Int, cfg["L"])
    collections = 30
    test_seed = parse(Int, cfg["test_data_seed"])

    Random.seed!(test_seed)
    _, testing_data, A, B = dataGeneration(problem_instance, 0, Ntest, N_xi_per_x, σ, p, L, collections)

    pairs = collect(testing_data)
    sort!(pairs, by = x -> tuple(x[1]...))
    contexts = [copy(first(pair)) for pair in pairs]
    tensors = [copy(last(pair)) for pair in pairs]

    samples_clean = Vector{Matrix{Float64}}(undef, length(contexts))
    noise_mats = Vector{Matrix{Float64}}(undef, length(contexts))

    for (i, ξ_tensor) in enumerate(tensors)
        ξ_slice = ξ_tensor[1, 1:scenario_samples, :]
        sample_mat = permutedims(ξ_slice, (2, 1))
        means = compute_means_for_context(contexts[i], A, B, p, L)
        noise_mats[i] = (sample_mat .- means) ./ σ
        samples_clean[i] = sample_mat
    end

    perturbed_contexts = [x .+ perturb_scale .* randn(perturb_rng, length(x)) for x in contexts]
    samples_pert = Vector{Matrix{Float64}}(undef, length(contexts))
    for i in eachindex(contexts)
        means = compute_means_for_context(perturbed_contexts[i], A, B, p, L)
        samples_pert[i] = generate_demand_samples(means, noise_mats[i], σ)
    end

    DatasetPack(contexts, perturbed_contexts, samples_clean, samples_pert)
end

function build_two_stage(problem_instance::ResourceAllocationProblem, xi_samples::Matrix{Float64})
    A = problem_instance.s1_constraint_matrix
    b = problem_instance.s1_constraint_vector
    c = problem_instance.s1_cost_vector
    sample_count = size(xi_samples, 2)

    Ws_list = Vector{Matrix{Float64}}(undef, sample_count)
    Ts_list = Vector{Matrix{Float64}}(undef, sample_count)
    hs_list = Vector{Vector{Float64}}(undef, sample_count)
    qs_list = Vector{Vector{Float64}}(undef, sample_count)

    for k in 1:sample_count
        W, T, h, q = scenario_realization(problem_instance, xi_samples[:, k])
        Ws_list[k] = W
        Ts_list[k] = T
        hs_list[k] = h
        qs_list[k] = q
    end

    Ws = cat(Ws_list...; dims = 3)
    Ts = cat(Ts_list...; dims = 3)
    hs = hcat(hs_list...)
    qs = hcat(qs_list...)

    two_slp = TwoStageSLP(A, b, c, Ws, Ts, hs, qs)
    margins = (1e-7, 5e-7, 1e-6, 5e-6, 1e-5)
    can_lp = CanLP(two_slp)
    for margin in margins
        solver = instance -> solve_canonical_lp(instance; feasibility_margin = margin)
        try
            opt_val = optimal_value(can_lp, solver; feasibility_margin = margin)
            return two_slp, opt_val
        catch err
            if err isa ErrorException && (occursin("Infeasible: max |Ax - b|", err.msg) || occursin("Decision is not feasible", err.msg))
                continue
            else
                rethrow(err)
            end
        end
    end
    error("Unable to solve canonical LP for scenario collection")
end

function compute_surrogate_decision(problem_instance::ResourceAllocationProblem, reg_param::Float64, scenario_collection::Matrix{Float64})
    A = problem_instance.s1_constraint_matrix
    b = problem_instance.s1_constraint_vector
    c = problem_instance.s1_cost_vector
    Ws, Ts, hs, qs = scenario_collection_realization(problem_instance, scenario_collection)
    surrogate_problem = LogBarCanLP(TwoStageSLP(A, b, c, Ws, Ts, hs, qs), reg_param)
    decision, _ = LogBarCanLP_standard_solver(surrogate_problem)
    decision[1:length(c)]
end

function evaluate_model_on_dataset(problem_instance::ResourceAllocationProblem, model, data::DatasetPack;
        reg_param_surr::Float64, reg_param_ref::Float64, sample_limit::Int)
    total = min(sample_limit, length(data.contexts_clean))
    gaps = Float64[]
    rel_gaps = Float64[]
    abs_gaps = Float64[]
    opt_costs = Float64[]
    model_costs = Float64[]

    for idx in 1:total
        x = data.contexts_clean[idx]
        xi_samples = data.samples_clean[idx]
        x_mat = reshape(x, :, 1)
        prediction = model(x_mat)
        surrogate_decision = compute_surrogate_decision(problem_instance, reg_param_surr, prediction)
        two_slp, opt_cost = build_two_stage(problem_instance, xi_samples)
        evaluated_cost = s1_cost(two_slp, surrogate_decision, reg_param_ref)
        gap = evaluated_cost - opt_cost
        rel_gap = gap / max(abs(opt_cost), eps())
        push!(gaps, gap)
        push!(rel_gaps, rel_gap)
        push!(abs_gaps, abs(gap))
        push!(opt_costs, opt_cost)
        push!(model_costs, evaluated_cost)
    end

    EvaluationMetrics(mean(gaps), mean(rel_gaps), mean(abs_gaps), mean(opt_costs), mean(model_costs), gaps, rel_gaps)
end

function evaluate_model_on_perturbed_dataset(problem_instance::ResourceAllocationProblem, model, data::DatasetPack;
        reg_param_surr::Float64, reg_param_ref::Float64, sample_limit::Int)
    total = min(sample_limit, length(data.contexts_perturbed))
    gaps = Float64[]
    rel_gaps = Float64[]
    abs_gaps = Float64[]
    opt_costs = Float64[]
    model_costs = Float64[]

    for idx in 1:total
        x = data.contexts_perturbed[idx]
        xi_samples = data.samples_perturbed[idx]
        x_mat = reshape(x, :, 1)
        prediction = model(x_mat)
        surrogate_decision = compute_surrogate_decision(problem_instance, reg_param_surr, prediction)
        two_slp, opt_cost = build_two_stage(problem_instance, xi_samples)
        evaluated_cost = s1_cost(two_slp, surrogate_decision, reg_param_ref)
        gap = evaluated_cost - opt_cost
        rel_gap = gap / max(abs(opt_cost), eps())
        push!(gaps, gap)
        push!(rel_gaps, rel_gap)
        push!(abs_gaps, abs(gap))
        push!(opt_costs, opt_cost)
        push!(model_costs, evaluated_cost)
    end

    EvaluationMetrics(mean(gaps), mean(rel_gaps), mean(abs_gaps), mean(opt_costs), mean(model_costs), gaps, rel_gaps)
end

function evaluate_saa_baseline(problem_instance::ResourceAllocationProblem, data::DatasetPack;
        sample_limit::Int, perturbed::Bool=false)
    contexts = perturbed ? data.contexts_perturbed : data.contexts_clean
    samples = perturbed ? data.samples_perturbed : data.samples_clean
    total = min(sample_limit, length(contexts))
    gaps = Float64[]
    rel_gaps = Float64[]
    abs_gaps = Float64[]
    opt_costs = Float64[]
    model_costs = Float64[]

    for idx in 1:total
        xi_samples = samples[idx]
        two_slp, opt_cost = build_two_stage(problem_instance, xi_samples)
        decision = compute_surrogate_decision(problem_instance, 0.0, xi_samples)
        evaluated_cost = s1_cost(two_slp, decision, 0.0)
        gap = evaluated_cost - opt_cost
        rel_gap = gap / max(abs(opt_cost), eps())
        push!(gaps, gap)
        push!(rel_gaps, rel_gap)
        push!(abs_gaps, abs(gap))
        push!(opt_costs, opt_cost)
        push!(model_costs, evaluated_cost)
    end

    EvaluationMetrics(mean(gaps), mean(rel_gaps), mean(abs_gaps), mean(opt_costs), mean(model_costs), gaps, rel_gaps)
end

function evaluate_cross_dataset_performance(problem_instance::ResourceAllocationProblem, model, data::DatasetPack;
        reg_param_surr::Float64, reg_param_ref::Float64, sample_limit::Int)
    total = min(sample_limit, length(data.contexts_clean))

    detail_rows = DataFrame(
        context_index = Int[],
        gap_clean = Float64[],
        gap_perturbed = Float64[],
        gap_delta = Float64[],
        relative_gap_clean = Float64[],
        relative_gap_perturbed = Float64[],
        relative_gap_delta = Float64[],
        cost_clean = Float64[],
        cost_perturbed = Float64[],
        saa_cost_perturbed = Float64[],
        gap_vs_saa = Float64[]
    )

    for idx in 1:total
        x_clean = data.contexts_clean[idx]
        xi_clean = data.samples_clean[idx]
        xi_perturbed = data.samples_perturbed[idx]

        x_mat = reshape(x_clean, :, 1)
        prediction = model(x_mat)
        surrogate_decision = compute_surrogate_decision(problem_instance, reg_param_surr, prediction)

        clean_two_slp, clean_opt = build_two_stage(problem_instance, xi_clean)
        clean_cost = s1_cost(clean_two_slp, surrogate_decision, reg_param_ref)
        clean_gap = clean_cost - clean_opt
        clean_rel_gap = clean_gap / max(abs(clean_opt), eps())

        pert_two_slp, pert_opt = build_two_stage(problem_instance, xi_perturbed)
        pert_cost = s1_cost(pert_two_slp, surrogate_decision, reg_param_ref)
        pert_gap = pert_cost - pert_opt
        pert_rel_gap = pert_gap / max(abs(pert_opt), eps())

        saa_decision = compute_surrogate_decision(problem_instance, 0.0, xi_perturbed)
        saa_cost = s1_cost(pert_two_slp, saa_decision, 0.0)
        gap_vs_saa = pert_cost - saa_cost

        push!(detail_rows, (
            context_index = idx,
            gap_clean = clean_gap,
            gap_perturbed = pert_gap,
            gap_delta = pert_gap - clean_gap,
            relative_gap_clean = clean_rel_gap,
            relative_gap_perturbed = pert_rel_gap,
            relative_gap_delta = pert_rel_gap - clean_rel_gap,
            cost_clean = clean_cost,
            cost_perturbed = pert_cost,
            saa_cost_perturbed = saa_cost,
            gap_vs_saa = gap_vs_saa
        ))
    end

    if nrow(detail_rows) == 0
        summary = DataFrame(
            mean_gap_clean = [NaN],
            mean_gap_perturbed = [NaN],
            mean_gap_delta = [NaN],
            mean_relative_gap_clean = [NaN],
            mean_relative_gap_perturbed = [NaN],
            mean_relative_gap_delta = [NaN],
            mean_gap_vs_saa = [NaN]
        )
    else
        summary = DataFrame(
            mean_gap_clean = [mean(detail_rows.gap_clean)],
            mean_gap_perturbed = [mean(detail_rows.gap_perturbed)],
            mean_gap_delta = [mean(detail_rows.gap_delta)],
            mean_relative_gap_clean = [mean(detail_rows.relative_gap_clean)],
            mean_relative_gap_perturbed = [mean(detail_rows.relative_gap_perturbed)],
            mean_relative_gap_delta = [mean(detail_rows.relative_gap_delta)],
            mean_gap_vs_saa = [mean(detail_rows.gap_vs_saa)]
        )
    end

    (; summary, details = detail_rows)
end

function evaluate_scenario(problem_instance::ResourceAllocationProblem, info::TrialInfo;
        sample_limit::Int, scenario_samples::Int, perturb_seed::Int)
    println(@sprintf("Evaluating scenario count %d (trial %02d)", info.scenario_count, info.trial_index))
    dummy_model = construct_neural_network(problem_instance; nr_of_scenarios = info.scenario_count)
    reshape_layer = dummy_model.layers[end]
    reshape_symbol = typeof(reshape_layer).name.name
    for missing_sym in (Symbol("#62#63"), Symbol("#65#66"))
        if !isdefined(ProblemBasedScenarioGeneration, missing_sym) && isdefined(ProblemBasedScenarioGeneration, reshape_symbol)
            ProblemBasedScenarioGeneration.eval(:(const $(missing_sym) = $(reshape_symbol)))
        end
    end
    model = load_trained_model(info.model_path)

    reg_param_ref = 0.0
    reg_param_surr = info.reg_param_final

    noise_rng_train = MersenneTwister(perturb_seed + 1000 * info.scenario_count)
    perturb_rng_train = MersenneTwister(perturb_seed + 2000 * info.scenario_count)
    perturb_rng_test = MersenneTwister(perturb_seed + 3000 * info.scenario_count)

    perturb_scale = 10.0 * info.noise_scale
    train_data = regenerate_training_data(problem_instance, info, scenario_samples, perturb_scale, perturb_rng_train, noise_rng_train)
    test_data = regenerate_testing_data(problem_instance, info, scenario_samples, perturb_scale, perturb_rng_test)

    println("  Evaluating model on training data (clean)")
    train_clean_metrics = evaluate_model_on_dataset(problem_instance, model, train_data;
        reg_param_surr = reg_param_surr, reg_param_ref = reg_param_ref, sample_limit = sample_limit)
    println("  Evaluating model on training data (perturbed)")
    train_pert_metrics = evaluate_model_on_perturbed_dataset(problem_instance, model, train_data;
        reg_param_surr = reg_param_surr, reg_param_ref = reg_param_ref, sample_limit = sample_limit)

    println("  Evaluating model on testing data (clean)")
    test_clean_metrics = evaluate_model_on_dataset(problem_instance, model, test_data;
        reg_param_surr = reg_param_surr, reg_param_ref = reg_param_ref, sample_limit = sample_limit)
    println("  Evaluating model on testing data (perturbed)")
    test_pert_metrics = evaluate_model_on_perturbed_dataset(problem_instance, model, test_data;
        reg_param_surr = reg_param_surr, reg_param_ref = reg_param_ref, sample_limit = sample_limit)

    println("  Evaluating SAA baseline (training clean)")
    saa_train_clean = evaluate_saa_baseline(problem_instance, train_data; sample_limit = sample_limit)
    println("  Evaluating SAA baseline (training perturbed)")
    saa_train_pert = evaluate_saa_baseline(problem_instance, train_data; sample_limit = sample_limit, perturbed = true)
    println("  Evaluating SAA baseline (testing clean)")
    saa_test_clean = evaluate_saa_baseline(problem_instance, test_data; sample_limit = sample_limit)
    println("  Evaluating SAA baseline (testing perturbed)")
    saa_test_pert = evaluate_saa_baseline(problem_instance, test_data; sample_limit = sample_limit, perturbed = true)

    rows = DataFrame(model = String[], scenario_count = Int[], trial = Int[],
        dataset = String[], perturbation = String[],
        mean_gap = Float64[], mean_relative_gap = Float64[])

    model_label = info.scenario_count == 1 ? "single" : "triple"

    push!(rows, (model_label, info.scenario_count, info.trial_index, "training", "clean",
        train_clean_metrics.mean_gap, train_clean_metrics.mean_relative_gap))
    push!(rows, (model_label, info.scenario_count, info.trial_index, "training", "perturbed",
        train_pert_metrics.mean_gap, train_pert_metrics.mean_relative_gap))
    push!(rows, (model_label, info.scenario_count, info.trial_index, "testing", "clean",
        test_clean_metrics.mean_gap, test_clean_metrics.mean_relative_gap))
    push!(rows, (model_label, info.scenario_count, info.trial_index, "testing", "perturbed",
        test_pert_metrics.mean_gap, test_pert_metrics.mean_relative_gap))

    push!(rows, ("saa", info.scenario_count, info.trial_index, "training", "clean",
        saa_train_clean.mean_gap, saa_train_clean.mean_relative_gap))
    push!(rows, ("saa", info.scenario_count, info.trial_index, "training", "perturbed",
        saa_train_pert.mean_gap, saa_train_pert.mean_relative_gap))
    push!(rows, ("saa", info.scenario_count, info.trial_index, "testing", "clean",
        saa_test_clean.mean_gap, saa_test_clean.mean_relative_gap))
    push!(rows, ("saa", info.scenario_count, info.trial_index, "testing", "perturbed",
        saa_test_pert.mean_gap, saa_test_pert.mean_relative_gap))

    diff_rows = DataFrame(model = String[], scenario_count = Int[], trial = Int[],
        dataset = String[], gap_delta = Float64[], relative_gap_delta = Float64[])

    gap_delta_train = train_pert_metrics.mean_gap - train_clean_metrics.mean_gap
    gap_delta_test = test_pert_metrics.mean_gap - test_clean_metrics.mean_gap
    rel_delta_train = train_pert_metrics.mean_relative_gap - train_clean_metrics.mean_relative_gap
    rel_delta_test = test_pert_metrics.mean_relative_gap - test_clean_metrics.mean_relative_gap

    push!(diff_rows, (model_label, info.scenario_count, info.trial_index, "training", gap_delta_train, rel_delta_train))
    push!(diff_rows, (model_label, info.scenario_count, info.trial_index, "testing", gap_delta_test, rel_delta_test))
    push!(diff_rows, ("saa", info.scenario_count, info.trial_index, "training",
        saa_train_pert.mean_gap - saa_train_clean.mean_gap,
        saa_train_pert.mean_relative_gap - saa_train_clean.mean_relative_gap))
    push!(diff_rows, ("saa", info.scenario_count, info.trial_index, "testing",
        saa_test_pert.mean_gap - saa_test_clean.mean_gap,
        saa_test_pert.mean_relative_gap - saa_test_clean.mean_relative_gap))

    println("  Evaluating cross-deployment robustness (clean decisions on perturbed contexts)")
    cross_metrics = evaluate_cross_dataset_performance(problem_instance, model, test_data;
        reg_param_surr = reg_param_surr, reg_param_ref = reg_param_ref, sample_limit = sample_limit)

    cross_summary = cross_metrics.summary
    cross_summary[!, :model] = fill(model_label, nrow(cross_summary))
    cross_summary[!, :scenario_count] = fill(info.scenario_count, nrow(cross_summary))
    cross_summary[!, :trial] = fill(info.trial_index, nrow(cross_summary))
    cross_summary[!, :dataset] = fill("testing", nrow(cross_summary))
    cross_summary = select(cross_summary, :model, :scenario_count, :trial, :dataset,
        :mean_gap_clean, :mean_gap_perturbed, :mean_gap_delta,
        :mean_relative_gap_clean, :mean_relative_gap_perturbed, :mean_relative_gap_delta,
        :mean_gap_vs_saa)

    cross_details = cross_metrics.details
    cross_details[!, :model] = fill(model_label, nrow(cross_details))
    cross_details[!, :scenario_count] = fill(info.scenario_count, nrow(cross_details))
    cross_details[!, :trial] = fill(info.trial_index, nrow(cross_details))
    cross_details[!, :dataset] = fill("testing", nrow(cross_details))
    cross_details = select(cross_details, :model, :scenario_count, :trial, :dataset, :context_index,
        :gap_clean, :gap_perturbed, :gap_delta,
        :relative_gap_clean, :relative_gap_perturbed, :relative_gap_delta,
        :cost_clean, :cost_perturbed, :saa_cost_perturbed, :gap_vs_saa)

    (; summary = rows, deltas = diff_rows, cross_summary, cross_details)
end

function main()
    options = parse_cli_args(ARGS)
    sample_limit = parse(Int, get(options, "sample-limit", "30"))
    scenario_samples = parse(Int, get(options, "scenario-samples", "30"))
    perturb_seed = parse(Int, get(options, "perturb-seed", "2025"))

    problem_instance = create_problem_instance()

    scenario_counts = [1, 3]
    all_summary = DataFrame(model = String[], scenario_count = Int[], trial = Int[],
        dataset = String[], perturbation = String[],
        mean_gap = Float64[], mean_relative_gap = Float64[])

    all_deltas = DataFrame(model = String[], scenario_count = Int[], trial = Int[],
        dataset = String[], gap_delta = Float64[], relative_gap_delta = Float64[])

    all_cross_summary = DataFrame(model = String[], scenario_count = Int[], trial = Int[],
        dataset = String[], mean_gap_clean = Float64[], mean_gap_perturbed = Float64[],
        mean_gap_delta = Float64[], mean_relative_gap_clean = Float64[],
        mean_relative_gap_perturbed = Float64[], mean_relative_gap_delta = Float64[],
        mean_gap_vs_saa = Float64[])

    all_cross_details = DataFrame(model = String[], scenario_count = Int[], trial = Int[],
        dataset = String[], context_index = Int[],
        gap_clean = Float64[], gap_perturbed = Float64[], gap_delta = Float64[],
        relative_gap_clean = Float64[], relative_gap_perturbed = Float64[],
        relative_gap_delta = Float64[], cost_clean = Float64[], cost_perturbed = Float64[],
        saa_cost_perturbed = Float64[], gap_vs_saa = Float64[])

    for sc in scenario_counts
        info = select_best_trial(sc)
        results = evaluate_scenario(problem_instance, info;
            sample_limit = sample_limit, scenario_samples = scenario_samples,
            perturb_seed = perturb_seed)
        all_summary = vcat(all_summary, results.summary; cols = :union)
        all_deltas = vcat(all_deltas, results.deltas; cols = :union)
        all_cross_summary = vcat(all_cross_summary, results.cross_summary; cols = :union)
        all_cross_details = vcat(all_cross_details, results.cross_details; cols = :union)
    end

    println("\n=== Summary of Optimality Gaps ===")
    for row in eachrow(all_summary)
        println(@sprintf("Scenario %d (trial %02d) | %s | %s | %s: mean gap = %.4f, mean rel gap = %.4f",
            row.scenario_count, row.trial, row.model, row.dataset, row.perturbation,
            row.mean_gap, row.mean_relative_gap))
    end

    println("\n=== Gap Differences (Perturbed - Clean) ===")
    for row in eachrow(all_deltas)
        println(@sprintf("Scenario %d (trial %02d) | %s | %s: Δgap = %.4f, Δrel gap = %.4f",
            row.scenario_count, row.trial, row.model, row.dataset,
            row.gap_delta, row.relative_gap_delta))
    end

    if nrow(all_cross_summary) > 0
        println("\n=== Cross-Deployment Robustness (Clean decision evaluated on perturbed contexts) ===")
        for row in eachrow(all_cross_summary)
            println(@sprintf("Scenario %d (trial %02d) | %s | %s: clean gap = %.4f, pert gap = %.4f, Δgap = %.4f, gap vs SAA = %.4f",
                row.scenario_count, row.trial, row.model, row.dataset,
                row.mean_gap_clean, row.mean_gap_perturbed, row.mean_gap_delta, row.mean_gap_vs_saa))
        end
    end

    output_dir = joinpath(@__DIR__, "outputs")
    mkpath(output_dir)
    CSV.write(joinpath(output_dir, "gap_summary.csv"), all_summary)
    CSV.write(joinpath(output_dir, "gap_deltas.csv"), all_deltas)
    CSV.write(joinpath(output_dir, "cross_deployment_summary.csv"), all_cross_summary)
    CSV.write(joinpath(output_dir, "cross_deployment_details.csv"), all_cross_details)
    println("\nSaved results to $(output_dir)")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
