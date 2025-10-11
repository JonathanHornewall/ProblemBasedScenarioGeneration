import Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

using LinearAlgebra
using Random
using Statistics
using Printf
using Serialization
using Flux

using ProblemBasedScenarioGeneration
using ProblemBasedScenarioGeneration: ResourceAllocationProblemData, ResourceAllocationProblem,
    dataGeneration, scenario_realization, TwoStageSLP, CanLP, optimal_value,
    s1_cost, loss, relative_loss, construct_neural_network,
    LogBarCanLP, LogBarCanLP_standard_solver

import Flux: params, Optimise, Adam

include(joinpath(@__DIR__, "..", "parameters.jl"))

cz_vec, qw_vec, ρ_vec = vec(cz), vec(qw), vec(ρᵢ)

struct KNNAnchor
    idxs::Vector{Vector{Int}}
    mu_list::Vector{Matrix{Float64}}  # each is J×1
end

"""
    build_knn_anchor(xs::Vector{Vector{Float64}}, xis::Vector{Matrix{Float64}}; k::Int=10)

Precompute kNN neighbor indices and local mean μ̂(x) for each training context.
"""
function build_knn_anchor(xs::Vector{Vector{Float64}}, xis::Vector{Matrix{Float64}}; k::Int=10)
    n = length(xs)
    # Pre-sort indices by x to ensure deterministic neighbor selection
    order = collect(1:n)
    sort!(order, by = i -> tuple(xs[i]...))
    xs_sorted = xs[order]
    xis_sorted = xis[order]

    idxs = Vector{Vector{Int}}(undef, n)
    mu_list = Vector{Matrix{Float64}}(undef, n)
    for (pos, i) in enumerate(order)
        x = xs_sorted[pos]
        # compute distances to others (in sorted space)
        dists = [(j, LinearAlgebra.norm(x .- xs_sorted[j])) for j in 1:n if j != pos]
        sort!(dists, by = t -> t[2])
        take = first(dists, min(k, length(dists)))
        neigh_sorted_idx = [t[1] for t in take]
        # map back to original indexing
        neigh_orig_idx = [order[j] for j in neigh_sorted_idx]
        idxs[i] = neigh_orig_idx
        # average neighbor ξ vectors
        J = size(xis_sorted[1], 1)
        mu = zeros(J, 1)
        for j in neigh_sorted_idx
            mu .+= xis_sorted[j]
        end
        mu ./= max(length(neigh_sorted_idx), 1)
        mu_list[i] = mu
    end
    return KNNAnchor(idxs, mu_list)
end

"""
Simple dropout-augmented network for resource allocation.
"""
function construct_regularized_network(instance::ResourceAllocationProblem; nr_of_scenarios::Int=1, pdrop::Float64=0.1)
    scenario_dim = size(instance.problem_data.service_rate_parameters, 2)
    output_dim = scenario_dim * nr_of_scenarios
    m = Chain(
        Dense(3, 128, relu), Dropout(pdrop),
        Dense(128, 128, relu), Dropout(pdrop),
        Dense(128, 128, relu), Dropout(pdrop),
        Dense(128, output_dim, relu),
        x -> reshape(x, scenario_dim, nr_of_scenarios)
    ) |> f64
    return m
end

"""
Compute batch decision-focused loss with regularization terms:
- decision_loss: original loss()
- anchor (kNN mean) penalty
- consistency penalty under x-perturbations
- L2 weight decay
"""
function batch_total_loss(problem_instance, model, xs_batch::Vector{Vector{Float64}}, xis_batch::Vector{Matrix{Float64}},
        reg_param_surr::Float64, reg_param_prim::Float64,
        anchor::KNNAnchor, λ_anchor::Float64, λ_cons::Float64, λ_l2::Float64, σ_x::Float64)

    # Decision-focused component
    dec_losses = Float64[]
    anchor_losses = Float64[]
    cons_losses = Float64[]

    for (x, ξ_true) in zip(xs_batch, xis_batch)
        x_mat = reshape(x, :, 1)
        ξ_pred = model(x_mat)
        # Decision-focused
        push!(dec_losses, loss(problem_instance, reg_param_surr, reg_param_prim, ξ_pred, ξ_true))
        # Anchor to local mean
        μ̂ = anchor.mu_list[findfirst(y -> y === x, xs_batch) === nothing ? 1 : findfirst(y -> y === x, xs_batch)]
        push!(anchor_losses, sum((ξ_pred .- μ̂).^2) / length(ξ_pred))
        # Consistency under x-perturbation
        if σ_x > 0
            x_pert = x .+ σ_x .* randn(length(x))
            ξ_pred_pert = model(reshape(x_pert, :, 1))
            push!(cons_losses, sum((ξ_pred_pert .- ξ_pred).^2) / length(ξ_pred))
        end
    end

    L_dec = mean(dec_losses)
    L_anchor = isempty(anchor_losses) ? 0.0 : mean(anchor_losses)
    L_cons = isempty(cons_losses) ? 0.0 : mean(cons_losses)

    # L2 regularization
    L_l2 = sum(sum(abs2, p) for p in Flux.params(model))
    return L_dec + λ_anchor * L_anchor + λ_cons * L_cons + λ_l2 * L_l2,
           (L_dec=L_dec, L_anchor=L_anchor, L_cons=L_cons, L_l2=L_l2)
end

function evaluate_on_contexts(problem_instance, model, xs::Vector{Vector{Float64}}, A, B; σ::Float64=5.0, p::Int=2, L::Int=3, scenario_samples::Int=30)
    # Build SAA batches for each context and compute regret
    gaps = Float64[]
    rels = Float64[]
    for x in xs
        # Build scenario collection by sampling noise around mean
        means = zeros(size(B,1))
        for j in 1:length(means)
            total = A[j]
            for ℓ in 1:L
                total += B[j, ℓ] * (x[ℓ])^p
            end
            means[j] = total
        end
        ξ_mat = repeat(means, 1, scenario_samples) .+ σ .* randn(length(means), scenario_samples)

        # Build TwoStageSLP
        A1 = problem_instance.s1_constraint_matrix
        b1 = problem_instance.s1_constraint_vector
        c1 = problem_instance.s1_cost_vector
        Ws_list, Ts_list, hs_list, qs_list = Any[], Any[], Any[], Any[]
        for k in 1:scenario_samples
            W, T, h, q = scenario_realization(problem_instance, ξ_mat[:, k])
            push!(Ws_list, W); push!(Ts_list, T); push!(hs_list, h); push!(qs_list, q)
        end
        Ws = cat(Ws_list...; dims=3)
        Ts = cat(Ts_list...; dims=3)
        hs = hcat(hs_list...)
        qs = hcat(qs_list...)
        two_slp = TwoStageSLP(A1, b1, c1, Ws, Ts, hs, qs)
        opt_cost = optimal_value(CanLP(two_slp))

        # Model decision
        ξ_pred = model(reshape(x, :, 1))
        # Convert predicted scenario to surrogate decision
        Wp, Tp, hp, qp = scenario_realization(problem_instance, vec(ξ_pred))
        Wps = cat(Wp; dims=3)
        Tps = cat(Tp; dims=3)
        hps = reshape(hp, :, 1)
        qps = reshape(qp, :, 1)
        decision, _ = LogBarCanLP_standard_solver(LogBarCanLP(TwoStageSLP(A1, b1, c1, Wps, Tps, hps, qps), 0.0))
        z = decision[1:length(c1)]
        eval_cost = s1_cost(two_slp, z, 0.0)
        gap = eval_cost - opt_cost
        push!(gaps, gap)
        push!(rels, gap / max(abs(opt_cost), eps()))
    end
    return mean(gaps), mean(rels)
end

function main()
    # Hyperparameters
    Ntrain = 100
    Ntest = 30
    N_xi_per_x = 100
    σ = 5.0
    p = 2
    Ldeg = 3
    collections_per_sample = 1

    # Regularization weights
    λ_anchor = 1e-3
    λ_cons = 1e-3
    λ_l2 = 1e-4
    σ_x = 0.01
    k_neighbors = 10
    pdrop = 0.1

    # Training config
    batchsize = 10
    epochs = 40
    step_size = 1e-3
    val_split = 15  # number of contexts held out
    seed = 2025
    Random.seed!(seed)

    # Build problem
    problem_data = ResourceAllocationProblemData(μᵢⱼ, cz_vec, qw_vec, ρ_vec)
    problem_instance = ResourceAllocationProblem(problem_data)

    # Data
    train_dict, test_dict, A, B = dataGeneration(problem_instance, Ntrain, Ntest, N_xi_per_x, σ, p, Ldeg, collections_per_sample)
    train_pairs = collect(train_dict)
    xs = [copy(t[1]) for t in train_pairs]
    xis = [reshape(copy(t[2]), :, 1) for t in train_pairs]

    # Validation split
    idxs = collect(1:length(xs))
    shuffle!(idxs)
    val_idx = idxs[1:val_split]
    tr_idx = idxs[val_split+1:end]

    xs_tr = xs[tr_idx]; xis_tr = xis[tr_idx]
    xs_val = xs[val_idx]; xis_val = xis[val_idx]

    # kNN anchor built on full training set (excluding val)
    anchor = build_knn_anchor(xs_tr, xis_tr; k=k_neighbors)

    # Model
    model = construct_regularized_network(problem_instance; nr_of_scenarios=1, pdrop=pdrop)

    # Training
    best_val = Inf
    best_bytes = nothing

    println("Starting regularized training...")
    for epoch in 1:epochs
        # mini-batch SGD
        perm = randperm(length(xs_tr))
        for batch in Iterators.partition(perm, batchsize)
            xb = xs_tr[batch]; yb = xis_tr[batch]
            gs = Flux.gradient(Flux.params(model)) do
                L, _ = batch_total_loss(problem_instance, model, xb, yb,
                    0.01, 0.01, anchor, λ_anchor, λ_cons, λ_l2, σ_x)
                L
            end
            Flux.Optimise.update!(Flux.setup(Adam(step_size), model), model, gs)
        end

        # validation decision-focused loss (no anchors/consistency)
        val_losses = Float64[]
        for (x, ξ_true) in zip(xs_val, xis_val)
            push!(val_losses, loss(problem_instance, 0.01, 0.01, model(reshape(x, :, 1)), ξ_true))
        end
        Lval = mean(val_losses)
        println(@sprintf("Epoch %3d | val decision loss = %.6f", epoch, Lval))
        if Lval < best_val
            best_val = Lval
            io = IOBuffer()
            Serialization.serialize(io, model)
            best_bytes = take!(io)
        end
    end

    if best_bytes !== nothing
        model = Serialization.deserialize(IOBuffer(best_bytes))
        Serialization.serialize(joinpath(@__DIR__, "regularized_model.jls"), model)
        println("Saved best model to regularized_model.jls")
    end

    # Evaluation: SAA benchmark on 30 random training contexts and on OOS
    tr_sel = shuffle!(collect(1:length(xs)))[1:30]
    xs_train_sample = xs[tr_sel]

    # pick all OOS contexts
    xs_oos = [copy(t[1]) for t in collect(test_dict)]

    println("\n=== SAA benchmark (trained model) ===")
    m_gap_tr, m_rel_tr = evaluate_on_contexts(problem_instance, model, xs_train_sample, A, B; σ=σ, p=p, L=Ldeg, scenario_samples=30)
    println(@sprintf("Train contexts (subset 30): mean gap = %.4f, mean rel gap = %.4f", m_gap_tr, m_rel_tr))

    m_gap_oos, m_rel_oos = evaluate_on_contexts(problem_instance, model, xs_oos, A, B; σ=σ, p=p, L=Ldeg, scenario_samples=30)
    println(@sprintf("OOS contexts:           mean gap = %.4f, mean rel gap = %.4f", m_gap_oos, m_rel_oos))
end

main()

