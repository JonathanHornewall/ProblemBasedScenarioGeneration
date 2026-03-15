"""
Evaluation metrics and plotting for trained scenario generators.

Provides:
- `compute_evaluation_metrics`: compute per-sample regret, relative regret, scenario distance
- `print_evaluation_summary`: formatted summary with 95% CI
- `save_evaluation_csv`: per-sample CSV output
- Plotting functions: loss curve, regret histogram, boxplot, CDF, scenario scatter
"""

using Statistics, LinearAlgebra

# Headless GR backend for plot generation
ENV["GKSwstype"] = "100"

using Plots, StatsPlots

# -----------------------------------------------------------------------
# Scenario parameter extraction (inverse of scenario_realization)
# -----------------------------------------------------------------------

function _extract_scenario_params(::UnreliableNewsvendorProblem, sc::Scenario)
    D = -sc.h[1]
    U = -sc.T[2, 1]
    return [D, U]
end

function _extract_scenario_params(prob::ShipmentPlanningProblem, sc::Scenario)
    n_loc = size(prob.shipment_costs, 2)
    return sc.h[1:n_loc]
end

function _extract_scenario_params(prob::ResourceAllocationProblem, sc::Scenario)
    I = size(prob.T, 2)
    return sc.h[(I+1):end]
end

# -----------------------------------------------------------------------
# Metrics computation
# -----------------------------------------------------------------------

"""
    compute_evaluation_metrics(generator, prob, test_data, mu_surr, mu_prim) -> Dict

Compute per-sample evaluation metrics over test data.

Returns a Dict with keys:
- `regrets`: absolute decision regrets
- `rel_regrets`: relative decision regrets
- `l2_distances`: L2 distance between predicted and actual scenario params
- `predicted_params`: vector of predicted param vectors
- `actual_params`: vector of actual param vectors
- `n`: number of samples
"""
function compute_evaluation_metrics(
    generator,
    prob::ProblemInstance,
    test_data::Vector{<:Tuple},
    mu_surr::Float64,
    mu_prim::Float64
)
    n = length(test_data)
    regrets = Vector{Float64}(undef, n)
    rel_regrets = Vector{Float64}(undef, n)
    l2_distances = Vector{Float64}(undef, n)
    predicted_params = Vector{Vector{Float64}}(undef, n)
    actual_params = Vector{Vector{Float64}}(undef, n)

    for (i, (x, actual_sc)) in enumerate(test_data)
        raw = generator(x)
        pred_sc = _params_to_scenarios(prob, raw)

        regrets[i] = decision_regret(prob, mu_surr, mu_prim, pred_sc, actual_sc)
        rel_regrets[i] = relative_decision_regret(prob, mu_surr, mu_prim, pred_sc, actual_sc)

        # Extract scenario parameters for distance computation
        sc_dim = _scenario_param_dim(prob)
        pred_p = Float64.(raw[1:sc_dim])
        actual_p = _extract_scenario_params(prob, actual_sc)
        predicted_params[i] = pred_p
        actual_params[i] = actual_p
        l2_distances[i] = norm(pred_p - actual_p)
    end

    return Dict(
        "regrets"          => regrets,
        "rel_regrets"      => rel_regrets,
        "l2_distances"     => l2_distances,
        "predicted_params" => predicted_params,
        "actual_params"    => actual_params,
        "n"                => n,
    )
end

# -----------------------------------------------------------------------
# Summary printing
# -----------------------------------------------------------------------

"""
    print_evaluation_summary(io, metrics)

Print formatted evaluation summary with 95% confidence intervals.
"""
function print_evaluation_summary(io::IO, metrics::Dict)
    regrets = metrics["regrets"]
    rel_regrets = metrics["rel_regrets"]
    l2_dists = metrics["l2_distances"]
    n = metrics["n"]

    println(io, "\n=== Evaluation Results ($n samples) ===")

    # Decision regret
    m_r = mean(regrets)
    s_r = std(regrets)
    ci_r = 1.96 * s_r / sqrt(n)
    println(io, "\nDecision regret:")
    println(io, "  Mean:      $m_r")
    println(io, "  Std:       $s_r")
    println(io, "  95% CI:    [$( m_r - ci_r ), $( m_r + ci_r )]")
    println(io, "  Median:    $(median(regrets))")
    println(io, "  Min:       $(minimum(regrets))")
    println(io, "  Max:       $(maximum(regrets))")

    # Relative decision regret
    m_rr = mean(rel_regrets)
    s_rr = std(rel_regrets)
    ci_rr = 1.96 * s_rr / sqrt(n)
    println(io, "\nRelative decision regret:")
    println(io, "  Mean:      $m_rr")
    println(io, "  Std:       $s_rr")
    println(io, "  95% CI:    [$( m_rr - ci_rr ), $( m_rr + ci_rr )]")
    println(io, "  Median:    $(median(rel_regrets))")
    println(io, "  Min:       $(minimum(rel_regrets))")
    println(io, "  Max:       $(maximum(rel_regrets))")

    # Threshold fractions
    println(io, "\nThreshold fractions (relative regret):")
    for threshold in [0.01, 0.05, 0.10]
        frac = count(x -> abs(x) < threshold, rel_regrets) / n
        pct = round(frac * 100; digits=1)
        println(io, "  < $(Int(threshold*100))%:     $pct% of samples")
    end

    # Scenario L2 distance
    println(io, "\nScenario L2 distance:")
    println(io, "  Mean:      $(mean(l2_dists))")
    println(io, "  Std:       $(std(l2_dists))")
    println(io, "  Median:    $(median(l2_dists))")
end

function print_evaluation_summary(metrics::Dict)
    print_evaluation_summary(stdout, metrics)
end

# -----------------------------------------------------------------------
# CSV output
# -----------------------------------------------------------------------

"""
    save_evaluation_csv(metrics, path)

Save per-sample metrics to a CSV file.
"""
function save_evaluation_csv(metrics::Dict, path::String)
    n = metrics["n"]
    regrets = metrics["regrets"]
    rel_regrets = metrics["rel_regrets"]
    l2_dists = metrics["l2_distances"]

    open(path, "w") do io
        println(io, "sample,decision_regret,relative_decision_regret,l2_distance")
        for i in 1:n
            println(io, "$i,$(regrets[i]),$(rel_regrets[i]),$(l2_dists[i])")
        end
    end
end

# -----------------------------------------------------------------------
# Plotting functions
# -----------------------------------------------------------------------

"""
    plot_loss_curve(loss_history, path)

Line plot of training loss vs epoch.
"""
function plot_loss_curve(loss_history::Vector{Float64}, path::String)
    Plots.plot(
        1:length(loss_history), loss_history;
        xlabel = "Epoch",
        ylabel = "Loss (decision regret)",
        title  = "Training Loss",
        legend = false,
        linewidth = 2,
        size   = (800, 500),
    )
    Plots.savefig(path)
end

"""
    plot_regret_histogram(rel_regrets, ci, path)

Histogram of relative regret with vertical mean line and CI annotation.
`ci` is the half-width of the 95% confidence interval on the mean.
"""
function plot_regret_histogram(rel_regrets::Vector{Float64}, ci::Float64, path::String)
    m = mean(rel_regrets)
    Plots.histogram(
        rel_regrets;
        xlabel = "Relative decision regret",
        ylabel = "Count",
        title  = "Regret Distribution",
        legend = false,
        size   = (800, 500),
        alpha  = 0.7,
    )
    Plots.vline!([m]; linewidth=2, color=:red, linestyle=:dash)
    annotate!(m, Plots.ylims()[2] * 0.9,
        Plots.text("Mean: $(round(m; digits=4)) +/- $(round(ci; digits=4))", 8, :left, :red))
    Plots.savefig(path)
end

"""
    plot_regret_boxplot(rel_regrets, path)

Violin/boxplot of relative regret distribution.
"""
function plot_regret_boxplot(rel_regrets::Vector{Float64}, path::String)
    StatsPlots.violin(
        ["Relative Regret"], rel_regrets;
        ylabel = "Relative decision regret",
        title  = "Regret Distribution",
        legend = false,
        size   = (600, 500),
        alpha  = 0.7,
    )
    StatsPlots.boxplot!(["Relative Regret"], rel_regrets; alpha=0.5)
    Plots.savefig(path)
end

"""
    plot_regret_cdf(rel_regrets, path)

Empirical CDF of relative regret.
"""
function plot_regret_cdf(rel_regrets::Vector{Float64}, path::String)
    sorted = sort(abs.(rel_regrets))
    n = length(sorted)
    fractions = (1:n) ./ n
    Plots.plot(
        sorted, fractions;
        xlabel    = "Absolute relative regret",
        ylabel    = "Fraction of samples",
        title     = "Empirical CDF of Regret",
        legend    = false,
        linewidth = 2,
        size      = (800, 500),
    )
    # Reference lines at common thresholds
    for thr in [0.01, 0.05, 0.10]
        frac = count(<(thr), sorted) / n
        Plots.hline!([frac]; color=:gray, linestyle=:dot, alpha=0.5)
        Plots.vline!([thr]; color=:gray, linestyle=:dot, alpha=0.5)
    end
    Plots.savefig(path)
end

"""
    plot_scenario_scatter(predicted, actual, path; max_dims=8)

Predicted vs actual scenario parameters with 45-degree reference line.
Generates subplots per parameter dimension. Skipped if dim > max_dims.
"""
function plot_scenario_scatter(
    predicted::Vector{Vector{Float64}},
    actual::Vector{Vector{Float64}},
    path::String;
    max_dims::Int = 8
)
    dim = length(predicted[1])
    dim > max_dims && return nothing

    n_cols = min(dim, 4)
    n_rows = ceil(Int, dim / n_cols)

    plts = []
    for d in 1:dim
        pred_d = [p[d] for p in predicted]
        act_d  = [a[d] for a in actual]
        lo = min(minimum(pred_d), minimum(act_d))
        hi = max(maximum(pred_d), maximum(act_d))
        margin = 0.05 * (hi - lo + 1e-10)

        p = Plots.scatter(
            act_d, pred_d;
            xlabel    = "Actual",
            ylabel    = "Predicted",
            title     = "Param $d",
            legend    = false,
            alpha     = 0.6,
            markersize = 3,
        )
        Plots.plot!(p, [lo - margin, hi + margin], [lo - margin, hi + margin];
            color=:red, linestyle=:dash, linewidth=1.5)
        push!(plts, p)
    end

    combined = Plots.plot(plts...; layout=(n_rows, n_cols),
        size=(300 * n_cols, 300 * n_rows), plot_title="Scenario Parameters")
    Plots.savefig(combined, path)
end
