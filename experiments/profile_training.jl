#!/usr/bin/env julia
#
# Profile the training loop for all 3 problems.
#
# Measures wall-clock time of each pipeline stage:
#   1. Neural network forward pass
#   2. Scenario construction (scenario_realization)
#   3. Extensive form construction
#   4. Surrogate LP solve (solve_lp in surrogate_first_stage)
#   5. Recourse LP solve (solve_lp in recourse_cost)
#   6. Backward pass (Flux.gradient total minus forward)
#   7. Parameter update (Flux.update!)
#   8. Loss tracking (forward pass without gradient)
#   9. GC
#
# Outputs:
#   experiments/profile_report.txt   — text report
#   experiments/profile_plot.png     — stacked bar chart
#

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "src", "ProblemBasedScenarioGeneration"))

using ProblemBasedScenarioGeneration
using Flux, Statistics, Random, Dates

ENV["GKSwstype"] = "100"
using Plots
using StatsPlots

# ── Helpers ──────────────────────────────────────────────────────────────

"""
Instrument one gradient step, timing each sub-stage.
Returns a Dict of stage => seconds.
"""
function profile_one_step(model, prob, sample; mu_surr=1.0, mu_prim=1.0)
    x, actual_sc = sample
    timings = Dict{String,Float64}()

    # ── Forward-pass components (outside gradient, for isolated timing) ──

    # 1. Neural net forward
    t = @elapsed raw_params = model(x)
    timings["1_nn_forward"] = t

    # 2. Scenario construction
    t = @elapsed predicted_sc = ProblemBasedScenarioGeneration._params_to_scenarios(prob, raw_params)
    timings["2_scenario_construct"] = t

    # 3. Extensive form
    A1, b1, c1 = first_stage_data(prob)
    slp_surr = TwoStageSLP(Float64.(A1), Float64.(b1), Float64.(c1),
        [Scenario(Float64.(sc.W), Float64.(sc.T), Float64.(sc.h), Float64.(sc.q))
         for sc in predicted_sc])
    t = @elapsed (A_ext, b_ext, c_ext) = extensive_form(slp_surr)
    timings["3_extensive_form"] = t

    # 4. Surrogate LP solve
    t = @elapsed begin
        x_full, _ = solve_lp(A_ext, b_ext, c_ext)
        x1_hat = x_full[1:length(c1)]
    end
    timings["4_surrogate_lp"] = t

    # 5. Recourse LP solve
    slp_prim = TwoStageSLP(Float64.(A1), Float64.(b1), Float64.(c1),
        [Scenario(Float64.(actual_sc.W), Float64.(actual_sc.T),
                  Float64.(actual_sc.h), Float64.(actual_sc.q))])
    t = @elapsed begin
        for s in 1:length(slp_prim.scenarios)
            sc = slp_prim.scenarios[s]
            b_rec = sc.h - sc.T * x1_hat
            solve_lp(sc.W, b_rec, sc.q)
        end
    end
    timings["5_recourse_lp"] = t

    # ── Full gradient step ──

    opt_state = Flux.setup(Adam(1e-3), model)

    t_grad = @elapsed begin
        gs = Flux.gradient(model) do m
            rp = m(x)
            psc = ProblemBasedScenarioGeneration._params_to_scenarios(prob, rp)
            decision_regret(prob, mu_surr, mu_prim, psc, actual_sc)
        end
    end
    timings["6_full_gradient"] = t_grad

    # 7. Parameter update
    gmodel = gs isa Tuple ? gs[1] : gs
    t = @elapsed Flux.update!(opt_state, model, gmodel)
    timings["7_param_update"] = t

    # 8. Loss tracking (forward without gradient tape)
    t = @elapsed begin
        rp2 = model(x)
        psc2 = ProblemBasedScenarioGeneration._params_to_scenarios(prob, rp2)
        decision_regret(prob, mu_surr, mu_prim, psc2, actual_sc)
    end
    timings["8_loss_tracking"] = t

    # 9. GC
    t = @elapsed GC.gc()
    timings["9_gc"] = t

    # Derived: backward = full_gradient - forward components
    forward_total = timings["1_nn_forward"] + timings["2_scenario_construct"] +
                    timings["3_extensive_form"] + timings["4_surrogate_lp"] +
                    timings["5_recourse_lp"]
    timings["6b_backward_pass"] = max(0.0, t_grad - forward_total)
    timings["6a_forward_in_grad"] = t_grad - timings["6b_backward_pass"]

    return timings
end

"""
Run profiling for one problem: warm up, then collect N_SAMPLES timing measurements.
"""
function profile_problem(prob, prob_name; n_samples=20, hidden_dim=64, n_layers=2)
    println("\n", "="^60)
    println("Profiling: $prob_name")
    println("="^60)

    Random.seed!(42)
    dataset = generate_dataset(prob, n_samples + 5)
    model = build_generator(prob; nr_of_scenarios=1, hidden_dim=hidden_dim, n_layers=n_layers)

    # Warm-up (JIT compilation)
    println("  Warming up (JIT)...")
    for i in 1:3
        profile_one_step(model, prob, dataset[i])
    end

    # Collect timings
    println("  Collecting $n_samples samples...")
    all_timings = Dict{String,Vector{Float64}}()
    for i in 1:n_samples
        t = profile_one_step(model, prob, dataset[i + 3])
        for (k, v) in t
            push!(get!(all_timings, k, Float64[]), v)
        end
    end

    return all_timings
end

# ── Reporting ────────────────────────────────────────────────────────────

# Display order for the main pipeline stages
const STAGE_ORDER = [
    "1_nn_forward",
    "2_scenario_construct",
    "3_extensive_form",
    "4_surrogate_lp",
    "5_recourse_lp",
    "6b_backward_pass",
    "7_param_update",
    "8_loss_tracking",
    "9_gc",
]

const STAGE_LABELS = Dict(
    "1_nn_forward"        => "NN forward pass",
    "2_scenario_construct" => "Scenario construction",
    "3_extensive_form"     => "Extensive form build",
    "4_surrogate_lp"       => "Surrogate LP solve",
    "5_recourse_lp"        => "Recourse LP solve",
    "6_full_gradient"      => "Full gradient step",
    "6a_forward_in_grad"   => "Forward (in gradient)",
    "6b_backward_pass"     => "Backward pass (AD)",
    "7_param_update"       => "Parameter update",
    "8_loss_tracking"      => "Loss tracking (no grad)",
    "9_gc"                 => "Garbage collection",
)

function format_report(results::Dict{String, Dict{String,Vector{Float64}}})
    lines = String[]
    push!(lines, "=" ^ 80)
    push!(lines, "  TRAINING LOOP PROFILING REPORT")
    push!(lines, "  $(Dates.now())")
    push!(lines, "=" ^ 80)
    push!(lines, "")
    push!(lines, "Methodology: Each stage is timed independently with @elapsed over")
    push!(lines, "N=20 gradient steps per problem (after 3 warm-up steps for JIT).")
    push!(lines, "Architecture: hidden_dim=64, n_layers=2, activation=relu")
    push!(lines, "Settings: mu_surr=1.0, mu_prim=1.0, batchsize=1")
    push!(lines, "")

    for (prob_name, timings) in sort(collect(results); by=first)
        push!(lines, "-" ^ 80)
        push!(lines, "  Problem: $prob_name")
        push!(lines, "-" ^ 80)

        # Compute total per-step time
        total_per_step = sum(mean(timings[s]) for s in STAGE_ORDER)

        push!(lines, "")
        push!(lines, "  Total per gradient step: $(round(total_per_step * 1000; digits=1)) ms")
        push!(lines, "")
        push!(lines, "  Stage                      │  Mean (ms) │  Std (ms)  │  % of step │ Median (ms)")
        push!(lines, "  ───────────────────────────┼────────────┼────────────┼────────────┼────────────")

        for stage in STAGE_ORDER
            vals = timings[stage]
            m = mean(vals) * 1000
            s = std(vals) * 1000
            med = median(vals) * 1000
            pct = mean(vals) / total_per_step * 100
            label = rpad(get(STAGE_LABELS, stage, stage), 27)
            push!(lines, "  $label │ $(lpad(round(m; digits=2), 10)) │ $(lpad(round(s; digits=2), 10)) │ $(lpad(round(pct; digits=1), 9))% │ $(lpad(round(med; digits=2), 10))")
        end

        # Also show the full gradient step vs forward+backward breakdown
        push!(lines, "")
        full_grad = mean(timings["6_full_gradient"]) * 1000
        fwd_in_grad = mean(timings["6a_forward_in_grad"]) * 1000
        bwd = mean(timings["6b_backward_pass"]) * 1000
        push!(lines, "  Full gradient step: $(round(full_grad; digits=1)) ms")
        push!(lines, "    ├─ Forward (in grad):  $(round(fwd_in_grad; digits=1)) ms ($(round(fwd_in_grad/full_grad*100; digits=1))%)")
        push!(lines, "    └─ Backward (AD):      $(round(bwd; digits=1)) ms ($(round(bwd/full_grad*100; digits=1))%)")
        push!(lines, "")
    end

    push!(lines, "=" ^ 80)
    push!(lines, "  KEY FINDINGS")
    push!(lines, "=" ^ 80)
    push!(lines, "")

    # Find bottleneck per problem
    for (prob_name, timings) in sort(collect(results); by=first)
        total = sum(mean(timings[s]) for s in STAGE_ORDER)
        sorted_stages = sort(STAGE_ORDER, by=s -> -mean(timings[s]))
        top3 = sorted_stages[1:min(3, length(sorted_stages))]
        push!(lines, "  $prob_name ($(round(total*1000; digits=1)) ms/step):")
        for (i, s) in enumerate(top3)
            m = mean(timings[s]) * 1000
            pct = mean(timings[s]) / total * 100
            label = get(STAGE_LABELS, s, s)
            push!(lines, "    $i. $label: $(round(m; digits=1)) ms ($(round(pct; digits=1))%)")
        end
        push!(lines, "")
    end

    return join(lines, "\n")
end

# ── Plotting ─────────────────────────────────────────────────────────────

function generate_plot(results::Dict{String, Dict{String,Vector{Float64}}}, path::String)

    prob_names = sort(collect(keys(results)))
    n_probs = length(prob_names)
    n_stages = length(STAGE_ORDER)

    # Build data matrix: rows = problems, cols = stages
    data = zeros(n_probs, n_stages)
    for (i, pn) in enumerate(prob_names)
        for (j, s) in enumerate(STAGE_ORDER)
            data[i, j] = mean(results[pn][s]) * 1000  # ms
        end
    end

    labels = [get(STAGE_LABELS, s, s) for s in STAGE_ORDER]

    colors = [:dodgerblue, :seagreen, :goldenrod, :tomato, :darkorange,
              :mediumpurple, :gray60, :lightcoral, :lightsteelblue]

    p = groupedbar(
        data,
        bar_position = :stack,
        labels = reshape(labels, 1, :),
        xlabel = "",
        ylabel = "Time per gradient step (ms)",
        title = "Training Loop Profile — Time Breakdown by Stage",
        xticks = (1:n_probs, prob_names),
        legend = :outerright,
        size = (1100, 550),
        color = reshape(colors[1:n_stages], 1, :),
        linewidth = 0.5,
        bar_width = 0.6,
        left_margin = 10Plots.mm,
        right_margin = 35Plots.mm,
        bottom_margin = 10Plots.mm,
    )

    # Add total time annotation on top of each bar
    for (i, pn) in enumerate(prob_names)
        total_ms = sum(data[i, :])
        annotate!(p, i, total_ms + total_ms * 0.03, text("$(round(total_ms; digits=0)) ms", 8, :center))
    end

    savefig(p, path)
    println("Plot saved: $path")
end

# Also generate a percentage-based chart
function generate_pct_plot(results::Dict{String, Dict{String,Vector{Float64}}}, path::String)

    prob_names = sort(collect(keys(results)))
    n_probs = length(prob_names)
    n_stages = length(STAGE_ORDER)

    # Build percentage data: rows = problems, cols = stages
    data = zeros(n_probs, n_stages)
    for (i, pn) in enumerate(prob_names)
        total = sum(mean(results[pn][s]) for s in STAGE_ORDER)
        for (j, s) in enumerate(STAGE_ORDER)
            data[i, j] = mean(results[pn][s]) / total * 100
        end
    end

    labels = [get(STAGE_LABELS, s, s) for s in STAGE_ORDER]

    colors = [:dodgerblue, :seagreen, :goldenrod, :tomato, :darkorange,
              :mediumpurple, :gray60, :lightcoral, :lightsteelblue]

    p = groupedbar(
        data,
        bar_position = :stack,
        labels = reshape(labels, 1, :),
        xlabel = "",
        ylabel = "Percentage of gradient step (%)",
        title = "Training Loop Profile — Relative Time per Stage",
        xticks = (1:n_probs, prob_names),
        legend = :outerright,
        size = (1100, 550),
        color = reshape(colors[1:n_stages], 1, :),
        linewidth = 0.5,
        bar_width = 0.6,
        ylims = (0, 105),
        left_margin = 10Plots.mm,
        right_margin = 35Plots.mm,
        bottom_margin = 10Plots.mm,
    )

    savefig(p, path)
    println("Plot saved: $path")
end

# Compute-only plot (excludes GC to show actual computation breakdown)
function generate_compute_plot(results::Dict{String, Dict{String,Vector{Float64}}}, path::String)
    prob_names = sort(collect(keys(results)))
    n_probs = length(prob_names)

    # Stages excluding GC
    compute_stages = filter(s -> s != "9_gc", STAGE_ORDER)
    n_stages = length(compute_stages)

    # Build data
    data = zeros(n_probs, n_stages)
    gc_times = zeros(n_probs)
    for (i, pn) in enumerate(prob_names)
        for (j, s) in enumerate(compute_stages)
            data[i, j] = mean(results[pn][s]) * 1000
        end
        gc_times[i] = mean(results[pn]["9_gc"]) * 1000
    end

    labels = [get(STAGE_LABELS, s, s) for s in compute_stages]
    colors = [:dodgerblue, :seagreen, :goldenrod, :tomato, :darkorange,
              :mediumpurple, :gray60, :lightcoral]

    p = groupedbar(
        data,
        bar_position = :stack,
        labels = reshape(labels, 1, :),
        xlabel = "",
        ylabel = "Time (ms)",
        title = "Computation Time per Gradient Step (excluding GC)",
        xticks = (1:n_probs, prob_names),
        legend = :outerright,
        size = (1100, 550),
        color = reshape(colors[1:n_stages], 1, :),
        linewidth = 0.5,
        bar_width = 0.6,
        left_margin = 10Plots.mm,
        right_margin = 35Plots.mm,
        bottom_margin = 10Plots.mm,
    )

    # Annotate total compute time and GC time
    for (i, pn) in enumerate(prob_names)
        compute_ms = sum(data[i, :])
        annotate!(p, i, compute_ms + compute_ms * 0.05,
            text("$(round(compute_ms; digits=1)) ms compute\n+ $(round(gc_times[i]; digits=0)) ms GC", 7, :center))
    end

    savefig(p, path)
    println("Plot saved: $path")
end

# ── Main ─────────────────────────────────────────────────────────────────

function main()
    println("Training Loop Profiler")
    println("=" ^ 40)

    problems = Dict(
        "newsvendor"          => UnreliableNewsvendorProblem(),
        "resource_allocation" => ResourceAllocationProblem(),
        "shipment_planning"   => ShipmentPlanningProblem(),
    )

    results = Dict{String, Dict{String,Vector{Float64}}}()

    for name in sort(collect(keys(problems)))
        prob = problems[name]
        results[name] = profile_problem(prob, name; n_samples=20, hidden_dim=64, n_layers=2)
    end

    # Generate report
    report = format_report(results)
    println("\n", report)

    report_path = joinpath(@__DIR__, "profile_report.txt")
    open(report_path, "w") do f
        write(f, report)
    end
    println("Report saved: $report_path")

    # Generate plots
    plot_path = joinpath(@__DIR__, "profile_plot.png")
    generate_plot(results, plot_path)

    pct_path = joinpath(@__DIR__, "profile_pct_plot.png")
    generate_pct_plot(results, pct_path)

    # Generate compute-only plot (excluding GC) — the interesting breakdown
    compute_path = joinpath(@__DIR__, "profile_compute_only.png")
    generate_compute_plot(results, compute_path)

    println("\nDone!")
end

main()
