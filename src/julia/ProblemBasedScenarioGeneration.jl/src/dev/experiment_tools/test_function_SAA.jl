"""
Experimental SAA testing utilities, copied from scripts.

Notes:
- This copy keeps the original logic and dependencies (StatsPlots, JuMP, Gurobi, CSV, DataFrames).
- The optional M5+AD baseline (MAD_gaps) is loaded from a CSV if available.
  If the CSV is not found, plotting will proceed with the NN group only.
"""

using StatsPlots
using Statistics
using JuMP
import MathOptInterface as MOI
using Gurobi
using CSV
using DataFrames

function _load_MAD_gaps(; csv_path::AbstractString = "tests_SAA/df1.csv")
    try
        if isfile(csv_path)
            df = CSV.read(csv_path, DataFrame)
            filtered = filter(row -> row.T == 100 && row.method == "M5 + AD", df)
            return filtered.OoS
        else
            @warn "MAD_gaps CSV not found; skipping baseline" csv_path
            return Float64[]
        end
    catch e
        @warn "Failed to load MAD_gaps; skipping baseline" exception=(e, catch_backtrace())
        return Float64[]
    end
end

const MAD_gaps = _load_MAD_gaps()

function testing_SAA(problem_instance, model, dataset_testing, reg_param_surr, reg_param_ref, N_xi_per_x)
    UCB_list = []

    # dataset_testing provides pairs (x, ξ) where ξ has shape 30×N_xi_per_x×30
    for (x, ξ) in dataset_testing
        # Determine optimal cost
        A, b, c = problem_instance.s1_constraint_matrix, problem_instance.s1_constraint_vector, problem_instance.s1_cost_vector

        list_gaps, list_costs = [], []

        for m in 1:30
            Ws, Ts, hs, qs = [], [], [], []
            for k in 1:N_xi_per_x
                W, T, h, q = scenario_realization(problem_instance, ξ[m, k, :])
                push!(Ws, W); push!(Ts, T); push!(hs, h); push!(qs, q)
            end

            # Convert vectors to proper 3D/2D arrays for TwoStageSLP
            Ws_array = cat(Ws..., dims = 3)
            Ts_array = cat(Ts..., dims = 3)
            hs_array = hcat(hs...)
            qs_array = hcat(qs...)

            two_slp = TwoStageSLP(A, b, c, Ws_array, Ts_array, hs_array, qs_array)
            can_lp = CanLP(two_slp)
            opt_cost = optimal_value(can_lp)

            ξ_hat = model(x)
            # Reshape the neural network output from a vector to a matrix with one column
            ξ_hat_matrix = reshape(ξ_hat, :, 1)
            surrogate_decision = surrogate_solution(problem_instance, ξ_hat_matrix, reg_param_surr)

            evaluated_cost = s1_cost(two_slp, surrogate_decision, reg_param_ref)

            push!(list_gaps, evaluated_cost - opt_cost)
            push!(list_costs, evaluated_cost)

            println("evaluated_cost: ", evaluated_cost, " optimal_cost: ", opt_cost, " gap: ", (evaluated_cost - opt_cost) / abs(opt_cost))
        end

        # compute 99% confidence upper bound for x
        cost_mean = mean(list_costs)
        UCB = (100 / abs(cost_mean)) * ((1 / 30) * sum(list_gaps[k] + 2.462 * sqrt((var(list_gaps) / 30)) for k in 1:30))
        push!(UCB_list, UCB)
    end

    # Boxplot: always show NN; show M5+AD if available
    groups = fill("NN", length(UCB_list))
    all_data = copy(UCB_list)
    if !isempty(MAD_gaps)
        groups = vcat(groups, fill("M5 + AD", length(MAD_gaps)))
        append!(all_data, MAD_gaps)
    end

    plot = boxplot(groups, all_data,
        legend = false,
        title = "",
        xlabel = "",
        ylabel = "Gap",
    )

    display(plot)
    savefig(plot, "gap_boxplot.pdf")
end

function gurobi_solver(A, b, c, Ws, Ts, hs, qs, first_stage_decision)
    # Compute the cost with a fixed or unfixed first stage decision
    n = length(c)              # number of first stage decision variables
    m = size(Ws[1], 2)         # number of second stage decision variables
    S = length(Ws)             # number of scenarios

    model = Model(Gurobi.Optimizer)
    set_optimizer_attribute(model, "OutputFlag", 0)

    @variable(model, z[1:n] >= 0)             # first stage decision variable
    @variable(model, u[1:S, 1:m] >= 0)        # second stage decision variable

    @constraint(model, A * z .== b)
    if first_stage_decision !== nothing
        @constraint(model, z == first_stage_decision)
    end

    for s in 1:S
        @constraint(model, Ws[s] * u[s, :] + Ts[s] * z .== hs[s])
    end

    @objective(model, Min,
        sum(c[i] * z[i] for i in 1:n) + (1 / S) * sum(qs[s][i] * u[s, i] for s in 1:S, i in 1:m))

    optimize!(model)

    ts = termination_status(model)
    if !(ts in (MOI.OPTIMAL, MOI.LOCALLY_SOLVED, MOI.ALMOST_LOCALLY_SOLVED))
        error("No feasible/optimal solution: $(ts) — $(MOI.get(model, MOI.RawStatusString()))")
    end

    return objective_value(model)
end
