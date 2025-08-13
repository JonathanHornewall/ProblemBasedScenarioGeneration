using StatsPlots

using CSV
using DataFrames

df = CSV.read("scripts/df1.csv", DataFrame)

filtered = filter(row -> row.T == 100 && row.method == "M5 + AD", df)

MAD_gaps = filtered.OoS


function primal_problem_cost_SAA(problem_instance::ResourceAllocationProblem, Ws, Ts, hs, qs, regularization_parameter, first_stage_decision)
    A, b, c = problem_instance.s1_constraint_matrix, problem_instance.s1_constraint_vector, problem_instance.s1_cost_vector
    twoslp = TwoStageSLP(A, b, c, Ws, Ts, hs, qs)
    cost = cost_2s_LogBarCanLP(twoslp, first_stage_decision, regularization_parameter)
    return cost
end

function testing_SAA(problem_instance, model, dataset_testing, regularization_parameter, N_xi_per_x, Noutofsamples)
    list_gaps = [] 


    for (x, ξ) in dataset_testing
        
        # Determine optimal cost
        A, b, c = problem_instance.s1_constraint_matrix, problem_instance.s1_constraint_vector, problem_instance.s1_cost_vector
        Ws, Ts, hs, qs = [], [], [], []
        for k in 1:N_xi_per_x
            W, T, h, q = scenario_realization(problem_instance, ξ[k,:])
            push!(Ws,W)
            push!(Ts,T)
            push!(hs,h)
            push!(qs,q)
        end
        twoslp = TwoStageSLP(A, b, c, Ws, Ts, hs, qs)
        logbarlp = LogBarCanLP(twoslp, regularization_parameter)
        opt_cost = optimal_value(logbarlp)

        ξ_hat = model(x)
        surrogate_decision = surrogate_solution(problem_instance, ξ_hat, regularization_parameter)

        evaluated_cost = primal_problem_cost_SAA(problem_instance, Ws, Ts, hs, qs, regularization_parameter, surrogate_decision)


        push!(list_gaps, evaluated_cost - opt_cost)
        println("evaluated_cost", evaluated_cost, " optimal_cost", opt_cost, "gap", evaluated_cost - opt_cost)
    end

    losses = randn(100) .+ 0.5  # example data

    # Boxplot with one group
    all_data = vcat(list_gaps, MAD_gaps)
    groups = vcat(fill("NN", length(list_gaps)), fill("M5 + AD", length(MAD_gaps)))

    plot = boxplot(groups, all_data,
        legend = false,
        title = "Boxplot NN vs M5 + AD",
        xlabel = "Method",
        ylabel = "Value",
    )

    display(plot)
    savefig(plot, "gap_boxplot.pdf") 

end