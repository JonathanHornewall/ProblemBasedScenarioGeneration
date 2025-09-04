using StatsPlots
using JuMP
using Gurobi

using CSV
using DataFrames

# retrieve results for M5 +AD 
df = CSV.read("/home/delannoypavysol/ProblemBasedScenarioGeneration/ProblemBasedScenarioGeneration/scripts/resource_allocation_prototype/tests_SAA/df2.csv", DataFrame)

filtered_MAD = filter(row -> row.T == 100 && row.method == "M5 + AD", df)
MAD_gaps = filtered_MAD.OoS

filtered_LS = filter(row -> row.T == 100 && row.method == "LS", df)
LS_gaps = filtered_LS.OoS

filtered_ER_SAA = filter(row -> row.T == 100 && row.method == "ER-SAA", df)
ER_SAA_gaps = filtered_ER_SAA.OoS

filtered_AD = filter(row -> row.T == 100 && row.method == "AD", df)
AD_gaps = filtered_AD.OoS

filtered_SAA = filter(row -> row.T == 100 && row.method == "SAA", df)
SAA_gaps = filtered_SAA.OoS

filtered_KNN = filter(row -> row.T == 100 && row.method == "KNN", df)
KNN_gaps = filtered_KNN.OoS

filtered_CART = filter(row -> row.T == 100 && row.method == "CART", df)
CART_gaps = filtered_CART.OoS


function testing_SAA(problem_instance, model, dataset_testing, reg_param_surr, reg_param_ref, N_xi_per_x)
    
    UCB_list = []

    # I modified the code for data generation so that we test using the same algorithm as tito's paper:
    # we generate 30 convariate x
    # for each of them, ξ is of size 30*1000*30
    # ξ[m,k,:] is the sample k for the SAA with 1000 sample for test m with covariate x


    for (x, ξ) in dataset_testing
        
        # Determine optimal cost
        A, b, c = problem_instance.s1_constraint_matrix, problem_instance.s1_constraint_vector, problem_instance.s1_cost_vector

        list_gaps , list_costs = [] , []

        for m in 1:30
            Ws, Ts, hs, qs = [], [], [], []
            for k in 1:N_xi_per_x
                W, T, h, q = scenario_realization(problem_instance, ξ[m,k,:])
                push!(Ws,W)
                push!(Ts,T)
                push!(hs,h)
                push!(qs,q)
            end
            
            # Convert vectors to proper 3D/2D arrays for TwoStageSLP
            # Ws should be (m_2, n_2, S), Ts should be (m_2, n_1, S)
            # hs should be (m_2, S), qs should be (n_2, S)
            Ws_array = cat(Ws..., dims=3)
            Ts_array = cat(Ts..., dims=3)
            hs_array = hcat(hs...)
            qs_array = hcat(qs...)
            
            # opt_cost = gurobi_solver(A, b, c, Ws, Ts, hs, qs, nothing)

            two_slp = TwoStageSLP(A, b, c, Ws_array, Ts_array, hs_array, qs_array)
            can_lp = CanLP(two_slp)
            opt_cost = optimal_value(can_lp)

            ξ_hat = model(x)
            # Reshape the neural network output from a vector to a matrix with one column
            # Each column represents one scenario, so for single scenario we need 30×1 matrix
            ξ_hat_matrix = reshape(ξ_hat, :, 1)
            surrogate_decision = surrogate_solution(problem_instance, reg_param_surr, ξ_hat)

            evaluated_cost = s1_cost(two_slp, surrogate_decision, reg_param_ref)

            #evaluated_cost = gurobi_solver(A, b, c, Ws, Ts, hs, qs, surrogate_decision)

            push!(list_gaps, evaluated_cost - opt_cost)

            # in the algorithm of tito's paper, we divide by the evaluated cost. But in the original code provided by tito we divide by the optimal cost.
            # to be clarified
            
            push!(list_costs, opt_cost) 

            println("evaluated_cost: ", evaluated_cost, " optimal_cost: ", opt_cost, " gap: ", (evaluated_cost - opt_cost)/abs(opt_cost))

        end

        # compute 99% confidence upper bound for x
        cost_mean = mean(list_costs)
        UCB = (100/abs(cost_mean))*(  (1/30)*sum( list_gaps[k] + 2.462*sqrt( (var(list_gaps)/30))  for k in 1:30) )
            
        push!(UCB_list, UCB)
        
    end


    # Boxplot with one group
    all_data = vcat(UCB_list, MAD_gaps, ER_SAA_gaps, AD_gaps, SAA_gaps, KNN_gaps, CART_gaps, LS_gaps)
    groups = vcat(fill("NN", length(UCB_list)), fill("M5 + AD", length(MAD_gaps)), fill("ER + SAA", length(ER_SAA_gaps)), fill("AD", length(AD_gaps)), fill("SAA", length(SAA_gaps)), fill("KNN", length(KNN_gaps)), fill("CART", length(CART_gaps)), fill("LS", length(LS_gaps)))

    plot = boxplot(groups, all_data,
        legend = false,
        title = "",
        xlabel = "",
        ylabel = "Gap",
    )

    mean_list = mean(UCB_list)
    println("mean UCB: ", mean_list)

    display(plot)
    savefig(plot, "gap_boxplot.pdf") 

end

function gurobi_solver(A, b, c, Ws, Ts, hs, qs, first_stage_decision)

    # this function computed the cost with a fixed or unfixed first stage decision 

    n = length(c)  # number of first stage decision variables
    m = size(Ws[1], 2) # number of second stage decision variables 
    S = length(Ws) # number of scenarios

    # model 
    model = Model(Gurobi.Optimizer)
    set_optimizer_attribute(model, "OutputFlag", 0)

    @variable(model, z[1:n] >= 0) #first stage decision variable
    @variable(model, u[1:S,1:m] >= 0) #second stage decision variable

    @constraint(model, A*z .== b)

    if first_stage_decision !== nothing
        @constraint(model, z == first_stage_decision) # fix first stage decision
    end

    for s in 1:S
        @constraint(model, Ws[s] * u[s,:] + Ts[s]*z .== hs[s]) 
    end

    @objective(model, Min,
        sum(c[i] * z[i] for i in 1:n) +
        (1/S)*sum(qs[s][i] * u[s,i] for s in 1:S, i in 1:m))

    optimize!(model)

    ts = termination_status(model)
    if !(ts in (MOI.OPTIMAL, MOI.LOCALLY_SOLVED, MOI.ALMOST_LOCALLY_SOLVED))
        error("No feasible/optimal solution: $(ts) — $(MOI.get(model, MOI.RawStatusString()))")
    end

    cost = objective_value(model)

    return cost
end

