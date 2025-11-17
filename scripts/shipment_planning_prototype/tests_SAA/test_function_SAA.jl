using StatsPlots
using CSV
using DataFrames
using ProblemBasedScenarioGeneration
using ProblemBasedScenarioGeneration: cost, s1_cost, scenario_realization,
    TwoStageSLP, CanLP, solve_canonical_lp

function optimal_value(instance::CanLP, solver=solve_canonical_lp; feasibility_margin::Real = 1e-8)
    optimal_solution, _ = solver(instance)
    cost(instance, optimal_solution; feasibility_margin = feasibility_margin)
end

function _scenario_tables(problem_instance, scenario_matrix)
    scenario_results = [scenario_realization(problem_instance, scenario) for scenario in eachcol(scenario_matrix)]
    W_list = [result[1] for result in scenario_results]
    T_list = [result[2] for result in scenario_results]
    h_list = [result[3] for result in scenario_results]
    q_list = [result[4] for result in scenario_results]
    Ws = cat(W_list..., dims=3)
    Ts = cat(T_list..., dims=3)
    hs = hcat(h_list...)
    qs = hcat(q_list...)
    return Ws, Ts, hs, qs
end

function _as_matrix(sample)
    if ndims(sample) == 1
        reshape(sample, :, 1)
    elseif ndims(sample) == 2
        sample
    elseif ndims(sample) == 3
        collections, N_xi, nloc = size(sample)
        reshape(permutedims(sample, (3, 1, 2)), nloc, :)
    else
        error("Unsupported scenario tensor dimensions")
    end
end

function testing_SAA(problem_instance, model, dataset_testing, reg_param_surr, reg_param_ref, N_xi_per_x)
    UCB_list = Float64[]
    first_tensor = first(values(dataset_testing))
    collections_per_sample = size(first_tensor, 1)
    A, b, c = ProblemBasedScenarioGeneration.return_first_stage_parameters(problem_instance)

    for (x, ξ_tensor) in dataset_testing
        gaps = Float64[]
        opt_costs = Float64[]

        for m in 1:collections_per_sample
            Ws = Matrix{Float64}[]
            Ts = Matrix{Float64}[]
            hs = Vector{Float64}[]
            qs = Vector{Float64}[]
            for k in 1:N_xi_per_x
                scenario = vec(ξ_tensor[m, k, :])
                W, T, h, q = scenario_realization(problem_instance, scenario)
                push!(Ws, W); push!(Ts, T); push!(hs, h); push!(qs, q)
            end
            W_array = cat(Ws..., dims=3)
            T_array = cat(Ts..., dims=3)
            h_array = hcat(hs...)
            q_array = hcat(qs...)
            two_slp = TwoStageSLP(A, b, c, W_array, T_array, h_array, q_array)
            opt_cost = optimal_value(CanLP(two_slp))

            ξ̂ = reshape(model(x), :, 1)
            surrogate_decision = surrogate_solution(problem_instance, reg_param_surr, ξ̂)
            evaluated_cost = s1_cost(two_slp, surrogate_decision, reg_param_ref)

            push!(gaps, evaluated_cost - opt_cost)
            push!(opt_costs, opt_cost)
        end

        cost_mean = mean(opt_costs)
        gap_var = length(gaps) > 1 ? var(gaps) : 0.0
        UCB = (100 / abs(cost_mean)) *
            ((1 / collections_per_sample) *
             sum(gaps[k] + 2.462 * sqrt(gap_var / collections_per_sample) for k in 1:collections_per_sample))
        push!(UCB_list, UCB)
    end

    df_path = joinpath(@__DIR__, "df_ship.csv")
    df =
        if isfile(df_path)
            tmp = CSV.read(df_path, DataFrame)
            nrow(tmp) == 0 ? DataFrame(method=String[], OoS=Float64[]) : tmp
        else
            DataFrame(method=String[], OoS=Float64[])
        end
    clean = UCB_list[.!isnan.(UCB_list)]
    if !isempty(clean)
        new_rows = DataFrame(method = fill("NN-shipment", length(clean)), OoS = clean)
        append!(df, new_rows)
    end
    CSV.write(df_path, df)

    if nrow(df) > 0
        df = dropmissing(df, :OoS)
        df = df[.!isnan.(df.OoS), :]
    end

    if nrow(df) > 0
        plotfile = "shipment_gap_boxplot.pdf"
        @df df boxplot(:method, :OoS, group=:method, legend=false, colour=[:blue :orange])
        savefig(plotfile)
    end

    return clean
end
