function testing(problem_instance, model, dataset_testing, regularization_parameter)
    total_gap = 0.0

    for (x, ξ) in dataset_testing
        ξ_hat = model(x)      
        evaluated_cost = loss(problem_instance, regularization_parameter, ξ_hat, ξ)
        
        # Determine optimal cost
        A, b, c = problem_instance.s1_constraint_matrix, problem_instance.s1_constraint_vector, problem_instance.s1_cost_vector
        W, T, h, q = scenario_realization(problem_instance, ξ)
        twoslp = TwoStageSLP(A, b, c, [W], [T], [h], [q])
        logbarlp = LogBarCanLP(twoslp, regularization_parameter)
        opt_cost = optimal_value(logbarlp)


        total_gap += (evaluated_cost - opt_cost)/abs(opt_cost)
        println("evaluated_cost", evaluated_cost, " optimal_cost", opt_cost, "gap", (evaluated_cost - opt_cost)/abs(opt_cost))
    end
    Noutofsamples = length(dataset_testing)
    average_gap = total_gap / Noutofsamples
    println("Test gap: ", average_gap)

    return average_gap

end