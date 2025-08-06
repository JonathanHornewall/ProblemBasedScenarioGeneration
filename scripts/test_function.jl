function testing(problem_instance, model, dataset_testing)
    total_gap = 0.0

    for (ξ, x) in dataset_testing
        
        ξ_hat = model(x)      
        W_hat, T_hat, h_hat, q_hat = scenario_realization(problem_instance, ξ_hat)      

        z_hat = surrogate_decision(x, W_hat, T_hat, h_hat, q_hat)  

        cost = primal_problem_cost(problem_instance, x, z_hat)
        opt_cost = primal_problem_cost(problem_instance, x) # we need to change that to use the name od the function that doesnt take z as input

        total_gap += (cost - opt_cost)/opt_cost
    end
    Noutofsamples = length(dataset_testing)
    average_gap = total_gap / Noutofsamples
    println("Test gap: ", average_gap)

    return average_gap

end