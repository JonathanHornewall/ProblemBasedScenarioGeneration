function testing(problem_instance, model, dataset_testing, reg_param_surr, reg_param_ref)
    total_gap = 0.0
    
    # Debug first sample in detail
    first_sample = true

    for (x, ξ) in dataset_testing
        ξ_hat = model(x)      
        evaluated_cost = loss(problem_instance, reg_param_surr, reg_param_ref, ξ_hat, ξ)
        
        # Determine optimal cost
        A, b, c = problem_instance.s1_constraint_matrix, problem_instance.s1_constraint_vector, problem_instance.s1_cost_vector
        W, T, h, q = scenario_realization(problem_instance, ξ)
        twoslp = TwoStageSLP(A, b, c, [W], [T], [h], [q])
        
        logbarlp = LogBarCanLP(twoslp, reg_param_ref)
        
        # Debug: check what the solver returns
        if first_sample
            println("Debug: reg_param_ref = $reg_param_ref")
            println("Debug: LogBarCanLP regularization parameters: $(logbarlp.regularization_parameters[1:5])...")
        end
        
        opt_cost = optimal_value(logbarlp)

        # Detailed debugging for first sample
        if first_sample
            println("\n=== DETAILED DEBUGGING OF FIRST SAMPLE ===")
            println("Regularization parameters:")
            println("  reg_param_surr: $reg_param_surr")
            println("  reg_param_ref: $reg_param_ref")
            
            println("\nProblem dimensions:")
            println("  First stage: $(length(c)) variables, $(length(b)) constraints")
            println("  Second stage: $(size(W, 1)) constraints, $(size(W, 2)) variables")
            
            println("\nTwoStageSLP probabilities: $(twoslp.ps)")
            
            println("\nLogBarCanLP regularization parameters:")
            println("  First stage: $(logbarlp.regularization_parameters[1:length(c)])")
            println("  Second stage: $(logbarlp.regularization_parameters[length(c)+1:end])")
            
            # Break down the cost calculation
            surrogate_decision = surrogate_solution(problem_instance, ξ_hat, reg_param_surr)
            println("\nSurrogate decision: $(surrogate_decision[1:5])...")
            
            # Manual calculation to compare
            s1_lp = CanLP(A, b, c)
            s1_reg_lp = LogBarCanLP(s1_lp, reg_param_surr)
            s1_cost = cost(s1_reg_lp, surrogate_decision)
            println("First-stage cost: $s1_cost")
            
            # Second-stage cost
            s2_constraint_matrix = W
            s2_constraint_vector = h - T * surrogate_decision
            s2_cost_vector = q * twoslp.ps[1]
            s2_lp = CanLP(s2_constraint_matrix, s2_constraint_vector, s2_cost_vector)
            s2_reg_lp = LogBarCanLP(s2_lp, reg_param_surr * twoslp.ps[1])
            
            println("Second-stage regularization: $(reg_param_surr * twoslp.ps[1])")
            println("Second-stage cost vector (scaled): $(s2_cost_vector[1:5])...")
            
            # Solve second-stage optimally
            optimal_s2_decision, _ = LogBarCanLP_standard_solver(s2_reg_lp)
            s2_cost = cost(s2_reg_lp, optimal_s2_decision)
            println("Second-stage cost: $s2_cost")
            
            # Manual total
            manual_total = s1_cost + s2_cost
            println("Manual calculation total: $manual_total")
            println("evaluated_cost (from loss function): $evaluated_cost")
            println("Difference: $(manual_total - evaluated_cost)")
            
            # Check if there's a mismatch in regularization
            println("\nRegularization mismatch check:")
            println("  In manual calculation:")
            println("    First stage: $reg_param_surr")
            println("    Second stage: $(reg_param_surr * twoslp.ps[1])")
            println("  In LogBarCanLP:")
            println("    First stage: $(logbarlp.regularization_parameters[1])")
            println("    Second stage: $(logbarlp.regularization_parameters[length(c)+1])")
            
            first_sample = false
            println("=== END DEBUGGING ===\n")
        end

        total_gap += (evaluated_cost - opt_cost)/abs(opt_cost)
        println("evaluated_cost", evaluated_cost, " optimal_cost", opt_cost, "gap", (evaluated_cost - opt_cost)/abs(opt_cost))
    end
    Noutofsamples = length(dataset_testing)
    average_gap = total_gap / Noutofsamples
    println("Test gap: ", average_gap)

    return average_gap

end