using FiniteDiff
using ProblemBasedScenarioGeneration
using ProblemBasedScenarioGeneration: convert_standard_to_canonical_form, CanLP, LogBarCanLP, LogBarCanLP_standard_solver, KKT
using ProblemBasedScenarioGeneration: convert_decision_standard_to_canonical, diff_opt_A, diff_opt_b, diff_opt_c
using ProblemBasedScenarioGeneration: diff_KKT_Y, diff_KKT_b, diff_cache_computation, diff_opt_b

function main()
    # Create a problem instance
    A = [1 0; -1 0; 0 1; 0 -1]
    b = [1,  1,  1,  1]
    c = [1, 1]
    m, n = size(A)
    A_can, b_can, c_can = convert_standard_to_canonical_form(A, b, c)  # Convert to canonical form and convert to float
    m_can, n_can = size(A_can)
    lp_instance = CanLP(A_can, b_can, c_can)
    reg_lp_instance = LogBarCanLP(lp_instance, 1.0)

    x_test_standard = zeros(Float64, n)  # Initialize decision variable in standard form
    x_test_can = convert_decision_standard_to_canonical(A, b, x_test_standard)  # Convert decision variable to canonical form
    λ_test_can = zeros(Float64, m_can)  # Initialize dual variable

    #test = KKT(reg_lp_instance, x_test_can, λ_test_can)  # Check KKT conditions

    #function make_KKT_test(reg_lp_instance)
    #   n = length(reg_lp_instance.linear_program.cost_vector)
    #   m = length(reg_lp_instance.linear_program.constraint_vector)
    #   KKT_test(Y) = KKT(reg_lp_instance, Y[1:n], Y[n + 1:end])  # Y is a concatenation of the primal and dual variables
    #   return KKT_test
    #end

    #KKT_test = make_KKT_test(reg_lp_instance)  # Create the KKT test function
    #Y_test = [x_test_can; λ_test_can]  # Concatenate primal and dual variables for testing
    # Test derivatives
    #@show KKT_test(Y_test)  # Show KKT test output
    #@show FiniteDiff.finite_difference_jacobian(KKT_test, Y_test)

    #@show diff_KKT_Y(reg_lp_instance, x_test_can)  # Show the KKT matrix
    #@show diff_KKT_b(reg_lp_instance, x_test_can, λ_test_can)  # Show the derivative of KKT with respect to b
    #@show diff_cache_computation(reg_lp_instance)  # Show the cached computation results
    #@show diff_opt_b(reg_lp_instance)  # Show the derivative of optimal solution with respect to b

    function opt_with_b(b)
        # Solve the LP with the modified constraint vector b
        lp_instance = CanLP(reg_lp_instance.linear_program.constraint_matrix, b, reg_lp_instance.linear_program.cost_vector)
        reg_lp_instance = LogBarCanLP(lp_instance, reg_lp_instance.regularization_parameters)
        optimal_state, optimal_dual = LogBarCanLP_standard_solver(reg_lp_instance)
        return optimal_state
    end
    b_test = [1.0, 1.0, 1.0, 1.0]  # Test constraint vector
    stepsize = 1e-6  # Step size for finite differences
    println("Optimal state for b = $b_test: ")
    optimal_state = opt_with_b(b_test)  # Get the optimal state for the test
    println("Differential of optimal solution with respect to b according to our calculations: ")
    println(diff_opt_b(reg_lp_instance))
    println("Differential of optimal solution with respect to b according to finite differences: ")
    println(FiniteDiff.finite_difference_jacobian(opt_with_b, b_test, absstep=stepsize))
    println("Maximum difference between our calculations and finite differences: ")
    println(maximum(abs.(diff_opt_b(reg_lp_instance) - FiniteDiff.finite_difference_jacobian(opt_with_b, b_test, absstep=stepsize))))  # Check the maximum difference
    println("Maximum values:: ")
    println(maximum(abs.(diff_opt_b(reg_lp_instance))))  # Check the maximum absolute difference
    println(maximum(abs.(FiniteDiff.finite_difference_jacobian(opt_with_b, b_test; absstep=stepsize))))  # Check the maximum absolute difference for finite differences
end
        

main()