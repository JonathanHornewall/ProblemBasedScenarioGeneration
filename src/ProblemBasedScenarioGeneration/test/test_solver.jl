@testset "Testing LogBarCanLP solver" begin
    # Create a problem instance
    A = [1 0; -1 0; 0 1; 0 -1; 1 1]
    b = [1,  1,  1,  1, 1]
    c = [1, 1]
    mu = 1.0
    A, b, c = convert_standard_to_canonical_form(A, b, c)
    lp_instance = CanLP(A, b, c)
    reg_lp_instance = LogBarCanLP(lp_instance, mu)
    x_opt, lambda_opt = LogBarCanLP_standard_solver(reg_lp_instance)
    @test KKT(reg_lp_instance, x_opt, lambda_opt) ≈ zeros(length(b) + length(c)) atol = 1e-8
    @test lambda_opt ≈ A' \ (c .- mu ./ x_opt ) atol = 1e-8

    #=
    rel_lp_instance = LogBarCanLP(lp_instance, 0.0)  # Regularization parameter is zero
    x_opt_rel, lambda_opt_rel = LogBarCanLP_standard_solver(rel_lp_instance)
    # @test A*x_opt_rel ≈ b atol = 1e-8  # Check primal feasibility
    # @test A'*lambda_opt_rel ≈ c atol = 1e-8  # Check dual feasibility
    =#
end

@testset "Testing canonical LP solver" begin
    # Test the new canonical LP solver with a simple LP already in canonical form
    # Simple problem: min x1 + x2 s.t. x1 + x2 = 1, x1, x2 >= 0
    A_simple = [1.0 1.0]
    b_simple = [1.0]
    c_simple = [1.0, 1.0]
    
    can_lp_simple = CanLP(A_simple, b_simple, c_simple)
    
    # Solve using the canonical LP solver
    x_opt, lambda_opt = solve_canonical_lp(can_lp_simple)
    
    # Test primal feasibility: Ax ≈ b
    @test A_simple * x_opt ≈ b_simple atol = 1e-8
    
    # Test dual feasibility: A'λ ≤ c 
    @test all(A_simple' * lambda_opt .<= c_simple .+ 1e-8)
    
    # Test non-negativity of primal solution
    @test all(x_opt .>= 0)
    
    # Test that the solution is reasonable (objective value should be finite)
    @test isfinite(sum(c_simple .* x_opt))
    
    # Test with another simple canonical form LP
    A2 = [1.0 0.0; 0.0 1.0]
    b2 = [2.0, 3.0]
    c2 = [1.0, 2.0]

    can_lp_2 = CanLP(A2, b2, c2)
    
    x_opt2, lambda_opt2 = solve_canonical_lp(can_lp_2)
    
    # Test primal feasibility: Ax ≈ b
    @test A2 * x_opt2 ≈ b2 atol = 1e-8
    
    # Test dual feasibility: A'λ ≤ c 
    @test all(A2' * lambda_opt2 .<= c2 .+ 1e-8)
    
    # Test non-negativity of primal solution
    @test all(x_opt2 .>= 0)
    
    # Test that the solution is reasonable (objective value should be finite)
    @test isfinite(sum(c2 .* x_opt2))
    
    # Test with the original problem converted to canonical form
    A = [1 0; -1 0; 0 1; 0 -1; 1 1]
    b = [1,  1,  1,  1, 1]
    c = [1, 1]
    
    # Convert to canonical form using the regular (non-regularized) utility function
    A_can, b_can, c_can = convert_standard_to_canonical_form_regular(A, b, c)
    
    lp_instance = CanLP(A_can, b_can, c_can)
    # Solve using the LogBarCanLP standard solver
    reg_lp_instance = LogBarCanLP(lp_instance, 0.0)  # Regularization parameter is zero
    x_opt, lambda_opt = LogBarCanLP_standard_solver(reg_lp_instance)
    
    # Test primal feasibility: A_can * x ≈ b_can
    @test A_can * x_opt ≈ b_can atol = 1e-8
    
    # Test dual feasibility: A_can' * λ ≤ c_can (using the CONVERTED system)
    @test all(A_can' * lambda_opt .<= c_can .+ 1e-8)
    
    # Test non-negativity of primal solution
    @test all(x_opt .>= 0)
    
    # Test that the solution is reasonable (objective value should be finite)
    @test isfinite(sum(c_can .* x_opt))
end