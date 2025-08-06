@testset "LogBarCanLP derivatives" begin
    # Create a problem instance
    A = float([1 0; -1 0; 0 1; 0 -1])
    b = float([1,  1,  1,  1])
    c = float([1, 1])
    A, b, c = convert_standard_to_canonical_form(A, b, c)
    lp_instance = CanLP(A, b, c)
    reg_lp_instance = LogBarCanLP(lp_instance, 1.0)
    
    # Test derivatives
    x_opt, lambda_opt = LogBarCanLP_standard_solver(reg_lp_instance)
    
    @test diff_opt_A(reg_lp_instance) ≈ zeros(size(A)) atol = 1e-8
    @test diff_opt_b(reg_lp_instance) ≈ zeros(length(b)) atol = 1e-8
    @test diff_opt_c(reg_lp_instance) ≈ zeros(length(c)) atol = 1e-8
end