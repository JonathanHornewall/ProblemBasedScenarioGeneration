@testset "Testing LogBarCanLP solver" begin
    # Create a problem instance
    A = [1 0; -1 0; 0 1; 0 -1; 1 1]
    b = [1,  1,  1,  1, 1]
    c = [1, 1]
    A, b, c = convert_standard_to_canonical_form(A, b, c)
    lp_instance = CanLP(A, b, c)
    reg_lp_instance = LogBarCanLP(lp_instance, 1.0)
    x_opt, lambda_opt = LogBarCanLP_standard_solver(reg_lp_instance)
    @test KKT(reg_lp_instance, x_opt, lambda_opt) ≈ zeros(length(b) + length(c)) atol = 1e-8
end