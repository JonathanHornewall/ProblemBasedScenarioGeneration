struct TestLPSolver <: LPSolver end
struct TestLogBarSolver <: LogBarSolver end

ContextualDFL.solve(::TestLPSolver, lp::LP; kwargs...) = (; method=:lp, lp, kwargs)
ContextualDFL.solve(::TestLogBarSolver, lp::LP; μ=nothing, kwargs...) =
    (; method=:log_barrier, lp, μ, kwargs)

@testset "base LP suite" begin
    @testset "LP construction" begin
        lp = LP(c=[1.0, 2.0])

        @test size(lp.A_eq) == (0, 2)
        @test size(lp.A_ineq) == (0, 2)
        @test isempty(lp.b_eq)
        @test isempty(lp.b_ineq)
        @test lp.c == [1.0, 2.0]
        @test eltype(lp.A_eq) == Float64
        @test eltype(lp.b_eq) == Float64

        zero_variable_lp = LP()

        @test size(zero_variable_lp.A_eq) == (0, 0)
        @test size(zero_variable_lp.A_ineq) == (0, 0)
        @test isempty(zero_variable_lp.b_eq)
        @test isempty(zero_variable_lp.b_ineq)
        @test isempty(zero_variable_lp.c)

        matrix_only_lp = LP(A_eq=[1.0 2.0; 3.0 4.0])

        @test matrix_only_lp.A_eq == [1.0 2.0; 3.0 4.0]
        @test matrix_only_lp.b_eq == zeros(2)
        @test matrix_only_lp.c == zeros(2)
        @test size(matrix_only_lp.A_ineq) == (0, 2)

        @test_throws ArgumentError LP(b_eq=[1.0], c=[1.0, 2.0])
        @test_throws DimensionMismatch LP(A_eq=[1.0 2.0], c=[1.0])
        @test_throws DimensionMismatch LP(A_eq=[1.0 2.0], b_eq=[1.0, 2.0], c=[1.0, 2.0])
    end

    @testset "solver dispatch" begin
        lp = LP(c=[1.0, 2.0])
        log_barrier_lp = LP(A_ineq=[1.0 0.0], b_ineq=[10.0], c=[1.0, 2.0])
        solver = Solver(TestLogBarSolver(), TestLPSolver())

        lp_solution = solve(solver, lp; warm_start=:basis)
        log_barrier_solution = solve(solver, log_barrier_lp; μ=0.5, max_iter=100)

        @test lp_solution.method == :lp
        @test lp_solution.lp === lp
        @test lp_solution.kwargs[:warm_start] == :basis

        @test log_barrier_solution.method == :log_barrier
        @test log_barrier_solution.lp === log_barrier_lp
        @test log_barrier_solution.μ == [0.5]
        @test log_barrier_solution.kwargs[:max_iter] == 100

        vector_barrier_solution = solve(solver, log_barrier_lp; μ=[0.25])
        zero_vector_solution = solve(solver, log_barrier_lp; μ=zeros(1))

        @test vector_barrier_solution.method == :log_barrier
        @test vector_barrier_solution.μ == [0.25]
        @test zero_vector_solution.method == :lp
        @test_throws DimensionMismatch solve(solver, log_barrier_lp; μ=[0.25, 0.5])
    end

    @testset "infeasible solves throw" begin
        infeasible_lp = LP(
            A_ineq=reshape([1.0, -1.0], 2, 1),
            b_ineq=[0.0, -1.0],
            c=[0.0],
        )

        @test_throws ErrorException solve(TEST_HIGHS_SOLVER, infeasible_lp)
        @test_throws ErrorException solve(TEST_SOLVER, infeasible_lp)
        @test_throws ErrorException solve(TEST_SOLVER, infeasible_lp; μ=1.0, max_iter=50)
    end

    @testset "geometric LP cases" begin
        square_A, square_b = square_2d()
        case_2_A = [square_A; 1.0 1.0]
        case_2_b = [square_b; 0.5]
        case_3_A = [
            case_2_A
            1.0 -1.0
            -1.0 0.4
            -0.3 -1.0
        ]
        case_3_b = [case_2_b; 1.0; 1.2; 1.4]

        simplex_2_A, simplex_2_b = nonnegative_orthant(2)
        simplex_5_A, simplex_5_b = nonnegative_orthant(5)

        tube_A = [square_A zeros(4)]
        tube_eq = [0.0 0.0 1.0]

        tilted_A = [case_3_A zeros(size(case_3_A, 1))]
        tilted_eq = [-0.2 0.1 1.0]
        tilted_expected = [-0.5, 1.0, 0.8]
        tilted_extra_A = [
            tilted_A
            0.2 1.0 0.0
            -0.4 -0.6 0.0
        ]
        tilted_extra_b = [case_3_b; 1.5; 1.3]

        cases = [
            (;
                name="case 1: square in 2D",
                lp=LP(A_ineq=square_A, b_ineq=square_b, c=[-1.0, -2.0]),
                expected_status="OPTIMAL",
                expected_z=[1.0, 1.0],
            ),
            (;
                name="case 2: square in 2D with one cut",
                lp=LP(A_ineq=case_2_A, b_ineq=case_2_b, c=[-1.0, -2.0]),
                expected_status="OPTIMAL",
                expected_z=[-0.5, 1.0],
            ),
            (;
                name="case 3: square in 2D with several cuts",
                lp=LP(A_ineq=case_3_A, b_ineq=case_3_b, c=[-1.0, -2.0]),
                expected_status="OPTIMAL",
                expected_z=[-0.5, 1.0],
            ),
            (;
                name="case 4: simplex in dimension 2",
                lp=LP(
                    A_eq=[1.0 1.0],
                    b_eq=[1.0],
                    A_ineq=simplex_2_A,
                    b_ineq=simplex_2_b,
                    c=[1.0, 2.0],
                ),
                expected_status="OPTIMAL",
                expected_z=[1.0, 0.0],
            ),
            (;
                name="case 5: simplex in dimension 5",
                lp=LP(
                    A_eq=ones(1, 5),
                    b_eq=[1.0],
                    A_ineq=simplex_5_A,
                    b_ineq=simplex_5_b,
                    c=collect(1.0:5.0),
                ),
                expected_status="OPTIMAL",
                expected_z=[1.0, 0.0, 0.0, 0.0, 0.0],
            ),
            (;
                name="case 6: square tube",
                lp=LP(
                    A_eq=tube_eq,
                    b_eq=[1.0],
                    A_ineq=tube_A,
                    b_ineq=square_b,
                    c=[-1.0, -2.0, 0.0],
                ),
                expected_status="OPTIMAL",
                expected_z=[1.0, 1.0, 1.0],
            ),
            (;
                name="case 7: square tube with tilted base floor",
                lp=LP(
                    A_eq=tilted_eq,
                    b_eq=[1.0],
                    A_ineq=tilted_A,
                    b_ineq=case_3_b,
                    c=[-1.0, -2.0, 0.0],
                ),
                expected_status="OPTIMAL",
                expected_z=tilted_expected,
            ),
            (;
                name="case 8: tilted square tube with extra cuts",
                lp=LP(
                    A_eq=tilted_eq,
                    b_eq=[1.0],
                    A_ineq=tilted_extra_A,
                    b_ineq=tilted_extra_b,
                    c=[-1.0, -2.0, 0.0],
                ),
                expected_status="OPTIMAL",
                expected_z=tilted_expected,
            ),
        ]

        for case in cases
            run_smooth_case(case)
        end

        for solver in (TEST_SOLVER, TEST_HIGHS_SOLVER)
            case_2_solution = solve(solver, cases[2].lp).z
            case_3_solution = solve(solver, cases[3].lp).z
            @test case_3_solution ≈ case_2_solution atol = 1e-8
        end
    end
end
