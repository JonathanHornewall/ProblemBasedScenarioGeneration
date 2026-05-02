@testset "extended LP suite" begin
    @testset "equality-only sanity cases" begin
        singleton_case = (;
            name="E1 singleton equality system",
            lp=LP(
                A_eq=[1.0 1.0; 1.0 -1.0],
                b_eq=[1.0, 0.0],
                c=[3.0, -2.0],
            ),
            expected_status="OPTIMAL",
            expected_z=[0.5, 0.5],
        )
        assert_lp_case_with_highs(singleton_case)
        singleton_barrier = solve(TEST_SOLVER, singleton_case.lp; μ=1.0)
        @test is_optimal_status(singleton_barrier.status)
        @test singleton_barrier.z ≈ singleton_case.expected_z atol = 1e-8

        redundant_case = (;
            name="E2 redundant equality system",
            lp=LP(
                A_eq=[1.0 1.0; 1.0 -1.0; 2.0 2.0],
                b_eq=[1.0, 0.0, 2.0],
                c=[3.0, -2.0],
            ),
            expected_status="OPTIMAL",
            expected_z=[0.5, 0.5],
        )
        assert_lp_case_with_highs(redundant_case)

        unbounded_case = (;
            name="E3 equality-only unbounded LP",
            lp=LP(A_eq=[1.0 1.0], b_eq=[1.0], c=[1.0, 0.0]),
            expected_status="DUAL_INFEASIBLE",
        )
        assert_lp_case_with_highs(unbounded_case)

        degenerate_case = (;
            name="E4 equality-only degenerate optimal face",
            lp=LP(A_eq=[1.0 1.0], b_eq=[1.0], c=[1.0, 1.0]),
            expected_status="OPTIMAL",
        )
        assert_lp_case_with_highs(degenerate_case)
    end

    @testset "equality and inequality barrier cases" begin
        slice_A, slice_b = box_constraints(fill(0.1, 3), fill(0.8, 3))
        tilted_A, tilted_b = box_constraints(fill(-1.0, 3), fill(1.0, 3))
        simplex_3_A, simplex_3_b = nonnegative_orthant(3)

        cases = [
            (;
                name="E6: affine slice of a box",
                lp=LP(
                    A_eq=ones(1, 3),
                    b_eq=[1.0],
                    A_ineq=slice_A,
                    b_ineq=slice_b,
                    c=[1.0, 2.0, 3.0],
                ),
                expected_status="OPTIMAL",
                expected_z=[0.8, 0.1, 0.1],
            ),
            (;
                name="E7: tilted affine slice of a box",
                lp=LP(
                    A_eq=[1.0 2.0 3.0],
                    b_eq=[1.0],
                    A_ineq=tilted_A,
                    b_ineq=tilted_b,
                    c=[-1.0, 0.5, 2.0],
                ),
                expected_status="OPTIMAL",
            ),
            (;
                name="E8: simplex with one oblique cut",
                lp=LP(
                    A_eq=ones(1, 3),
                    b_eq=[1.0],
                    A_ineq=[simplex_3_A; 1.0 2.0 0.0],
                    b_ineq=[simplex_3_b; 0.8],
                    c=[2.0, -2.0, 1.0],
                ),
                expected_status="OPTIMAL",
                expected_z=[0.0, 0.4, 0.6],
            ),
        ]

        for case in cases
            run_smooth_case(case)
        end
    end

    @testset "realistic equality-structured toy LPs" begin
        transport_A = [
            1.0 1.0 1.0 0.0 0.0 0.0
            0.0 0.0 0.0 1.0 1.0 1.0
            1.0 0.0 0.0 1.0 0.0 0.0
            0.0 1.0 0.0 0.0 1.0 0.0
        ]
        transport_A_redundant = [
            transport_A
            0.0 0.0 1.0 0.0 0.0 1.0
        ]
        transport_b = [1.0, 2.0, 0.5, 1.0]
        transport_b_redundant = [transport_b; 1.5]
        transport_ineq_A, transport_ineq_b = nonnegative_orthant(6)

        network_A = [
            1.0 1.0 0.0 0.0 0.0
            -1.0 0.0 1.0 1.0 0.0
            0.0 -1.0 -1.0 0.0 1.0
        ]
        network_ineq_A, network_ineq_b = box_constraints(zeros(5), ones(5))

        inventory_A_eq = [
            -1.0 0.0 0.0 1.0 0.0 0.0
            0.0 -1.0 0.0 -1.0 1.0 0.0
            0.0 0.0 -1.0 0.0 -1.0 1.0
        ]
        inventory_b_eq = [-0.3, -0.4, -0.2]
        production_upper = [Matrix{Float64}(I, 3, 3) zeros(3, 3)]
        production_nonnegative = [-Matrix{Float64}(I, 3, 3) zeros(3, 3)]
        inventory_nonnegative = [zeros(3, 3) -Matrix{Float64}(I, 3, 3)]
        inventory_A_ineq = [
            production_upper
            production_nonnegative
            inventory_nonnegative
        ]
        inventory_b_ineq = [ones(3); zeros(3); zeros(3)]

        cases = [
            (;
                name="E9: transportation polytope, full-row-rank equalities",
                lp=LP(
                    A_eq=transport_A,
                    b_eq=transport_b,
                    A_ineq=transport_ineq_A,
                    b_ineq=transport_ineq_b,
                    c=[1.0, 4.0, 2.0, 3.0, 1.0, 5.0],
                ),
                expected_status="OPTIMAL",
            ),
            (;
                name="E10: network flow with capacities",
                lp=LP(
                    A_eq=network_A,
                    b_eq=[1.0, 0.0, 0.0],
                    A_ineq=network_ineq_A,
                    b_ineq=network_ineq_b,
                    c=[1.0, 2.0, 0.5, 3.0, 1.0],
                ),
                expected_status="OPTIMAL",
            ),
            (;
                name="E11: inventory balance model",
                lp=LP(
                    A_eq=inventory_A_eq,
                    b_eq=inventory_b_eq,
                    A_ineq=inventory_A_ineq,
                    b_ineq=inventory_b_ineq,
                    c=[1.0, 1.2, 1.1, 0.1, 0.1, 0.1],
                ),
                expected_status="OPTIMAL",
            ),
        ]

        for case in cases
            run_smooth_case(case)
        end

        rank_deficient_transport = LP(
            A_eq=transport_A_redundant,
            b_eq=transport_b_redundant,
            A_ineq=transport_ineq_A,
            b_ineq=transport_ineq_b,
            c=[1.0, 4.0, 2.0, 3.0, 1.0, 5.0],
        )
        assert_lp_case_with_highs((;
            name="E9 redundant transportation polytope",
            lp=rank_deficient_transport,
            expected_status="OPTIMAL",
        ))

        for μ in TEST_BARRIER_MUS
            barrier_result =
                solve(TEST_SOLVER, rank_deficient_transport; μ=μ, tol=1e-10, max_iter=1_000)
            @test is_optimal_status(barrier_result.status)
            @test_throws ArgumentError construct_jacobian(
                TEST_SOLVER,
                rank_deficient_transport,
                μ;
                pre_computed=barrier_result.z,
            )
        end
    end
end
