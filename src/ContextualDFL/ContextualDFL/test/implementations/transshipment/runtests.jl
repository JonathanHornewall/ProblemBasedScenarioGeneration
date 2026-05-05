using ContextualDFL
using LinearAlgebra
using Random
using Test

@testset "transshipment implementation" begin
    problem = TransShipmentProblem()

    @test problem.data.source_repository == "https://github.com/USC3DLAB/SD"
    @test isfile(problem.data.core_path)
    @test isfile(problem.data.time_path)
    @test isfile(problem.data.stochastic_path)
    @test length(problem.data.first_stage_variables) == 7
    @test length(problem.data.second_stage_variables) == 77
    @test isempty(problem.data.first_stage_rows)
    @test length(problem.data.second_stage_rows) == 35
    @test length(problem.data.random_rhs_entries) == 7
    @test length(problem.data.random_objective_entries) == 7
    @test first(problem.data.first_stage_variables) == "orderUp(0)"
    @test first(problem.data.second_stage_variables) == "begEnd(0)"
    @test first(problem.data.second_stage_rows) == "initInv(0)"

    @test size(problem.stochastic_program.A_eq) == (0, 7)
    @test size(problem.stochastic_program.A_ineq) == (7, 7)
    @test problem.stochastic_program.A_ineq == -Matrix{Float64}(I, 7, 7)
    @test size(problem.base_scenario.W_eq) == (35, 77)
    @test size(problem.base_scenario.T_eq) == (35, 7)
    @test size(problem.base_scenario.W_ineq) == (77, 77)
    @test problem.base_scenario.W_ineq == -Matrix{Float64}(I, 77, 77)

    mean_parameters = transshipment_mean_parameters(problem)
    @test mean_parameters.rhs == [100.0, 200.0, 150.0, 170.0, 180.0, 170.0, 170.0]
    @test mean_parameters.q == [4.0, 4.2, 4.4, 4.6, 4.8, 5.0, 5.2]

    arrays = transshipment_scenario_arrays(problem, [mean_parameters])
    @test size(arrays[1]) == (35, 77, 1)
    @test size(arrays[3]) == (35, 7, 1)
    @test size(arrays[5]) == (35, 1)
    @test size(arrays[7]) == (77, 1)

    block_lp = transshipment_mean_lp(problem)
    direct_lp = transshipment_direct_mean_lp(problem)
    @test block_lp.A_eq == direct_lp.A_eq
    @test block_lp.A_ineq == direct_lp.A_ineq
    @test block_lp.b_eq == direct_lp.b_eq
    @test block_lp.b_ineq == direct_lp.b_ineq
    @test block_lp.c == direct_lp.c

    decoder = TransShipmentScenarioDecoder(problem)
    perturbed_rhs = copy(mean_parameters.rhs)
    perturbed_rhs[4] += 1.0
    rhs_arrays = transshipment_scenario_arrays(problem, [(; rhs=perturbed_rhs, q=mean_parameters.q)])
    @test count(!iszero, vec(rhs_arrays[5] - arrays[5])) == 1
    @test all(iszero, vec(rhs_arrays[7] - arrays[7]))

    perturbed_q = copy(mean_parameters.q)
    perturbed_q[4] += 1.0
    q_arrays = transshipment_scenario_arrays(problem, [(; rhs=mean_parameters.rhs, q=perturbed_q)])
    @test count(!iszero, vec(q_arrays[7] - arrays[7])) == 1
    @test all(iszero, vec(q_arrays[5] - arrays[5]))

    compact_scenario = ContextualDFL.ParametricScenario(;
        h_eq_xi=mean_parameters.rhs .+ 1.0,
        q_xi=mean_parameters.q .+ 1.0,
    )
    compact_arrays = decode_scenario_collection(decoder, [compact_scenario])
    @test count(!iszero, vec(compact_arrays[5] - arrays[5])) == 7
    @test count(!iszero, vec(compact_arrays[7] - arrays[7])) == 7

    vector_arrays = decode_scenario_collection(
        decoder,
        vcat(mean_parameters.rhs, mean_parameters.q);
        nr_scenarios=1,
    )
    @test vector_arrays[5] == arrays[5]
    @test vector_arrays[7] == arrays[7]

    sampled = sample_transshipment_parameters(
        problem;
        rng=Random.MersenneTwister(11),
        truncate_at_zero=true,
    )
    @test length(sampled.rhs) == 7
    @test length(sampled.q) == 7
    @test all(>=(0.0), sampled.rhs)
    @test all(>=(0.0), sampled.q)

    report = validate_transshipment_problem(problem)
    @test report.dimensions == (; n1=7, n2=77, m1=0, m2=35)
    @test report.random_rhs_entries == 7
    @test report.random_objective_entries == 7
    @test all(values(report.perturbation_report))
    @test length(report.solve_reports) == 4
    @test all(item -> item.status in ("OPTIMAL", "LOCALLY_SOLVED"), report.solve_reports)
    @test all(item -> item.max_equality_residual <= 1e-6, report.solve_reports)
    @test all(item -> item.max_inequality_violation <= 1e-6, report.solve_reports)

    W_eq, W_ineq, T_eq, T_ineq, h_eq, h_ineq, q = arrays
    solver = Solver(IpoptSolver(), HiGHSSolver())
    z, y = solve(
        solver,
        problem.stochastic_program,
        W_eq,
        W_ineq,
        T_eq,
        T_ineq,
        h_eq,
        h_ineq,
        q;
        μ=0.0,
        ρ=0.0,
    )[1:2]
    @test length(z) == 7
    @test size(y) == (77, 1)
    @test minimum(z) >= -1e-7
    @test minimum(y) >= -1e-7
end
