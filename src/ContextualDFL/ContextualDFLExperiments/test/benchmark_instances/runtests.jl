using ContextualDFL
using ContextualDFLExperiments
using LinearAlgebra
using Random
using Test

function benchmark_solver()
    return ContextualDFL.Solver(ContextualDFL.IpoptSolver(), ContextualDFL.HiGHSSolver())
end

function assert_one_scenario_solve(problem, decoder; seed=1)
    solver = benchmark_solver()
    program = stochastic_program(problem)
    data = generate_benchmark_dataset(
        problem;
        n_contexts=1,
        scenarios_per_context=1,
        seed=seed,
    )
    arrays = ContextualDFL.decode_scenario_collection(decoder, data[1].scenario_parameters)
    z, y = ContextualDFL.solve(solver, program, arrays...)[1:2]
    @test all(isfinite, z)
    @test all(isfinite, y)
    return data
end

function smoke_saa_knn(problem, decoder; seed=1, train_contexts=3)
    solver = benchmark_solver()
    program = stochastic_program(problem)
    train = generate_benchmark_dataset(
        problem;
        n_contexts=train_contexts,
        scenarios_per_context=1,
        seed=seed,
    )
    test = generate_benchmark_dataset(
        problem;
        n_contexts=1,
        scenarios_per_context=1,
        seed=seed + 1,
    )

    saa = SampleAverageApproximationPolicy(train, solver, program, decoder)
    knn = KNearestNeighborsPolicy(train, solver, program, decoder; k=1)

    @test all(isfinite, infer(saa, test[1].context))
    @test all(isfinite, infer(knn, test[1].context))

    optimal = solve_dataset_to_optimality(test, program, decoder, solver)
    saa_eval = evaluate_policy_against_optimum(
        saa,
        test,
        program,
        decoder,
        solver;
        optimal_results=optimal,
    )
    @test isfinite(saa_eval.metrics.test_policy_value_mean)
    @test isfinite(saa_eval.metrics.test_regret_mean)
end

function smoke_regression_policies(
    problem,
    decoder;
    seed=10,
    target_component=:h_eq_xi,
    postprocess_prediction=identity,
)
    solver = benchmark_solver()
    program = stochastic_program(problem)
    train = generate_benchmark_dataset(
        problem;
        n_contexts=5,
        scenarios_per_context=1,
        seed=seed,
    )

    ls = LeastSquaresPolicy(
        train,
        solver,
        program,
        decoder;
        target_component=target_component,
        postprocess_prediction=postprocess_prediction,
    )
    er = ResidualSampleAverageApproximationPolicy(
        train,
        solver,
        program,
        decoder;
        target_component=target_component,
        postprocess_prediction=postprocess_prediction,
    )

    @test all(isfinite, infer(ls, train[1].context))
    @test all(isfinite, infer(er, train[1].context))
end

@testset "benchmark instances" begin
    @testset "shipment planning" begin
        problem = ShipmentPlanningProblem()
        program = stochastic_program(problem)
        base = base_scenario(problem)
        decoder = ShipmentPlanningParametricDecoder(problem)

        @test size(program.A_ineq) == (5, 5)
        @test size(base.W_eq) == (17, 82)
        @test size(base.W_ineq) == (82, 82)
        @test size(base.T_eq) == (17, 5)
        @test length(base.h_eq) == 17
        @test length(base.q) == 82

        context = generate_benchmark_contexts(
            problem;
            n_contexts=1,
            rng=Random.MersenneTwister(1),
        )[1]
        scenario = generate_benchmark_scenarios(
            problem,
            context;
            n_scenarios=1,
            rng=Random.MersenneTwister(2),
        )[1]
        arrays = ContextualDFL.decode_scenario_collection(decoder, [scenario])
        @test size(arrays[1]) == (17, 82, 1)
        @test size(arrays[3]) == (17, 5, 1)
        @test size(arrays[5]) == (17, 1)
        @test all(>=(1e-6), scenario.h_eq_xi[1:problem.demand_count])

        assert_one_scenario_solve(problem, decoder; seed=3)
        smoke_saa_knn(problem, decoder; seed=4)

        shipment_postprocess = target -> begin
            values = Float64.(target)
            values[1:problem.demand_count] = max.(values[1:problem.demand_count], 1e-6)
            values[(problem.demand_count + 1):end] .= 0.0
            values
        end
        smoke_regression_policies(
            problem,
            decoder;
            seed=5,
            target_component=:h_eq_xi,
            postprocess_prediction=shipment_postprocess,
        )
    end

    @testset "transshipment variants" begin
        for variant in (:q_only, :h_only, :h_and_q)
            problem = TransShipmentExperimentProblem(; variant=variant)
            decoder = transshipment_decoder(problem)
            program = stochastic_program(problem)
            context = generate_benchmark_contexts(
                problem;
                n_contexts=1,
                rng=Random.MersenneTwister(11),
            )[1]
            scenario = generate_benchmark_scenarios(
                problem,
                context;
                n_scenarios=1,
                rng=Random.MersenneTwister(12),
            )[1]
            mean_parameters = ContextualDFL.transshipment_mean_parameters(problem.core_problem)
            arrays = ContextualDFL.decode_scenario_collection(decoder, [scenario])

            @test length(context) == 3
            @test length(scenario.h_eq_xi) == 7
            @test length(scenario.q_xi) == 7
            @test all(>(0.0), scenario.h_eq_xi)
            @test all(>(0.0), scenario.q_xi)
            @test size(arrays[1]) == (35, 77, 1)
            @test size(arrays[3]) == (35, 7, 1)
            @test size(arrays[5]) == (35, 1)
            @test size(arrays[7]) == (77, 1)
            @test size(program.A_ineq) == (7, 7)

            if variant == :q_only
                scenarios = generate_benchmark_scenarios(
                    problem,
                    context;
                    n_scenarios=3,
                    rng=Random.MersenneTwister(13),
                )
                @test all(s -> s.h_eq_xi == mean_parameters.rhs, scenarios)
                @test length(unique([s.q_xi[1] for s in scenarios])) > 1
            end

            assert_one_scenario_solve(problem, decoder; seed=14)
            smoke_saa_knn(problem, decoder; seed=15, train_contexts=2)

            if variant == :q_only
                smoke_regression_policies(
                    problem,
                    decoder;
                    seed=16,
                    target_component=:q_xi,
                    postprocess_prediction=target -> max.(target, 1e-4),
                )
            elseif variant == :h_only
                smoke_regression_policies(
                    problem,
                    decoder;
                    seed=17,
                    target_component=:h_eq_xi,
                    postprocess_prediction=target -> max.(target, 1e-4),
                )
            end
        end
    end

    @testset "random yield" begin
        problem = RandomYieldProblem(; r=5, a=10, K_support=5)
        decoder = RandomYieldParametricDecoder(problem)
        context = generate_benchmark_contexts(
            problem;
            n_contexts=1,
            rng=Random.MersenneTwister(21),
        )[1]

        probabilities = random_yield_probabilities(problem, context)
        @test length(probabilities) == 5
        @test all(>=(0.0), probabilities)
        @test sum(probabilities) ≈ 1.0

        support = random_yield_support_scenarios(problem, context)
        @test length(support) == 5
        @test support[1].W_eq_xi == base_scenario(problem).W_eq

        scenario = sample_random_yield_scenario(
            problem,
            context;
            rng=Random.MersenneTwister(22),
        )
        @test size(scenario.W_eq_xi) == (5, 20)
        arrays = ContextualDFL.decode_scenario_collection(decoder, [scenario])
        @test size(arrays[1]) == (5, 20, 1)
        @test size(arrays[3]) == (5, 5, 1)
        @test size(arrays[5]) == (5, 1)

        assert_one_scenario_solve(problem, decoder; seed=23)
        smoke_saa_knn(problem, decoder; seed=24)
    end
end
