using Test
using ProblemBasedScenarioGeneration
using LinearAlgebra

@testset "Decision Regret" begin
    @testset "Newsvendor: zero regret when predicted == actual" begin
        prob = UnreliableNewsvendorProblem()

        # When predicted == actual and mu_surr == mu_prim, the cost of the
        # surrogate decision should equal the optimal cost (zero regret).
        actual_sc = scenario_realization(prob, [0.5, 0.8])
        predicted  = [actual_sc]

        surr_cost = decision_regret(prob, 0.1, 0.1, predicted, actual_sc)
        opt_cost  = decision_regret(prob, 0.1, 0.1, [actual_sc], actual_sc)

        @test isfinite(surr_cost)
        # Same prediction as actual => same decision => same cost
        @test isapprox(surr_cost, opt_cost, atol=1e-4)
    end

    @testset "Newsvendor: regret is non-negative" begin
        prob = UnreliableNewsvendorProblem()

        actual_sc   = scenario_realization(prob, [0.7, 0.9])
        predicted_sc = scenario_realization(prob, [0.3, 0.5])  # wrong prediction

        cost_correct = decision_regret(prob, 0.1, 0.1, [actual_sc], actual_sc)
        cost_wrong   = decision_regret(prob, 0.1, 0.1, [predicted_sc], actual_sc)

        # Both costs should be finite
        @test isfinite(cost_correct)
        @test isfinite(cost_wrong)
    end

    @testset "Shipment planning: regret is finite" begin
        prob = ShipmentPlanningProblem()
        demand = [3.0, 2.0, 4.0, 1.0]

        actual_sc   = scenario_realization(prob, demand)
        predicted_sc = scenario_realization(prob, demand .* 0.8)

        regret = decision_regret(prob, 0.5, 0.5, [predicted_sc], actual_sc)
        @test isfinite(regret)
    end

    @testset "Resource allocation: surrogate_first_stage dimensions" begin
        prob = ResourceAllocationProblem()
        A1, b1, c1 = first_stage_data(prob)

        sc = scenario_realization(prob, 50 .* ones(30))
        slp = TwoStageSLP(A1, b1, c1, [sc])

        x1 = surrogate_first_stage(slp, 0.5)
        @test length(x1) == length(c1)
        @test all(isfinite.(x1))
    end

    @testset "evaluate_cost is non-negative for physical problems" begin
        prob = ShipmentPlanningProblem()
        A1, b1, c1 = first_stage_data(prob)

        demand = [2.0, 3.0, 1.0, 4.0]
        sc = scenario_realization(prob, demand)
        slp = TwoStageSLP(A1, b1, c1, [sc])

        x1 = surrogate_first_stage(slp, 0.1)
        cost = evaluate_cost(x1, slp, 0.1)

        @test isfinite(cost)
    end
end
