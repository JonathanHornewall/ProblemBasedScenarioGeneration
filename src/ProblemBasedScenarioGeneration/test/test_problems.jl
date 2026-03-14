using Test
using ProblemBasedScenarioGeneration
using LinearAlgebra

@testset "Problem Instances" begin
    @testset "ResourceAllocationProblem" begin
        prob = ResourceAllocationProblem()

        A1, b1, c1 = first_stage_data(prob)
        @test size(A1, 2) == 20    # I=20 clients
        @test length(c1) == 20
        @test noise_pattern(prob) == H_ONLY

        # Scenario realization
        demand = ones(30)           # J=30 resources
        sc = scenario_realization(prob, demand)
        @test sc isa Scenario
        @test size(sc.W, 2) > 0    # n2 > 0
        @test length(sc.h) == size(sc.W, 1)  # m2 consistent
        @test length(sc.q) == size(sc.W, 2)

        # Generate dataset
        dataset = generate_dataset(prob, 5)
        @test length(dataset) == 5
        for (x, sc2) in dataset
            @test length(x) == 3   # context dim = 3
            @test sc2 isa Scenario
        end
    end

    @testset "ShipmentPlanningProblem" begin
        prob = ShipmentPlanningProblem()

        A1, b1, c1 = first_stage_data(prob)
        @test size(A1, 2) == 12    # 12 warehouses
        @test length(c1) == 12
        @test noise_pattern(prob) == H_ONLY

        # Scenario realization
        demand = ones(4)           # 4 locations
        sc = scenario_realization(prob, demand)
        @test sc isa Scenario
        @test size(sc.W, 1) == 12 + 4   # m2 = demand + supply rows
        @test length(sc.h) == size(sc.W, 1)
        @test length(sc.q) == size(sc.W, 2)

        # Generate dataset
        dataset = generate_dataset(prob, 5)
        @test length(dataset) == 5
        for (x, sc2) in dataset
            @test length(x) == 3
            @test sc2 isa Scenario
        end
    end

    @testset "UnreliableNewsvendorProblem" begin
        prob = UnreliableNewsvendorProblem()

        A1, b1, c1 = first_stage_data(prob)
        @test size(A1, 2) == 1    # 1D first stage
        @test noise_pattern(prob) == WH

        # Scenario realization
        param = [0.7, 0.8]   # D=0.7, U=0.8
        sc = scenario_realization(prob, param)
        @test sc isa Scenario
        @test size(sc.W) == (2, 3)
        @test size(sc.T) == (2, 1)
        @test length(sc.h) == 2
        @test length(sc.q) == 3

        # h should encode demand correctly
        @test sc.h[1] ≈ -0.7   # h = [-D; 0]
        @test sc.h[2] ≈ 0.0

        # T should encode reliability
        @test sc.T[2, 1] ≈ -0.8  # T = [0; -U]

        # Generate dataset
        dataset = generate_dataset(prob, 10)
        @test length(dataset) == 10
        for (x, sc2) in dataset
            @test length(x) == 1   # context = [D]
            @test sc2 isa Scenario
        end
    end

    @testset "TwoStageSLP construction" begin
        prob = ShipmentPlanningProblem()
        A1, b1, c1 = first_stage_data(prob)

        # Build scenarios
        scenarios = [scenario_realization(prob, rand(4)) for _ in 1:3]
        slp = TwoStageSLP(A1, b1, c1, scenarios)

        @test length(slp.scenarios) == 3
        @test isapprox(sum(slp.p), 1.0, atol=1e-10)  # equiprobable

        # Extensive form
        A_ext, b_ext, c_ext = extensive_form(slp)
        n2 = size(scenarios[1].W, 2)
        expected_vars = length(c1) + 3 * n2
        @test length(c_ext) == expected_vars
    end
end
