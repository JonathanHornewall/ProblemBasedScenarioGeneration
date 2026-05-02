using ChainRulesCore
using ContextualDFL
using Flux
using LinearAlgebra
using Random
using Statistics
using Test

include("resource_allocation_instance.jl")

@testset "resource_allocation_training" begin
    @testset "instance import, decoder, and data generation" begin
        imported = imported_resource_allocation_data()

        @test size(imported.service_rate_parameters) == (20, 30)
        @test length(imported.first_stage_costs) == 20
        @test length(imported.second_stage_costs) == 30
        @test length(imported.yield_parameters) == 20
        @test all(>=(0.0), imported.service_rate_parameters)
        @test all(>(0.0), imported.first_stage_costs)
        @test all(>(0.0), imported.second_stage_costs)

        instance = resource_allocation_instance()
        resource_count, demand_count = size(instance.problem_data.service_rate_parameters)
        recourse_variables = demand_count + resource_count * demand_count + resource_count + demand_count
        recourse_rows = resource_count + demand_count

        @test instance.legacy_first_stage.A == zeros(1, resource_count)
        @test instance.legacy_first_stage.b == [0.0]
        @test instance.legacy_first_stage.c == instance.problem_data.first_stage_costs
        @test size(instance.stochastic_program.A_ineq) == (resource_count, resource_count)
        @test instance.stochastic_program.A_ineq == -Matrix{Float64}(I, resource_count, resource_count)
        @test instance.stochastic_program.c == instance.problem_data.first_stage_costs

        @test size(instance.base_scenario.W_eq) == (recourse_rows, recourse_variables)
        @test size(instance.base_scenario.T_eq) == (recourse_rows, resource_count)
        @test size(instance.base_scenario.W_ineq) == (recourse_variables, recourse_variables)
        @test instance.base_scenario.W_ineq == -Matrix{Float64}(I, recourse_variables, recourse_variables)
        @test instance.base_scenario.q[1:demand_count] == instance.problem_data.second_stage_costs
        @test all(iszero, instance.base_scenario.q[(demand_count + 1):end])

        decoder = ResourceAllocationDemandDecoder(instance)
        demand = collect(1.0:demand_count)
        W_eq, W_ineq, T_eq, T_ineq, h_eq, h_ineq, q = decoder((; h_eq=demand))

        @test W_eq === instance.base_scenario.W_eq
        @test W_ineq === instance.base_scenario.W_ineq
        @test T_eq === instance.base_scenario.T_eq
        @test T_ineq === instance.base_scenario.T_ineq
        @test h_eq[1:resource_count] == zeros(resource_count)
        @test h_eq[(resource_count + 1):end] == demand
        @test h_ineq === instance.base_scenario.h_ineq
        @test q === instance.base_scenario.q

        generated = generate_resource_allocation_context_scenarios(
            instance;
            n_contexts=5,
            n_scenarios=3,
            sigma=1.0,
            p=1.0,
            L=3,
            rng=Random.MersenneTwister(7),
        )

        @test size(generated.x_array) == (3, 5)
        @test length(generated.scenario_collections) == 5
        @test length(generated.data) == 5
        for scenario_collection in generated.scenario_collections
            @test length(scenario_collection) == 3
            for scenario in scenario_collection
                @test propertynames(scenario) == (:h_eq,)
                @test length(scenario.h_eq) == demand_count
                @test all(isfinite, scenario.h_eq)
            end
        end

        arrays = decoded_resource_allocation_arrays(decoder, generated.scenario_collections[1])
        @test size(arrays[1]) == (recourse_rows, recourse_variables, 3)
        @test size(arrays[3]) == (recourse_rows, resource_count, 3)
        @test size(arrays[5]) == (recourse_rows, 3)
        @test arrays[5][1:resource_count, :] == zeros(resource_count, 3)
        @test arrays[5][(resource_count + 1):end, 1] == generated.scenario_collections[1][1].h_eq

        _, pullback = ChainRulesCore.rrule(
            decode_scenario_collection,
            decoder,
            generated.scenario_collections[1],
        )
        dh_eq = ones(recourse_rows, 3)
        tangents = pullback((
            zeros(size(arrays[1])),
            zeros(size(arrays[2])),
            zeros(size(arrays[3])),
            zeros(size(arrays[4])),
            dh_eq,
            zeros(size(arrays[6])),
            zeros(size(arrays[7])),
        ))
        @test tangents[3][1].h_eq == ones(demand_count)
        @test tangents[3][2].h_eq == ones(demand_count)
    end

    @testset "resource allocation LP, stochastic solve, and cost" begin
        instance = resource_allocation_instance(resource_indices=1:4, demand_indices=1:5)
        solver = Solver(IpoptSolver(), HiGHSSolver())
        generated = generate_resource_allocation_context_scenarios(
            instance;
            n_contexts=2,
            n_scenarios=2,
            sigma=0.25,
            p=1.0,
            L=3,
            rng=Random.MersenneTwister(11),
        )
        arrays = resource_allocation_scenario_arrays(instance, generated.scenario_collections[1])
        W_eq, W_ineq, T_eq, T_ineq, h_eq, h_ineq, q = arrays
        probabilities = [0.4, 0.6]

        lp = construct_lp(
            instance.stochastic_program,
            W_eq,
            W_ineq,
            T_eq,
            T_ineq,
            h_eq,
            h_ineq,
            q;
            probabilities=probabilities,
        )
        lp_result = solve(HiGHSSolver(), lp)

        @test status_is_optimal(lp_result.status)
        assert_resource_allocation_feasible(lp, lp_result.z)
        @test lp_result.objective_value ≈ dot(lp.c, lp_result.z) atol = 1e-7
        @test minimum(lp_result.z) >= -1e-7

        z, y, λ_b_eq, λ_b_ineq, λ_h_eq, λ_h_ineq = solve(
            solver,
            instance.stochastic_program,
            W_eq,
            W_ineq,
            T_eq,
            T_ineq,
            h_eq,
            h_ineq,
            q;
            probabilities=probabilities,
        )

        @test length(z) == size(instance.problem_data.service_rate_parameters, 1)
        @test size(y) == (size(q, 1), size(q, 2))
        @test isempty(λ_b_eq)
        @test length(λ_b_ineq) == length(instance.stochastic_program.b_ineq)
        @test size(λ_h_eq) == size(h_eq)
        @test size(λ_h_ineq) == size(h_ineq)
        @test minimum(z) >= -1e-7
        @test minimum(y) >= -1e-7

        value = cost_function(
            instance.stochastic_program,
            solver,
            z,
            W_eq,
            W_ineq,
            T_eq,
            T_ineq,
            h_eq,
            h_ineq,
            q;
            probabilities=probabilities,
        )
        @test value ≈ lp_result.objective_value atol = 1e-6
    end

    @testset "resource allocation differentiation and rrules" begin
        instance = resource_allocation_instance(resource_indices=1:3, demand_indices=1:4)
        solver = Solver(IpoptSolver(), HiGHSSolver())
        generated = generate_resource_allocation_context_scenarios(
            instance;
            n_contexts=1,
            n_scenarios=1,
            sigma=0.1,
            p=1.0,
            L=3,
            rng=Random.MersenneTwister(13),
        )
        W_eq, W_ineq, T_eq, T_ineq, h_eq, h_ineq, q =
            resource_allocation_scenario_arrays(instance, generated.scenario_collections[1])
        program = instance.stochastic_program
        μ = 0.25

        lp = construct_lp(program, W_eq, W_ineq, T_eq, T_ineq, h_eq, h_ineq, q)
        base_result = solve(solver, lp; μ=μ, tol=1e-9)
        @test status_is_optimal(base_result.status)
        assert_resource_allocation_feasible(lp, base_result.z; atol=1e-5)

        dc = vec(deterministic_resource_allocation_direction((length(lp.c),); scale=0.02, phase=0.1))
        db_eq = vec(deterministic_resource_allocation_direction((length(lp.b_eq),); scale=0.02, phase=0.3))
        db_ineq = vec(deterministic_resource_allocation_direction((length(lp.b_ineq),); scale=0.02, phase=0.5))
        dz = diff_solve(
            solver,
            lp,
            μ;
            pre_computed=base_result,
            dc=dc,
            db_eq=db_eq,
            db_ineq=db_ineq,
            tol=1e-9,
        )

        ϵ = 1e-4
        lp_plus = LP(
            A_eq=lp.A_eq,
            A_ineq=lp.A_ineq,
            b_eq=lp.b_eq + ϵ .* db_eq,
            b_ineq=lp.b_ineq + ϵ .* db_ineq,
            c=lp.c + ϵ .* dc,
        )
        lp_minus = LP(
            A_eq=lp.A_eq,
            A_ineq=lp.A_ineq,
            b_eq=lp.b_eq - ϵ .* db_eq,
            b_ineq=lp.b_ineq - ϵ .* db_ineq,
            c=lp.c - ϵ .* dc,
        )
        finite_difference_dz =
            (solve(solver, lp_plus; μ=μ, tol=1e-9).z - solve(solver, lp_minus; μ=μ, tol=1e-9).z) ./ (2ϵ)
        @test dz ≈ finite_difference_dz atol = 2e-3 rtol = 2e-2

        z_for_cost = fill(15.0, length(program.c))
        value, cost_pullback = ChainRulesCore.rrule(
            cost_function,
            program,
            solver,
            z_for_cost,
            W_eq,
            W_ineq,
            T_eq,
            T_ineq,
            h_eq,
            h_ineq,
            q;
            μ=μ,
            tol=1e-9,
        )
        dz_cost = cost_pullback(1.0)[4]
        direction = vec(deterministic_resource_allocation_direction(size(z_for_cost); scale=0.1, phase=0.7))
        finite_difference_cost = (
            cost_function(program, solver, z_for_cost + ϵ .* direction, W_eq, W_ineq, T_eq, T_ineq, h_eq, h_ineq, q; μ=μ, tol=1e-9) -
            cost_function(program, solver, z_for_cost - ϵ .* direction, W_eq, W_ineq, T_eq, T_ineq, h_eq, h_ineq, q; μ=μ, tol=1e-9)
        ) / (2ϵ)

        @test value isa Number
        @test dot(dz_cost, direction) ≈ finite_difference_cost atol = 2e-3 rtol = 2e-2

        output, solve_pullback = ChainRulesCore.rrule(
            solve,
            solver,
            program,
            W_eq,
            W_ineq,
            T_eq,
            T_ineq,
            h_eq,
            h_ineq,
            q;
            μ=μ,
            tol=1e-9,
        )
        dy_tangent = deterministic_resource_allocation_direction(size(output[2]); scale=0.03, phase=0.2)
        solve_tangents = solve_pullback((
            zeros(size(output[1])),
            dy_tangent,
            zeros(size(output[3])),
            zeros(size(output[4])),
            zeros(size(output[5])),
            zeros(size(output[6])),
        ))
        dh_eq_tangent = solve_tangents[8]
        h_direction = zeros(size(h_eq))
        resource_count = size(instance.problem_data.service_rate_parameters, 1)
        h_direction[(resource_count + 1):end, :] .=
            deterministic_resource_allocation_direction(
                size(view(h_direction, (resource_count + 1):size(h_direction, 1), :));
                scale=0.1,
                phase=1.3,
            )

        function solve_scalar(h_eq_candidate)
            candidate_output = solve(
                solver,
                program,
                W_eq,
                W_ineq,
                T_eq,
                T_ineq,
                h_eq_candidate,
                h_ineq,
                q;
                μ=μ,
                tol=1e-9,
            )
            return sum(candidate_output[2] .* dy_tangent)
        end

        finite_difference_solve =
            (solve_scalar(h_eq + ϵ .* h_direction) - solve_scalar(h_eq - ϵ .* h_direction)) / (2ϵ)
        @test sum(dh_eq_tangent .* h_direction) ≈ finite_difference_solve atol = 3e-3 rtol = 3e-2
    end

    @testset "train! loop learns resource allocation demand scenarios" begin
        Random.seed!(23)
        instance = resource_allocation_instance()
        n_scenarios = 2
        generated = generate_resource_allocation_context_scenarios(
            instance;
            n_contexts=24,
            n_scenarios=n_scenarios,
            sigma=0.5,
            p=1.0,
            L=3,
            rng=Random.MersenneTwister(23),
        )
        model = construct_resource_allocation_neural_net(instance; n_scenarios=n_scenarios)
        initial_loss = mean_resource_allocation_training_loss(model, generated.data)

        result = train!(
            model,
            resource_allocation_training_loss,
            relative_resource_allocation_training_loss,
            fill(0.0, 12),
            generated.data;
            opt=Flux.Adam(1e-3),
            epochs=12,
            batchsize=4,
            display_iterations=true,
            display_plot=false,
            shuffle=true,
            rng=Random.MersenneTwister(29),
        )

        final_loss = mean_resource_allocation_training_loss(model, generated.data)
        history_losses = [row.loss for row in result.history]
        history_display_losses = [row.display_loss for row in result.history]

        @test length(result.history) == 12
        @test all(isfinite, history_losses)
        @test all(isfinite, history_display_losses)
        @test final_loss < initial_loss
        @test last(history_display_losses) < first(history_display_losses)
        @test minimum(history_losses) <= first(history_losses)
    end
end
