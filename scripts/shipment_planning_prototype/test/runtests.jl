using Test
using Statistics
using FiniteDifferences
using Zygote
using ProblemBasedScenarioGeneration
using ProblemBasedScenarioGeneration: dataGeneration, loss, relative_loss
using Flux

include(joinpath(@__DIR__, "..", "custom_code", "neural_net.jl"))
include(joinpath(@__DIR__, "..", "parameters.jl"))

const problem_instance = ShipmentPlanningProblem(shipment_problem_data)

function _scenario_dim(instance)
    size(instance.problem_data.shipment_costs, 2)
end

@testset "Shipment surrogate rrule" begin
    scenario_dim = _scenario_dim(problem_instance)
    scenario_vec = 80 .+ randn(scenario_dim)

    function surrogate_sum(vec)
        scenario_mat = reshape(vec, scenario_dim, 1)
        sum(surrogate_solution(problem_instance, 0.5, scenario_mat))
    end

    fdm = FiniteDifferences.central_fdm(5, 1)
    fd_grad = first(FiniteDifferences.grad(fdm, surrogate_sum, scenario_vec))
    ad_grad = first(Zygote.gradient(surrogate_sum, scenario_vec))

    @test maximum(abs.(fd_grad .- ad_grad)) < 1e-5
end

@testset "Shipment training decreases relative loss" begin
    cfg = shipment_training_config
    train_data, _ = dataGeneration(
        problem_instance,
        cfg[:Ntraining_samples] ÷ 4,
        1,
        max(5, cfg[:N_xi_per_x] ÷ 10),
        cfg[:sigma],
        cfg[:seasonal_scale],
        cfg[:trend_decay];
        collections_per_sample = 1
    )

    model = construct_neural_network(problem_instance; nr_of_scenarios = 1)
    reg_param = 0.5

    function _as_matrix(v)
        ndims(v) == 1 ? reshape(v, :, 1) : v
    end

    function avg_relative_loss(dataset)
        vals = Float64[]
        for (x, ξ) in dataset
            ξ̂ = model(x)
            push!(vals, relative_loss(problem_instance, reg_param, reg_param, ξ̂, _as_matrix(ξ)))
        end
        mean(vals)
    end

    before_loss = avg_relative_loss(train_data)

    input_loss(ξ_output, ξ_actual) = loss(problem_instance, reg_param, reg_param, ξ_output, ξ_actual)
    input_relative_loss(ξ_output, ξ_actual) = relative_loss(problem_instance, reg_param, reg_param, ξ_output, ξ_actual)
    train!(input_loss, input_relative_loss, model, train_data;
           opt = Adam(5e-4), epochs = 2, batchsize = 2, display_iterations = false)

    after_loss = avg_relative_loss(train_data)
    @test after_loss < before_loss
end
