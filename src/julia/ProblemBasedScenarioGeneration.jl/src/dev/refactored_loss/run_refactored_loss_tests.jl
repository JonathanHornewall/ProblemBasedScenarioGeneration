using Pkg
env_override = get(ENV, "PBSG_REFACTORED_ENV", nothing)
if env_override === nothing
    Pkg.activate(joinpath(@__DIR__, "..", "..", ".."))
else
    Pkg.activate(env_override)
end

using Random
using LinearAlgebra
using Statistics
using BenchmarkTools
using Zygote
using Flux
using Profile

using ProblemBasedScenarioGeneration
include(joinpath(@__DIR__, "RefactoredLoss.jl"))
using .RefactoredLoss

module CustomResourceAllocationLoss
using ..ProblemBasedScenarioGeneration
using ..ProblemBasedScenarioGeneration: diff_opt_b, scenario_collection_realization, LogBarCanLP, TwoStageSLP, s1_cost, diff_s1_cost, LogBarCanLP_standard_solver, ProblemInstanceC2SCanLP, ResourceAllocationProblem
using ChainRulesCore
const _prototype_dir = normpath(joinpath(@__DIR__, "..", "..", "..", "..", "..", "..",
    "scripts", "resource_allocation_prototype"))
include(joinpath(_prototype_dir, "custom_code", "neural_net.jl"))
end
using .CustomResourceAllocationLoss

const _prototype_dir = CustomResourceAllocationLoss._prototype_dir

const PBSG = ProblemBasedScenarioGeneration

const resource_alloc_scenario_type = PBSG.ScenarioType(:H)

has_return_fn = isdefined(PBSG, :return_scenario_type)
has_resource_alloc_method = has_return_fn && hasmethod(PBSG.return_scenario_type, Tuple{PBSG.ResourceAllocationProblem})
if !has_resource_alloc_method
    @info "Defining return_scenario_type for ResourceAllocationProblem" 
    @eval PBSG const _resource_alloc_scenario_type = $resource_alloc_scenario_type
    @eval PBSG function return_scenario_type(::ResourceAllocationProblem)
        return _resource_alloc_scenario_type
    end
end

function build_sample_problem()
    I, J = 3, 3
    service_rate = 0.8 .+ 0.1 .* rand(I, J)
    first_stage_costs = 5 .+ rand(I)
    second_stage_costs = 2 .+ rand(J)
    yield_parameters = 0.5 .+ 0.1 .* rand(I)
    data = PBSG.ResourceAllocationProblemData(service_rate, first_stage_costs, second_stage_costs, yield_parameters)
    return PBSG.ResourceAllocationProblem(data)
end

function finite_difference(f, x, direction; eps=1e-5)
    return (f(x .+ eps .* direction) - f(x .- eps .* direction)) / (2eps)
end

Random.seed!(42)
problem = build_sample_problem()
J = size(problem.problem_data.service_rate_parameters, 2)
S = 1
scenario_collection = 1 .+ rand(J, S)
actual_scenario_collection = 1 .+ rand(J, S)
reg_param_surr = 1.0
reg_param_prim = 1.0

loss_old = PBSG.loss(problem, reg_param_surr, reg_param_prim, scenario_collection, actual_scenario_collection)
loss_new = refactored_loss(problem, reg_param_surr, reg_param_prim, scenario_collection, actual_scenario_collection)
loss_custom = CustomResourceAllocationLoss.loss(problem, reg_param_surr, reg_param_prim, scenario_collection, actual_scenario_collection)

println("Loss difference: ", abs(loss_old - loss_new))
println("Custom vs new loss difference: ", abs(loss_custom - loss_new))

old_grad = Zygote.gradient(sc -> PBSG.loss(problem, reg_param_surr, reg_param_prim, sc, actual_scenario_collection), scenario_collection)[1]
new_grad = Zygote.gradient(sc -> refactored_loss(problem, reg_param_surr, reg_param_prim, sc, actual_scenario_collection), scenario_collection)[1]
custom_grad = Zygote.gradient(sc -> CustomResourceAllocationLoss.loss(problem, reg_param_surr, reg_param_prim, sc, actual_scenario_collection), scenario_collection)[1]

grad_diff = norm(old_grad .- new_grad) / max(norm(old_grad), 1e-12)
println("Relative gradient difference: ", grad_diff)
grad_diff_custom = norm(custom_grad .- new_grad) / max(norm(new_grad), 1e-12)
println("Custom vs new relative gradient difference: ", grad_diff_custom)

direction = randn(size(scenario_collection))
fd = finite_difference(sc -> refactored_loss(problem, reg_param_surr, reg_param_prim, sc, actual_scenario_collection), scenario_collection, direction)
ad = sum(new_grad .* direction)
println("Finite difference check: ", abs(fd - ad))
fd_custom = finite_difference(sc -> CustomResourceAllocationLoss.loss(problem, reg_param_surr, reg_param_prim, sc, actual_scenario_collection), scenario_collection, direction)
ad_custom = sum(custom_grad .* direction)
println("Custom finite difference check: ", abs(fd_custom - ad_custom))

old_time = @elapsed Zygote.gradient(sc -> PBSG.loss(problem, reg_param_surr, reg_param_prim, sc, actual_scenario_collection), scenario_collection)
new_time = @elapsed Zygote.gradient(sc -> refactored_loss(problem, reg_param_surr, reg_param_prim, sc, actual_scenario_collection), scenario_collection)
custom_time = @elapsed Zygote.gradient(sc -> CustomResourceAllocationLoss.loss(problem, reg_param_surr, reg_param_prim, sc, actual_scenario_collection), scenario_collection)

println("Old gradient time: ", old_time)
println("New gradient time: ", new_time)
println("Speedup factor: ", old_time / new_time)
println("Custom gradient time: ", custom_time)
println("Custom/New speedup factor: ", custom_time / new_time)

function gradient_timing(problem, scenario_collection, actual_scenario_collection;
                         repeats=3, reg_param_surr=1.0, reg_param_prim=1.0)
    t0 = time()
    for _ in 1:repeats
        Zygote.gradient(sc -> refactored_loss(problem, reg_param_surr, reg_param_prim, sc, actual_scenario_collection),
                        scenario_collection)
    end
    ref_time = (time() - t0) / repeats

    t0 = time()
    for _ in 1:repeats
        Zygote.gradient(sc -> CustomResourceAllocationLoss.loss(problem, reg_param_surr, reg_param_prim, sc, actual_scenario_collection),
                        scenario_collection)
    end
    custom_time = (time() - t0) / repeats

    return ref_time, custom_time
end

function benchmark_speed(; samples=4, repeats=3, scenarios=1)
    ref_times = Float64[]
    custom_times = Float64[]

    for seed in 1:samples
        Random.seed!(seed + 123)
        problem = build_sample_problem()
        J = size(problem.problem_data.service_rate_parameters, 2)
        scenario_collection = 1 .+ rand(J, scenarios)
        actual_scenario_collection = 1 .+ rand(J, scenarios)
        ref_t, custom_t = gradient_timing(problem, scenario_collection, actual_scenario_collection;
                                          repeats=repeats)
        push!(ref_times, ref_t)
        push!(custom_times, custom_t)
    end

    println("\nSpeed benchmark over $(samples) samples (each averaged over $(repeats) repeats):")
    println("Refactored median time: ", median(ref_times))
    println("Custom median time:     ", median(custom_times))
    ratios = ref_times ./ custom_times
    println("Median ref/custom ratio: ", median(ratios))
    println("Mean ref/custom ratio:   ", mean(ratios))
end

benchmark_speed()

function mini_training_time(loss_function; epochs::Int=5, samples::Int=20, repeats::Int=3)
    total = 0.0
    for _ in 1:repeats
        Random.seed!(42)
        problem = build_sample_problem()
        model = ProblemBasedScenarioGeneration.construct_neural_network(problem; nr_of_scenarios=1)
        opt = Flux.Optimisers.Adam(1e-3)
        opt_state = Flux.Optimisers.setup(opt, model)
        dataset = [(rand(size(problem.problem_data.service_rate_parameters, 1)),
                    1 .+ rand(size(problem.problem_data.service_rate_parameters, 2), 1))
                   for _ in 1:samples]
        reg_param_surr = 0.1
        reg_param_prim = 0.0
        t0 = time()
        for _ in 1:epochs
            for (x, actual) in dataset
                grad = Flux.gradient(model) do m
                    ξ_output = m(x)
                    ξ_output = reshape(ξ_output, :, size(actual, 2))
                    loss_function(problem, reg_param_surr, reg_param_prim, ξ_output, actual)
                end
                opt_state, model = Flux.Optimisers.update(opt_state, model, grad[1])
            end
        end
        total += time() - t0
    end
    total / repeats
end

ref_train = mini_training_time(refactored_loss)
custom_train = mini_training_time((problem, r_s, r_p, predicted, actual) ->
    CustomResourceAllocationLoss.loss(problem, r_s, r_p, predicted, actual))

println("\nMini training runtime (averaged over repeats):")
println("Refactored loss time: ", ref_train)
println("Custom loss time:     ", custom_train)
println("Ref/Custom ratio:     ", ref_train / custom_train)

function profile_mini_training(loss_function; epochs::Int=1, samples::Int=5)
    Profile.clear()
    Random.seed!(11)
    problem = build_sample_problem()
    model = ProblemBasedScenarioGeneration.construct_neural_network(problem; nr_of_scenarios=1)
    opt = Flux.Optimisers.Adam(1e-3)
    opt_state = Flux.Optimisers.setup(opt, model)
    dataset = [(rand(size(problem.problem_data.service_rate_parameters, 1)),
                1 .+ rand(size(problem.problem_data.service_rate_parameters, 2), 1))
               for _ in 1:samples]
    reg_param_surr = 0.1
    reg_param_prim = 0.0

    Profile.@profile begin
        for _ in 1:epochs
            for (x, actual) in dataset
                grad = Flux.gradient(model) do m
                    ξ_output = m(x)
                    ξ_output = reshape(ξ_output, :, size(actual, 2))
                    loss_function(problem, reg_param_surr, reg_param_prim, ξ_output, actual)
                end
                opt_state, model = Flux.Optimisers.update(opt_state, model, grad[1])
            end
        end
    end

    Profile.print(; maxdepth=12, sortedby=:count)
end

println("\nProfile (refactored loss, mini training):")
profile_mini_training(refactored_loss)

println("\nProfile (custom loss, mini training):")
profile_mini_training((problem, r_s, r_p, predicted, actual) ->
    CustomResourceAllocationLoss.loss(problem, r_s, r_p, predicted, actual))
