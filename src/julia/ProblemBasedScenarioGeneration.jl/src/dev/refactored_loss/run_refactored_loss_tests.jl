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

using ProblemBasedScenarioGeneration
include(joinpath(@__DIR__, "RefactoredLoss.jl"))
using .RefactoredLoss

module CustomResourceAllocationLoss
using ..ProblemBasedScenarioGeneration
using ..ProblemBasedScenarioGeneration: diff_opt_b, scenario_collection_realization, LogBarCanLP, TwoStageSLP, s1_cost, diff_s1_cost, LogBarCanLP_standard_solver, ProblemInstanceC2SCanLP, ResourceAllocationProblem
using ChainRulesCore
include(joinpath(@__DIR__, "..", "..", "..", "..", "..", "..", "scripts", "resource_allocation_prototype", "custom_code", "neural_net.jl"))
end
using .CustomResourceAllocationLoss

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

old_time = @belapsed Zygote.gradient(sc -> PBSG.loss($problem, $reg_param_surr, $reg_param_prim, sc, $actual_scenario_collection), $scenario_collection)
new_time = @belapsed Zygote.gradient(sc -> refactored_loss($problem, $reg_param_surr, $reg_param_prim, sc, $actual_scenario_collection), $scenario_collection)
custom_time = @belapsed Zygote.gradient(sc -> CustomResourceAllocationLoss.loss($problem, $reg_param_surr, $reg_param_prim, sc, $actual_scenario_collection), $scenario_collection)

println("Old gradient time: ", old_time)
println("New gradient time: ", new_time)
println("Speedup factor: ", old_time / new_time)
println("Custom gradient time: ", custom_time)
println("Custom/New speedup factor: ", custom_time / new_time)
