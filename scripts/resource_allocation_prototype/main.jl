# Tests for the first experiments
using ProblemBasedScenarioGeneration
using LinearAlgebra
using Flux, ChainRulesCore, ChainRulesTestUtils, FiniteDifferences
using DataLoaders: DataLoader
using SparseArrays
using Statistics
using Plots: plot
using ProblemBasedScenarioGeneration: LogBarCanLP, TwoStageSLP, LogBarCanLP_standard_solver, ResourceAllocationProblemData, 
ResourceAllocationProblem, scenario_realization, dataGeneration, cost, cost_2s_LogBarCanLP, optimal_value,
diff_cost_2s_LogBarCanLP, diff_opt_b, train_model!, CanLP
import ProblemBasedScenarioGeneration: primal_problem_cost

import Flux: params, gradient, Optimise, Adam   # error if any of these re-appear


# Import data
include("parameters.jl")
cz, qw, ρᵢ, = vec(cz), vec(qw), vec(ρᵢ)

include("neural_net.jl")
include("training.jl")
include("test_function.jl")

function main()

problem_data = ResourceAllocationProblemData(μᵢⱼ, cz, qw, ρᵢ)
problem_instance = ResourceAllocationProblem(problem_data)

# Testing rrules


test_scenario = ones(30)
scenario_parameter = float(test_scenario)
reg_param_surr_test = 1.5
reg_param_prim_test = 1.5



#=
# Helper: if a function returns (J, y0), take the first item; otherwise return as-is.
takeJ(x) = x isa Tuple ? x[1] : x

# Only the scenario vector varies
g = x -> surrogate_solution(problem_instance, x, reg_param_surr_test, LogBarCanLP_standard_solver)

# Tiny one-sided steps to stay feasible; no adaptive probing
fdm = FiniteDifferences.forward_fdm(5, 1; max_range=1e-10, factor=5, adapt=0)

println("=== Jacobian check (FD vs your derivative) ===")
J_fd = takeJ( FiniteDifferences.jacobian(fdm, g, scenario_parameter) )
J_ad = derivative_surrogate_solution(problem_instance, scenario_parameter, reg_param_surr_test,
                                     LogBarCanLP_standard_solver)

@assert isa(J_fd, AbstractMatrix) "J_fd is not a matrix; got $(typeof(J_fd))"
@assert isa(J_ad, AbstractMatrix) "J_ad is not a matrix; got $(typeof(J_ad))"
@assert size(J_fd) == size(J_ad) "shape mismatch: size(J_fd)=$(size(J_fd)) vs size(J_ad)=$(size(J_ad))"

println("size(J) = ", size(J_fd))
println("∞-norm error:      ", norm(J_fd - J_ad, Inf))
println("relative ∞-norm:   ", norm(J_fd - J_ad, Inf) / max(1e-12, norm(J_fd, Inf)))
println("rank(J_ad):        ", rank(J_ad))
println("std of first 5 cols of J_ad: ", [std(view(J_ad, :, j)) for j in 1:min(5, size(J_ad,2))])

println("\n=== VJP check (what test_rrule compares) ===")
y0 = g(scenario_parameter)
ȳ  = randn(length(y0))  # arbitrary test cotangent

# AD VJP via your rrule
y_primal, pb = ChainRulesCore.rrule(surrogate_solution,
                                    problem_instance, scenario_parameter, reg_param_surr_test,
                                    LogBarCanLP_standard_solver)
Δf, Δprob, Δx_ad, Δμ, Δsolver = pb(ȳ)

# FD VJP
Δx_fd = takeJ( FiniteDifferences.j′vp(fdm, g, ȳ, scenario_parameter) )

@assert isa(Δx_ad, AbstractVector) "Δx_ad not a vector; got $(typeof(Δx_ad))"
@assert isa(Δx_fd, AbstractVector) "Δx_fd not a vector; got $(typeof(Δx_fd))"
@assert length(Δx_ad) == length(Δx_fd) "length mismatch: $(length(Δx_ad)) vs $(length(Δx_fd))"

println("VJP ∞-norm error:   ", norm(Δx_ad - Δx_fd, Inf))
println("VJP relative error: ", norm(Δx_ad - Δx_fd, Inf) / max(1e-12, norm(Δx_fd, Inf)))

=#





# Generate data (add comment later to explain the parameters)
Nsamples = 10
Noutofsamples = 10
sigma = 5
p =1
L = 3
Σ = 3


data_set_training, data_set_testing =  dataGeneration(problem_instance, 100, 10, 5, 1, 3, 3)


# Train the neural network model
reg_param_surr = 1.0
reg_param_prim = 0.0
reg_param_ref = 0.0

# Store regularization parameters for saving
reg_params = Dict(
    "reg_param_surr" => reg_param_surr,
    "reg_param_prim" => reg_param_prim,
    "reg_param_ref" => reg_param_ref
)


println("Starting training...")
train!(problem_instance::ResourceAllocationProblem, reg_param_surr, reg_param_ref, model, data_set_training; 
       opt = Adam(1e-3), epochs = 1000, display_iterations = true, 
       save_model = true, model_save_path = "trained_model.jls")

println("Training completed!")

# Save the complete experiment state
save_experiment_state(model, data_set_training, data_set_testing, problem_instance, reg_params, 
                     filepath = "experiment_state.jls")

# Test the trained model
println("Testing the trained model...")
test_result = testing(problem_instance, model, data_set_testing, reg_param_surr, reg_param_ref)
println("Test result: ", test_result)

println("Experiment completed and saved!")
end

main()