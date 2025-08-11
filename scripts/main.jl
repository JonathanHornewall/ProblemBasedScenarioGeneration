# Tests for the first experiments
using ProblemBasedScenarioGeneration
using Flux, ChainRulesCore, ChainRulesTestUtils
using DataLoaders: DataLoader
using SparseArrays
using Statistics
using Plots: plot
using ProblemBasedScenarioGeneration: LogBarCanLP, TwoStageSLP, LogBarCanLP_standard_solver, ResourceAllocationProblemData, 
ResourceAllocationProblem, scenario_realization, dataGeneration, cost, cost_2s_LogBarCanLP, optimal_value,
diff_cost_2s_LogBarCanLP, diff_opt_b, train_model!
import ProblemBasedScenarioGeneration: primal_problem_cost

import Flux: params, gradient, Optimise, Adam   # error if any of these re-appear


# Import data
include("parameters.jl")
cz, qw, ρᵢ, = vec(cz), vec(qw), vec(ρᵢ)

include("neural_net.jl")
include("training.jl")
include("test_function.jl")



problem_data = ResourceAllocationProblemData(μᵢⱼ, cz, qw, ρᵢ)
problem_instance = ResourceAllocationProblem(problem_data)

# Testing rrules

test_scenario = ones(30)
regularization_parameter = 1.0

#test_rrule(surrogate_solution, problem_instance, test_scenario, regularization_parameter, LogBarCanLP_standard_solver)

# Generate data (add comment later to explain the parameters)
Nsamples = 10
Noutofsamples = 10
σ = 5
p =1
L = 3
Σ = 3


data_set_training, data_set_testing =  dataGeneration(problem_instance, 100, 10, 5, 1, 3, 3)


# Train the neural network model
regularization_parameter = 1.0

# Test the value function computation
#ξ_test = first(values(data_set_training))
#W, T, h, q = scenario_realization(problem_instance, ξ_test)
#A, b, c = problem_instance.s1_constraint_matrix, problem_instance.s1_constraint_vector, problem_instance.s1_cost_vector
# Compute log-barrier way
#twoslp = TwoStageSLP(A, b, c, [W], [T], [h], [q])
#logbarlp = LogBarCanLP(twoslp, regularization_parameter)
#logbar_compute = optimal_value(logbarlp, LogBarCanLP_standard_solver)
# compute primal way
#optimal_solution, optimal_dual = LogBarCanLP_standard_solver(logbarlp)
#first_stage_decision = optimal_solution[1:length(c)]
#primal_compute = primal_problem_cost(problem_instance, ξ_test, regularization_parameter, first_stage_decision)
#println("Log-barrier computed value: ", logbar_compute)
#println("Primal computed value: ", primal_compute)

train!(problem_instance::ResourceAllocationProblem, regularization_parameter, model, data_set_training; opt = Adam(1e-3), epochs = 20, display_iterations = true)

print(testing(problem_instance, model, data_set_testing, regularization_parameter))

