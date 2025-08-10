# Tests for the first experiments
using ProblemBasedScenarioGeneration
using Flux, ChainRulesCore
using DataLoaders: DataLoader
using SparseArrays
using Statistics
using Plots: plot
using ProblemBasedScenarioGeneration: LogBarCanLP, TwoStageSLP, LogBarCanLP_standard_solver, ResourceAllocationProblemData, 
ResourceAllocationProblem, scenario_realization, dataGeneration, cost, cost_2s_LogBarCanLP, diff_cost_2s_LogBarCanLP, train_model!
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

# Generate data (add comment later to explain the parameters)
Nsamples = 10
Noutofsamples = 10
σ = 5
p =1 
L = 3
Σ = 3

data_set_training, data_set_testing =  dataGeneration(problem_instance, 10, 10, 5, 1, 3, 3)

# Train the neural network model
regularization_parameter = 1.0

train!(problem_instance::ResourceAllocationProblem, regularization_parameter, model, data_set_training; opt = Adam(1e-3), epochs = 5, display_iterations = true)

print(testing(problem_instance, model, data_set_testing, regularization_parameter))

