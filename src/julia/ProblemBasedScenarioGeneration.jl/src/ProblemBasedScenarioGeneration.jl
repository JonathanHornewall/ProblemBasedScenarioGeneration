module ProblemBasedScenarioGeneration
using Random, Distributions, Statistics
using LinearAlgebra
using Einsum
using JuMP, Ipopt  # Optimization tools 
using Flux, ChainRulesCore
using DataLoaders: DataLoader
using SparseArrays
using Statistics
using Serialization
using Plots
# Here we add the "include" statements in appropriate order.

# include("lp_structs.jl")
include("differentitation/differentials_logbar_lp.jl")
include("solvers/can_lp_solver.jl")
include("solvers/log_bar_linprog_solvers.jl")
include("differentitation/2sp_differentials.jl")

include("problem_instances/problem_instances.jl")
include("neural_net/loss.jl")
include("neural_net/training.jl")
include("neural_net/load_parameters.jl")

include("utils.jl")

# Inclusions for specific problem instances
include("problem_instances/resource_allocation/resource_allocation_problem.jl")
include("problem_instances/resource_allocation/data_generation.jl")

export ProblemInstanceC2SCanLP
export manual_C2SCanLP
export Scenario

export ResourceAllocationProblem
export ResourceAllocationProblemData

export construct_neural_network
export train!
export loss  # To compare with out of sample data
export relative_loss

export save_trained_model, load_trained_model, save_training_data, load_training_data, save_experiment_state, 
load_experiment_state, load_and_continue_experiment, continue_training, compare_models

export solve_canonical_lp
export convert_standard_to_canonical_form_regular

# Export types and functions needed for neural network differentiation
export TwoStageSLP, LogBarCanLP, CanLP
export LogBarCanLP_standard_solver, LogBarCanLP_standard_solver_primal
export s1_cost, diff_s1_cost
export diff_cache_computation, diff_opt
export scenario_collection_realization, surrogate_solution

end # module ProblemBasedScenarioGeneration
