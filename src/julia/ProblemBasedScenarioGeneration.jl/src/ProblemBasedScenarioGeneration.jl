module ProblemBasedScenarioGeneration
using Random, Distributions, Statistics
using LinearAlgebra
using Einsum
using JuMP, Ipopt  # Optimization tools 
using Flux, ChainRulesCore
using DataLoaders: DataLoader
using SparseArrays
# Here we add the "include" statements in appropriate order.
include("lp_structs.jl")
include("differentitation/differentials_logbar_lp.jl")
include("solvers/log_bar_linprog_solvers.jl")
include("differentitation/2sp_differentials.jl")

include("problem_instances/problem_instances.jl")
include("neural_net_constructor.jl")
include("learning.jl")

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
export train_model!
export loss  # To compare with out of sample data

end # module ProblemBasedScenarioGeneration
