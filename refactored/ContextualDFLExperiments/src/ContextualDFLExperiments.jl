module ContextualDFLExperiments

using ContextualDFL
using Dates
using LinearAlgebra
using Random
using SparseArrays
using Statistics

not_implemented(name) = throw(ErrorException("$(name) is not implemented yet."))

include("benchmark.jl")
include("diagnostics.jl")
include("problem_instances/ProblemInstance.jl")

export benchmark, run_benchmark
export stochasticity_diagnostics, value_of_stochasticity
export ProblemInstance
export A, b, c
export W_base, T_base, h_base, q_base
export get_A, get_b, get_c
export get_W_base, get_T_base, get_h_base, get_q_base
export context_sampler, scenario_sampler, scenario_parametrization

end
