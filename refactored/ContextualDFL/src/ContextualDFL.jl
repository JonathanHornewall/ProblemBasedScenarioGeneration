module ContextualDFL

using ChainRulesCore
using LinearAlgebra
using Random
using SparseArrays
using Statistics

const COMPONENT_SYMBOLS = (:W_eq, :W_ineq, :T_eq, :T_ineq, :h, :q)

not_implemented(name) = throw(ErrorException("$(name) is not implemented yet."))

include("linear_programming/LP.jl")
include("linear_programming/solver.jl")
include("linear_programming/diff_lp.jl")
include("linear_programming/solvers/glpk/GLPKSolver.jl")
include("linear_programming/solvers/gurobi/GurobiSolver.jl")

include("stochastic_programming/StochasticProgram.jl")
include("stochastic_programming/construct_lp.jl")
include("stochastic_programming/solve.jl")
include("stochastic_programming/solve_rrule.jl")
include("stochastic_programming/cost_function.jl")
include("stochastic_programming/cost_function_rrule.jl")

include("scenario_decoders/ComponentDecoder.jl")
include("scenario_decoders/BaseScenario.jl")
include("scenario_decoders/DecoderStrategy.jl")
include("scenario_decoders/ScenarioDecoder.jl")
include("scenario_decoders/DataSetScenarioDecoder.jl")

include("DFLScenarioGenerator.jl")

include("learning/DataSet.jl")
include("learning/loss_functions/Loss.jl")
include("learning/loss_functions/DflScenLoss.jl")
include("learning/loss_functions/dfl_c_loss.jl")
include("learning/loss_functions/MSE_scen_loss.jl")
include("learning/loss_functions/projected_z_loss.jl")
include("learning/train.jl")
include("learning/utils/test.jl")
include("learning/utils/hyper_parameter_helpers/schedules.jl")

include("data_generation/context_sampling/ContextSampler.jl")
include("data_generation/scenario_sampling/ScenarioSampler.jl")

export COMPONENT_SYMBOLS

export DFLScenarioGenerator

export DataSet
export train
export test
export LossFunction
export DflScenLoss, DFLScenarioLoss
export DflCLoss, DFLCLoss
export MSEScenLoss, MSEScenarioLoss
export ProjectedZLoss
export constant_schedule, linear_schedule, geometric_schedule
export make_mu_schedule, make_rho_schedule
export make_batch_size_schedule, make_step_size_schedule

export ScenarioDecoder, TrivialDecoder
export DataSetScenarioDecoder
export BaseScenario
export DecoderStrategy
export ComponentDecoder, DefaultComponentDecoder, EmptyComponentDecoder

export ContextSampler, generate_context_set
export ScenarioSampler, generate_scenario_set

export StochasticProgram
export construct_lp
export solve
export solve_rrule
export cost_function, scenario_wise_cost
export cost_function_rrule

export LP
export Solver, SolverConfig, SolverStrategy
export LPImplementation, ConcreteLPImplementation
export implement, get_implementation
export differentiate_solve
export GLPKSolver, GLPKImplementation
export GurobiSolver, GurobiImplementation

end
