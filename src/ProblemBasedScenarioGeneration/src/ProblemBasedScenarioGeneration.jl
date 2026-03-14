"""
ProblemBasedScenarioGeneration

A Julia package for problem-based scenario generation in two-stage stochastic
linear programming. Trains neural networks to generate scenario sets that
minimize downstream decision cost (decision-focused learning).

Architecture:
- `core/`: data structures (Scenario, TwoStageSLP, canonical form)
- `solvers/`: LP solver (HiGHS) and log-barrier Newton solver
- `diff/`: implicit differentiation and ChainRules rrules
- `problems/`: concrete problem instances
- `loss/`: decision regret loss
- `models/`: neural network builders
- `training/`: training loops and continuation schedule
"""
module ProblemBasedScenarioGeneration

using LinearAlgebra
using SparseArrays
using Statistics
using Random
using Distributions
using Flux
using ChainRulesCore
using JuMP
using HiGHS

# -----------------------------------------------------------------------
# Core data structures
# -----------------------------------------------------------------------
include("core/scenario.jl")
include("core/two_stage_lp.jl")
include("core/canonical_form.jl")

# -----------------------------------------------------------------------
# Solvers
# -----------------------------------------------------------------------
include("solvers/lp_solver.jl")
include("solvers/barrier_solver.jl")

# -----------------------------------------------------------------------
# Implicit differentiation
# -----------------------------------------------------------------------
include("diff/implicit_diff.jl")
include("diff/barrier_rrule.jl")
include("diff/subgradient_rrule.jl")

# -----------------------------------------------------------------------
# Problem instances
# -----------------------------------------------------------------------
include("problems/interface.jl")
include("problems/resource_allocation.jl")
include("problems/shipment_planning.jl")
include("problems/newsvendor.jl")

# -----------------------------------------------------------------------
# Loss
# -----------------------------------------------------------------------
include("loss/decision_regret.jl")

# -----------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------
include("models/scenario_generator.jl")
include("models/output_heads.jl")

# -----------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------
include("training/trainer.jl")
include("training/continuation.jl")

# -----------------------------------------------------------------------
# Exports
# -----------------------------------------------------------------------

# Core types
export Scenario, scenario_from_tuple
export TwoStageSLP
export to_canonical, extensive_form, to_canonical_decision

# Solvers
export solve_lp, solve_lp_primal
export BarrierCache, solve_barrier, kkt_residual

# Implicit diff
export implicit_diff_h, implicit_diff_q, recourse_multiplier

# Problems
export ProblemInstance, NoisePattern
export H_ONLY, Q_ONLY, W_ONLY, WH, WQ, WHQ
export first_stage_data, scenario_realization, generate_dataset, noise_pattern
export ResourceAllocationProblem
export ShipmentPlanningProblem
export UnreliableNewsvendorProblem

# Loss
export surrogate_first_stage, decision_regret, relative_decision_regret
export evaluate_cost, recourse_cost

# Models
export build_generator, build_output_head, build_full_model
export _scenario_param_dim, _context_dim

# Training
export train!, continuation_train!

end  # module ProblemBasedScenarioGeneration
