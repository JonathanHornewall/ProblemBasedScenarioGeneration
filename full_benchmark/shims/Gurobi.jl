__precompile__(false)

module Gurobi

using GLPK

struct Env end

Optimizer(::Env) = GLPK.Optimizer()
Optimizer() = GLPK.Optimizer()

end # module
