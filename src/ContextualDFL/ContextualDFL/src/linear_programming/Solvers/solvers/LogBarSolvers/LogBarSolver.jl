abstract type LogBarSolver end

solve(solver::LogBarSolver, lp::LP; μ=nothing, kwargs...) =
    error("Log-barrier LP solving is not defined for $(typeof(solver)).")
