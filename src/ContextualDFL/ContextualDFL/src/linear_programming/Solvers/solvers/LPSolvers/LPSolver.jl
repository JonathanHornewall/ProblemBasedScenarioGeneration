abstract type LPSolver end

solve(solver::LPSolver, lp::LP; kwargs...) =
    error("LP solving is not defined for $(typeof(solver)).")
