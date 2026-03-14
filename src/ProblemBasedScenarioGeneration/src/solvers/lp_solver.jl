"""
LP solver using JuMP/HiGHS for canonical-form linear programs.
min c'x  s.t. Ax = b, x ≥ 0

This is used for evaluation (unsmoothed LP), not training.
"""

using JuMP, HiGHS

"""
    solve_lp(A, b, c; tol=1e-9) -> (x::Vector, lambda::Vector)

Solve the canonical LP `min c'x  s.t. Ax = b, x ≥ 0` using HiGHS.

Returns primal solution `x` and dual variable `lambda` for the equality
constraints. Throws on infeasibility or unboundedness.
"""
function solve_lp(A, b, c; tol=1e-9)
    m, n = size(A)
    length(b) == m || error("Dimension mismatch: A is $(m)×$(n) but b has length $(length(b))")
    length(c) == n || error("Dimension mismatch: A is $(m)×$(n) but c has length $(length(c))")

    model = Model(HiGHS.Optimizer)
    set_silent(model)
    set_optimizer_attribute(model, "primal_feasibility_tolerance", tol)
    set_optimizer_attribute(model, "dual_feasibility_tolerance", tol)

    @variable(model, x[1:n] >= 0)
    con = @constraint(model, A * x .== b)
    @objective(model, Min, dot(c, x))

    optimize!(model)

    ts = termination_status(model)
    if !(ts in (MOI.OPTIMAL, MOI.LOCALLY_SOLVED, MOI.ALMOST_LOCALLY_SOLVED))
        error("solve_lp: no optimal solution — status $ts ($(MOI.get(model, MOI.RawStatusString())))")
    end

    x_opt = value.(x)
    lambda_opt = dual.(con)

    # Sanity check
    if maximum(abs.(A * x_opt .- b)) > 1e-6
        @warn "solve_lp: primal feasibility violation $(maximum(abs.(A*x_opt .- b)))"
    end

    return x_opt, lambda_opt
end

"""
    solve_lp_primal(A, b, c; tol=1e-9) -> Vector

Primal-only variant of `solve_lp`.
"""
solve_lp_primal(A, b, c; tol=1e-9) = solve_lp(A, b, c; tol=tol)[1]
