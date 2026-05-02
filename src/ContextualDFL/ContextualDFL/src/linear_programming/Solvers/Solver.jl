struct Solver{TLogBarSolver<:LogBarSolver,TLPSolver<:LPSolver}
    log_bar_solver::TLogBarSolver
    lp_solver::TLPSolver
end

function solve(solver::Solver, lp::LP; μ=0, kwargs...)
    if iszero(μ)
        return solve(solver.lp_solver, lp; kwargs...)
    end

    return solve(solver.log_bar_solver, lp; μ=μ, kwargs...)
end
