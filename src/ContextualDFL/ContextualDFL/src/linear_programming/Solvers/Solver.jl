struct Solver{TLogBarSolver<:LogBarSolver,TLPSolver<:LPSolver}
    log_bar_solver::TLogBarSolver
    lp_solver::TLPSolver
end

function solve(solver::Solver, lp::LP; μ=0, kwargs...)
    μ_vector = _barrier_parameter_vector(lp, μ)

    if _is_zero_barrier_parameter(μ_vector)
        return solve(solver.lp_solver, lp; kwargs...)
    end

    return solve(solver.log_bar_solver, lp; μ=μ_vector, kwargs...)
end
