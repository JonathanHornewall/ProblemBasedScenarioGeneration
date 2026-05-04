struct Solver{TLogBarSolver<:LogBarSolver,TLPSolver<:LPSolver}
    log_bar_solver::TLogBarSolver
    lp_solver::TLPSolver
end

function solve(solver::Solver, lp::LP; μ=0, ρ=0, rho=ρ, kwargs...)
    μ_vector = _barrier_parameter_vector(lp, μ)
    ρ_vector = _quadratic_parameter_vector(lp, rho)

    if _is_zero_barrier_parameter(μ_vector) && _is_zero_quadratic_parameter(ρ_vector)
        return solve(solver.lp_solver, lp; kwargs...)
    end

    return solve(solver.log_bar_solver, lp; μ=μ_vector, ρ=ρ_vector, kwargs...)
end
