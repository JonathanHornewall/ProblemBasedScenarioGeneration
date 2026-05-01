struct ProjectedZLoss{S,P} <: LossFunction
    solver::S
    program::P
end

function (loss::ProjectedZLoss)(program, xi, xi_tilde, mu, rho)
    program = loss.program
    solver = loss.solver
    z_actual = solve(program, solver, xi; config=(mu=mu,)).first_stage_decision
    z_pred = solve(program, solver, xi_tilde; config=(mu=mu,)).first_stage_decision
    return sum(abs2, z_actual .- z_pred) / length(z_actual)
end
