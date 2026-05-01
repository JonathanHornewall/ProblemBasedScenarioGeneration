struct DflCLoss{S,P,R,M} <: LossFunction
    solver::S
    program::P
    rho::R
    mu::M
end

const DFLCLoss = DflCLoss

function (loss::DflCLoss)(program, xi, xi_tilde, mu, rho)
    program = loss.program
    solver = loss.solver
    mu_value = mu === nothing ? loss.mu : mu
    rho_value = rho === nothing ? loss.rho : rho
    z = solve(program, solver, xi_tilde; config=(mu=mu_value,)).first_stage_decision
    return cost_function(program, z, xi; solver=GLPKSolver(mu=rho_value))
end
