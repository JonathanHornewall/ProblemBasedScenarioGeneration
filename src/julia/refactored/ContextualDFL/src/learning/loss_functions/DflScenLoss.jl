struct DflScenLoss{R,S,P} <: LossFunction
    scenario_realizer::R
    solver::S
    program::P
end

const DFLScenarioLoss = DflScenLoss

DflScenLoss(solver::S, program::P) where {S,P} = DflScenLoss(nothing, solver, program)

function (loss::DflScenLoss)(program, xi, xi_tilde, mu, rho)
    program = loss.program
    solver = loss.solver
    predicted = loss.scenario_realizer === nothing ? xi_tilde : loss.scenario_realizer(xi_tilde)
    actual = loss.scenario_realizer === nothing ? xi : loss.scenario_realizer(xi)
    z = solve(program, solver, predicted; config=(mu=mu,)).first_stage_decision
    return cost_function(program, z, actual; solver=GLPKSolver(mu=rho))
end
