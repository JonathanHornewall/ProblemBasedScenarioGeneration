struct DflScenLoss{R,S,P} <: LossFunction
    scenario_realizer::R
    solver::S
    program::P
end

const DFLScenarioLoss = DflScenLoss

function (loss::DflScenLoss)(program, xi, xi_tilde, mu, rho)
    return not_implemented(:DflScenLoss)
end
