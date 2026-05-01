struct MSEScenLoss <: LossFunction end

const MSEScenarioLoss = MSEScenLoss

function (loss::MSEScenLoss)(program, xi, xi_tilde, mu, rho)
    return not_implemented(:MSEScenLoss)
end
