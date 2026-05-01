struct MSEScenLoss <: LossFunction end

const MSEScenarioLoss = MSEScenLoss

function (loss::MSEScenLoss)(program, xi, xi_tilde, mu, rho)
    return _scenario_mse(xi, xi_tilde)
end
