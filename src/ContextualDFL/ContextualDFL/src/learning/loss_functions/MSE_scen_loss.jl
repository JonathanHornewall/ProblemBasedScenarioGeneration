struct MSEScenLoss <: LossFunction end

(loss::MSEScenLoss)(
    program::StochasticProgram,
    input_scenario_parameter,
    reference_scenario_parameters,
    mu;
    kwargs...,
) =
    error("MSE scenario loss has not been implemented yet.")
