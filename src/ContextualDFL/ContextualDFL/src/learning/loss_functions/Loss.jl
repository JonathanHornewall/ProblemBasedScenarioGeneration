abstract type LossFunction end

(loss::LossFunction)(
    input_scenario_parameter,
    reference_scenario_parameters,
    mu;
    kwargs...,
) =
    error("Loss evaluation is not defined for $(typeof(loss)).")
