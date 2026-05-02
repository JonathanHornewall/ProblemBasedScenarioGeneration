struct ProjectedZLoss{
    TSolver<:Solver,
    TProgram<:StochasticProgram,
} <: LossFunction
    solver::TSolver
    program::TProgram
end

(loss::ProjectedZLoss)(
    program::StochasticProgram,
    input_scenario_parameter,
    reference_scenario_parameters,
    mu;
    kwargs...,
) =
    error("Projected-z loss has not been implemented yet.")
