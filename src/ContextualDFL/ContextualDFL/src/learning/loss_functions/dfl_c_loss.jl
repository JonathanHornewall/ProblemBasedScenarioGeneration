struct DflCLoss{
    TSolver<:Solver,
    TProgram<:StochasticProgram,
    TMu,
} <: LossFunction
    solver::TSolver
    program::TProgram
    mu::TMu
end

(loss::DflCLoss)(
    input_scenario_parameter,
    reference_scenario_parameters,
    mu;
    kwargs...,
) =
    error("DFL cost loss has not been implemented yet.")
