struct DflScenLoss{
    TInputScenarioDecoder<:ScenarioDecoder,
    TReferenceScenarioDecoder<:ScenarioDecoder,
    TSolver<:Solver,
    TProgram<:StochasticProgram,
} <: LossFunction
    input_scenario_decoder::TInputScenarioDecoder
    reference_scenario_decoder::TReferenceScenarioDecoder
    solver::TSolver
    program::TProgram
end

function (loss::DflScenLoss)(
    program::StochasticProgram,
    input_scenario_parameter_collection,
    reference_scenario_parameter_collection,
    mu;
    probabilities=nothing,
    kwargs...,
)
    W_eq, W_ineq, T_eq, T_ineq, h_eq, h_ineq, q =
        decode_scenario_collection(
            loss.input_scenario_decoder,
            input_scenario_parameter_collection,
        )
    z, _, _, _, _, _ = solve(
        loss.solver,
        program,
        W_eq,
        W_ineq,
        T_eq,
        T_ineq,
        h_eq,
        h_ineq,
        q;
        probabilities=probabilities,
        μ=mu,
        kwargs...,
    )

    W_eq, W_ineq, T_eq, T_ineq, h_eq, h_ineq, q =
        decode_scenario_collection(
            loss.reference_scenario_decoder,
            reference_scenario_parameter_collection,
        )
    return cost_function(
        program,
        loss.solver,
        z,
        W_eq,
        W_ineq,
        T_eq,
        T_ineq,
        h_eq,
        h_ineq,
        q;
        probabilities=probabilities,
        μ=mu,
        kwargs...,
    )
end

(loss::DflScenLoss)(
    input_scenario_parameter_collection,
    reference_scenario_parameter_collection,
    mu;
    kwargs...,
) =
    loss(
        loss.program,
        input_scenario_parameter_collection,
        reference_scenario_parameter_collection,
        mu;
        kwargs...,
    )
