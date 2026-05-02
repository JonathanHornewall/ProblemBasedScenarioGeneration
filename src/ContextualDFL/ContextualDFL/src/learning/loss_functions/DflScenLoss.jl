struct DflScenLoss{
    TInputScenarioDecoder<:VectorDecoder,
    TReferenceScenarioDecoder<:ScenarioDecoder,
    TSolver<:Solver,
    TProgram<:StochasticProgram,
} <: LossFunction
    input_scenario_decoder::TInputScenarioDecoder
    reference_scenario_decoder::TReferenceScenarioDecoder
    solver::TSolver
    program::TProgram
    nr_scenarios::Int
end

function DflScenLoss(
    input_scenario_decoder::VectorDecoder,
    reference_scenario_decoder::ScenarioDecoder,
    solver::Solver,
    program::StochasticProgram;
    nr_scenarios::Integer=1,
)
    nr_scenarios > 0 || throw(ArgumentError("nr_scenarios must be a positive integer."))
    return DflScenLoss{
        typeof(input_scenario_decoder),
        typeof(reference_scenario_decoder),
        typeof(solver),
        typeof(program),
    }(
        input_scenario_decoder,
        reference_scenario_decoder,
        solver,
        program,
        Int(nr_scenarios),
    )
end

function (loss::DflScenLoss)(
    input_scenario_parameter_collection,
    reference_scenario_parameter_collection,
    mu_in=0,
    mu_ref=mu_in;
    probabilities=nothing,
    nr_scenarios=loss.nr_scenarios,
    kwargs...,
)
    W_eq, W_ineq, T_eq, T_ineq, h_eq, h_ineq, q =
        decode_scenario_collection(
            loss.input_scenario_decoder,
            input_scenario_parameter_collection,
            nr_scenarios=nr_scenarios,
        )
    z, _, _, _, _, _ = solve(
        loss.solver,
        loss.program,
        W_eq,
        W_ineq,
        T_eq,
        T_ineq,
        h_eq,
        h_ineq,
        q;
        probabilities=probabilities,
        μ=mu_in,
        kwargs...,
    )

    W_eq, W_ineq, T_eq, T_ineq, h_eq, h_ineq, q =
        decode_scenario_collection(
            loss.reference_scenario_decoder,
            reference_scenario_parameter_collection,
        )
    return cost_function(
        loss.program,
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
        μ=mu_ref,
        kwargs...,
    )
end
