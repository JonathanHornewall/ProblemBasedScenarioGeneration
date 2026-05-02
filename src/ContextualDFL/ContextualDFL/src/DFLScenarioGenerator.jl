struct DFLScenarioGenerator{
    LEARNED_COMPONENTS,
    TScenarioDecoder<:ScenarioDecoder,
    TSolver<:Solver,
    TNeuralNet,
    TProgram<:StochasticProgram,
}
    NNScenarioDecoder::TScenarioDecoder
    Solver::TSolver
    neural_net::TNeuralNet
    program::TProgram
end

function DFLScenarioGenerator(
    nn_scenario_decoder::ScenarioDecoder,
    solver::Solver,
    neural_net,
    program::StochasticProgram,
    learned_components::Tuple{Vararg{Symbol}}=(),
)
    return DFLScenarioGenerator{
        learned_components,
        typeof(nn_scenario_decoder),
        typeof(solver),
        typeof(neural_net),
        typeof(program),
    }(nn_scenario_decoder, solver, neural_net, program)
end

DFLScenarioGenerator(;
    NNScenarioDecoder,
    Solver,
    neural_net,
    program,
    learned_components=(),
) = DFLScenarioGenerator(NNScenarioDecoder, Solver, neural_net, program, learned_components)

learned_components(::DFLScenarioGenerator{LEARNED_COMPONENTS}) where {LEARNED_COMPONENTS} =
    LEARNED_COMPONENTS

(generator::DFLScenarioGenerator)(context) =
    error("Scenario generation from context has not been implemented yet.")
