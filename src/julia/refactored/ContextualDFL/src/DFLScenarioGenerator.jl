struct DFLScenarioGenerator{LearnedComponents,D<:ScenarioDecoder,S,N,P<:StochasticProgram}
    scenario_decoder::D
    solver::S
    neural_net::N
    program::P
end

function DFLScenarioGenerator(scenario_decoder::D, solver::S, neural_net::N, program::P) where {D<:ScenarioDecoder,S,N,P<:StochasticProgram}
    return DFLScenarioGenerator{Any,D,S,N,P}(scenario_decoder, solver, neural_net, program)
end

function (generator::DFLScenarioGenerator)(context)
    return generator.scenario_decoder(generator.neural_net(context))
end
