struct DFLScenarioGenerator{LearnedComponents,D<:ScenarioDecoder,S,N,P<:StochasticProgram}
    scenario_decoder::D
    solver::S
    neural_net::N
    program::P
end

function (generator::DFLScenarioGenerator)(context)
    return not_implemented(:DFLScenarioGenerator)
end
