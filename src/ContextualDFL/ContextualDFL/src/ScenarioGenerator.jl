struct ScenarioGenerator{TNeuralNet,TDecoder<:VectorDecoder}
    neural_net::TNeuralNet
    scenario_decoder::TDecoder
end

ScenarioGenerator(; neural_net, scenario_decoder) =
    ScenarioGenerator(neural_net, scenario_decoder)

(generator::ScenarioGenerator)(context) =
    generator.scenario_decoder(generator.neural_net(context))
