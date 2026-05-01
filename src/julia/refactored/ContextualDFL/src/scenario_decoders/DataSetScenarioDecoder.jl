struct DataSetScenarioDecoder{ChangingComponents,S<:DecoderStrategy,B<:BaseScenario} <: ScenarioDecoder
    decoder_strategy::S
    base_scenario::B
    changing_components::ChangingComponents
end

function (decoder::DataSetScenarioDecoder)(xi)
    return not_implemented(:DataSetScenarioDecoder)
end
