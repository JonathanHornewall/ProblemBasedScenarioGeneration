abstract type ScenarioDecoder end

struct TrivialDecoder <: ScenarioDecoder end

function (decoder::ScenarioDecoder)(xi)
    return not_implemented(:ScenarioDecoder)
end

function (decoder::TrivialDecoder)(xi)
    return xi
end
