abstract type ScenarioDataGenerator end

(generator::ScenarioDataGenerator)(context) =
    error("Scenario data generation is not defined for $(typeof(generator)).")
