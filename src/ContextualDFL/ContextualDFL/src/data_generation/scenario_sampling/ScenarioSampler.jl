abstract type ScenarioSampler end

(sampler::ScenarioSampler)(context; kwargs...) =
    error("Scenario sampling is not defined for $(typeof(sampler)).")

generate_scenario_set(sampler::ScenarioSampler, context_data, scenarios_per_context; kwargs...) =
    error("Scenario-set generation has not been implemented yet.")
