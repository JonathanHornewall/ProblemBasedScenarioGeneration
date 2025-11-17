using ProblemBasedScenarioGeneration

const shipment_problem_data = ProblemBasedScenarioGeneration.shipment_planning_problem_data

const shipment_training_config = Dict(
    :Ntraining_samples => 100,
    :Ntesting_samples => 30,
    :N_xi_per_x => 100,
    :sigma => 5.0,
    :seasonal_scale => 12.0,
    :trend_decay => 2.0,
    :collections_per_sample => 1
)
