# Placeholder parameters for the bike sharing reallocation problem in Homem-de-Mello et al. (2024).
# Replace the illustrative values with empirical estimates when they become available.

# Cost of positioning one bike at each station during the nightly rebalancing cycle (in arbitrary units).
bike_sharing_first_stage_costs = [0.9, 1.1, 1.0, 1.2, 0.95]

# Cost of performing an emergency relocation trip per station.
bike_sharing_emergency_relocation_costs = [3.5, 3.8, 3.6, 3.9, 3.7]

# Penalty for unmet customer demand at each station.
bike_sharing_lost_demand_penalties = [6.0, 6.2, 6.5, 6.1, 6.4]

# Dimensionality of the contextual feature vector (e.g., [peak-hour indicator, precipitation, temperature, special-event flag]).
bike_sharing_context_dimension = 4

# Example instantiation (after loading the module):
# using ProblemBasedScenarioGeneration
# data = BikeSharingReallocationProblemData(bike_sharing_first_stage_costs,
#                                           bike_sharing_emergency_relocation_costs,
#                                           bike_sharing_lost_demand_penalties;
#                                           context_dimension=bike_sharing_context_dimension)
# bike_problem = BikeSharingReallocationProblem(data)
