# Placeholder parameters for the shipment planning problem described by Homem-de-Mello et al. (2024).
# The article does not provide explicit numerical values, so the numbers below are illustrative and
# heavily commented to clarify their intended meaning.  Replace them with calibrated data when available.

# Planned transportation cost per market in thousands of dollars.
shipment_planning_first_stage_costs = [1.8, 2.1, 1.6, 2.4]

# Emergency shipping / lost-sales penalty per market (thousands of dollars per unit of unmet demand).
shipment_planning_recourse_penalties = [4.5, 5.0, 4.0, 5.5]

# Dimensionality of the contextual features (e.g., [fuel price index, demand forecast, port congestion]).
shipment_planning_context_dimension = 3

# Example of how to instantiate the problem once the module is loaded:
# using ProblemBasedScenarioGeneration
# data = ShipmentPlanningProblemData(shipment_planning_first_stage_costs,
#                                    shipment_planning_recourse_penalties;
#                                    context_dimension=shipment_planning_context_dimension)
# shipment_problem = ShipmentPlanningProblem(data)
