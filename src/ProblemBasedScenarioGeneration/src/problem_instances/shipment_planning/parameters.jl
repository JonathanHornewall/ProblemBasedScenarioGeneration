# Canonical parameter set for the shipment planning problem following
# Homem-de-Mello et al. (2024) and the supporting email from Tito.

const shipment_planning_production_cost = 5.0
const shipment_planning_emergency_cost = 100.0

const shipment_planning_distance_matrix = transpose([
    0.15     1.3124   1.85     1.3124;
    0.50026  0.93408  1.7874   1.6039;
    0.93408  0.50026  1.6039   1.7874;
    1.3124   0.15     1.3124   1.85;
    1.6039   0.50026  0.93408  1.7874;
    1.7874   0.93408  0.50026  1.6039;
    1.85     1.3124   0.15     1.3124;
    1.7874   1.6039   0.50026  0.93408;
    1.6039   1.7874   0.93408  0.50026;
    1.3124   1.85     1.3124   0.15;
    0.93408  1.7874   1.6039   0.50026;
    0.50026  1.6039   1.7874   0.93408
])

const shipment_planning_shipping_costs = 10 .* shipment_planning_distance_matrix
const shipment_planning_context_dimension = 3

const shipment_planning_problem_data = ShipmentPlanningProblemData(
    fill(shipment_planning_production_cost, size(shipment_planning_shipping_costs, 1)),
    fill(shipment_planning_emergency_cost, size(shipment_planning_shipping_costs, 1)),
    shipment_planning_shipping_costs;
    context_dimension = shipment_planning_context_dimension
)
