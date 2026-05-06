module ContextualDFLExperiments

import ChainRulesCore
import ContextualDFL

export ContextDataGenerator,
    KNearestNeighborsPolicy,
    LeastSquaresPolicy,
    Policy,
    ProgramInstance,
    ResidualSampleAverageApproximationPolicy,
    ResourceAllocationContextDataGenerator,
    ResourceAllocationDemandParametricDecoder,
    ResourceAllocationDemandVectorDecoder,
    ResourceAllocationEconomicCostVectorDecoder,
    ResourceAllocationOriginalCostVectorDecoder,
    ResourceAllocationProblem,
    ResourceAllocationProblemData,
    ResourceAllocationScenarioDataGenerator,
    RandomYieldHVectorDecoder,
    RandomYieldParametricDecoder,
    RandomYieldPositiveQVectorDecoder,
    RandomYieldProblem,
    ScenarioDataGenerator,
    ScenarioGenerationPolicy,
    SampleAverageApproximationPolicy,
    ShipmentPlanningDemandVectorDecoder,
    ShipmentPlanningEconomicShippingCostVectorDecoder,
    ShipmentPlanningParametricDecoder,
    ShipmentPlanningPositiveDemandVectorDecoder,
    ShipmentPlanningPositiveShippingCostVectorDecoder,
    ShipmentPlanningProblem,
    base_scenario,
    default_resource_allocation_problem_data,
    default_knn_k,
    evaluate_policy,
    evaluate_policy_against_optimum,
    generate_benchmark_contexts,
    generate_benchmark_dataset,
    generate_benchmark_scenarios,
    generate_contextual_data_set,
    generate_decision_set,
    infer,
    random_yield_probabilities,
    random_yield_support_scenarios,
    sample_random_yield_scenario,
    solve_dataset_to_optimality,
    summarize_regret,
    summarize_values,
    stochastic_program,
    transshipment_decoder,
    TransShipmentComponentVectorDecoder,
    TransShipmentExperimentProblem,
    TransShipmentPositiveHQVectorDecoder,
    TransShipmentPositiveHVectorDecoder,
    TransShipmentPositiveQVectorDecoder,
    UnreliableNewsvendorParametricDecoder,
    UnreliableNewsvendorProblem,
    UnreliableNewsvendorProblemData,
    unreliable_newsvendor_scenario

include("data_generation/generate_contextual_data_set.jl")
include("data_generation/benchmark_dataset.jl")
include("data_generation/contextual_generators/ContextDataGenerator.jl")
include("data_generation/scenario_generators/ScenarioDataGenerator.jl")
include("program_instance/ProgramInstance.jl")
include("testing/policies/Policy.jl")
include("testing/policies/ScenarioGenerationPolicy.jl")
include("testing/policies/BaselinePolicies.jl")
include("testing/evaluation/evaluation.jl")

include("implementations/decoder_utils.jl")

include("implementations/resource_allocation_problem/problem_data/parameters.jl")
include("implementations/resource_allocation_problem/program_instance/ResourceAllocationProblem.jl")
include("implementations/resource_allocation_problem/scenario_decoders/ResourceAllocationDemandDecoders.jl")
include("implementations/resource_allocation_problem/scenario_decoders/ResourceAllocationCostDecoders.jl")
include("implementations/resource_allocation_problem/data_generators/ResourceAllocationContextDataGenerator.jl")
include("implementations/resource_allocation_problem/data_generators/ResourceAllocationScenarioDataGenerator.jl")

include("implementations/shipment_planning/ShipmentPlanningProblem.jl")
include("implementations/shipment_planning/ShipmentPlanningDataGenerators.jl")
include("implementations/shipment_planning/ShipmentPlanningDecoders.jl")

include("implementations/transshipment_problem/TransShipmentExperimentProblem.jl")
include("implementations/transshipment_problem/TransShipmentDataGenerators.jl")
include("implementations/transshipment_problem/TransShipmentDecoders.jl")

include("implementations/random_yield_problem/RandomYieldProblem.jl")
include("implementations/random_yield_problem/RandomYieldDataGenerators.jl")
include("implementations/random_yield_problem/RandomYieldDecoders.jl")

include("implementations/unreliable_newsvendor/UnreliableNewsvendorProblem.jl")
include("implementations/unreliable_newsvendor/UnreliableNewsvendorDataGenerators.jl")
include("implementations/unreliable_newsvendor/UnreliableNewsvendorDecoders.jl")

end
