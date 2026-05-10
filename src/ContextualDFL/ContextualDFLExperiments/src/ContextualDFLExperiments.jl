module ContextualDFLExperiments

import ChainRulesCore
import ContextualDFL

export ContextDataGenerator,
    AdaptiveDecisionTreePolicy,
    CARTPolicy,
    DecisionFocusedLinearPolicy,
    KNearestNeighborsPolicy,
    LeastSquaresPolicy,
    LexSPOLinearPolicy,
    M5ADPolicy,
    Policy,
    ProgramInstance,
    ResidualSampleAverageApproximationPolicy,
    ResourceAllocationContextDataGenerator,
    ResourceAllocationDemandParametricDecoder,
    ResourceAllocationDemandVectorDecoder,
    ResourceAllocationEconomicCostVectorDecoder,
    ResourceAllocationFullCostVectorDecoder,
    ResourceAllocationOriginalCostVectorDecoder,
    ResourceAllocationPhysicalCostVectorDecoder,
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
    convert_base_scenario_to_equality_form,
    convert_datapoint_to_equality_form,
    convert_dataset_to_equality_form,
    decode_q_conversion_arrays,
    full_base_scenario_arrays,
    convert_datapoint_to_q,
    convert_dataset_to_q,
    q_lower_bound_from_converted_dataset,
    LowerBoundedQDecoder,
    make_spoplus_q_loss,
    prepare_spoplus_q_dataset,
    convert_datapoint_to_h,
    convert_dataset_to_h,
    DecisionOptimalHDecoder,
    make_decision_h_loss,
    prepare_decision_h_dataset,
    prepare_decision_optimal_h_dataset,
    prepare_decision_optimal_dataset,
    convert_datapoint_to_decision_optimal,
    convert_dataset_to_decision_optimal,
    make_decision_equivalent_dataset,
    nonnegative_prediction_penalty_transform,
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
    UnreliableNewsvendorParameterVectorDecoder,
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
include("decision_optimal_q_conversion.jl")

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
