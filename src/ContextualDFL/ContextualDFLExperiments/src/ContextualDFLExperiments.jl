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
    ResourceAllocationProblem,
    ResourceAllocationProblemData,
    ResourceAllocationScenarioDataGenerator,
    ScenarioDataGenerator,
    ScenarioGenerationPolicy,
    SampleAverageApproximationPolicy,
    base_scenario,
    default_resource_allocation_problem_data,
    default_knn_k,
    evaluate_policy,
    evaluate_policy_against_optimum,
    generate_contextual_data_set,
    generate_decision_set,
    infer,
    solve_dataset_to_optimality,
    summarize_regret,
    summarize_values,
    stochastic_program

include("data_generation/generate_contextual_data_set.jl")
include("data_generation/contextual_generators/ContextDataGenerator.jl")
include("data_generation/scenario_generators/ScenarioDataGenerator.jl")
include("program_instance/ProgramInstance.jl")
include("testing/policies/Policy.jl")
include("testing/policies/ScenarioGenerationPolicy.jl")
include("testing/policies/BaselinePolicies.jl")
include("testing/evaluation/evaluation.jl")

include("implementations/resource_allocation_problem/problem_data/parameters.jl")
include("implementations/resource_allocation_problem/program_instance/ResourceAllocationProblem.jl")
include("implementations/resource_allocation_problem/scenario_decoders/ResourceAllocationDemandDecoders.jl")
include("implementations/resource_allocation_problem/data_generators/ResourceAllocationContextDataGenerator.jl")
include("implementations/resource_allocation_problem/data_generators/ResourceAllocationScenarioDataGenerator.jl")

end
