module ContextualDFLExperiments

import ContextualDFL

export ContextDataGenerator,
    Policy,
    ProgramInstance,
    ResourceAllocationContextDataGenerator,
    ResourceAllocationDemandDecoder,
    ResourceAllocationProblem,
    ResourceAllocationProblemData,
    ResourceAllocationScenarioDataGenerator,
    ScenarioDataGenerator,
    ScenarioGenerationPolicy,
    base_scenario,
    default_resource_allocation_problem_data,
    evaluate_policy,
    generate_contextual_data_set,
    generate_decision_set,
    infer,
    solve_dataset_to_optimality,
    stochastic_program

include("data_generation/generate_contextual_data_set.jl")
include("data_generation/contextual_generators/ContextDataGenerator.jl")
include("data_generation/scenario_generators/ScenarioDataGenerator.jl")
include("program_instance/ProgramInstance.jl")
include("testing/policies/Policy.jl")
include("testing/policies/ScenarioGenerationPolicy.jl")
include("testing/evaluation/evaluation.jl")

include("implementations/resource_allocation_problem/problem_data/parameters.jl")
include("implementations/resource_allocation_problem/program_instance/ResourceAllocationProblem.jl")
include("implementations/resource_allocation_problem/scenario_decoders/ResourceAllocationDemandDecoder.jl")
include("implementations/resource_allocation_problem/data_generators/ResourceAllocationContextDataGenerator.jl")
include("implementations/resource_allocation_problem/data_generators/ResourceAllocationScenarioDataGenerator.jl")

end
