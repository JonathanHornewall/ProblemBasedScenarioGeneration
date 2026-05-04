import ContextualDFL
import ContextualDFLExperiments
import ContextualDFLTraining
import Random
import Serialization
import SHA

const EXPERIMENT_ID = "resource_allocation/experiment_1"
const EXPERIMENT_NAME = "resource_allocation_experiment_1"
const DEFAULT_TEST_DATA_SEED = 1
const DEFAULT_TEST_DATA_SET_SIZE = 30

const DEFAULT_TRAINING_CONTEXT_COUNT = 150
const DEFAULT_TRAINING_SCENARIOS_PER_CONTEXT = 1
const DEFAULT_COLLECTION_DUPLICATES_PER_CONTEXT = 1
const DEFAULT_VALIDATION_FRACTION = 0.13333333333333333
const DEFAULT_GENERATED_SPLIT_TEST_FRACTION = 0.20
const DEFAULT_TEST_SCENARIOS_PER_CONTEXT = 100
const DEFAULT_TRAINING_DATA_SEED = 1

const DEMAND_SIGMA = 5.0
const DEMAND_POWER = 2.0
const CONTEXT_TERMS = 3

experiment_id() = EXPERIMENT_ID
experiment_name() = EXPERIMENT_NAME
experiment_module_name() = :ResourceAllocationExperiment1
artifact_dir() = joinpath(@__DIR__, "artifacts")
test_data_dir() = joinpath(artifact_dir(), "test_data")
test_data_path(seed::Integer=DEFAULT_TEST_DATA_SEED) =
    joinpath(test_data_dir(), "test_data_seed$(Int(seed)).jls")
test_optimal_results_path(seed::Integer=DEFAULT_TEST_DATA_SEED) =
    joinpath(test_data_dir(), "optimal_solutions_seed$(Int(seed)).jls")

function experiment_overrides(; overrides...)
    return merge(
        (;
            experiment_id=EXPERIMENT_ID,
            experiment_name=EXPERIMENT_NAME,
            problem=:resource_allocation,
            mlflow_dataset_name=EXPERIMENT_NAME,
        ),
        NamedTuple(overrides),
    )
end

function problem_config()
    return (;
        problem=:resource_allocation,
        solver=:highs,
    )
end

function problem_identity_config()
    return (;
        problem=:resource_allocation,
        demand_sigma=DEMAND_SIGMA,
        sigma=DEMAND_SIGMA,
        demand_power=DEMAND_POWER,
        context_terms=CONTEXT_TERMS,
    )
end

function training_data_defaults()
    return (;
        training_context_count=DEFAULT_TRAINING_CONTEXT_COUNT,
        training_scenarios_per_context=DEFAULT_TRAINING_SCENARIOS_PER_CONTEXT,
        collection_duplicates_per_context=DEFAULT_COLLECTION_DUPLICATES_PER_CONTEXT,
        validation_fraction=DEFAULT_VALIDATION_FRACTION,
        generated_split_test_fraction=DEFAULT_GENERATED_SPLIT_TEST_FRACTION,
        training_data_seed=DEFAULT_TRAINING_DATA_SEED,
    )
end

function base_config(; overrides...)
    return merge(
        ContextualDFLTraining.DEFAULT_RUN_SETTINGS,
        problem_config(),
        training_data_defaults(),
        experiment_overrides(; overrides...),
    )
end

function test_data_config(;
    seed=DEFAULT_TEST_DATA_SEED,
    data_set_size=DEFAULT_TEST_DATA_SET_SIZE,
    overrides...,
)
    return base_config(;
        seed=Int(seed),
        test_data_seed=Int(seed),
        data_set_size=Int(data_set_size),
        overrides...,
    )
end

function profile_config(; overrides...)
    return merge(
        base_config(;
            epochs=100,
            warmup_epochs=2,
            mu=1.0,
            mu_start=1.0,
            mu_end=1.0,
            mu_schedule=:constant,
            learning_rate=1e-3,
            hidden_size=128,
            depth=2,
            batch_size=64,
            dropout=0.0,
            seed=3,
            run_id="profile_standard_seed3",
            base_run_id="profile_standard_seed3",
            candidate_name="profile_standard_seed3",
            method_variant="profiling",
            overrides...,
        ),
        (;
            mlflow_dataset_context="profiling",
            mlflow_source_name="ContextualDFLTraining/profile_training.jl",
            mlflow_params=(;
                profile_target="ContextualDFL.train!",
                profile_loss="ContextualDFL.DflScenLoss",
                profile_progress_logged_by="remote_worker",
            ),
        ),
    )
end

problem_data() = ContextualDFLExperiments.default_resource_allocation_problem_data()

ResourceAllocationProblem(
    data::ContextualDFLExperiments.ResourceAllocationProblemData=problem_data(),
) = ContextualDFLExperiments.ResourceAllocationProblem(data)

problem() = ResourceAllocationProblem(problem_data())

program(problem_instance=problem()) = ContextualDFLExperiments.stochastic_program(problem_instance)

solver(config=base_config()) = ContextualDFLTraining.build_solver(config)

scenario_decoder(problem_instance=problem()) =
    ContextualDFLExperiments.ResourceAllocationDemandVectorDecoder(problem_instance)

reference_scenario_decoder(problem_instance=problem()) =
    ContextualDFLExperiments.ResourceAllocationDemandParametricDecoder(problem_instance)

ContextDataGenerator(; rng::Random.AbstractRNG=Random.default_rng()) =
    ContextualDFLExperiments.ResourceAllocationContextDataGenerator(rng=rng)

function ScenarioDataGenerator(
    problem_instance=problem();
    rng::Random.AbstractRNG=Random.default_rng(),
)
    return ContextualDFLExperiments.ResourceAllocationScenarioDataGenerator(
        problem_instance;
        sigma=DEMAND_SIGMA,
        p=DEMAND_POWER,
        L=CONTEXT_TERMS,
        rng=rng,
    )
end

function problem_objects(config=base_config())
    problem_instance = problem()
    return (;
        problem=problem_instance,
        program=program(problem_instance),
        solver=solver(config),
        scenario_decoder=scenario_decoder(problem_instance),
        reference_scenario_decoder=reference_scenario_decoder(problem_instance),
    )
end

function demand_count(problem_instance)
    return size(problem_instance.problem_data.service_rate_parameters, 2)
end

function resource_count(problem_instance)
    return size(problem_instance.problem_data.service_rate_parameters, 1)
end

function training_context_count(config)
    return Int(
        ContextualDFLTraining.config_value(
            config,
            :training_context_count,
            ContextualDFLTraining.config_value(
                config,
                :Nr_contexts,
                DEFAULT_TRAINING_CONTEXT_COUNT,
            ),
        ),
    )
end

function training_scenarios_per_context(config)
    return Int(
        ContextualDFLTraining.config_value(
            config,
            :training_scenarios_per_context,
            ContextualDFLTraining.config_value(
                config,
                :scenarios_per_context,
                ContextualDFLTraining.config_value(
                    config,
                    :nr_scenarios,
                    DEFAULT_TRAINING_SCENARIOS_PER_CONTEXT,
                ),
            ),
        ),
    )
end

function collection_duplicates_count(config)
    return Int(
        ContextualDFLTraining.config_value(
            config,
            :collection_duplicates_per_context,
            DEFAULT_COLLECTION_DUPLICATES_PER_CONTEXT,
        ),
    )
end

function validation_split_fraction(config)
    return Float64(
        ContextualDFLTraining.config_value(
            config,
            :validation_fraction,
            DEFAULT_VALIDATION_FRACTION,
        ),
    )
end

function generated_split_test_fraction(config)
    return Float64(
        ContextualDFLTraining.config_value(
            config,
            :generated_split_test_fraction,
            ContextualDFLTraining.config_value(
                config,
                :test_fraction,
                DEFAULT_GENERATED_SPLIT_TEST_FRACTION,
            ),
        ),
    )
end

function test_scenarios_per_context(config)
    return Int(
        ContextualDFLTraining.config_value(
            config,
            :test_scenarios_per_context,
            ContextualDFLTraining.config_value(
                config,
                :scenarios_per_context,
                ContextualDFLTraining.config_value(
                    config,
                    :nr_scenarios,
                    DEFAULT_TEST_SCENARIOS_PER_CONTEXT,
                ),
            ),
        ),
    )
end

function dataset_scenarios_per_context(dataset, config)
    isempty(dataset) && return training_scenarios_per_context(config)
    return length(first(dataset).scenario_parameters)
end

function generate_dataset(
    problem_instance,
    rng::Random.AbstractRNG;
    context_count::Integer=DEFAULT_TRAINING_CONTEXT_COUNT,
    scenario_count::Integer=DEFAULT_TRAINING_SCENARIOS_PER_CONTEXT,
    duplicates_per_context::Integer=DEFAULT_COLLECTION_DUPLICATES_PER_CONTEXT,
)
    context_count > 0 || throw(ArgumentError("context_count must be positive."))
    scenario_count > 0 || throw(ArgumentError("scenario_count must be positive."))
    duplicates_per_context > 0 ||
        throw(ArgumentError("duplicates_per_context must be positive."))

    context_generator = ContextDataGenerator(rng=rng)
    scenario_generator = ScenarioDataGenerator(problem_instance; rng=rng)

    contexts = Vector{Vector{Float64}}()
    scenario_collections = Vector{Vector{ContextualDFL.ParametricScenario}}()

    for _ in 1:Int(context_count)
        context = Vector{Float64}(context_generator())
        for _ in 1:Int(duplicates_per_context)
            push!(contexts, copy(context))
            push!(
                scenario_collections,
                [scenario_generator(context) for _ in 1:Int(scenario_count)],
            )
        end
    end

    return ContextualDFLExperiments.generate_contextual_data_set(
        contexts,
        scenario_collections,
    )
end

function generated_training_splits(
    problem_instance,
    config,
    rng::Random.AbstractRNG;
    test_fraction::Real=0.0,
)
    dataset = generate_dataset(
        problem_instance,
        rng;
        context_count=training_context_count(config),
        scenario_count=training_scenarios_per_context(config),
        duplicates_per_context=collection_duplicates_count(config),
    )
    return ContextualDFLTraining.split_contextual_dataset(
        dataset;
        validation_fraction=validation_split_fraction(config),
        test_fraction=Float64(test_fraction),
        rng=rng,
    )
end

function generated_test_artifact(config)
    Bool(ContextualDFLTraining.config_value(config, :use_generated_test_data_artifact, true)) ||
        return nothing
    spec = ContextualDFLTraining.experiment_from_config(config)
    spec === nothing && return nothing
    isempty(ContextualDFLTraining.test_artifact_paths(spec, "test_data")) && return nothing
    return ContextualDFLTraining.load_test_data_artifact(spec)
end

function test_data_artifact_metadata(config)
    artifact = generated_test_artifact(config)
    return artifact === nothing ? (; source=:generated_split) : artifact.metadata
end

function data_splits(problem_instance, config, rng::Random.AbstractRNG)
    test_artifact = generated_test_artifact(config)
    split_test_fraction =
        test_artifact === nothing ? generated_split_test_fraction(config) : 0.0
    generated_splits = generated_training_splits(
        problem_instance,
        config,
        rng;
        test_fraction=split_test_fraction,
    )
    return (;
        train=generated_splits.train,
        validation=generated_splits.validation,
        test=test_artifact === nothing ? generated_splits.test : test_artifact.dataset,
    )
end

function training_data_seed(config)
    return Int(
        ContextualDFLTraining.config_value(
            config,
            :training_data_seed,
            DEFAULT_TRAINING_DATA_SEED,
        ),
    )
end

function training_data(config)
    rng = Random.MersenneTwister(training_data_seed(config))
    return data_splits(problem(), config, rng)
end

function training_dataset_name(config)
    explicit_name =
        ContextualDFLTraining.config_value(config, :training_dataset_name, nothing)
    isnothing(explicit_name) || return string(explicit_name)
    return join(
        (
            EXPERIMENT_NAME,
            "ctx$(training_context_count(config))",
            "scen$(training_scenarios_per_context(config))",
            "dup$(collection_duplicates_count(config))",
            "training_seed$(training_data_seed(config))",
        ),
        "-",
    )
end

function demand_from_scenario(scenario::ContextualDFL.ParametricScenario)
    isempty(scenario.h_eq_xi) &&
        throw(ArgumentError("expected a demand vector in h_eq_xi"))
    return scenario.h_eq_xi
end

function target_from_contextual_point(point)
    isempty(point.scenario_parameters) &&
        throw(ArgumentError("contextual data point has no scenario parameters."))
    return reduce(vcat, (demand_from_scenario(scenario) for scenario in point.scenario_parameters))
end

function problem_instance_id(problem_instance)
    values = (
        "service_rate=$(vec(problem_instance.problem_data.service_rate_parameters))",
        "first_stage=$(problem_instance.problem_data.first_stage_costs)",
        "second_stage=$(problem_instance.problem_data.second_stage_costs)",
        "yield=$(problem_instance.problem_data.yield_parameters)",
        "demand_sigma=$(DEMAND_SIGMA)",
        "demand_power=$(DEMAND_POWER)",
        "context_terms=$(CONTEXT_TERMS)",
    )
    return "sha256:" * SHA.bytes2hex(SHA.sha256(join(values, "\n")))
end

function serialized_digest(value)
    io = IOBuffer()
    Serialization.serialize(io, value)
    return "sha1:" * SHA.bytes2hex(SHA.sha1(take!(io)))
end

function test_artifact_identity(config)
    artifact = generated_test_artifact(config)
    artifact === nothing && return (; source=:generated_split)
    return (;
        source=:artifact,
        path=artifact.metadata.path,
        seed=artifact.metadata.seed,
        data_set_size=artifact.metadata.data_set_size,
        dataset_digest=artifact.metadata.dataset_digest,
    )
end

function training_data_identity(config)
    test_artifact = test_artifact_identity(config)
    identity = (;
        experiment_id=EXPERIMENT_ID,
        problem_instance_id=problem_instance_id(problem()),
        training_data_seed=training_data_seed(config),
        training_context_count=training_context_count(config),
        training_scenarios_per_context=training_scenarios_per_context(config),
        collection_duplicates_per_context=collection_duplicates_count(config),
        validation_fraction=validation_split_fraction(config),
        demand_sigma=DEMAND_SIGMA,
        demand_power=DEMAND_POWER,
        context_terms=CONTEXT_TERMS,
        test_artifact=test_artifact,
    )
    return test_artifact.source == :generated_split ?
        merge(
            identity,
            (; generated_split_test_fraction=generated_split_test_fraction(config)),
        ) :
        identity
end

function problem_metadata(problem_instance)
    return (;
        problem="resource_allocation",
        instance_id=problem_instance_id(problem_instance),
        resource_count=resource_count(problem_instance),
        demand_count=demand_count(problem_instance),
        service_rate_shape=size(problem_instance.problem_data.service_rate_parameters),
        demand_sigma=DEMAND_SIGMA,
        demand_power=DEMAND_POWER,
        context_terms=CONTEXT_TERMS,
    )
end

function data_metadata(splits, config)
    test_artifact =
        ContextualDFLTraining.config_value(config, :test_data_artifact, NamedTuple())
    test_source = hasproperty(test_artifact, :path) ? :artifact : :generated_split
    dataset_recipe = join(
        (
            "ContextualDFLTraining.experiment",
            "experiment_id=$(EXPERIMENT_ID)",
            "training_data_seed=$(training_data_seed(config))",
            "training_context_count=$(training_context_count(config))",
            "training_scenarios_per_context=$(training_scenarios_per_context(config))",
            "collection_duplicates_per_context=$(collection_duplicates_count(config))",
            "validation_fraction=$(validation_split_fraction(config))",
            "test_source=$(test_source)",
        ),
        ";",
    )

    return (;
        generator="ContextualDFLExperiments.resource_allocation",
        dataset_recipe=dataset_recipe,
        train_size=length(splits.train),
        validation_size=length(splits.validation),
        test_size=length(splits.test),
        context_dimension=isempty(splits.train) ? 0 : length(first(splits.train).context),
        target_dimension=isempty(splits.train) ? 0 : length(target_from_contextual_point(first(splits.train))),
        training_context_count=training_context_count(config),
        training_scenarios_per_context=training_scenarios_per_context(config),
        collection_duplicates_per_context=collection_duplicates_count(config),
        validation_fraction=validation_split_fraction(config),
        generated_split_test_fraction=
            test_source == :generated_split ? generated_split_test_fraction(config) : 0.0,
        test_source=test_source,
        train_context_seed=training_data_seed(config),
        train_scenario_seed=training_data_seed(config),
        split_seed=training_data_seed(config),
        test_data_artifact=get(Dict(pairs(test_artifact)), :path, ""),
    )
end

function model_metadata(model, problem_instance, splits, config)
    scenario_count = dataset_scenarios_per_context(splits.train, config)
    return (;
        architecture="Flux.Chain",
        depth=Int(config.depth),
        width=Int(config.hidden_size),
        activation=string(ContextualDFLTraining.config_value(config, :activation, "relu")),
        output_activation="softplus",
        dropout=Float64(config.dropout),
        initialization_seed=Int(config.seed),
        input_dimension=isempty(splits.train) ? 0 : length(first(splits.train).context),
        output_dimension=demand_count(problem_instance) * scenario_count,
    )
end

function test_data_bundle(
    config;
    seed=DEFAULT_TEST_DATA_SEED,
    data_set_size=DEFAULT_TEST_DATA_SET_SIZE,
)
    data_set_size > 0 || throw(ArgumentError("data_set_size must be positive."))
    objects = problem_objects(config)
    rng = Random.MersenneTwister(Int(seed))
    dataset = generate_dataset(
        objects.problem,
        rng;
        context_count=Int(data_set_size),
        scenario_count=test_scenarios_per_context(config),
        duplicates_per_context=1,
    )
    return merge(
        objects,
        (;
            dataset=dataset,
            problem_metadata=problem_metadata(objects.problem),
            data_metadata=(;
                generator="ContextualDFLExperiments.resource_allocation",
                dataset_name=EXPERIMENT_NAME,
                dataset_digest=serialized_digest(dataset),
                dataset_source=join(
                    (
                        "ContextualDFLTraining.experiment_test_data",
                        "experiment_id=$(EXPERIMENT_ID)",
                        "seed=$(Int(seed))",
                        "data_set_size=$(Int(data_set_size))",
                    ),
                    ";",
                ),
                data_set_size=Int(data_set_size),
                context_dimension=isempty(dataset) ? 0 : length(first(dataset).context),
                target_dimension=isempty(dataset) ? 0 : length(target_from_contextual_point(first(dataset))),
                scenarios_per_context=test_scenarios_per_context(config),
            ),
        ),
    )
end

training_objects(config) = training_objects(config, training_data(config))

function training_objects(config, data)
    objects = problem_objects(config)
    test_data_artifact = test_data_artifact_metadata(config)
    scenario_count = dataset_scenarios_per_context(data.train, config)
    model_initialization_seed = Int(config.seed)
    effective_config = merge(
        config,
        (; nr_scenarios=scenario_count, model_initialization_seed=model_initialization_seed),
    )

    neural_net = ContextualDFLTraining.build_neural_net(
        length(first(data.train).context),
        demand_count(objects.problem) * scenario_count;
        hidden_size=Int(config.hidden_size),
        depth=Int(config.depth),
        dropout=Float64(config.dropout),
        activation=ContextualDFLTraining.config_value(config, :activation, :relu),
        seed=model_initialization_seed,
    )
    generator = ContextualDFL.ScenarioGenerator(
        neural_net=neural_net,
        scenario_decoder=objects.scenario_decoder,
    )
    loss = ContextualDFLTraining.build_loss(
        effective_config,
        objects.scenario_decoder,
        objects.reference_scenario_decoder,
        objects.solver,
        objects.program,
    )

    return merge(
        objects,
        (;
            loss=loss,
            scenario_generator=generator,
            data=data,
            target_extractor=ContextualDFLTraining.LatestExperimentFunction(
                @__MODULE__,
                :target_from_contextual_point,
            ),
            test_data_artifact=test_data_artifact,
            problem_metadata=problem_metadata(objects.problem),
            data_metadata=data_metadata(
                data,
                merge(config, (; test_data_artifact=test_data_artifact)),
            ),
            model_metadata=model_metadata(neural_net, objects.problem, data, effective_config),
            schedules=(;
                mu=ContextualDFLTraining.ConstantSchedule(config.mu),
                rho=ContextualDFLTraining.ConstantSchedule(config.rho),
                batch_size=ContextualDFLTraining.ConstantSchedule(config.batch_size),
                step_size=ContextualDFLTraining.ConstantSchedule(config.learning_rate),
            ),
        ),
    )
end

function optimality_splits(objects, config)
    return ContextualDFLTraining.optimality_evaluation_datasets(objects, config)
end

function optimal_results_path(split_name::Symbol)
    return joinpath(artifact_dir(), "optimal_solutions", string(split_name) * ".jls")
end
