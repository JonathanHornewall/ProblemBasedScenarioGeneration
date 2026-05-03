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

const Nr_contexts = 150
const scenarios_per_context = 1
const collection_duplicates_per_context = 1
const validation_fraction = 0.13333333333333333
const test_fraction = 0.20

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
        Nr_contexts=Nr_contexts,
        scenarios_per_context=scenarios_per_context,
        collection_duplicates_per_context=collection_duplicates_per_context,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
        nr_scenarios=scenarios_per_context,
        solver=:highs,
    )
end

function problem_identity_config()
    return merge(
        problem_config(),
        (;
            demand_sigma=DEMAND_SIGMA,
            sigma=DEMAND_SIGMA,
            demand_power=DEMAND_POWER,
            context_terms=CONTEXT_TERMS,
            n_samples=Nr_contexts,
        ),
    )
end

function base_config(; overrides...)
    return merge(
        ContextualDFLTraining.DEFAULT_RUN_SETTINGS,
        problem_config(),
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

function generate_dataset(
    problem_instance,
    rng::Random.AbstractRNG;
    context_count::Integer=Nr_contexts,
    scenario_count::Integer=scenarios_per_context,
    duplicates_per_context::Integer=collection_duplicates_per_context,
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

function generated_training_splits(problem_instance, config, rng::Random.AbstractRNG)
    dataset = generate_dataset(
        problem_instance,
        rng;
        context_count=Int(config.Nr_contexts),
        scenario_count=Int(config.scenarios_per_context),
        duplicates_per_context=Int(config.collection_duplicates_per_context),
    )
    return ContextualDFLTraining.split_contextual_dataset(
        dataset;
        validation_fraction=config.validation_fraction,
        test_fraction=config.test_fraction,
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

function data_splits(problem_instance, config, rng::Random.AbstractRNG)
    generated_splits = generated_training_splits(problem_instance, config, rng)
    test_artifact = generated_test_artifact(config)
    return (;
        data=(;
            train=generated_splits.train,
            validation=generated_splits.validation,
            test=test_artifact === nothing ? generated_splits.test : test_artifact.dataset,
        ),
        test_data_artifact=test_artifact === nothing ?
            (; source=:generated_split) :
            test_artifact.metadata,
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

target_extractor(point) = Base.invokelatest(target_from_contextual_point, point)

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

function split_digest(splits)
    return serialized_digest((; train=splits.train, validation=splits.validation, test=splits.test))
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
    return (;
        generator="ContextualDFLExperiments.resource_allocation",
        dataset_name=EXPERIMENT_NAME,
        dataset_digest=split_digest(splits),
        dataset_source=join(
            (
                "ContextualDFLTraining.experiment",
                "experiment_id=$(EXPERIMENT_ID)",
                "seed=$(config.seed)",
                "Nr_contexts=$(config.Nr_contexts)",
                "scenarios_per_context=$(config.scenarios_per_context)",
                "collection_duplicates_per_context=$(config.collection_duplicates_per_context)",
            ),
            ";",
        ),
        train_size=length(splits.train),
        validation_size=length(splits.validation),
        test_size=length(splits.test),
        context_dimension=isempty(splits.train) ? 0 : length(first(splits.train).context),
        target_dimension=isempty(splits.train) ? 0 : length(target_from_contextual_point(first(splits.train))),
        Nr_contexts=Int(config.Nr_contexts),
        scenarios_per_context=Int(config.scenarios_per_context),
        collection_duplicates_per_context=Int(config.collection_duplicates_per_context),
        validation_fraction=Float64(config.validation_fraction),
        test_fraction=Float64(config.test_fraction),
        train_context_seed=Int(config.seed),
        train_scenario_seed=Int(config.seed),
        split_seed=Int(config.seed),
        test_data_artifact=get(
            Dict(pairs(ContextualDFLTraining.config_value(config, :test_data_artifact, NamedTuple()))),
            :path,
            "",
        ),
    )
end

function model_metadata(model, problem_instance, splits, config)
    return (;
        architecture="Flux.Chain",
        depth=Int(config.depth),
        width=Int(config.hidden_size),
        activation=string(ContextualDFLTraining.config_value(config, :activation, "relu")),
        output_activation="softplus",
        dropout=Float64(config.dropout),
        input_dimension=isempty(splits.train) ? 0 : length(first(splits.train).context),
        output_dimension=demand_count(problem_instance) * Int(config.nr_scenarios),
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
        scenario_count=Int(config.scenarios_per_context),
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
                scenarios_per_context=Int(config.scenarios_per_context),
            ),
        ),
    )
end

function training_objects(config)
    rng = Random.MersenneTwister(Int(config.seed))
    objects = problem_objects(config)
    splits = data_splits(objects.problem, config, rng)
    data = splits.data

    neural_net = ContextualDFLTraining.build_neural_net(
        length(first(data.train).context),
        demand_count(objects.problem) * Int(config.nr_scenarios);
        hidden_size=Int(config.hidden_size),
        depth=Int(config.depth),
        dropout=Float64(config.dropout),
    )
    generator = ContextualDFL.ScenarioGenerator(
        neural_net=neural_net,
        scenario_decoder=objects.scenario_decoder,
    )
    loss = ContextualDFLTraining.build_loss(
        config,
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
            target_extractor=target_extractor,
            test_data_artifact=splits.test_data_artifact,
            problem_metadata=problem_metadata(objects.problem),
            data_metadata=data_metadata(data, merge(config, (; test_data_artifact=splits.test_data_artifact))),
            model_metadata=model_metadata(neural_net, objects.problem, data, config),
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
