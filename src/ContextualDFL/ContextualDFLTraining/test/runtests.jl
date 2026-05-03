using ContextualDFLTraining
using ContextualDFL
using ContextualDFLExperiments
using Test

mutable struct FakeRun
    params::Vector{Tuple{String,String}}
    metrics::Vector{Tuple{String,Float64,Int}}
    tags::Vector{Tuple{String,String}}
    inputs::Vector{Any}
    artifacts::Vector{Tuple{String,Vector{UInt8}}}
    events::Vector{Symbol}
end

@testset "ContextualDFLTraining experiments" begin
    spec = ContextualDFLTraining.load_experiment("ResourceAllocationExperiment1")
    @test spec.id == "resource_allocation/experiment_1"
    @test spec.name == "resource_allocation_experiment_1"
    @test ContextualDFLTraining.experiment_base_config(spec).experiment_id == spec.id
    @test isabspath(ContextualDFLTraining.optimal_results_path(spec, :test))
    @test !ContextualDFLTraining.experiment_has_function(spec, :grid_configs)
    @test !ContextualDFLTraining.experiment_has_function(spec, :smoke_configs)
    @test !isdefined(ContextualDFLTraining, :experiment_grid_configs)
    @test !isdefined(ContextualDFLTraining, :experiment_smoke_configs)
    @test !isdefined(ContextualDFLTraining, :experiment_problem_identity)
    @test !isdefined(ContextualDFLTraining, :resource_allocation_training_objects)
    @test !isdefined(ContextualDFLTraining, :resource_allocation_test_data_bundle)
    @test_throws ArgumentError ContextualDFLTraining.training_objects_for_config((; seed=1))

    config = merge(
        ContextualDFLTraining.experiment_base_config(spec),
        (; optimality_evaluation=false, use_generated_test_data_artifact=false),
    )
    @test hasproperty(config, :Nr_contexts)
    @test hasproperty(config, :scenarios_per_context)
    @test hasproperty(config, :collection_duplicates_per_context)
    @test !hasproperty(config, :n_samples)
    @test !hasproperty(config, :sigma)
    @test !hasproperty(config, :demand_power)
    @test !hasproperty(config, :context_terms)

    objects = ContextualDFLTraining.training_objects_for_config(config)
    @test objects.problem isa ResourceAllocationProblem
    @test objects.program isa ContextualDFL.StochasticProgram
    @test objects.solver isa ContextualDFL.Solver
    @test objects.scenario_decoder isa ResourceAllocationDemandVectorDecoder
    @test objects.reference_scenario_decoder isa ResourceAllocationDemandParametricDecoder
    @test objects.loss isa ContextualDFL.DflScenLoss
    @test hasproperty(objects, :target_extractor)
    @test objects.problem_metadata.problem == "resource_allocation"
    @test objects.data_metadata.Nr_contexts == config.Nr_contexts
    @test length(objects.data.train) == 100
    @test length(objects.data.validation) == 20
    @test length(objects.data.test) == 30

    mktempdir() do dir
        config_dir = joinpath(dir, "toy")
        mkpath(config_dir)
        config_path = joinpath(config_dir, "Config.jl")
        write(
            config_path,
            """
            import ContextualDFLTraining

            experiment_id() = "toy/experiment"
            experiment_name() = "toy_experiment"
            experiment_module_name() = :ToyExperiment
            artifact_dir() = joinpath(@__DIR__, "artifacts")
            base_config() = (; experiment_id=experiment_id())
            training_objects(config) = nothing
            optimality_splits(objects, config) = Pair{Symbol,Any}[]
            optimal_results_path(split_name::Symbol) =
                joinpath(artifact_dir(), string(split_name) * ".jls")
            """,
        )

        toy_spec = ContextualDFLTraining.load_experiment(config_path)
        dataset = [(; context=[1.0], scenario_parameters=[2.0])]
        results = [(; objective_value=3.0)]
        path = ContextualDFLTraining.save_optimal_results!(
            toy_spec,
            :test,
            results;
            dataset=dataset,
        )

        @test isfile(path)
        @test ContextualDFLTraining.load_optimal_results(
            toy_spec,
            :test;
            dataset=dataset,
        ) == results
        @test_throws ArgumentError ContextualDFLTraining.load_optimal_results(
            toy_spec,
            :train;
            dataset=dataset,
        )
    end
end

@testset "ContextualDFLTraining generated test data" begin
    script_module = Module(:GenerateTestDataScriptTest)
    Base.include(
        script_module,
        joinpath(dirname(dirname(pathof(ContextualDFLTraining))), "generate_test_data.jl"),
    )
    parsed = getfield(script_module, :parse_commandline)(
        ["--experiment", "resource_allocation/experiment_1"],
    )
    @test parsed["seed"] == 1
    @test parsed["data-set-size"] == 30

    resource_spec = ContextualDFLTraining.load_experiment("resource_allocation/experiment_1")
    resource_bundle = ContextualDFLTraining.experiment_test_data_bundle(
        resource_spec;
        seed=5,
        data_set_size=2,
    )
    @test length(resource_bundle.dataset) == 2
    @test resource_bundle.problem isa ResourceAllocationProblem
    @test resource_bundle.data_metadata.data_set_size == 2

    mktempdir() do dir
        config_dir = joinpath(dir, "toy_generated")
        mkpath(config_dir)
        config_path = joinpath(config_dir, "Config.jl")
        write(
            config_path,
            """
            experiment_id() = "toy/generated"
            experiment_name() = "toy_generated"
            experiment_module_name() = :ToyGeneratedExperiment
            artifact_dir() = joinpath(@__DIR__, "artifacts")
            test_data_dir() = joinpath(artifact_dir(), "test_data")
            test_data_path(seed::Integer) =
                joinpath(test_data_dir(), "test_data_seed\$(Int(seed)).jls")
            test_optimal_results_path(seed::Integer) =
                joinpath(test_data_dir(), "optimal_solutions_seed\$(Int(seed)).jls")
            base_config() = (; experiment_id=experiment_id())
            training_objects(config) = nothing
            optimality_splits(objects, config) = Pair{Symbol,Any}[]
            optimal_results_path(split_name::Symbol) =
                joinpath(artifact_dir(), "legacy", string(split_name) * ".jls")
            """,
        )

        spec = ContextualDFLTraining.load_experiment(config_path)
        dataset = [
            (; context=[Float64(index)], scenario_parameters=[Float64(index + 1)]) for
            index in 1:3
        ]
        results = [(; objective_value=Float64(index)) for index in 1:3]

        data_path = ContextualDFLTraining.save_test_data!(
            spec,
            7,
            dataset;
            data_set_size=3,
        )
        optimal_path = ContextualDFLTraining.save_test_optimal_results!(
            spec,
            7,
            results;
            dataset=dataset,
            data_set_size=3,
        )

        @test basename(data_path) == "test_data_seed7.jls"
        @test basename(optimal_path) == "optimal_solutions_seed7.jls"
        artifact = ContextualDFLTraining.load_test_data_artifact(spec)
        @test artifact.dataset == dataset
        @test artifact.metadata.seed == 7
        @test artifact.metadata.data_set_size == 3
        @test ContextualDFLTraining.load_test_data(spec) == dataset
        @test ContextualDFLTraining.load_optimal_results(spec, :test; dataset=dataset) ==
              results
        @test_throws ArgumentError ContextualDFLTraining.load_optimal_results(
            spec,
            :test;
            dataset=dataset[1:2],
        )

        cp(data_path, ContextualDFLTraining.test_data_path(spec, 8))
        @test_throws ArgumentError ContextualDFLTraining.load_test_data(spec)
    end
end

@testset "ContextualDFLTraining grid file config" begin
    gridsearch_module = Module(:GridSearchScriptTest)
    Core.eval(gridsearch_module, :(using Base))
    Core.eval(gridsearch_module, :(include(path) = Base.include($gridsearch_module, path)))
    Base.include(
        gridsearch_module,
        joinpath(dirname(dirname(pathof(ContextualDFLTraining))), "gridsearch.jl"),
    )
    grid_load_experiment = getfield(gridsearch_module, :load_experiment)
    grid_load_grid_config = getfield(gridsearch_module, :load_grid_config)
    selected_grid = getfield(gridsearch_module, :selected_grid)

    experiment = grid_load_experiment("resource_allocation/experiment_1")

    @testset "bundled resource allocation configs" begin
        default_spec = grid_load_grid_config(
            joinpath(experiment.root_dir, "grid_configs", "default.yaml"),
        )
        smoke_spec = grid_load_grid_config(
            joinpath(experiment.root_dir, "grid_configs", "smoke.yaml"),
        )

        default_configs = selected_grid(experiment, default_spec)
        smoke_configs = selected_grid(experiment, smoke_spec)
        @test length(default_configs) == 24
        @test length(smoke_configs) == 1
        for config in vcat(default_configs, smoke_configs)
            @test !hasproperty(config, :n_samples)
            @test !hasproperty(config, :sigma)
            @test !hasproperty(config, :demand_power)
            @test !hasproperty(config, :context_terms)
            @test hasproperty(config, :Nr_contexts)
            @test hasproperty(config, :scenarios_per_context)
            @test hasproperty(config, :collection_duplicates_per_context)
        end
    end

    mktempdir() do dir
        yaml_path = joinpath(dir, "grid.yaml")
        write(
            yaml_path,
            """
            version: 1
            name: yaml_grid
            base:
              epochs: 3
              optimality_evaluation: false
            fixed:
              depth: 1
              batch_size: 4
              dropout: 0.0
            grid:
              learning_rate: [0.001, 0.0005]
              hidden_size: [16, 32]
              seed: [1]
            schedules:
              mu:
                kind: geometric
                start: 1.0
                stop: 0.01
              mu_ref:
                kind: match_input
            run_id_template: "{name}_{index}_{hash}"
            """,
        )

        spec = grid_load_grid_config(yaml_path)
        configs = selected_grid(experiment, spec)
        resolved_json = ContextualDFLTraining.resolved_grid_json(configs)
        digest = ContextualDFLTraining.grid_config_digest(configs)

        @test spec.format == :yaml
        @test length(configs) == 4
        @test all(config -> config.experiment_id == experiment.id, configs)
        @test all(config -> config.optimality_evaluation == false, configs)
        @test Set(config.learning_rate for config in configs) == Set([0.001, 0.0005])
        @test Set(config.hidden_size for config in configs) == Set([16, 32])
        @test all(config -> config.mu_schedule == :geometric, configs)
        @test all(config -> config.mu_start == 1.0, configs)
        @test all(config -> config.mu_end == 0.01, configs)
        @test all(config -> config.mu_ref_schedule == :match_input, configs)
        @test all(config -> startswith(config.run_id, "yaml_grid_"), configs)
        @test startswith(digest, "sha256:")
        @test all(config -> config.grid_config_digest == digest, configs)
        @test occursin("\"grid_config_name\"", resolved_json)
        @test !occursin("grid_config_digest", resolved_json)

        write(
            yaml_path,
            """

            version: 1
            name: yaml_grid
            base:
              epochs: 3
              optimality_evaluation: false
            fixed:
              depth: 1
              batch_size: 4
              dropout: 0.0
            grid:
              learning_rate: [0.001, 0.0005]
              hidden_size: [16, 32]
              seed: [1]
            schedules:
              mu:
                kind: geometric
                start: 1.0
                stop: 0.01
              mu_ref:
                kind: match_input
            run_id_template: "{name}_{index}_{hash}"

            """,
        )
        blank_line_configs = selected_grid(
            experiment,
            grid_load_grid_config(yaml_path),
        )
        @test ContextualDFLTraining.resolved_grid_json(blank_line_configs) == resolved_json
        @test ContextualDFLTraining.grid_config_digest(blank_line_configs) == digest
    end

    mktempdir() do dir
        for problem_key in (
            "Nr_contexts",
            "nr_scenarios",
            "solver",
            "n_samples",
            "sigma",
            "demand_power",
            "context_terms",
        )
            invalid_problem_key_path = joinpath(dir, "problem_key_$(problem_key).yaml")
            write(
                invalid_problem_key_path,
                """
                version: 1
                name: invalid_problem_key
                fixed:
                  learning_rate: 0.001
                  hidden_size: 16
                  depth: 1
                  batch_size: 4
                  dropout: 0.0
                  $(problem_key): 16
                grid:
                  seed: [1]
                """,
            )

            @test_throws ArgumentError selected_grid(
                experiment,
                grid_load_grid_config(invalid_problem_key_path),
            )
        end
    end

    mktempdir() do dir
        json_path = joinpath(dir, "grid.json")
        write(
            json_path,
            """
            {
              "version": 1,
              "name": "json_grid",
              "fixed": {
                "learning_rate": 0.001,
                "hidden_size": 16,
                "depth": 1,
                "batch_size": 4,
                "dropout": 0.0
              },
              "grid": {
                "seed": [1, 2]
              },
              "schedules": {
                "mu": {"kind": "constant", "value": 0.25}
              }
            }
            """,
        )

        spec = grid_load_grid_config(json_path)
        configs = selected_grid(experiment, spec)

        @test spec.format == :json
        @test length(configs) == 2
        @test Set(config.seed for config in configs) == Set([1, 2])
        @test all(config -> config.mu_schedule == :constant, configs)
        @test all(config -> config.mu == 0.25, configs)
    end

    mktempdir() do dir
        piecewise_path = joinpath(dir, "piecewise.yaml")
        write(
            piecewise_path,
            """
            version: 1
            name: manual_schedule_grid
            base:
              epochs: 6
            fixed:
              learning_rate: 0.001
              hidden_size: 16
              depth: 1
              batch_size: 4
              dropout: 0.0
            grid:
              seed: [1]
            schedules:
              mu:
                kind: piecewise
                segments:
                  - epochs: 2
                    value: 1.0
                  - epochs: 3
                    value: 0.9
                  - epochs: 1
                    value: 0.4
              mu_ref:
                kind: values
                values: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
            """,
        )

        spec = grid_load_grid_config(piecewise_path)
        config = only(selected_grid(experiment, spec))

        @test config.mu_schedule == [1.0, 1.0, 0.9, 0.9, 0.9, 0.4]
        @test config.mu_ref_schedule == [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        @test ContextualDFLTraining.mu_schedule_for_config(config) ==
              [1.0, 1.0, 0.9, 0.9, 0.9, 0.4]
        @test ContextualDFLTraining.mu_ref_schedule_for_config(config) ==
              [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    end

    mktempdir() do dir
        values_path = joinpath(dir, "values.yaml")
        write(
            values_path,
            """
            version: 1
            name: values_schedule_grid
            base:
              epochs: 3
            fixed:
              learning_rate: 0.001
              hidden_size: 16
              depth: 1
              batch_size: 4
              dropout: 0.0
            grid:
              seed: [1]
            schedules:
              mu:
                kind: values
                values: [1.0, 0.5, 0.25]
            """,
        )

        config = only(
            selected_grid(
                experiment,
                grid_load_grid_config(values_path),
            ),
        )

        @test config.mu_schedule == [1.0, 0.5, 0.25]
        @test ContextualDFLTraining.mu_schedule_for_config(config) == [1.0, 0.5, 0.25]
        @test ContextualDFLTraining.mu_ref_schedule_for_config(config) == [1.0, 0.5, 0.25]
    end

    @testset "manual schedule validation" begin
        @test ContextualDFLTraining.mu_schedule_for_config(
            (; epochs=3, mu=0.0, mu_schedule=[1, 2, 3]),
        ) == [1.0, 2.0, 3.0]
        @test_throws ArgumentError ContextualDFLTraining.mu_schedule_for_config(
            (; epochs=3, mu=0.0, mu_schedule=[1, 2]),
        )

        mktempdir() do dir
            empty_values_path = joinpath(dir, "empty_values.yaml")
            write(
                empty_values_path,
                """
                version: 1
                schedules:
                  mu:
                    kind: values
                    values: []
                """,
            )
            @test_throws ArgumentError ContextualDFLTraining.load_grid_config(empty_values_path)

            empty_segments_path = joinpath(dir, "empty_segments.yaml")
            write(
                empty_segments_path,
                """
                version: 1
                schedules:
                  mu:
                    kind: piecewise
                    segments: []
                """,
            )
            @test_throws ArgumentError ContextualDFLTraining.load_grid_config(empty_segments_path)

            non_positive_path = joinpath(dir, "non_positive.yaml")
            write(
                non_positive_path,
                """
                version: 1
                schedules:
                  mu:
                    kind: piecewise
                    segments:
                      - epochs: 0
                        value: 1.0
                """,
            )
            @test_throws ArgumentError ContextualDFLTraining.load_grid_config(non_positive_path)
        end
    end

    mktempdir() do dir
        invalid_path = joinpath(dir, "invalid.yaml")
        write(
            invalid_path,
            """
            version: 1
            surprise: true
            """,
        )

        @test_throws ArgumentError ContextualDFLTraining.load_grid_config(invalid_path)
    end
end

FakeRun() = FakeRun(
    Tuple{String,String}[],
    Tuple{String,Float64,Int}[],
    Tuple{String,String}[],
    Any[],
    Tuple{String,Vector{UInt8}}[],
    Symbol[],
)

struct FakeMLFlow end

function ContextualDFLTraining.logparam(::FakeMLFlow, run::FakeRun, key, value)
    value isa String || throw(ArgumentError("MLflow params must be strings."))
    push!(run.params, (string(key), value))
    push!(run.events, :param)
    return nothing
end

function ContextualDFLTraining.logmetric(
    ::FakeMLFlow,
    run::FakeRun,
    key,
    value;
    step,
    timestamp=missing,
)
    value isa Float64 || throw(ArgumentError("MLflow metrics must be Float64."))
    timestamp === missing || timestamp isa Int64 ||
        throw(ArgumentError("MLflow metric timestamps must be Int64."))
    push!(run.metrics, (string(key), value, Int(step)))
    push!(run.events, :metric)
    return nothing
end

function ContextualDFLTraining.logbatch(::FakeMLFlow, run::FakeRun; metrics=[], params=[], tags=[])
    for metric in metrics
        step = getproperty(metric, :step)
        push!(
            run.metrics,
            (
                string(getproperty(metric, :key)),
                Float64(getproperty(metric, :value)),
                step === nothing ? 0 : Int(step),
            ),
        )
    end

    for param in params
        push!(
            run.params,
            (string(getproperty(param, :key)), string(getproperty(param, :value))),
        )
    end

    for tag in tags
        push!(
            run.tags,
            (string(getproperty(tag, :key)), string(getproperty(tag, :value))),
        )
    end

    push!(run.events, :batch)
    return nothing
end

function ContextualDFLTraining.setruntag(::FakeMLFlow, run::FakeRun, key, value)
    push!(run.tags, (string(key), string(value)))
    push!(run.events, :tag)
    return nothing
end

function ContextualDFLTraining.loginputs(::FakeMLFlow, run::FakeRun; datasets)
    append!(run.inputs, datasets)
    push!(run.events, :input)
    return nothing
end

function ContextualDFLTraining.uploadartifact(
    ::FakeMLFlow,
    artifact_path::AbstractString,
    data::Vector{UInt8},
)
    push!(GLOBAL_ARTIFACT_RUN[], (string(artifact_path), data))
    return nothing
end

const GLOBAL_ARTIFACT_RUN = Ref{Vector{Tuple{String,Vector{UInt8}}}}(
    Tuple{String,Vector{UInt8}}[],
)

@testset "ContextualDFLTraining MLflow support" begin
    @testset "logs params and epoch metrics" begin
        mlf = FakeMLFlow()
        run = FakeRun()

        ContextualDFLTraining.log_mlflow_params!(
            mlf,
            run,
            "model",
            (; depth=2, hidden_size=64, nested=(; activation=:relu), skipped=[1, 2]),
        )
        ContextualDFLTraining.log_mlflow_epoch!(
            mlf,
            run,
            2,
            1.5,
            2.5,
            (; mu=0.1, mu_in=0.1, mu_ref=0.0, iterations=3, epoch_seconds=0.25),
        )

        params = Dict(run.params)
        @test params["model_depth"] == "2"
        @test params["model_hidden_size"] == "64"
        @test params["model_nested_activation"] == "relu"
        @test !haskey(params, "model_skipped")

        metrics = Dict(metric[1] => metric[2] for metric in run.metrics)
        @test metrics["loss"] == 1.5
        @test metrics["epoch_mu_in"] == 0.1
        @test metrics["epoch_mu_ref"] == 0.0
        @test metrics["epoch_iterations"] == 3.0
        @test metrics["epoch_seconds"] == 0.25
        @test !haskey(metrics, "display_loss")
        @test !haskey(metrics, "epoch_mu")
        @test all(metric -> metric[3] == 2, run.metrics)
        @test count(==(:batch), run.events) == 1
    end

    @testset "logs evaluation metrics, datasets, tags, and artifacts" begin
        mlf = FakeMLFlow()
        run = FakeRun()
        empty!(GLOBAL_ARTIFACT_RUN[])

        mktempdir() do dir
            artifact_path = joinpath(dir, "report.txt")
            write(artifact_path, "ok")
            ContextualDFLTraining.log_mlflow_evaluation_result!(
                mlf,
                run,
                "",
                (; metrics=(; validation_mse=1.25), artifacts=(; report=artifact_path)),
            )
        end
        append!(run.artifacts, GLOBAL_ARTIFACT_RUN[])

        ContextualDFLTraining.log_mlflow_source_tags!(
            mlf,
            run;
            source_name="ContextualDFLTraining/gridsearch.jl",
            source_type="LOCAL",
            source_git_commit="abc123",
        )
        ContextualDFLTraining.log_mlflow_dataset!(
            mlf,
            run;
            dataset_name="resource_allocation_generated",
            dataset_digest="sha256:test",
            dataset_source_type="generated",
            dataset_source="generated:test",
            dataset_context="training",
        )

        @test ("validation_mse", 1.25, 0) in run.metrics
        @test only(run.artifacts)[1] == "report"
        @test !isempty(only(run.artifacts)[2])
        @test Dict(run.tags)["mlflow.source.name"] == "ContextualDFLTraining/gridsearch.jl"
        @test length(run.inputs) == 1
    end
end
