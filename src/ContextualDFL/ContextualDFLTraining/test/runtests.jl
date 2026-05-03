using ContextualDFLTraining
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
    @test length(ContextualDFLTraining.experiment_smoke_configs(spec)) == 1
    @test isabspath(ContextualDFLTraining.optimal_results_path(spec, :test))

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
            grid_configs(; kwargs...) = [base_config()]
            smoke_configs(; kwargs...) = [base_config()]
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
