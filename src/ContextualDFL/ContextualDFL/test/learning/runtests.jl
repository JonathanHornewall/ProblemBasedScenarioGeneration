import Flux

module FakeMLFlowClient

const RunStatus = (; FINISHED=:FINISHED, FAILED=:FAILED)

mutable struct FakeRun
    experiment_id
    start_time
    end_time
    params::Vector{Tuple{String,String}}
    metrics::Vector{Tuple{String,Float64,Int}}
    tags::Vector{Tuple{String,String}}
    inputs::Vector
    artifacts::Vector{Tuple{String,Vector{UInt8}}}
    events::Vector{Symbol}
    status
end

struct Tag
    key::String
    value::String
end

struct Dataset
    name::String
    digest::String
    source_type::String
    source::String
    schema
    profile
end

struct DatasetInput
    tags::Vector{Tag}
    dataset::Dataset
end

mutable struct FakeMLFlow
    runs::Vector{FakeRun}
end

FakeMLFlow() = FakeMLFlow(FakeRun[])

function createrun(mlf::FakeMLFlow, experiment_id; start_time=missing)
    run = FakeRun(
        experiment_id,
        start_time,
        missing,
        Tuple{String,String}[],
        Tuple{String,Float64,Int}[],
        Tuple{String,String}[],
        Any[],
        Tuple{String,Vector{UInt8}}[],
        Symbol[],
        nothing,
    )
    push!(mlf.runs, run)
    return run
end

function logparam(::FakeMLFlow, run::FakeRun, key, value)
    value isa String || throw(ArgumentError("MLflow params must be strings."))
    push!(run.params, (key, value))
    push!(run.events, :param)
    return nothing
end

function logmetric(::FakeMLFlow, run::FakeRun, key, value; step, timestamp=missing)
    value isa Float64 || throw(ArgumentError("MLflow metrics must be Float64."))
    timestamp === missing || timestamp isa Int64 ||
        throw(ArgumentError("MLflow metric timestamps must be Int64."))
    push!(run.metrics, (key, value, Int(step)))
    push!(run.events, :metric)
    return nothing
end

function setruntag(::FakeMLFlow, run::FakeRun, key, value)
    push!(run.tags, (string(key), string(value)))
    push!(run.events, :tag)
    return nothing
end

function loginputs(::FakeMLFlow, run::FakeRun; datasets)
    append!(run.inputs, datasets)
    push!(run.events, :input)
    return nothing
end

function loginputs(::FakeMLFlow, run::FakeRun, datasets)
    append!(run.inputs, datasets)
    push!(run.events, :input)
    return nothing
end

function uploadartifact(mlf::FakeMLFlow, artifact_path, data::Vector{UInt8})
    run = only(mlf.runs)
    push!(run.artifacts, (string(artifact_path), data))
    push!(run.events, :artifact)
    return nothing
end

function updaterun(::FakeMLFlow, run::FakeRun; status, end_time=missing)
    run.status = status
    run.end_time = end_time
    push!(run.events, :status)
    return run
end

end

function supervised_dataset(inputs, targets)
    points = [
        ContextualDataPoint(Float32[input], [ParametricScenario(h_eq_xi=Float32[target])])
        for (input, target) in zip(inputs, targets)
    ]
    return ContextualDataSet{eltype(points)}(points)
end

target_vector(scenario_parameters) = only(scenario_parameters).h_eq_xi

@testset "learning" begin
    @testset "train! calls epoch callback" begin
        model = Flux.Chain(Flux.Dense(1 => 1))
        data = supervised_dataset(1:4, 2:2:8)
        loss(prediction, scenario_parameters, mu_in, mu_ref; kwargs...) =
            sum(abs2, prediction .- target_vector(scenario_parameters))
        callbacks = NamedTuple[]

        result = train!(
            model,
            loss,
            nothing,
            fill(0.0, 2),
            data;
            optimizer_type=Flux.Descent,
            learning_rate=1e-4,
            epochs=2,
            batchsize=2,
            on_epoch_end=(epoch, loss_value, display_loss) -> push!(
                callbacks,
                (; epoch=epoch, loss=loss_value, display_loss=display_loss),
            ),
        )

        @test length(result.history) == 2
        @test length(callbacks) == 2
        @test [callback.epoch for callback in callbacks] == [1, 2]
        @test all(callback -> callback.loss isa Float64, callbacks)
        @test all(callback -> callback.display_loss isa Float64, callbacks)
    end

    @testset "train! defaults mu_ref_schedule to mu_in_schedule" begin
        model = Flux.Chain(Flux.Dense(1 => 1))
        data = supervised_dataset(1:4, 2:2:8)
        mu_schedule = [0.1, 0.2]
        loss(prediction, scenario_parameters, mu_in, mu_ref; kwargs...) =
            sum(abs2, prediction .- target_vector(scenario_parameters))

        result = train!(
            model,
            loss,
            nothing,
            mu_schedule,
            data;
            optimizer_type=Flux.Descent,
            learning_rate=1e-4,
            epochs=length(mu_schedule),
            batchsize=2,
        )

        @test [row.mu_in for row in result.history] == mu_schedule
        @test [row.mu_ref for row in result.history] == mu_schedule
        @test [row.mu for row in result.history] == mu_schedule
    end

    @testset "train! rejects non-finite training loss" begin
        model = Flux.Chain(Flux.Dense(1 => 1))
        data = supervised_dataset(1:1, 2:2)
        nan_loss(prediction, scenario_parameters, mu_in, mu_ref; kwargs...) =
            sum(prediction) * Float32(NaN)
        callback_count = Ref(0)

        @test_throws DomainError train!(
            model,
            nan_loss,
            nothing,
            fill(0.0, 1),
            data;
            optimizer_type=Flux.Descent,
            learning_rate=1.0,
            epochs=1,
            batchsize=1,
            on_epoch_end=(args...) -> (callback_count[] += 1),
        )

        @test callback_count[] == 0
        @test all(parameter -> all(isfinite, parameter), Flux.trainables(model))
    end

    @testset "train_with_mlflow! logs epoch params and metrics" begin
        mlf = FakeMLFlowClient.FakeMLFlow()
        model = Flux.Chain(Flux.Dense(1 => 1))
        data = supervised_dataset(1:4, 2:2:8)
        loss(prediction, scenario_parameters, mu_in, mu_ref; kwargs...) =
            sum(abs2, prediction .- target_vector(scenario_parameters))
        mu_schedule = [0.1, 0.2]
        epoch_metric_counts = Int[]

        result = mktempdir() do dir
            model_path = joinpath(dir, "trained_model.jls")
            train_with_mlflow!(
                mlf,
                "experiment-1",
                model,
                loss,
                mu_schedule,
                data;
                learning_rate=1e-4,
                optimizer_type=Flux.Descent,
                epochs=2,
                batchsize=2,
                shuffle=false,
                reset_optimizer_each_epoch=true,
                source_name="train_test.jl",
                source_git_commit="abc123",
                dataset_name="training-data",
                dataset_digest="sha256:test",
                dataset_source="/tmp/train.csv",
                save_model=true,
                model_save_path=model_path,
                model_artifact_path="models/trained_model.jls",
                on_epoch_end=(epoch, loss_value, display_loss) ->
                    push!(epoch_metric_counts, length(only(mlf.runs).metrics)),
            )
        end

        run = only(mlf.runs)
        params = Dict(run.params)
        tags = Dict(run.tags)

        @test run.experiment_id == "experiment-1"
        @test run.start_time isa Int64
        @test run.end_time isa Int64
        @test run.end_time >= run.start_time
        @test params["learning_rate"] == string(1e-4)
        @test params["optimizer_type"] == string(Flux.Descent)
        @test params["epochs"] == "2"
        @test params["batchsize"] == "2"
        @test params["mu_in_schedule"] == string(mu_schedule)
        @test params["mu_ref_schedule"] == string(mu_schedule)
        @test params["shuffle"] == "false"
        @test params["reset_optimizer_each_epoch"] == "true"
        @test tags["mlflow.source.name"] == "train_test.jl"
        @test tags["mlflow.source.type"] == "LOCAL"
        @test tags["mlflow.source.git.commit"] == "abc123"

        @test length(result.history) == 2
        @test [row.mu_ref for row in result.history] == mu_schedule
        @test run.status === FakeMLFlowClient.RunStatus.FINISHED
        loss_metrics = filter(metric -> metric[1] == "loss", run.metrics)
        display_loss_metrics = filter(metric -> metric[1] == "display_loss", run.metrics)
        @test length(loss_metrics) == 2
        @test length(display_loss_metrics) == 2
        @test all(metric -> metric[2] isa Float64, run.metrics)
        @test [metric[3] for metric in loss_metrics] == [1, 2]
        @test [metric[3] for metric in display_loss_metrics] == [1, 2]
        @test sort(unique(metric[3] for metric in run.metrics)) == [1, 2]
        @test epoch_metric_counts == [7, 14]
        @test run.events[end] === :status
        @test count(==(:metric), run.events) == 14
        @test length(run.inputs) == 1
        @test only(run.inputs).dataset.name == "training-data"
        @test only(run.inputs).dataset.digest == "sha256:test"
        @test only(run.inputs).tags[1].key == "context"
        @test only(run.inputs).tags[1].value == "training"
        @test length(run.artifacts) == 1
        @test run.artifacts[1][1] == "models/trained_model.jls"
        @test !isempty(run.artifacts[1][2])
    end

    @testset "train_with_mlflow! marks failed runs" begin
        mlf = FakeMLFlowClient.FakeMLFlow()
        model = Flux.Chain(Flux.Dense(1 => 1))
        data = supervised_dataset(1:1, 2:2)
        failing_loss(prediction, scenario_parameters, mu_in, mu_ref; kwargs...) =
            error("intentional training failure")

        @test_throws ErrorException train_with_mlflow!(
            mlf,
            "experiment-2",
            model,
            failing_loss,
            fill(0.0, 1),
            data;
            optimizer_type=Flux.Descent,
            epochs=1,
            batchsize=1,
        )

        run = only(mlf.runs)
        @test run.status === FakeMLFlowClient.RunStatus.FAILED
        @test run.end_time isa Int64
        @test run.events[end] === :status
    end
end
