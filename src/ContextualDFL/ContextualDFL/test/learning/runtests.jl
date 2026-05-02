import Flux

module FakeMLFlowClient

const RunStatus = (; FINISHED=:FINISHED, FAILED=:FAILED)

mutable struct FakeRun
    experiment_id
    params::Vector{Tuple{String,String}}
    metrics::Vector{Tuple{String,Float64,Int}}
    events::Vector{Symbol}
    status
end

mutable struct FakeMLFlow
    runs::Vector{FakeRun}
end

FakeMLFlow() = FakeMLFlow(FakeRun[])

function createrun(mlf::FakeMLFlow, experiment_id)
    run = FakeRun(
        experiment_id,
        Tuple{String,String}[],
        Tuple{String,Float64,Int}[],
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

function logmetric(::FakeMLFlow, run::FakeRun, key, value; step)
    value isa Float64 || throw(ArgumentError("MLflow metrics must be Float64."))
    push!(run.metrics, (key, value, Int(step)))
    push!(run.events, :metric)
    return nothing
end

function updaterun(::FakeMLFlow, run::FakeRun; status)
    run.status = status
    push!(run.events, :status)
    return run
end

end

@testset "learning" begin
    @testset "train! calls epoch callback" begin
        model = Flux.Chain(Flux.Dense(1 => 1))
        data = (
            reshape(Float32[1, 2, 3, 4], 1, 4),
            reshape(Float32[2, 4, 6, 8], 1, 4),
        )
        loss(prediction, target) = sum(abs2, prediction .- target)
        callbacks = NamedTuple[]

        result = train!(
            loss,
            model,
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

    @testset "train_with_mlflow! logs live params and metrics" begin
        mlf = FakeMLFlowClient.FakeMLFlow()
        model = Flux.Chain(Flux.Dense(1 => 1))
        data = (
            reshape(Float32[1, 2, 3, 4], 1, 4),
            reshape(Float32[2, 4, 6, 8], 1, 4),
        )
        loss(prediction, target) = sum(abs2, prediction .- target)
        live_metric_counts = Int[]

        result = train_with_mlflow!(
            mlf,
            "experiment-1",
            loss,
            model,
            data;
            learning_rate=1e-4,
            optimizer_type=Flux.Descent,
            epochs=2,
            batchsize=2,
            shuffle=false,
            reset_optimizer_each_epoch=true,
            on_epoch_end=(epoch, loss_value, display_loss) ->
                push!(live_metric_counts, length(only(mlf.runs).metrics)),
        )

        run = only(mlf.runs)
        params = Dict(run.params)

        @test run.experiment_id == "experiment-1"
        @test params["learning_rate"] == string(1e-4)
        @test params["optimizer_type"] == string(Flux.Descent)
        @test params["epochs"] == "2"
        @test params["batchsize"] == "2"
        @test params["shuffle"] == "false"
        @test params["reset_optimizer_each_epoch"] == "true"

        @test length(result.history) == 2
        @test run.status === FakeMLFlowClient.RunStatus.FINISHED
        @test count(metric -> metric[1] == "loss", run.metrics) == 2
        @test count(metric -> metric[1] == "display_loss", run.metrics) == 2
        @test all(metric -> metric[2] isa Float64, run.metrics)
        @test sort(unique(metric[3] for metric in run.metrics)) == [1, 2]
        @test live_metric_counts == [2, 4]
        @test run.events[end] === :status
        @test count(==(:metric), run.events) == 4
    end

    @testset "train_with_mlflow! marks failed runs" begin
        mlf = FakeMLFlowClient.FakeMLFlow()
        model = Flux.Chain(Flux.Dense(1 => 1))
        data = (reshape(Float32[1], 1, 1), reshape(Float32[2], 1, 1))
        failing_loss(prediction, target) = error("intentional training failure")

        @test_throws ErrorException train_with_mlflow!(
            mlf,
            "experiment-2",
            failing_loss,
            model,
            data;
            optimizer_type=Flux.Descent,
            epochs=1,
            batchsize=1,
        )

        run = only(mlf.runs)
        @test run.status === FakeMLFlowClient.RunStatus.FAILED
        @test run.events[end] === :status
    end
end
