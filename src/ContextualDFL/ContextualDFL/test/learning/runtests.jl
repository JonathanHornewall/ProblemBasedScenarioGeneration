import Flux

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
            on_epoch_end=(epoch, loss_value, display_loss, metadata) -> push!(
                callbacks,
                (;
                    epoch=epoch,
                    loss=loss_value,
                    display_loss=display_loss,
                    metadata=metadata,
                ),
            ),
        )

        @test length(result.history) == 2
        @test length(callbacks) == 2
        @test [callback.epoch for callback in callbacks] == [1, 2]
        @test all(callback -> callback.loss isa Float64, callbacks)
        @test all(callback -> callback.display_loss isa Float64, callbacks)
        @test [callback.metadata.epoch for callback in callbacks] == [1, 2]
        @test [callback.metadata.iterations for callback in callbacks] == [2, 2]
        @test all(callback -> callback.metadata.epoch_seconds >= 0, callbacks)
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

end
