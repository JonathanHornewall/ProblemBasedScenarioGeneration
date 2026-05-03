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

    @testset "train! smooth display uses cached references" begin
        model = Flux.Chain(Flux.Dense(1 => 1))
        data = supervised_dataset(1:4, 2:2:8)
        mu_in_schedule = [0.1, 0.2]
        mu_ref_schedule = [0.1, 0.2]
        reference_calls = Ref(0)
        relative_calls = Ref(0)

        function loss(input, scenario_parameters, mu_in, mu_ref; kwargs...)
            input === target_vector(scenario_parameters) && (reference_calls[] += 1)
            return sum(abs2, input) + sum(target_vector(scenario_parameters)) + mu_in + mu_ref
        end
        relative_loss(args...; kwargs...) = (relative_calls[] += 1)
        display_reference_input(point) = target_vector(point.scenario_parameters)

        result = train!(
            model,
            loss,
            relative_loss,
            mu_in_schedule,
            mu_ref_schedule,
            data;
            optimizer_type=Flux.Descent,
            learning_rate=1e-4,
            epochs=length(mu_in_schedule),
            batchsize=2,
            display_smooth=true,
            display_reference_input=display_reference_input,
        )

        @test reference_calls[] == length(unique(mu_ref_schedule)) * length(data)
        @test relative_calls[] == 0
        @test all(row -> row.display_loss isa Float64, result.history)
        @test all(row -> row.real_display_loss === nothing, result.history)
    end

    @testset "train! display modes validate reference input" begin
        model = Flux.Chain(Flux.Dense(1 => 1))
        data = supervised_dataset(1:1, 2:2)
        loss(input, scenario_parameters, mu_in, mu_ref; kwargs...) =
            sum(abs2, input) + sum(target_vector(scenario_parameters))
        display_reference_input(point) = target_vector(point.scenario_parameters)

        @test_throws ArgumentError train!(
            model,
            loss,
            nothing,
            fill(0.0, 1),
            data;
            optimizer_type=Flux.Descent,
            learning_rate=1e-4,
            epochs=1,
            batchsize=1,
            display_smooth=true,
        )
        @test_throws ArgumentError train!(
            model,
            loss,
            nothing,
            fill(0.0, 1),
            data;
            optimizer_type=Flux.Descent,
            learning_rate=1e-4,
            epochs=1,
            batchsize=1,
            display_real=1,
        )
        @test_throws ArgumentError train!(
            model,
            loss,
            nothing,
            fill(0.0, 1),
            data;
            optimizer_type=Flux.Descent,
            learning_rate=1e-4,
            epochs=1,
            batchsize=1,
            display_real=0,
            display_reference_input=display_reference_input,
        )
    end

    @testset "train! real display runs on requested epochs" begin
        model = Flux.Chain(Flux.Dense(1 => 1))
        data = supervised_dataset(1:2, 2:2:4)
        mu_schedule = [0.3, 0.2, 0.1]
        reference_calls = Ref(0)
        real_calls = Ref(0)

        function loss(input, scenario_parameters, mu_in, mu_ref; kwargs...)
            if input === target_vector(scenario_parameters)
                reference_calls[] += 1
            elseif mu_ref == 0.0
                real_calls[] += 1
            end
            return sum(abs2, input) + sum(target_vector(scenario_parameters)) + mu_in + mu_ref
        end
        display_reference_input(point) = target_vector(point.scenario_parameters)

        result = train!(
            model,
            loss,
            nothing,
            mu_schedule,
            data;
            optimizer_type=Flux.Descent,
            learning_rate=1e-4,
            epochs=length(mu_schedule),
            batchsize=1,
            display_real=2,
            display_reference_input=display_reference_input,
        )

        @test reference_calls[] == length(data)
        @test real_calls[] == length(data)
        @test result.history[1].real_display_loss === nothing
        @test result.history[2].real_display_loss isa Float64
        @test result.history[3].real_display_loss === nothing
    end

end
