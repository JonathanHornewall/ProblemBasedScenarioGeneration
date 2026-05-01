function benchmark(
    problem_instance,
    scenario_generator,
    loss,
    data_set;
    save_path=nothing,
    kwargs...,
)
    training_result = if haskey(kwargs, :train_args)
        train(scenario_generator, loss, data_set, kwargs[:train_args]...)
    else
        nothing
    end
    return (
        problem_instance=problem_instance,
        training_result=training_result,
        save_path=save_path,
    )
end

function run_benchmark(args...; kwargs...)
    return benchmark(args...; kwargs...)
end
