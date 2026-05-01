using Flux

function train(
    scenario_generator::DFLScenarioGenerator,
    loss::LossFunction,
    data_set::DataSet,
    scenario_decoder::DataSetScenarioDecoder,
    mu_schedule,
    rho_schedule,
    batch_size_schedule,
    step_size_schedule;
    epochs=nothing,
    optimizer=Flux.Adam,
    rng=Random.default_rng(),
    verbose::Bool=false,
)
    n = length(data_set)
    n > 0 || error("Cannot train on an empty DataSet.")
    epochs = epochs === nothing ? _infer_epochs(mu_schedule, rho_schedule, batch_size_schedule, step_size_schedule) : epochs
    model = scenario_generator.neural_net
    history = Float64[]
    opt_state = nothing

    for epoch in 1:epochs
        mu = _schedule_value(mu_schedule, epoch)
        rho = _schedule_value(rho_schedule, epoch)
        batch_size = max(1, Int(round(_schedule_value(batch_size_schedule, epoch))))
        step_size = _schedule_value(step_size_schedule, epoch)
        opt = optimizer(step_size)
        opt_state = Flux.setup(opt, model)

        indices = randperm(rng, n)
        batch_losses = Float64[]
        for batch in Iterators.partition(indices, batch_size)
            local batch_loss_value
            gradient = Flux.gradient(model) do m
                generator = DFLScenarioGenerator(
                    scenario_generator.scenario_decoder,
                    scenario_generator.solver,
                    m,
                    scenario_generator.program,
                )
                batch_loss = sum(batch) do idx
                    row = data_set[idx]
                    actual = scenario_decoder(row)
                    predicted = generator(row.x)
                    loss(scenario_generator.program, actual, predicted, mu, rho)
                end / length(batch)
                batch_loss_value = batch_loss
                return batch_loss
            end
            gmodel = gradient isa Tuple ? gradient[1] : gradient
            Flux.update!(opt_state, model, gmodel)
            push!(batch_losses, Float64(batch_loss_value))
        end

        epoch_loss = mean(batch_losses)
        push!(history, epoch_loss)
        verbose && @info "epoch=$epoch loss=$epoch_loss mu=$mu rho=$rho batch_size=$batch_size step_size=$step_size"
    end

    return (loss_history=history, model=model, opt_state=opt_state)
end

function _schedule_value(schedule, i::Integer)
    schedule isa Number && return schedule
    try
        return schedule(i)
    catch
        return schedule[i]
    end
end

function _infer_epochs(schedules...)
    lengths = Int[]
    for schedule in schedules
        schedule isa Number && continue
        try
            push!(lengths, length(schedule))
        catch
        end
    end
    return isempty(lengths) ? 1 : maximum(lengths)
end
