import Flux
import Plots
import Random
import Serialization
import Statistics

# %%% Core training loop

"""
    train!(
        neural_net,
        loss,
        relative_loss,
        mu_in_schedule,
        mu_ref_schedule,
        data_set;
        kwargs...,
    )

Flux training loop for a contextual scenario dataset.

Each data point must have a context vector and a scenario-parameter collection.
At epoch `k`, `loss` is called as:

    loss(neural_net(context), scenario_parameters, mu_in_schedule[k], mu_ref_schedule[k])
"""
function train!(
    neural_net,
    loss,
    relative_loss,
    mu_in_schedule::AbstractVector,
    mu_ref_schedule::AbstractVector,
    data_set;
    opt=nothing,
    optimizer_type=Flux.Adam,
    learning_rate=1e-3,
    epochs::Integer=length(mu_in_schedule),
    batchsize::Integer=1,
    display_iterations::Bool=false,
    verbose::Bool=display_iterations,
    display_plot::Bool=display_iterations,
    save_model::Bool=false,
    model_save_path::AbstractString="trained_model.jls",
    shuffle::Bool=false,
    rng::Random.AbstractRNG=Random.default_rng(),
    opt_state=nothing,
    reset_optimizer_each_epoch::Bool=false,
    on_epoch_end=nothing,
    nr_scenarios=nothing,
    rho_in_schedule=nothing,
    rho_ref_schedule=nothing,
    display_smooth::Bool=false,
    display_real=nothing,
    display_reference_input=nothing,
)
    epochs >= 0 || throw(ArgumentError("epochs must be non-negative."))
    batchsize > 0 || throw(ArgumentError("batchsize must be positive."))
    length(mu_in_schedule) == epochs ||
        throw(ArgumentError("mu_in_schedule must have one value per epoch."))
    length(mu_ref_schedule) == epochs ||
        throw(ArgumentError("mu_ref_schedule must have one value per epoch."))
    rho_in_values = _optional_schedule_values(rho_in_schedule, epochs, :rho_in_schedule)
    rho_ref_values = _optional_schedule_values(
        isnothing(rho_ref_schedule) ? rho_in_schedule : rho_ref_schedule,
        epochs,
        :rho_ref_schedule,
    )
    pass_rho = any(!iszero, rho_in_values) || any(!iszero, rho_ref_values)
    isempty(data_set) && throw(ArgumentError("training data must not be empty."))
    _validate_nr_scenarios(nr_scenarios)
    _validate_display_options(display_smooth, display_real, display_reference_input)

    optimizer = isnothing(opt) ? _make_optimizer(optimizer_type, learning_rate) : opt
    state = isnothing(opt_state) ? Flux.setup(optimizer, neural_net) : opt_state
    show_progress = display_iterations || verbose
    loss_kwargs = _training_loss_kwargs(nr_scenarios)
    display_reference_cache = Dict{Any,Vector{Float64}}()

    if display_smooth
        for (mu_ref, rho_ref) in unique(collect(zip(mu_ref_schedule, rho_ref_values)))
            _display_reference_values!(
                display_reference_cache,
                loss,
                data_set,
                display_reference_input,
                mu_ref;
                rho_ref=rho_ref,
                pass_rho=pass_rho,
                loss_kwargs=loss_kwargs,
            )
        end
    end

    real_reference_values = if isnothing(display_real)
        nothing
    else
        _display_reference_values!(
            display_reference_cache,
            loss,
            data_set,
            display_reference_input,
            0.0;
            rho_ref=0.0,
            pass_rho=pass_rho,
            loss_kwargs=loss_kwargs,
        )
    end

    history = NamedTuple[]
    displayed_epoch_losses = Float64[]

    for epoch_number in 1:epochs
        epoch_started = time()
        mu_in = mu_in_schedule[epoch_number]
        mu_ref = mu_ref_schedule[epoch_number]
        rho_in = rho_in_values[epoch_number]
        rho_ref = rho_ref_values[epoch_number]
        epoch_loss_kwargs = _with_rho_loss_kwargs(loss_kwargs, pass_rho, rho_in, rho_ref)

        if reset_optimizer_each_epoch
            state = Flux.setup(optimizer, neural_net)
        end

        show_progress && print("Epoch ", epoch_number)
        epoch_losses = Float64[]
        epoch_display_losses = Float64[]

        indices = shuffle ? Random.randperm(rng, length(data_set)) : collect(eachindex(data_set))
        for idxs_iter in Iterators.partition(indices, batchsize)
            idxs = collect(idxs_iter)

            loss_value, gradients = Flux.withgradient(neural_net) do trainable_neural_net
                Statistics.mean(
                    loss(
                        trainable_neural_net(_context(data_set[index])),
                        _scenario_parameters(data_set[index]),
                        mu_in,
                        mu_ref;
                        epoch_loss_kwargs...,
                    )
                    for index in idxs
                )
            end
            iteration_number = length(epoch_losses) + 1
            loss_float = _checked_loss_float(
                loss_value,
                "training loss";
                epoch=epoch_number,
                iteration=iteration_number,
                mu_in=mu_in,
                mu_ref=mu_ref,
            )
            Flux.update!(state, neural_net, gradients[1])
            push!(epoch_losses, loss_float)

            if display_smooth
                reference_key = _display_reference_cache_key(mu_ref, rho_ref, pass_rho)
                reference_mean = Statistics.mean(
                    display_reference_cache[reference_key][index] for index in idxs
                )
                display_loss = _relative_display_loss(loss_float, reference_mean)
                push!(
                    epoch_display_losses,
                    _checked_loss_float(
                        display_loss,
                        "smooth display loss";
                        epoch=epoch_number,
                        iteration=iteration_number,
                        mu_in=mu_in,
                        mu_ref=mu_ref,
                    ),
                )
            elseif show_progress || !isnothing(relative_loss)
                display_loss_function = isnothing(relative_loss) ? loss : relative_loss
                display_loss = Statistics.mean(
                    display_loss_function(
                        neural_net(_context(data_set[index])),
                        _scenario_parameters(data_set[index]),
                        mu_in,
                        mu_ref;
                        epoch_loss_kwargs...,
                    )
                    for index in idxs
                )
                push!(
                    epoch_display_losses,
                    _checked_loss_float(
                        display_loss,
                        "display loss";
                        epoch=epoch_number,
                        iteration=iteration_number,
                        mu_in=mu_in,
                        mu_ref=mu_ref,
                    ),
                )
            end
        end

        average_loss = Statistics.mean(epoch_losses)
        average_display_loss = isempty(epoch_display_losses) ?
            average_loss :
            Statistics.mean(epoch_display_losses)
        real_display_loss = if isnothing(display_real) ||
                               epoch_number % Int(display_real) != 0
            nothing
        else
            real_loss = Statistics.mean(
                loss(
                    neural_net(_context(data_point)),
                    _scenario_parameters(data_point),
                    mu_in,
                    0.0;
                    _with_rho_loss_kwargs(loss_kwargs, pass_rho, rho_in, 0.0)...,
                )
                for data_point in data_set
            )
            real_loss_float = _checked_loss_float(
                real_loss,
                "real display loss";
                epoch=epoch_number,
                iteration=length(epoch_losses),
                mu_in=mu_in,
                mu_ref=0.0,
            )
            real_reference_mean = Statistics.mean(real_reference_values)
            _checked_loss_float(
                _relative_display_loss(real_loss_float, real_reference_mean),
                "relative real display loss";
                epoch=epoch_number,
                iteration=length(epoch_losses),
                mu_in=mu_in,
                mu_ref=0.0,
            )
        end
        epoch_seconds = time() - epoch_started
        epoch_metadata = (;
            epoch=Int(epoch_number),
            mu=mu_in,
            mu_in=mu_in,
            mu_ref=mu_ref,
            rho_in=rho_in,
            rho_ref=rho_ref,
            iterations=length(epoch_losses),
            epoch_seconds=epoch_seconds,
            real_display_loss=real_display_loss,
        )

        if show_progress
            print(" with avg loss ", average_display_loss)
            isnothing(real_display_loss) || print(" real loss ", real_display_loss)
            println(" (", length(epoch_display_losses), " iterations)")
            push!(displayed_epoch_losses, average_display_loss)
        end

        if !isnothing(on_epoch_end)
            _call_epoch_callback(
                on_epoch_end,
                Int(epoch_number),
                average_loss,
                average_display_loss,
                epoch_metadata,
            )
        end

        push!(
            history,
            (;
                epoch=Int(epoch_number),
                mu=mu_in,
                mu_in=mu_in,
                mu_ref=mu_ref,
                rho_in=rho_in,
                rho_ref=rho_ref,
                loss=average_loss,
                display_loss=average_display_loss,
                real_display_loss=real_display_loss,
                iterations=length(epoch_losses),
                epoch_seconds=epoch_seconds,
            ),
        )
    end

    # %%% Optional model storage
    if save_model
        Serialization.serialize(model_save_path, neural_net)
        println("Model saved to: $model_save_path")
    end

    # %%% Optional training-loss plot
    if display_plot && show_progress && !isempty(displayed_epoch_losses)
        plt = Plots.plot(
            1:length(displayed_epoch_losses),
            displayed_epoch_losses;
            xlabel="Epoch",
            ylabel="Loss",
            title="Training Loss",
        )
        display(plt)
    end

    return (; model=neural_net, history=history, opt_state=state)
end

train!(
    neural_net,
    loss,
    relative_loss,
    mu_in_schedule::AbstractVector,
    data_set;
    mu_ref_schedule=nothing,
    kwargs...,
) =
    train!(
        neural_net,
        loss,
        relative_loss,
        mu_in_schedule,
        _default_mu_ref_schedule(mu_in_schedule, mu_ref_schedule),
        data_set;
        kwargs...,
    )

train!(
    neural_net,
    loss,
    mu_schedule::AbstractVector,
    data_set;
    kwargs...,
) =
    train!(
        neural_net,
        loss,
        nothing,
        mu_schedule,
        data_set;
        kwargs...,
    )

train!(
    neural_net,
    loss,
    mu_in_schedule::AbstractVector,
    mu_ref_schedule::AbstractVector,
    data_set;
    kwargs...,
) =
    train!(
        neural_net,
        loss,
        nothing,
        mu_in_schedule,
        mu_ref_schedule,
        data_set;
        kwargs...,
    )

# %%% Small core helpers

_context(data_point::ContextualDataPoint) = data_point.context
_context(data_point::Tuple) = data_point[1]

_scenario_parameters(data_point::ContextualDataPoint) = data_point.scenario_parameters
_scenario_parameters(data_point::Tuple) = data_point[2]

function _validate_nr_scenarios(nr_scenarios)
    isnothing(nr_scenarios) && return nothing
    nr_scenarios isa Integer && nr_scenarios > 0 ||
        throw(ArgumentError("nr_scenarios must be a positive integer."))
    return nothing
end

_training_loss_kwargs(nr_scenarios) =
    isnothing(nr_scenarios) ? NamedTuple() : (; nr_scenarios=Int(nr_scenarios))

function _optional_schedule_values(schedule, epochs, name::Symbol)
    if isnothing(schedule)
        return zeros(Float64, epochs)
    end

    length(schedule) == epochs ||
        throw(ArgumentError("$(name) must have one value per epoch."))
    return collect(schedule)
end

function _with_rho_loss_kwargs(loss_kwargs, pass_rho, rho_in, rho_ref)
    pass_rho || return loss_kwargs
    return merge(loss_kwargs, (; rho_in=rho_in, rho_ref=rho_ref))
end

_default_mu_ref_schedule(mu_in_schedule, mu_ref_schedule) =
    isnothing(mu_ref_schedule) ? mu_in_schedule : mu_ref_schedule

function _validate_display_options(display_smooth, display_real, display_reference_input)
    if !isnothing(display_real)
        display_real isa Integer && !(display_real isa Bool) && display_real > 0 ||
            throw(ArgumentError("display_real must be nothing or a positive integer."))
    end

    if (display_smooth || !isnothing(display_real)) && isnothing(display_reference_input)
        throw(
            ArgumentError(
                "display_smooth/display_real require display_reference_input, " *
                "a function from data point to exact-scenario decoder input.",
            ),
        )
    end

    return nothing
end

function _display_reference_values!(
    cache,
    loss,
    data_set,
    display_reference_input,
    mu_ref;
    rho_ref=0,
    pass_rho=false,
    loss_kwargs,
)
    cache_key = _display_reference_cache_key(mu_ref, rho_ref, pass_rho)
    haskey(cache, cache_key) && return cache[cache_key]
    reference_loss_kwargs = _with_rho_loss_kwargs(loss_kwargs, pass_rho, rho_ref, rho_ref)

    # Reference inputs represent the exact scenario, so this is the cached baseline.
    values = [
        _checked_loss_float(
            loss(
                display_reference_input(data_point),
                _scenario_parameters(data_point),
                mu_ref,
                mu_ref;
                reference_loss_kwargs...,
            ),
            "display reference loss";
            epoch=0,
            iteration=index,
            mu_in=mu_ref,
            mu_ref=mu_ref,
        ) for (index, data_point) in enumerate(data_set)
    ]
    cache[cache_key] = values
    return values
end

_display_reference_cache_key(mu_ref, rho_ref, pass_rho) =
    pass_rho ? (mu_ref, rho_ref) : mu_ref

_relative_display_loss(loss_value, reference_value) =
    (Float64(loss_value) - Float64(reference_value)) / abs(Float64(reference_value))

_float(value::Number) = Float64(value)
_float(value::AbstractArray) = Float64(only(value))

function _checked_loss_float(value, label; epoch, iteration, mu=nothing, mu_in=mu, mu_ref=0)
    float_value = _float(value)
    isfinite(float_value) || throw(
        DomainError(
            float_value,
            "$label became non-finite at epoch=$(epoch) iteration=$(iteration) mu_in=$(mu_in) mu_ref=$(mu_ref)",
        ),
    )
    return float_value
end

function _call_epoch_callback(callback, epoch, loss_value, display_loss, metadata)
    if applicable(callback, epoch, loss_value, display_loss, metadata)
        return callback(epoch, loss_value, display_loss, metadata)
    end
    return callback(epoch, loss_value, display_loss)
end

# %%% Optimizer helpers

function _make_optimizer(optimizer_type::Symbol, learning_rate)
    if optimizer_type === :adam
        return Flux.Adam(learning_rate)
    elseif optimizer_type in (:descent, :sgd)
        return Flux.Descent(learning_rate)
    elseif optimizer_type === :rmsprop
        return Flux.RMSProp(learning_rate)
    end

    throw(ArgumentError("unsupported optimizer_type `$optimizer_type`."))
end

_make_optimizer(optimizer_type, learning_rate) = optimizer_type(learning_rate)
