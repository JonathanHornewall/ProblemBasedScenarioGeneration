import Flux
import Dates
import Plots
import Random
import Serialization
import Statistics

"""
    train!(loss, relative_loss, model, data; kwargs...)

Generic Flux training loop for a callable `model`.

`loss` is called as `loss(model(x), y)` on each sample in a mini-batch.
`relative_loss`, when provided, is used for the progress display in the same
way as the old ProblemBasedScenarioGeneration training loop.
"""
function train!(
    loss,
    relative_loss,
    model,
    data;
    opt=nothing,
    optimizer_type=Flux.Adam,
    learning_rate=1e-3,
    epochs::Integer=1,
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
)
    epochs >= 0 || throw(ArgumentError("epochs must be non-negative."))
    batchsize > 0 || throw(ArgumentError("batchsize must be positive."))

    xs, ys = _training_samples(data)
    length(xs) == length(ys) ||
        throw(ArgumentError("training inputs and targets must have the same length."))
    isempty(xs) && throw(ArgumentError("training data must not be empty."))

    optimizer = isnothing(opt) ? _make_optimizer(optimizer_type, learning_rate) : opt
    state = isnothing(opt_state) ? Flux.setup(optimizer, model) : opt_state
    show_progress = display_iterations || verbose

    history = NamedTuple[]
    displayed_epoch_losses = Float64[]

    for epoch_number in 1:epochs
        if reset_optimizer_each_epoch
            state = Flux.setup(optimizer, model)
        end

        show_progress && print("Epoch ", epoch_number)
        epoch_losses = Float64[]
        epoch_display_losses = Float64[]

        indices = shuffle ? Random.randperm(rng, length(xs)) : collect(eachindex(xs))
        for idxs_iter in Iterators.partition(indices, batchsize)
            idxs = collect(idxs_iter)
            x_batch = _batch_data(xs, idxs)
            y_batch = _batch_data(ys, idxs)

            loss_value, gradients = Flux.withgradient(model) do trainable_model
                _mean_sample_loss(loss, trainable_model, x_batch, y_batch)
            end
            Flux.update!(state, model, gradients[1])
            push!(epoch_losses, _float(loss_value))

            if show_progress
                display_loss = isnothing(relative_loss) ?
                    _mean_sample_loss(loss, model, x_batch, y_batch) :
                    _mean_sample_loss(relative_loss, model, x_batch, y_batch)
                push!(epoch_display_losses, _float(display_loss))
            end
        end

        average_loss = Statistics.mean(epoch_losses)
        average_display_loss = isempty(epoch_display_losses) ?
            average_loss :
            Statistics.mean(epoch_display_losses)

        if show_progress
            println(
                " with avg loss ",
                average_display_loss,
                " (",
                length(epoch_display_losses),
                " iterations)",
            )
            push!(displayed_epoch_losses, average_display_loss)
        end

        if !isnothing(on_epoch_end)
            on_epoch_end(Int(epoch_number), average_loss, average_display_loss)
        end

        push!(
            history,
            (;
                epoch=Int(epoch_number),
                loss=average_loss,
                display_loss=average_display_loss,
                iterations=length(epoch_losses),
            ),
        )
    end

    if save_model
        Serialization.serialize(model_save_path, model)
        println("Model saved to: $model_save_path")
    end

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

    return (; model=model, history=history, opt_state=state)
end

train!(loss, model, data; kwargs...) =
    train!(loss, nothing, model, data; kwargs...)

function train_with_mlflow!(
    mlf,
    experiment_id,
    loss,
    model,
    data;
    relative_loss=nothing,
    learning_rate=1e-3,
    optimizer_type=Flux.Adam,
    epochs::Integer=10,
    batchsize::Integer=32,
    shuffle::Bool=false,
    reset_optimizer_each_epoch::Bool=false,
    on_epoch_end=nothing,
    source_name=nothing,
    source_type="LOCAL",
    source_git_commit=_git_commit_or_nothing(),
    log_source_tags::Bool=true,
    dataset_inputs=nothing,
    dataset_name=nothing,
    dataset_digest=nothing,
    dataset_source_type="local",
    dataset_source=nothing,
    dataset_context="training",
    save_model::Bool=false,
    model_save_path::AbstractString="trained_model.jls",
    upload_model_artifact::Bool=save_model,
    model_artifact_path::AbstractString=basename(model_save_path),
    kwargs...,
)
    mlflow = parentmodule(typeof(mlf))
    run = getproperty(mlflow, :createrun)(
        mlf,
        experiment_id;
        start_time=_unix_milliseconds(),
    )

    logparam = getproperty(mlflow, :logparam)
    logparam(mlf, run, "learning_rate", string(learning_rate))
    logparam(mlf, run, "optimizer_type", string(optimizer_type))
    logparam(mlf, run, "epochs", string(epochs))
    logparam(mlf, run, "batchsize", string(batchsize))
    logparam(mlf, run, "shuffle", string(shuffle))
    logparam(mlf, run, "reset_optimizer_each_epoch", string(reset_optimizer_each_epoch))

    if log_source_tags
        _log_mlflow_source_tags!(
            mlflow,
            mlf,
            run;
            source_name=source_name,
            source_type=source_type,
            source_git_commit=source_git_commit,
        )
    end

    _log_mlflow_dataset!(
        mlflow,
        mlf,
        run;
        dataset_inputs=dataset_inputs,
        dataset_name=dataset_name,
        dataset_digest=dataset_digest,
        dataset_source_type=dataset_source_type,
        dataset_source=dataset_source,
        dataset_context=dataset_context,
    )

    logmetric = getproperty(mlflow, :logmetric)
    training_succeeded = false

    try
        result = train!(
            loss,
            relative_loss,
            model,
            data;
            learning_rate=learning_rate,
            optimizer_type=optimizer_type,
            epochs=epochs,
            batchsize=batchsize,
            shuffle=shuffle,
            reset_optimizer_each_epoch=reset_optimizer_each_epoch,
            save_model=save_model,
            model_save_path=model_save_path,
            on_epoch_end=(epoch, loss_value, display_loss) -> begin
                logmetric(mlf, run, "loss", Float64(loss_value); step=epoch)
                logmetric(mlf, run, "display_loss", Float64(display_loss); step=epoch)
                if !isnothing(on_epoch_end)
                    on_epoch_end(epoch, loss_value, display_loss)
                end
            end,
            kwargs...,
        )

        if upload_model_artifact && isfile(model_save_path)
            _upload_mlflow_artifact!(
                mlflow,
                mlf,
                run,
                model_save_path;
                artifact_path=model_artifact_path,
            )
        end

        training_succeeded = true
        return result
    finally
        status = training_succeeded ?
            getproperty(getproperty(mlflow, :RunStatus), :FINISHED) :
            getproperty(getproperty(mlflow, :RunStatus), :FAILED)
        try
            getproperty(mlflow, :updaterun)(
                mlf,
                run;
                status=status,
                end_time=_unix_milliseconds(),
            )
        catch
            training_succeeded && rethrow()
        end
    end
end

function train(
    scenario_generator::DFLScenarioGenerator,
    loss::LossFunction,
    data_set::DataSet,
    scenario_decoder::ScenarioDecoder,
    mu_schedule,
    rho_schedule,
    batch_size_schedule,
    step_size_schedule;
    epochs::Integer=1,
    opt=nothing,
    optimizer_type=Flux.Adam,
    display_iterations::Bool=false,
    verbose::Bool=display_iterations,
    relative_loss=nothing,
    display_plot::Bool=display_iterations,
    save_model::Bool=false,
    model_save_path::AbstractString="trained_model.jls",
    shuffle::Bool=false,
    rng::Random.AbstractRNG=Random.default_rng(),
    opt_state=nothing,
    reset_optimizer_each_epoch::Bool=false,
    on_epoch_end=nothing,
    kwargs...,
)
    learning_rate = _schedule_value(step_size_schedule, 1)
    batchsize = Int(_schedule_value(batch_size_schedule, 1))
    optimizer = isnothing(opt) ? _make_optimizer(optimizer_type, learning_rate) : opt
    model = scenario_generator.neural_net

    train_loss(predicted, reference) = loss(
        scenario_generator.program,
        predicted,
        reference,
        _schedule_value(mu_schedule, 1);
        rho=_schedule_value(rho_schedule, 1),
        kwargs...,
    )

    data = _contextual_training_samples(data_set, scenario_decoder)
    return train!(
        train_loss,
        relative_loss,
        model,
        data;
        opt=optimizer,
        epochs=epochs,
        batchsize=batchsize,
        display_iterations=display_iterations,
        verbose=verbose,
        display_plot=display_plot,
        save_model=save_model,
        model_save_path=model_save_path,
        shuffle=shuffle,
        rng=rng,
        opt_state=opt_state,
        reset_optimizer_each_epoch=reset_optimizer_each_epoch,
        on_epoch_end=on_epoch_end,
    )
end

function _training_samples(data::AbstractDict)
    pairs = collect(data)
    return [pair.first for pair in pairs], [pair.second for pair in pairs]
end

function _training_samples(data::Tuple)
    length(data) == 2 ||
        throw(ArgumentError("tuple training data must be `(inputs, targets)`."))
    return _column_samples(data[1]), _column_samples(data[2])
end

function _training_samples(data)
    pairs = collect(data)
    xs = Any[]
    ys = Any[]

    for pair in pairs
        if pair isa Pair
            push!(xs, pair.first)
            push!(ys, pair.second)
            continue
        end

        length(pair) == 2 ||
            throw(ArgumentError("training data entries must be `(input, target)` pairs."))
        push!(xs, pair[1])
        push!(ys, pair[2])
    end

    return xs, ys
end

_column_samples(data::AbstractMatrix) = [view(data, :, i) for i in axes(data, 2)]

function _column_samples(data::AbstractVector)
    all(sample -> sample isa Number, data) &&
        return [view(data, i:i) for i in eachindex(data)]
    return collect(data)
end

function _batch_data(samples, idxs)
    all(index -> _batchable_sample(samples[index]), idxs) ||
        return [samples[index] for index in idxs]
    return hcat((samples[i] for i in idxs)...)
end

_batchable_sample(sample) =
    sample isa Number || (sample isa AbstractArray && eltype(sample) <: Number)

function _mean_sample_loss(loss, model, x_batch, y_batch)
    return Statistics.mean(
        loss(model(_sample(x_batch, i)), _sample(y_batch, i))
        for i in 1:_sample_count(x_batch)
    )
end

_sample_count(data::AbstractMatrix) = size(data, 2)
_sample_count(data::AbstractVector) = length(data)

_sample(data::AbstractMatrix, index) = view(data, :, index:index)
_sample(data::AbstractVector, index) = data[index]

_sample_column(data::AbstractMatrix, index) = view(data, :, index:index)
_sample_column(data::AbstractVector, index) = view(data, index:index)

_float(value::Number) = Float64(value)
_float(value::AbstractArray) = Float64(only(value))

_unix_milliseconds() = round(Int64, Dates.datetime2unix(Dates.now()) * 1000)

function _log_mlflow_source_tags!(
    mlflow,
    mlf,
    run;
    source_name,
    source_type,
    source_git_commit,
)
    isdefined(mlflow, :setruntag) || return nothing
    setruntag = getproperty(mlflow, :setruntag)

    actual_source_name = if isnothing(source_name)
        isempty(PROGRAM_FILE) ? "ContextualDFL.train_with_mlflow!" : PROGRAM_FILE
    else
        source_name
    end

    setruntag(mlf, run, "mlflow.source.name", string(actual_source_name))
    setruntag(mlf, run, "mlflow.source.type", string(source_type))
    isnothing(source_git_commit) ||
        setruntag(mlf, run, "mlflow.source.git.commit", string(source_git_commit))

    return nothing
end

function _log_mlflow_dataset!(
    mlflow,
    mlf,
    run;
    dataset_inputs,
    dataset_name,
    dataset_digest,
    dataset_source_type,
    dataset_source,
    dataset_context,
)
    isdefined(mlflow, :loginputs) || return nothing

    inputs = if !isnothing(dataset_inputs)
        dataset_inputs
    elseif !any(isnothing, (dataset_name, dataset_digest, dataset_source))
        dataset = getproperty(mlflow, :Dataset)(
            string(dataset_name),
            string(dataset_digest),
            string(dataset_source_type),
            string(dataset_source),
            nothing,
            nothing,
        )
        tag = getproperty(mlflow, :Tag)("context", string(dataset_context))
        [getproperty(mlflow, :DatasetInput)([tag], dataset)]
    else
        return nothing
    end

    loginputs = getproperty(mlflow, :loginputs)
    try
        loginputs(mlf, run; datasets=inputs)
    catch error
        error isa MethodError || rethrow()
        loginputs(mlf, run, inputs)
    end

    return nothing
end

function _upload_mlflow_artifact!(mlflow, mlf, run, path; artifact_path)
    isdefined(mlflow, :uploadartifact) || return nothing
    uploadartifact = getproperty(mlflow, :uploadartifact)

    if applicable(uploadartifact, mlf, run, path)
        uploadartifact(mlf, run, path)
    elseif applicable(uploadartifact, mlf, run, path, artifact_path)
        uploadartifact(mlf, run, path, artifact_path)
    else
        uploadartifact(mlf, string(artifact_path), read(path))
    end

    return nothing
end

function _git_commit_or_nothing()
    try
        return strip(read(pipeline(`git rev-parse HEAD`; stderr=devnull), String))
    catch
        return nothing
    end
end

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

function _schedule_value(schedule, epoch)
    if applicable(schedule, epoch)
        return schedule(epoch)
    elseif applicable(schedule)
        return schedule()
    else
        return schedule
    end
end

function _contextual_training_samples(data_set::DataSet, scenario_decoder::ScenarioDecoder)
    x_samples = _column_samples(data_set.x_data)
    references = [
        scenario_decoder(_dataset_scenario_parameter(data_set, i))
        for i in eachindex(x_samples)
    ]
    return collect(zip(x_samples, references))
end

function _dataset_scenario_parameter(data_set::DataSet, index)
    pairs = Pair{Symbol,Any}[]
    _push_sample!(pairs, :W, data_set.xi_W_data, index)
    _push_sample!(pairs, :T, data_set.xi_T_data, index)
    _push_sample!(pairs, :h, data_set.xi_h_data, index)
    _push_sample!(pairs, :q, data_set.xi_q_data, index)
    return (; pairs...)
end

_push_sample!(pairs, name, data::Nothing, index) = pairs

function _push_sample!(pairs, name, data, index)
    push!(pairs, name => _sample_column(data, index))
    return pairs
end
