import Flux
import Dates
import Plots
import Random
import Serialization
import Statistics

# %%% Core training loop

"""
    train!(
        loss,
        relative_loss,
        stochastic_program,
        mu_schedule,
        nr_scenarios,
        model,
        data_set;
        kwargs...,
    )

Flux training loop for a contextual scenario dataset.

Each data point must have a context vector and a scenario-parameter collection.
At epoch `k`, `loss` is called as:

    loss(stochastic_program, model(context), scenario_parameters, mu_schedule[k]; nr_scenarios=nr_scenarios)
"""
function train!(
    loss,
    relative_loss,
    stochastic_program::StochasticProgram,
    mu_schedule::AbstractVector,
    nr_scenarios::Integer,
    model,
    data_set::ContextualDataSet;
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
    nr_scenarios > 0 || throw(ArgumentError("nr_scenarios must be positive."))
    length(mu_schedule) == epochs ||
        throw(ArgumentError("mu_schedule must have one value per epoch."))
    isempty(data_set) && throw(ArgumentError("training data must not be empty."))

    optimizer = isnothing(opt) ? _make_optimizer(optimizer_type, learning_rate) : opt
    state = isnothing(opt_state) ? Flux.setup(optimizer, model) : opt_state
    show_progress = display_iterations || verbose
    nr_scenarios = Int(nr_scenarios)

    history = NamedTuple[]
    displayed_epoch_losses = Float64[]

    for epoch_number in 1:epochs
        mu = mu_schedule[epoch_number]

        if reset_optimizer_each_epoch
            state = Flux.setup(optimizer, model)
        end

        show_progress && print("Epoch ", epoch_number)
        epoch_losses = Float64[]
        epoch_display_losses = Float64[]

        indices = shuffle ? Random.randperm(rng, length(data_set)) : collect(eachindex(data_set))
        for idxs_iter in Iterators.partition(indices, batchsize)
            idxs = collect(idxs_iter)

            loss_value, gradients = Flux.withgradient(model) do trainable_model
                Statistics.mean(
                    loss(
                        stochastic_program,
                        trainable_model(data_set[index].context),
                        data_set[index].scenario_parameters,
                        mu;
                        nr_scenarios=nr_scenarios,
                    )
                    for index in idxs
                )
            end
            Flux.update!(state, model, gradients[1])
            push!(epoch_losses, _float(loss_value))

            if show_progress
                display_loss_function = isnothing(relative_loss) ? loss : relative_loss
                display_loss = Statistics.mean(
                    display_loss_function(
                        stochastic_program,
                        model(data_set[index].context),
                        data_set[index].scenario_parameters,
                        mu;
                        nr_scenarios=nr_scenarios,
                    )
                    for index in idxs
                )
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
                mu=mu,
                loss=average_loss,
                display_loss=average_display_loss,
                iterations=length(epoch_losses),
            ),
        )
    end

    # %%% Optional model storage
    if save_model
        Serialization.serialize(model_save_path, model)
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

    return (; model=model, history=history, opt_state=state)
end

train!(
    loss,
    stochastic_program::StochasticProgram,
    mu_schedule::AbstractVector,
    nr_scenarios::Integer,
    model,
    data_set::ContextualDataSet;
    kwargs...,
) =
    train!(
        loss,
        nothing,
        stochastic_program,
        mu_schedule,
        nr_scenarios,
        model,
        data_set;
        kwargs...,
    )

# %%% MLflow logging, model artifacts, and git metadata

function train_with_mlflow!(
    mlf,
    experiment_id,
    loss,
    stochastic_program::StochasticProgram,
    mu_schedule::AbstractVector,
    nr_scenarios::Integer,
    model,
    data_set::ContextualDataSet;
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
    logparam(mlf, run, "nr_scenarios", string(nr_scenarios))
    logparam(mlf, run, "mu_schedule", string(collect(mu_schedule)))
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
            stochastic_program,
            mu_schedule,
            nr_scenarios,
            model,
            data_set;
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

# %%% Small core helpers

_float(value::Number) = Float64(value)
_float(value::AbstractArray) = Float64(only(value))

# %%% MLflow helper functions

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
