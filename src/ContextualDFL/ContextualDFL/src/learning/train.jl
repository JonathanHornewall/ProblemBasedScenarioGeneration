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
)
    epochs >= 0 || throw(ArgumentError("epochs must be non-negative."))
    batchsize > 0 || throw(ArgumentError("batchsize must be positive."))
    length(mu_in_schedule) == epochs ||
        throw(ArgumentError("mu_in_schedule must have one value per epoch."))
    length(mu_ref_schedule) == epochs ||
        throw(ArgumentError("mu_ref_schedule must have one value per epoch."))
    isempty(data_set) && throw(ArgumentError("training data must not be empty."))
    _validate_nr_scenarios(nr_scenarios)

    optimizer = isnothing(opt) ? _make_optimizer(optimizer_type, learning_rate) : opt
    state = isnothing(opt_state) ? Flux.setup(optimizer, neural_net) : opt_state
    show_progress = display_iterations || verbose
    loss_kwargs = _training_loss_kwargs(nr_scenarios)

    history = NamedTuple[]
    displayed_epoch_losses = Float64[]

    for epoch_number in 1:epochs
        epoch_started = time()
        mu_in = mu_in_schedule[epoch_number]
        mu_ref = mu_ref_schedule[epoch_number]

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
                        loss_kwargs...,
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

            if show_progress || !isnothing(relative_loss)
                display_loss_function = isnothing(relative_loss) ? loss : relative_loss
                display_loss = Statistics.mean(
                    display_loss_function(
                        neural_net(_context(data_set[index])),
                        _scenario_parameters(data_set[index]),
                        mu_in,
                        mu_ref;
                        loss_kwargs...,
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
        epoch_seconds = time() - epoch_started
        epoch_metadata = (;
            epoch=Int(epoch_number),
            mu=mu_in,
            mu_in=mu_in,
            mu_ref=mu_ref,
            iterations=length(epoch_losses),
            epoch_seconds=epoch_seconds,
        )

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
                loss=average_loss,
                display_loss=average_display_loss,
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

# %%% MLflow logging, model artifacts, and git metadata

function train_with_mlflow!(
    mlf,
    experiment_id,
    neural_net,
    loss,
    mu_in_schedule::AbstractVector,
    mu_ref_schedule::AbstractVector,
    data_set;
    kwargs...,
)
    return train_with_mlflow!(
        mlf,
        experiment_id,
        neural_net,
        loss,
        mu_in_schedule,
        data_set;
        mu_ref_schedule=mu_ref_schedule,
        kwargs...,
    )
end

function train_with_mlflow!(
    mlf,
    experiment_id,
    neural_net,
    loss,
    mu_in_schedule::AbstractVector,
    data_set;
    mu_ref_schedule=nothing,
    relative_loss=nothing,
    learning_rate=1e-3,
    optimizer_type=Flux.Adam,
    epochs::Integer=length(mu_in_schedule),
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
    experiment_spec=NamedTuple(),
    data_spec=NamedTuple(),
    model_spec=NamedTuple(),
    method_spec=NamedTuple(),
    evaluation_callbacks=NamedTuple(),
    optional_evaluation_callbacks=NamedTuple(),
    nr_scenarios=nothing,
    kwargs...,
)
    mu_ref_schedule = _default_mu_ref_schedule(mu_in_schedule, mu_ref_schedule)
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
    logged_nr_scenarios = _logged_nr_scenarios(loss, nr_scenarios)
    isnothing(logged_nr_scenarios) ||
        logparam(mlf, run, "nr_scenarios", string(logged_nr_scenarios))
    logparam(mlf, run, "mu_in_schedule", string(collect(mu_in_schedule)))
    logparam(mlf, run, "mu_ref_schedule", string(collect(mu_ref_schedule)))
    logparam(mlf, run, "shuffle", string(shuffle))
    logparam(mlf, run, "reset_optimizer_each_epoch", string(reset_optimizer_each_epoch))
    _log_mlflow_params!(mlflow, mlf, run, "experiment", experiment_spec)
    _log_mlflow_params!(mlflow, mlf, run, "data", data_spec)
    _log_mlflow_params!(mlflow, mlf, run, "model", model_spec)
    _log_mlflow_params!(mlflow, mlf, run, "method", method_spec)

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
            neural_net,
            loss,
            relative_loss,
            mu_in_schedule,
            mu_ref_schedule,
            data_set;
            learning_rate=learning_rate,
            optimizer_type=optimizer_type,
            epochs=epochs,
            batchsize=batchsize,
            shuffle=shuffle,
            reset_optimizer_each_epoch=reset_optimizer_each_epoch,
            save_model=save_model,
            model_save_path=model_save_path,
            on_epoch_end=(epoch, loss_value, display_loss, metadata) -> begin
                timestamp = _unix_milliseconds()
                _log_mlflow_metric!(
                    logmetric,
                    mlf,
                    run,
                    "loss",
                    loss_value;
                    timestamp=timestamp,
                    step=epoch,
                )
                _log_mlflow_metric!(
                    logmetric,
                    mlf,
                    run,
                    "display_loss",
                    display_loss;
                    timestamp=timestamp,
                    step=epoch,
                )
                _log_mlflow_epoch_metadata!(
                    logmetric,
                    mlf,
                    run,
                    metadata;
                    timestamp=timestamp,
                    step=epoch,
                )
                if !isnothing(on_epoch_end)
                    _call_epoch_callback(
                        on_epoch_end,
                        epoch,
                        loss_value,
                        display_loss,
                        metadata,
                    )
                end
            end,
            nr_scenarios=nr_scenarios,
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

        _run_mlflow_evaluation_callbacks!(
            mlflow,
            mlf,
            run,
            evaluation_callbacks,
            result;
            optional=false,
        )
        _run_mlflow_evaluation_callbacks!(
            mlflow,
            mlf,
            run,
            optional_evaluation_callbacks,
            result;
            optional=true,
        )

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

_default_mu_ref_schedule(mu_in_schedule, mu_ref_schedule) =
    isnothing(mu_ref_schedule) ? mu_in_schedule : mu_ref_schedule

function _logged_nr_scenarios(loss, nr_scenarios)
    isnothing(nr_scenarios) || return Int(nr_scenarios)
    hasproperty(loss, :nr_scenarios) || return nothing
    return getproperty(loss, :nr_scenarios)
end

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

# %%% MLflow helper functions

# Explicit Unix epoch milliseconds. Do not derive MLflow timestamps from
# timezone-local DateTime values; MLflow expects epoch milliseconds.
_unix_milliseconds() = round(Int64, time() * 1000)

function _log_mlflow_metric!(logmetric, mlf, run, key, value; timestamp, step)
    return logmetric(
        mlf,
        run,
        key,
        Float64(value);
        timestamp=Int64(timestamp),
        step=Int(step),
    )
end

function _log_mlflow_params!(mlflow, mlf, run, prefix::AbstractString, values)
    isdefined(mlflow, :logparam) || return nothing
    params = Dict{String,String}()
    _flatten_mlflow_params!(params, prefix, values)

    logparam = getproperty(mlflow, :logparam)
    for key in sort!(collect(keys(params)))
        logparam(mlf, run, key, params[key])
    end

    return nothing
end

function _flatten_mlflow_params!(params::Dict{String,String}, prefix::AbstractString, values::NamedTuple)
    for key in keys(values)
        _flatten_mlflow_params!(params, _join_mlflow_key(prefix, key), getproperty(values, key))
    end
    return params
end

function _flatten_mlflow_params!(params::Dict{String,String}, prefix::AbstractString, values::AbstractDict)
    for (key, value) in values
        _flatten_mlflow_params!(params, _join_mlflow_key(prefix, key), value)
    end
    return params
end

function _flatten_mlflow_params!(params::Dict{String,String}, prefix::AbstractString, value)
    isempty(prefix) && return params
    _mlflow_param_value(value) || return params
    params[prefix] = string(value)
    return params
end

_mlflow_param_value(value) =
    value isa Number ||
    value isa Bool ||
    value isa Symbol ||
    value isa AbstractString

function _log_mlflow_epoch_metadata!(logmetric, mlf, run, metadata; timestamp, step)
    metadata isa NamedTuple || return nothing

    for (metric_name, field_name) in (
        ("epoch_seconds", :epoch_seconds),
        ("epoch_mu", :mu),
        ("epoch_mu_in", :mu_in),
        ("epoch_mu_ref", :mu_ref),
        ("epoch_iterations", :iterations),
    )
        haskey(metadata, field_name) || continue
        value = getproperty(metadata, field_name)
        _mlflow_metric_value(value) || continue
        _log_mlflow_metric!(
            logmetric,
            mlf,
            run,
            metric_name,
            value;
            timestamp=timestamp,
            step=step,
        )
    end

    return nothing
end

function _run_mlflow_evaluation_callbacks!(
    mlflow,
    mlf,
    run,
    callbacks,
    train_result;
    optional::Bool,
)
    isempty(_mlflow_callback_pairs(callbacks)) && return nothing

    for (name, callback) in _mlflow_callback_pairs(callbacks)
        try
            result = if applicable(callback, train_result)
                callback(train_result)
            else
                callback()
            end
            _log_mlflow_evaluation_result!(mlflow, mlf, run, string(name), result)
        catch error
            optional || rethrow()
            _tag_optional_evaluation_error!(mlflow, mlf, run, name, error)
        end
    end

    return nothing
end

_mlflow_callback_pairs(callbacks::NamedTuple) = collect(pairs(callbacks))
_mlflow_callback_pairs(callbacks::AbstractDict) = collect(pairs(callbacks))
_mlflow_callback_pairs(callbacks::Tuple) = collect(pairs(callbacks))
_mlflow_callback_pairs(callbacks::AbstractVector) = collect(pairs(callbacks))
_mlflow_callback_pairs(::Nothing) = Pair{Symbol,Any}[]

function _log_mlflow_evaluation_result!(mlflow, mlf, run, name::AbstractString, result)
    if result isa NamedTuple || result isa AbstractDict
        metrics = _evaluation_field(result, :metrics, result)
        artifacts = _evaluation_field(result, :artifacts, nothing)
        _log_mlflow_metrics!(mlflow, mlf, run, name, metrics; step=0)
        _log_mlflow_artifacts!(mlflow, mlf, run, artifacts)
    elseif _mlflow_metric_value(result)
        _log_mlflow_metrics!(mlflow, mlf, run, "", Dict(name => result); step=0)
    end

    return nothing
end

function _evaluation_field(values::NamedTuple, key::Symbol, default)
    return haskey(values, key) ? getproperty(values, key) : default
end

function _evaluation_field(values::AbstractDict, key::Symbol, default)
    return haskey(values, key) ? values[key] : get(values, string(key), default)
end

function _log_mlflow_metrics!(mlflow, mlf, run, prefix::AbstractString, values; step::Integer)
    isdefined(mlflow, :logmetric) || return nothing
    metrics = Dict{String,Float64}()
    _flatten_mlflow_metrics!(metrics, prefix, values)

    logmetric = getproperty(mlflow, :logmetric)
    timestamp = _unix_milliseconds()
    for key in sort!(collect(keys(metrics)))
        _log_mlflow_metric!(
            logmetric,
            mlf,
            run,
            key,
            metrics[key];
            timestamp=timestamp,
            step=step,
        )
    end

    return nothing
end

function _flatten_mlflow_metrics!(metrics::Dict{String,Float64}, prefix::AbstractString, values::NamedTuple)
    for key in keys(values)
        _flatten_mlflow_metrics!(metrics, _join_mlflow_key(prefix, key), getproperty(values, key))
    end
    return metrics
end

function _flatten_mlflow_metrics!(metrics::Dict{String,Float64}, prefix::AbstractString, values::AbstractDict)
    for (key, value) in values
        _flatten_mlflow_metrics!(metrics, _join_mlflow_key(prefix, key), value)
    end
    return metrics
end

function _flatten_mlflow_metrics!(metrics::Dict{String,Float64}, prefix::AbstractString, value)
    isempty(prefix) && return metrics
    _mlflow_metric_value(value) || return metrics
    metrics[prefix] = Float64(value)
    return metrics
end

function _mlflow_metric_value(value)
    value isa Bool && return false
    value isa Number || return false
    float_value = try
        Float64(value)
    catch
        return false
    end
    return isfinite(float_value)
end

function _log_mlflow_artifacts!(mlflow, mlf, run, artifacts)
    artifacts === nothing && return nothing

    if artifacts isa AbstractString
        isfile(artifacts) && _upload_mlflow_artifact!(mlflow, mlf, run, artifacts; artifact_path=basename(artifacts))
        return nothing
    elseif artifacts isa Pair
        path = last(artifacts)
        path isa AbstractString && isfile(path) &&
            _upload_mlflow_artifact!(mlflow, mlf, run, path; artifact_path=string(first(artifacts)))
        return nothing
    elseif artifacts isa NamedTuple || artifacts isa AbstractDict
        for (name, path) in pairs(artifacts)
            path isa AbstractString && isfile(path) &&
                _upload_mlflow_artifact!(mlflow, mlf, run, path; artifact_path=string(name))
        end
        return nothing
    elseif artifacts isa AbstractVector || artifacts isa Tuple
        for artifact in artifacts
            _log_mlflow_artifacts!(mlflow, mlf, run, artifact)
        end
    end

    return nothing
end

function _tag_optional_evaluation_error!(mlflow, mlf, run, name, error)
    isdefined(mlflow, :setruntag) || return nothing
    getproperty(mlflow, :setruntag)(
        mlf,
        run,
        "mlflow.optional_evaluation.$(name).error",
        sprint(showerror, error),
    )
    return nothing
end

function _join_mlflow_key(prefix::AbstractString, key)
    key_text = string(key)
    return isempty(prefix) ? key_text : prefix * "_" * key_text
end

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
