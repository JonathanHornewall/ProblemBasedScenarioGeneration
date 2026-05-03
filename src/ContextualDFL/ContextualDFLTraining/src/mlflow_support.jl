import MLFlowClient

const RunStatus = MLFlowClient.RunStatus
const MLFLOW_RETRY_ATTEMPTS = 8
const MLFLOW_RETRY_INITIAL_DELAY_SECONDS = 1.0
const MLFLOW_RETRY_BACKOFF = 1.5

function with_mlflow_retry(callback, operation)
    delay = MLFLOW_RETRY_INITIAL_DELAY_SECONDS
    for attempt in 1:MLFLOW_RETRY_ATTEMPTS
        try
            return callback()
        catch error
            attempt == MLFLOW_RETRY_ATTEMPTS && rethrow()
            @warn "MLflow $operation failed; retrying" attempt error=sprint(showerror, error)
            sleep(delay)
            delay *= MLFLOW_RETRY_BACKOFF
        end
    end
end

mutable struct NamedMLFlowClient
    client::MLFlowClient.MLFlow
    run_name::String
    tags::Dict{String,String}
    params::Dict{String,String}
    run::Any
end

function NamedMLFlowClient(; tracking_uri="", run_name, tags, params)
    client = isempty(string(tracking_uri)) ?
        MLFlowClient.MLFlow(; headers=mlflow_http_headers()) :
        MLFlowClient.MLFlow(string(tracking_uri); headers=mlflow_http_headers())
    return NamedMLFlowClient(
        client,
        string(run_name),
        string_dict(tags),
        string_dict(params),
        nothing,
    )
end

function createrun(mlf::NamedMLFlowClient, experiment_id; start_time=missing)
    run = with_mlflow_retry("create run") do
        MLFlowClient.createrun(
            mlf.client,
            string(experiment_id);
            run_name=mlf.run_name,
            start_time=start_time,
            tags=mlf.tags,
        )
    end
    mlf.run = run

    for key in sort!(collect(keys(mlf.params)))
        with_mlflow_retry("log param $key") do
            MLFlowClient.logparam(mlf.client, run, key, mlf.params[key])
        end
    end

    return run
end

function logparam(mlf::NamedMLFlowClient, run, key, value)
    return with_mlflow_retry("log param $key") do
        MLFlowClient.logparam(mlf.client, run, string(key), string(value))
    end
end

function logmetric(
    mlf::NamedMLFlowClient,
    run,
    key,
    value;
    step,
    timestamp=round(Int64, time() * 1000),
)
    return with_mlflow_retry("log metric $key") do
        MLFlowClient.logmetric(
            mlf.client,
            run,
            string(key),
            Float64(value);
            timestamp=Int64(timestamp),
            step=Int(step),
        )
    end
end

function logbatch(
    mlf::NamedMLFlowClient,
    run;
    metrics=MLFlowClient.Metric[],
    params=MLFlowClient.Param[],
    tags=MLFlowClient.Tag[],
)
    return with_mlflow_retry("log batch") do
        MLFlowClient.logbatch(mlf.client, run; metrics=metrics, params=params, tags=tags)
    end
end

function setruntag(mlf::NamedMLFlowClient, run, key, value)
    return with_mlflow_retry("set tag $key") do
        MLFlowClient.setruntag(mlf.client, run, string(key), string(value))
    end
end

function loginputs(mlf::NamedMLFlowClient, run; datasets)
    return with_mlflow_retry("log inputs") do
        MLFlowClient.loginputs(mlf.client, run, datasets)
    end
end

function loginputs(mlf::NamedMLFlowClient, run, datasets)
    return with_mlflow_retry("log inputs") do
        MLFlowClient.loginputs(mlf.client, run, datasets)
    end
end

const Dataset = MLFlowClient.Dataset
const DatasetInput = MLFlowClient.DatasetInput
const Metric = MLFlowClient.Metric
const Tag = MLFlowClient.Tag

function uploadartifact(mlf::NamedMLFlowClient, run, path)
    return uploadartifact(mlf, run, path, basename(path))
end

function uploadartifact(mlf::NamedMLFlowClient, run, path, artifact_path)
    return with_mlflow_retry("upload artifact $artifact_path") do
        MLFlowClient.uploadartifact(mlf.client, string(artifact_path), read(path))
    end
end

function uploadartifact(mlf::NamedMLFlowClient, artifact_path::AbstractString, data::Vector{UInt8})
    return with_mlflow_retry("upload artifact $artifact_path") do
        MLFlowClient.uploadartifact(mlf.client, string(artifact_path), data)
    end
end

function updaterun(mlf::NamedMLFlowClient, run; status, end_time=missing)
    return with_mlflow_retry("update run") do
        MLFlowClient.updaterun(mlf.client, run; status=status, end_time=end_time)
    end
end

function log_mlflow_metric!(mlf, run, key, value; timestamp, step)
    return logmetric(
        mlf,
        run,
        key,
        Float64(value);
        timestamp=Int64(timestamp),
        step=Int(step),
    )
end

function log_mlflow_epoch!(mlf, run, epoch, loss_value, _display_loss, metadata)
    timestamp = mlflow_unix_milliseconds()
    step = Int64(epoch)
    metrics = Metric[Metric("loss", Float64(loss_value), timestamp, step)]
    append!(metrics, mlflow_epoch_metadata_metrics(metadata; timestamp=timestamp, step=step))
    return logbatch(mlf, run; metrics=metrics)
end

function mlflow_epoch_metadata_metrics(metadata; timestamp, step)
    metadata isa NamedTuple || return Metric[]
    metrics = Metric[]

    for (metric_name, field_name) in (
        ("epoch_seconds", :epoch_seconds),
        ("epoch_mu_in", :mu_in),
        ("epoch_mu_ref", :mu_ref),
        ("epoch_iterations", :iterations),
    )
        field_name in keys(metadata) || continue
        value = getproperty(metadata, field_name)
        mlflow_metric_value(value) || continue
        push!(metrics, Metric(metric_name, Float64(value), Int64(timestamp), Int64(step)))
    end

    return metrics
end

function log_mlflow_params!(mlf, run, prefix::AbstractString, values)
    params = Dict{String,String}()
    flatten_mlflow_params!(params, prefix, values)

    for key in sort!(collect(keys(params)))
        logparam(mlf, run, key, params[key])
    end

    return nothing
end

function flatten_mlflow_params!(
    params::Dict{String,String},
    prefix::AbstractString,
    values::NamedTuple,
)
    for key in keys(values)
        flatten_mlflow_params!(params, join_mlflow_key(prefix, key), getproperty(values, key))
    end
    return params
end

function flatten_mlflow_params!(
    params::Dict{String,String},
    prefix::AbstractString,
    values::AbstractDict,
)
    for (key, value) in values
        flatten_mlflow_params!(params, join_mlflow_key(prefix, key), value)
    end
    return params
end

function flatten_mlflow_params!(params::Dict{String,String}, prefix::AbstractString, value)
    isempty(prefix) && return params
    mlflow_param_value(value) || return params
    params[prefix] = string(value)
    return params
end

function log_mlflow_evaluation_result!(mlf, run, name::AbstractString, result)
    if result isa NamedTuple || result isa AbstractDict
        metrics = mlflow_evaluation_field(result, :metrics, result)
        artifacts = mlflow_evaluation_field(result, :artifacts, nothing)
        log_mlflow_metrics!(mlf, run, name, metrics; step=0)
        log_mlflow_artifacts!(mlf, run, artifacts)
    elseif mlflow_metric_value(result)
        log_mlflow_metrics!(mlf, run, "", Dict(name => result); step=0)
    end

    return nothing
end

function mlflow_evaluation_field(values::NamedTuple, key::Symbol, default)
    return key in keys(values) ? getproperty(values, key) : default
end

function mlflow_evaluation_field(values::AbstractDict, key::Symbol, default)
    return haskey(values, key) ? values[key] : get(values, string(key), default)
end

function log_mlflow_metrics!(mlf, run, prefix::AbstractString, values; step::Integer)
    metrics = Dict{String,Float64}()
    flatten_mlflow_metrics!(metrics, prefix, values)

    timestamp = mlflow_unix_milliseconds()
    for key in sort!(collect(keys(metrics)))
        log_mlflow_metric!(
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

function flatten_mlflow_metrics!(
    metrics::Dict{String,Float64},
    prefix::AbstractString,
    values::NamedTuple,
)
    for key in keys(values)
        flatten_mlflow_metrics!(metrics, join_mlflow_key(prefix, key), getproperty(values, key))
    end
    return metrics
end

function flatten_mlflow_metrics!(
    metrics::Dict{String,Float64},
    prefix::AbstractString,
    values::AbstractDict,
)
    for (key, value) in values
        flatten_mlflow_metrics!(metrics, join_mlflow_key(prefix, key), value)
    end
    return metrics
end

function flatten_mlflow_metrics!(metrics::Dict{String,Float64}, prefix::AbstractString, value)
    isempty(prefix) && return metrics
    mlflow_metric_value(value) || return metrics
    metrics[prefix] = Float64(value)
    return metrics
end

function mlflow_metric_value(value)
    value isa Bool && return false
    value isa Number || return false
    float_value = try
        Float64(value)
    catch
        return false
    end
    return isfinite(float_value)
end

function log_mlflow_artifacts!(mlf, run, artifacts)
    artifacts === nothing && return nothing

    if artifacts isa AbstractString
        isfile(artifacts) && upload_mlflow_artifact!(mlf, run, artifacts; artifact_path=basename(artifacts))
        return nothing
    elseif artifacts isa Pair
        path = last(artifacts)
        path isa AbstractString && isfile(path) &&
            upload_mlflow_artifact!(mlf, run, path; artifact_path=string(first(artifacts)))
        return nothing
    elseif artifacts isa NamedTuple || artifacts isa AbstractDict
        for (name, path) in pairs(artifacts)
            path isa AbstractString && isfile(path) &&
                upload_mlflow_artifact!(mlf, run, path; artifact_path=string(name))
        end
        return nothing
    elseif artifacts isa AbstractVector || artifacts isa Tuple
        for artifact in artifacts
            log_mlflow_artifacts!(mlf, run, artifact)
        end
    end

    return nothing
end

function upload_mlflow_artifact!(mlf, run, path; artifact_path)
    if applicable(uploadartifact, mlf, run, path)
        uploadartifact(mlf, run, path)
    elseif applicable(uploadartifact, mlf, run, path, artifact_path)
        uploadartifact(mlf, run, path, artifact_path)
    else
        uploadartifact(mlf, string(artifact_path), read(path))
    end

    return nothing
end

function log_mlflow_source_tags!(
    mlf,
    run;
    source_name,
    source_type,
    source_git_commit,
)
    setruntag(mlf, run, "mlflow.source.name", string(source_name))
    setruntag(mlf, run, "mlflow.source.type", string(source_type))
    isnothing(source_git_commit) ||
        setruntag(mlf, run, "mlflow.source.git.commit", string(source_git_commit))

    return nothing
end

function log_mlflow_dataset!(
    mlf,
    run;
    dataset_inputs=nothing,
    dataset_name=nothing,
    dataset_digest=nothing,
    dataset_source_type="local",
    dataset_source=nothing,
    dataset_context="training",
)
    inputs = if !isnothing(dataset_inputs)
        dataset_inputs
    elseif !any(isnothing, (dataset_name, dataset_digest, dataset_source))
        dataset = Dataset(
            string(dataset_name),
            mlflow_dataset_digest_value(dataset_digest),
            string(dataset_source_type),
            string(dataset_source),
            nothing,
            nothing,
        )
        [DatasetInput([Tag("context", string(dataset_context))], dataset)]
    else
        return nothing
    end

    try
        loginputs(mlf, run; datasets=inputs)
    catch error
        error isa MethodError || rethrow()
        loginputs(mlf, run, inputs)
    end

    return nothing
end

function mlflow_dataset_digest_value(digest)
    value = string(digest)
    length(value) <= 36 && return value
    return bytes2hex(sha256(value))[1:32]
end

function tag_optional_mlflow_evaluation_error!(mlf, run, name, error)
    setruntag(
        mlf,
        run,
        "mlflow.optional_evaluation.$(name).error",
        sprint(showerror, error),
    )
    return nothing
end

function run_mlflow_evaluation_callbacks!(
    mlf,
    run,
    callbacks,
    train_result;
    optional::Bool,
)
    isempty(mlflow_callback_pairs(callbacks)) && return nothing

    for (name, callback) in mlflow_callback_pairs(callbacks)
        try
            result = applicable(callback, train_result) ? callback(train_result) : callback()
            log_mlflow_evaluation_result!(mlf, run, string(name), result)
        catch error
            optional || rethrow()
            tag_optional_mlflow_evaluation_error!(mlf, run, name, error)
        end
    end

    return nothing
end

mlflow_callback_pairs(callbacks::NamedTuple) = collect(pairs(callbacks))
mlflow_callback_pairs(callbacks::AbstractDict) = collect(pairs(callbacks))
mlflow_callback_pairs(callbacks::Tuple) = collect(pairs(callbacks))
mlflow_callback_pairs(callbacks::AbstractVector) = collect(pairs(callbacks))
mlflow_callback_pairs(::Nothing) = Pair{Symbol,Any}[]

function join_mlflow_key(prefix::AbstractString, key)
    key_text = string(key)
    return isempty(prefix) ? key_text : prefix * "_" * key_text
end

mlflow_unix_milliseconds() = round(Int64, time() * 1000)

function git_commit_or_nothing()
    try
        return strip(read(pipeline(`git rev-parse HEAD`; stderr=devnull), String))
    catch
        return nothing
    end
end

function mlflow_enabled(config)
    return Bool(config_value(config, :mlflow_enabled, false))
end

function mlflow_client_for_config(config)
    experiment_id = string(config_value(config, :mlflow_experiment_id, ""))
    isempty(experiment_id) &&
        throw(ArgumentError("mlflow_enabled=true requires config.mlflow_experiment_id"))

    run_name = string(
        config_value(
            config,
            :mlflow_run_name,
            config_value(config, :candidate_name, config_value(config, :run_id, "training-run")),
        ),
    )
    tracking_uri = string(config_value(config, :mlflow_tracking_uri, ""))

    mlf = NamedMLFlowClient(
        tracking_uri=tracking_uri,
        run_name=run_name,
        tags=mlflow_tags_for_config(config),
        params=mlflow_params_for_config(config),
    )
    return mlf, experiment_id
end

function mlflow_tags_for_config(config)
    tags = Dict{String,String}(
        "source" => "ContextualDFLTraining.gridsearch",
        "run_id" => string(config_value(config, :run_id, "")),
        "base_run_id" => string(config_value(config, :base_run_id, "")),
        "candidate_name" => string(config_value(config, :candidate_name, "")),
        "gridsearch_id" => string(config_value(config, :gridsearch_id, "")),
        "gridsearch_timestamp" => string(config_value(config, :gridsearch_timestamp, "")),
        "candidate_index" => string(config_value(config, :candidate_index, "")),
        "gridsearch_parent_run_id" => string(config_value(config, :mlflow_parent_run_id, "")),
        "mlflow.parentRunId" => string(config_value(config, :mlflow_parent_run_id, "")),
        "training_project" => "ContextualDFLTraining",
    )

    extra_tags = config_value(config, :mlflow_tags, nothing)
    add_string_pairs!(tags, extra_tags)
    return drop_empty_values(tags)
end

function mlflow_params_for_config(config)
    params = Dict{String,String}()
    config isa NamedTuple || return params

    for key in keys(config)
        key in (:mlflow_tags, :mlflow_tracking_uri) && continue
        value = getproperty(config, key)
        mlflow_param_value(value) || continue
        params["config_" * string(key)] = string(value)
    end

    return params
end

function config_value(config, key::Symbol, default)
    config isa NamedTuple || return default
    return key in keys(config) ? getproperty(config, key) : default
end

function string_dict(values)
    output = Dict{String,String}()
    add_string_pairs!(output, values)
    return output
end

function add_string_pairs!(output::Dict{String,String}, values::NamedTuple)
    for key in keys(values)
        output[string(key)] = string(getproperty(values, key))
    end
    return output
end

function add_string_pairs!(output::Dict{String,String}, values::AbstractDict)
    for (key, value) in values
        output[string(key)] = string(value)
    end
    return output
end

add_string_pairs!(output::Dict{String,String}, ::Nothing) = output

function add_string_pairs!(output::Dict{String,String}, values)
    throw(ArgumentError("MLflow tags/params must be a NamedTuple or Dict, got $(typeof(values))"))
end

function drop_empty_values(values::Dict{String,String})
    return Dict(key => value for (key, value) in values if !isempty(value))
end

mlflow_param_value(value) =
    value isa Number ||
    value isa Bool ||
    value isa Symbol ||
    value isa AbstractString

function mlflow_http_headers()
    return Dict("Connection" => "close")
end
