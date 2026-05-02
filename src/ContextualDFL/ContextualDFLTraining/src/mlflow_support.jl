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
