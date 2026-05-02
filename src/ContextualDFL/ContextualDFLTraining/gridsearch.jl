#!/usr/bin/env julia

using Dates
using Distributed
import MLFlowClient
using Sockets
using Statistics

include(joinpath(@__DIR__, "src", "grid_config.jl"))
include(joinpath(@__DIR__, "src", "csv_results.jl"))

const DEFAULT_REMOTE_PROJECT =
    "/home/rwl/ProblemBasedScenarioGeneration/src/ContextualDFL/ContextualDFLTraining"
const DEFAULT_REMOTE_JULIA = "/home/rwl/.juliaup/bin/julia"
const DEFAULT_MLFLOW_EXPERIMENT_ID = "2"
const DEFAULT_MLFLOW_EXPERIMENT_NAME = "ContextualDFLTraining"

function _contextualdfltraining_remote_eval(config)
    started_at = unix_milliseconds()
    try
        return Main.ContextualDFLTraining.train_and_evaluate(config)
    catch error
        return (;
            status="worker_error",
            run_id=(config isa NamedTuple && :run_id in keys(config)) ? config.run_id : "",
            config=config,
            worker=(;
                worker_id=Distributed.myid(),
                hostname=Sockets.gethostname(),
                pid=getpid(),
                julia_version=string(VERSION),
            ),
            final_metrics=NamedTuple(),
            epoch_history=Dict{Symbol,Any}[],
            error=sprint(showerror, error, catch_backtrace()),
            started_at=started_at,
            finished_at=unix_milliseconds(),
            elapsed_seconds=0.0,
        )
    end
end

function env_int(name, default)
    value = get(ENV, name, string(default))
    parsed = tryparse(Int, value)
    parsed === nothing && error("ENV[$name] must be an integer, got: $value")
    return parsed
end

function env_worker_count(name, default)
    value = lowercase(strip(get(ENV, name, string(default))))
    value == "auto" && return :auto

    parsed = tryparse(Int, value)
    parsed === nothing && error("ENV[$name] must be an integer or auto, got: $value")
    return parsed
end

function env_float(name, default)
    value = get(ENV, name, string(default))
    parsed = tryparse(Float64, value)
    parsed === nothing && error("ENV[$name] must be a number, got: $value")
    return parsed
end

function remote_worker_specs()
    return [
        ("rwl@gcp-big", env_worker_count("GCP_BIG_WORKERS", :auto)),
        ("rwl@gcp-small", env_worker_count("GCP_SMALL_WORKERS", :auto)),
    ]
end

function env_flag(name, default=false)
    value = lowercase(get(ENV, name, default ? "1" : "0"))
    return value in ("1", "true", "yes", "y")
end

function grid_mlflow_settings()
    enabled = env_flag("MLFLOW_ENABLED", true)
    return (;
        enabled=enabled,
        experiment_id=get(ENV, "MLFLOW_EXPERIMENT_ID", DEFAULT_MLFLOW_EXPERIMENT_ID),
        experiment_name=get(ENV, "MLFLOW_EXPERIMENT_NAME", DEFAULT_MLFLOW_EXPERIMENT_NAME),
        tracking_uri=get(ENV, "MLFLOW_TRACKING_URI", ""),
        upload_model_artifact=env_flag("MLFLOW_UPLOAD_MODEL_ARTIFACTS", false),
    )
end

function validate_mlflow_settings(settings)
    if settings.enabled && isempty(settings.experiment_id)
        error("MLFLOW_ENABLED=true requires MLFLOW_EXPERIMENT_ID.")
    end
    return nothing
end

function mlflow_client(settings)
    return isempty(string(settings.tracking_uri)) ?
        MLFlowClient.MLFlow(; headers=mlflow_http_headers()) :
        MLFlowClient.MLFlow(string(settings.tracking_uri); headers=mlflow_http_headers())
end

function ensure_clean_worker_start!()
    nprocs() == 1 ||
        error("Refusing to run with pre-existing workers. Start Julia without -p or --machine-file.")
end

function sync_code!()
    if env_flag("SKIP_SYNC", false)
        println("Skipping code sync because SKIP_SYNC is set.")
        return nothing
    end

    sync_script = joinpath(homedir(), "sync-julia-code.sh")
    isfile(sync_script) || error("sync script not found: $sync_script")
    println("Syncing code to remote machines with $sync_script")
    run(Cmd(`$sync_script`; dir=homedir()))
    return nothing
end

function add_remote_workers!()
    remote_project = get(ENV, "REMOTE_CONTEXTUAL_DFL_TRAINING_PROJECT", DEFAULT_REMOTE_PROJECT)
    remote_julia = get(ENV, "REMOTE_JULIA", DEFAULT_REMOTE_JULIA)

    for (host, count) in remote_worker_specs()
        count isa Integer && count <= 0 && continue
        println("Adding $count worker(s) on $host")
        addprocs(
            [(host, count)];
            exename=remote_julia,
            exeflags="--project=$(remote_project)",
            dir=remote_project,
            tunnel=true,
        )
    end

    remote_worker_ids = setdiff(workers(), [1])
    isempty(remote_worker_ids) && error("No remote workers were added.")
    return remote_worker_ids
end

function load_worker_stdlibs!()
    for process_id in procs()
        remotecall_fetch(process_id) do
            Core.eval(Main, :(using Dates))
            Core.eval(Main, :(using Distributed))
            Core.eval(Main, :(using Pkg))
            Core.eval(Main, :(using Sockets))
            return nothing
        end
    end
end

function assert_remote_only_workers!(remote_worker_ids)
    local_hostname = Sockets.gethostname()
    worker_hosts = Dict(
        worker => remotecall_fetch(() -> Sockets.gethostname(), worker) for
        worker in remote_worker_ids
    )
    local_workers = [
        worker for (worker, hostname) in worker_hosts if hostname == local_hostname
    ]

    isempty(local_workers) ||
        error("Refusing to run training on local worker(s): $(local_workers)")

    host_summary = join(
        ["$(worker)=>$(worker_hosts[worker])" for worker in sort(remote_worker_ids)],
        ", ",
    )
    println("Workers online: ", length(remote_worker_ids), " [", host_summary, "]")
    return worker_hosts
end

function load_training_project_on_workers!(remote_worker_ids)
    println("Instantiating and loading ContextualDFLTraining on remote workers")
    for worker in remote_worker_ids
        metadata = remotecall_fetch(worker) do
            Pkg.instantiate()
            Core.eval(Main, :(using ContextualDFLTraining))
            return (;
                worker_id=Distributed.myid(),
                hostname=Sockets.gethostname(),
                pid=getpid(),
            )
        end
        println("Loaded worker $(metadata.worker_id) on $(metadata.hostname), pid $(metadata.pid)")
    end
end

function define_remote_eval!()
    definition = quote
        function _contextualdfltraining_remote_eval(config)
            started_at = round(Int64, time() * 1000)
            try
                return Main.ContextualDFLTraining.train_and_evaluate(config)
            catch error
                return (;
                    status="worker_error",
                    run_id=(config isa NamedTuple && :run_id in keys(config)) ? config.run_id : "",
                    config=config,
                    worker=(;
                        worker_id=Distributed.myid(),
                        hostname=Sockets.gethostname(),
                        pid=getpid(),
                        julia_version=string(VERSION),
                    ),
                    final_metrics=NamedTuple(),
                    epoch_history=Dict{Symbol,Any}[],
                    error=sprint(showerror, error, catch_backtrace()),
                    started_at=started_at,
                    finished_at=round(Int64, time() * 1000),
                    elapsed_seconds=0.0,
                )
            end
        end
    end

    for worker in workers()
        remotecall_fetch(Core.eval, worker, Main, definition)
    end
end

function coordinator_error_result(config, worker, worker_hosts, status, error, backtrace, elapsed_seconds)
    return (;
        status=status,
        run_id=(config isa NamedTuple && :run_id in keys(config)) ? config.run_id : "",
        config=config,
        worker=(;
            worker_id=worker,
            hostname=get(worker_hosts, worker, ""),
            pid=missing,
            julia_version="",
        ),
        final_metrics=NamedTuple(),
        epoch_history=Dict{Symbol,Any}[],
        error=sprint(showerror, error, backtrace),
        started_at=unix_milliseconds(),
        finished_at=unix_milliseconds(),
        elapsed_seconds=elapsed_seconds,
    )
end

function transport_failure(error)
    return error isa Distributed.ProcessExitedException ||
        error isa EOFError ||
        error isa Base.IOError
end

function run_grid_on_remote_workers(remote_worker_ids, configs, worker_hosts)
    results = Vector{Any}(undef, length(configs))
    pending = Tuple{Int,Any}[(index, config) for (index, config) in enumerate(configs)]
    pending_lock = ReentrantLock()

    function next_pending!()
        lock(pending_lock)
        try
            isempty(pending) && return nothing
            return popfirst!(pending)
        finally
            unlock(pending_lock)
        end
    end

    tasks = [
        @async begin
            while true
                item = next_pending!()
                item === nothing && break

                index, config = item
                started = time()
                try
                    results[index] = remotecall_fetch(
                        _contextualdfltraining_remote_eval,
                        worker,
                        config,
                    )
                catch error
                    elapsed_seconds = time() - started
                    if transport_failure(error)
                        println(
                            "Worker $worker exited while running $(config.run_id); recording worker_lost and continuing.",
                        )
                        mark_mlflow_run_failed(config, "worker_lost")
                        results[index] = coordinator_error_result(
                            config,
                            worker,
                            worker_hosts,
                            "worker_lost",
                            error,
                            catch_backtrace(),
                            elapsed_seconds,
                        )
                        break
                    end

                    results[index] = coordinator_error_result(
                        config,
                        worker,
                        worker_hosts,
                        "coordinator_error",
                        error,
                        catch_backtrace(),
                        elapsed_seconds,
                    )
                end
            end
        end for worker in remote_worker_ids
    ]

    foreach(wait, tasks)

    for (index, config) in enumerate(configs)
        if !isassigned(results, index)
            results[index] = (;
                status="not_started",
                run_id=config.run_id,
                config=config,
                worker=NamedTuple(),
                final_metrics=NamedTuple(),
                epoch_history=Dict{Symbol,Any}[],
                error="No remote worker remained available for this configuration.",
                started_at=unix_milliseconds(),
                finished_at=unix_milliseconds(),
                elapsed_seconds=0.0,
            )
        end
    end

    return results
end

function mark_mlflow_run_failed(config, reason)
    config isa NamedTuple || return nothing
    (:mlflow_enabled in keys(config) && config.mlflow_enabled) || return nothing

    try
        uri = string(getproperty(config, :mlflow_tracking_uri))
        experiment_id = string(getproperty(config, :mlflow_experiment_id))
        run_name = string(getproperty(config, :mlflow_run_name))
        mlf = isempty(uri) ?
            MLFlowClient.MLFlow(; headers=mlflow_http_headers()) :
            MLFlowClient.MLFlow(uri; headers=mlflow_http_headers())
        filter = "tags.candidate_name = \"$(mlflow_filter_escape(run_name))\" and attributes.status = \"RUNNING\""
        runs, _ = MLFlowClient.searchruns(
            mlf;
            experiment_ids=[experiment_id],
            filter=filter,
            max_results=100,
        )

        for run in runs
            MLFlowClient.setruntag(mlf, run, "ContextualDFLTraining.coordinator_status", reason)
            MLFlowClient.updaterun(
                mlf,
                run;
                status=MLFlowClient.RunStatus.FAILED,
                end_time=unix_milliseconds(),
            )
        end
    catch error
        println("Could not mark MLflow run failed for $(config.run_id): ", sprint(showerror, error))
    end

    return nothing
end

function create_mlflow_grid_parent_run(settings, grid_id, timestamp, configs, worker_hosts)
    settings.enabled || return nothing

    mlf = mlflow_client(settings)
    tags = Dict(
        "gridsearch_id" => grid_id,
        "gridsearch_timestamp" => timestamp,
        "gridsearch_role" => "parent",
        "training_project" => "ContextualDFLTraining",
        "mlflow.experiment.name" => string(settings.experiment_name),
        "mlflow.source.name" => "ContextualDFLTraining/gridsearch.jl",
        "mlflow.source.type" => "LOCAL",
    )
    git_commit = git_commit_or_empty()
    isempty(git_commit) || (tags["mlflow.source.git.commit"] = git_commit)

    parent_params = grid_parent_params(grid_id, timestamp, configs, worker_hosts)

    run = MLFlowClient.createrun(
        mlf,
        string(settings.experiment_id);
        run_name=grid_id,
        start_time=unix_milliseconds(),
        tags=tags,
    )

    for (key, value) in parent_params
        MLFlowClient.logparam(mlf, run, key, value)
    end

    return (; client=mlf, run=run)
end

function close_mlflow_grid_parent_run(parent, results)
    parent === nothing && return nothing

    success = all(result -> getproperty(result, :status) == "ok", results)
    mark_failed_mlflow_candidates!(results)
    log_grid_aggregate_metrics!(parent.client, parent.run, results)
    MLFlowClient.updaterun(
        parent.client,
        parent.run;
        status=success ? MLFlowClient.RunStatus.FINISHED : MLFlowClient.RunStatus.FAILED,
        end_time=unix_milliseconds(),
    )
    return nothing
end

function mark_failed_mlflow_candidates!(results)
    for result in results
        getproperty(result, :status) == "ok" && continue
        config = getproperty(result, :config)
        mark_mlflow_run_failed(config, string(getproperty(result, :status)))
    end
    return nothing
end

function fail_mlflow_grid_parent_run(parent)
    parent === nothing && return nothing
    try
        MLFlowClient.updaterun(
            parent.client,
            parent.run;
            status=MLFlowClient.RunStatus.FAILED,
            end_time=unix_milliseconds(),
        )
    catch error
        println("Could not mark parent MLflow run failed: ", sprint(showerror, error))
    end
    return nothing
end

function grid_parent_params(grid_id, timestamp, configs, worker_hosts)
    params = Dict{String,String}(
        "gridsearch_id" => grid_id,
        "gridsearch_timestamp" => timestamp,
        "grid_candidate_count" => string(length(configs)),
        "grid_selected_grid" => env_flag("GRIDSEARCH_SMOKE", false) ? "smoke" : "default",
        "grid_worker_count" => string(length(worker_hosts)),
        "grid_worker_hosts" => join(sort!(unique(collect(values(worker_hosts)))), ","),
    )

    for (key, value) in grid_constant_config_values(configs)
        params["grid_constant_" * string(key)] = string(value)
    end

    variable_keys = grid_variable_config_keys(configs)
    params["grid_variable_keys"] = join(string.(variable_keys), ",")
    for key in variable_keys
        values = sort!(unique([string(getproperty(config, key)) for config in configs]))
        params["grid_variable_" * string(key) * "_count"] = string(length(values))
        length(values) <= 20 && (params["grid_variable_" * string(key) * "_values"] = join(values, ","))
    end

    return params
end

function grid_constant_config_values(configs)
    isempty(configs) && return Pair{Symbol,Any}[]
    constants = Pair{Symbol,Any}[]
    first_config = first(configs)

    for key in keys(first_config)
        value = getproperty(first_config, key)
        mlflow_scalar_value(value) || continue
        all(config -> hasproperty(config, key) && getproperty(config, key) == value, configs) ||
            continue
        push!(constants, key => value)
    end

    return constants
end

function grid_variable_config_keys(configs)
    isempty(configs) && return Symbol[]
    variable_keys = Symbol[]
    first_config = first(configs)

    for key in keys(first_config)
        value = getproperty(first_config, key)
        mlflow_scalar_value(value) || continue
        all(config -> hasproperty(config, key) && getproperty(config, key) == value, configs) &&
            continue
        push!(variable_keys, key)
    end

    return sort!(variable_keys; by=String)
end

function log_grid_aggregate_metrics!(mlf, run, results)
    metric_keys = Set{Symbol}()
    for result in results
        getproperty(result, :status) == "ok" || continue
        metrics = getproperty(result, :final_metrics)
        metrics isa NamedTuple || continue
        for key in keys(metrics)
            value = getproperty(metrics, key)
            mlflow_numeric_metric(value) && push!(metric_keys, key)
        end
    end

    for key in sort!(collect(metric_keys); by=String)
        values = Float64[
            Float64(getproperty(getproperty(result, :final_metrics), key)) for result in results if
            getproperty(result, :status) == "ok" &&
            getproperty(result, :final_metrics) isa NamedTuple &&
            hasproperty(getproperty(result, :final_metrics), key) &&
            mlflow_numeric_metric(getproperty(getproperty(result, :final_metrics), key))
        ]
        isempty(values) && continue

        prefix = "grid_" * string(key)
        timestamp = unix_milliseconds()
        MLFlowClient.logmetric(
            mlf,
            run,
            prefix * "_mean",
            mean(values);
            timestamp=timestamp,
            step=0,
        )
        MLFlowClient.logmetric(
            mlf,
            run,
            prefix * "_median",
            median(values);
            timestamp=timestamp,
            step=0,
        )
        MLFlowClient.logmetric(
            mlf,
            run,
            prefix * "_min",
            minimum(values);
            timestamp=timestamp,
            step=0,
        )
        MLFlowClient.logmetric(
            mlf,
            run,
            prefix * "_max",
            maximum(values);
            timestamp=timestamp,
            step=0,
        )
        MLFlowClient.logmetric(
            mlf,
            run,
            prefix * "_std",
            length(values) > 1 ? std(values) : 0.0;
            timestamp=timestamp,
            step=0,
        )
    end

    return nothing
end

function mlflow_parent_run_id(parent)
    parent === nothing && return ""
    try
        return string(parent.run.info.run_id)
    catch
        return ""
    end
end

mlflow_scalar_value(value) =
    value isa Number ||
    value isa Bool ||
    value isa Symbol ||
    value isa AbstractString

function mlflow_numeric_metric(value)
    value isa Bool && return false
    value isa Number || return false
    float_value = try
        Float64(value)
    catch
        return false
    end
    return isfinite(float_value)
end

function git_commit_or_empty()
    try
        return strip(read(pipeline(`git rev-parse HEAD`; stderr=devnull), String))
    catch
        return ""
    end
end

function mlflow_http_headers()
    return Dict("Connection" => "close")
end

function mlflow_filter_escape(value)
    return replace(string(value), "\\" => "\\\\", "\"" => "\\\"")
end

function selected_grid()
    overrides = grid_overrides_from_env()
    if env_flag("GRIDSEARCH_SMOKE", false)
        return smoke_grid(; overrides...)
    end
    return default_grid(; overrides...)
end

function grid_overrides_from_env()
    return (;
        optimality_evaluation=env_flag(
            "GRID_OPTIMALITY_EVALUATION",
            DEFAULT_RUN_SETTINGS.optimality_evaluation,
        ),
        optimality_test_sample_count=env_int(
            "GRID_OPTIMALITY_TEST_SAMPLE_COUNT",
            DEFAULT_RUN_SETTINGS.optimality_test_sample_count,
        ),
        optimality_train_sample_count=env_int(
            "GRID_OPTIMALITY_TRAIN_SAMPLE_COUNT",
            DEFAULT_RUN_SETTINGS.optimality_train_sample_count,
        ),
        optimality_validation_sample_count=env_int(
            "GRID_OPTIMALITY_VALIDATION_SAMPLE_COUNT",
            DEFAULT_RUN_SETTINGS.optimality_validation_sample_count,
        ),
        optimality_mu=env_float("GRID_OPTIMALITY_MU", DEFAULT_RUN_SETTINGS.optimality_mu),
    )
end

function gridsearch_id(timestamp::AbstractString)
    return "gridsearch_" * timestamp
end

function candidate_tag(index::Integer)
    return "candidate_" * lpad(string(index), 4, "0")
end

function base_run_id(config, index::Integer)
    if config isa NamedTuple && :run_id in keys(config)
        return string(config.run_id)
    end
    return candidate_tag(index)
end

function annotate_grid_config(
    config,
    index::Integer,
    timestamp::AbstractString,
    mlflow_settings=grid_mlflow_settings(),
    parent_run_id::AbstractString="",
    coordinator_hostname::AbstractString=Sockets.gethostname(),
)
    grid_id = gridsearch_id(timestamp)
    candidate = candidate_tag(index)
    previous_run_id = base_run_id(config, index)
    candidate_name = grid_id * "__" * candidate * "__" * previous_run_id

    return merge(
        config,
        (;
            run_id=candidate_name,
            base_run_id=previous_run_id,
            gridsearch_id=grid_id,
            gridsearch_timestamp=timestamp,
            candidate_index=Int(index),
            candidate_name=candidate_name,
            mlflow_enabled=mlflow_settings.enabled,
            mlflow_experiment_id=mlflow_settings.experiment_id,
            mlflow_experiment_name=mlflow_settings.experiment_name,
            mlflow_tracking_uri=mlflow_settings.tracking_uri,
            mlflow_upload_model_artifact=mlflow_settings.upload_model_artifact,
            mlflow_parent_run_id=parent_run_id,
            mlflow_run_name=candidate_name,
            coordinator_hostname=coordinator_hostname,
            mlflow_tags=(;
                gridsearch_id=grid_id,
                gridsearch_timestamp=timestamp,
                candidate_index=Int(index),
                base_run_id=previous_run_id,
                candidate_name=candidate_name,
                gridsearch_parent_run_id=parent_run_id,
                mlflow_parentRunId=parent_run_id,
                gridsearch_role="candidate",
            ),
        ),
    )
end

function annotate_grid_configs(
    configs,
    timestamp::AbstractString,
    mlflow_settings=grid_mlflow_settings(),
    parent_run_id::AbstractString="",
    coordinator_hostname::AbstractString=Sockets.gethostname(),
)
    return [
        annotate_grid_config(
            config,
            index,
            timestamp,
            mlflow_settings,
            parent_run_id,
            coordinator_hostname,
        ) for
        (index, config) in enumerate(configs)
    ]
end

function main()
    ensure_clean_worker_start!()
    sync_code!()
    remote_worker_ids = add_remote_workers!()
    load_worker_stdlibs!()
    worker_hosts = assert_remote_only_workers!(remote_worker_ids)
    load_training_project_on_workers!(remote_worker_ids)
    define_remote_eval!()

    timestamp = result_timestamp()
    grid_id = gridsearch_id(timestamp)
    mlflow_settings = grid_mlflow_settings()
    validate_mlflow_settings(mlflow_settings)
    base_configs = selected_grid()
    parent_run = create_mlflow_grid_parent_run(
        mlflow_settings,
        grid_id,
        timestamp,
        base_configs,
        worker_hosts,
    )
    parent_run_id = mlflow_parent_run_id(parent_run)
    configs = annotate_grid_configs(
        base_configs,
        timestamp,
        mlflow_settings,
        parent_run_id,
        Sockets.gethostname(),
    )
    println("Grid search id: $grid_id")
    if mlflow_settings.enabled
        println(
            "MLflow experiment id: $(mlflow_settings.experiment_id) ($(mlflow_settings.experiment_name))",
        )
    end
    println(
        "Running $(length(configs)) configuration(s) on $(length(remote_worker_ids)) remote worker(s)",
    )

    results = try
        run_grid_on_remote_workers(remote_worker_ids, configs, worker_hosts)
    catch error
        fail_mlflow_grid_parent_run(parent_run)
        rethrow()
    end
    close_mlflow_grid_parent_run(parent_run, results)

    output_dir = write_grid_results(
        results;
        configs=configs,
        output_root=joinpath(@__DIR__, "results"),
        timestamp=timestamp,
    )
    println("Wrote grid-search CSV results to $output_dir")

    failed_count = count(result -> result.status != "ok", results)
    failed_count > 0 && println("Recorded $failed_count failed configuration(s).")
    return output_dir
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
