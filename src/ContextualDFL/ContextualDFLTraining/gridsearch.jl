#!/usr/bin/env julia

using Dates
using Distributed
using ArgParse
import MLFlowClient
using Random
using SHA
using Sockets
using Statistics

include(joinpath(@__DIR__, "src", "run_defaults.jl"))
include(joinpath(@__DIR__, "src", "grid_config.jl"))
include(joinpath(@__DIR__, "src", "csv_results.jl"))
include(joinpath(@__DIR__, "src", "experiments", "ExperimentAPI.jl"))
include(joinpath(@__DIR__, "src", "grid_file_config.jl"))

const DEFAULT_REMOTE_PROJECT =
    "/home/rwl/ProblemBasedScenarioGeneration/src/ContextualDFL/ContextualDFLTraining"
const DEFAULT_REMOTE_JULIA = "/home/rwl/.juliaup/bin/julia"
const MLFLOW_RETRY_ATTEMPTS = 8
const MLFLOW_TRANSIENT_RETRY_ATTEMPTS = 30
const MLFLOW_RETRY_INITIAL_DELAY_SECONDS = 1.0
const MLFLOW_RETRY_BACKOFF = 1.5
const MLFLOW_RETRY_MAX_DELAY_SECONDS = 30.0
const GRID_CANDIDATE_START_STAGGER_SECONDS = 0.25
const GRID_TRAINING_DATA_SEED_MAX = typemax(Int32) - 1

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

function env_worker_count(name, default)
    value = lowercase(strip(get(ENV, name, string(default))))
    value == "auto" && return :auto

    parsed = tryparse(Int, value)
    parsed === nothing && error("ENV[$name] must be an integer or auto, got: $value")
    return parsed
end

function remote_worker_specs()
    return [
#        ("rwl@gcp-4c-1", env_worker_count("GCP_4C_1_WORKERS", :auto)),
#        ("rwl@gcp-4c-2", env_worker_count("GCP_4C_2_WORKERS", :auto)),
#        ("rwl@gcp-4c-3", env_worker_count("GCP_4C_3_WORKERS", :auto)),
#        ("rwl@gcp-4c-5", env_worker_count("GCP_4C_5_WORKERS", :auto)),
#        ("rwl@gcp-8c-1", env_worker_count("GCP_8C_1_WORKERS", :auto)),
#        ("rwl@gcp-8c-2", env_worker_count("GCP_8C_2_WORKERS", :auto)),
#        ("rwl@gcp-8c-3", env_worker_count("GCP_8C_3_WORKERS", :auto)),
#        ("rwl@gcp-8c-5", env_worker_count("GCP_8C_5_WORKERS", :auto)),
        ("rwl@gcp-16c-4", env_worker_count("GCP_16C_4_WORKERS", :auto)),
        ("rwl@ibm-96c-1", env_worker_count("IBM_96C_1_WORKERS", :auto)),
        ("rwl@ibm-96c-2", env_worker_count("IBM_96C_2_WORKERS", :auto)),
    ]
end

function env_flag(name, default=false)
    value = lowercase(get(ENV, name, default ? "1" : "0"))
    return value in ("1", "true", "yes", "y")
end

function deterministic_mlflow_experiment_id(experiment_id)
    digest = bytes2hex(sha1(string(experiment_id)))
    value = parse(UInt64, digest[1:15]; base=16)
    return string(1 + value % UInt64(9_000_000_000))
end

function deterministic_mlflow_experiment_name(experiment)
    return "ContextualDFLTraining/" * string(experiment.name)
end

function grid_mlflow_settings(experiment)
    deterministic_id = deterministic_mlflow_experiment_id(experiment.id)
    enabled = env_flag("MLFLOW_ENABLED", true)
    return (;
        enabled=enabled,
        experiment_id=deterministic_id,
        deterministic_experiment_id=deterministic_id,
        experiment_name=deterministic_mlflow_experiment_name(experiment),
        contextualdfl_experiment_id=experiment.id,
        contextualdfl_experiment_name=experiment.name,
        tracking_uri=get(ENV, "MLFLOW_TRACKING_URI", ""),
        upload_model_artifact=enabled && env_flag("MLFLOW_UPLOAD_MODEL_ARTIFACTS", true),
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

function missing_mlflow_experiment_error(error)
    message = lowercase(sprint(showerror, error))
    return occursin("resource_does_not_exist", message) ||
           occursin("does not exist", message) ||
           occursin("not found", message)
end

function ensure_mlflow_grid_experiment(settings)
    settings.enabled || return settings

    mlf = mlflow_client(settings)
    experiment_id = with_mlflow_retry("ensure MLflow experiment $(settings.experiment_name)") do
        try
            experiment = MLFlowClient.getexperimentbyname(
                mlf,
                string(settings.experiment_name),
            )
            string(experiment.experiment_id)
        catch error
            missing_mlflow_experiment_error(error) || rethrow()
            MLFlowClient.createexperiment(mlf, string(settings.experiment_name))
        end
    end

    experiment_tags = (
        "contextualdfl.experiment_id" => string(settings.contextualdfl_experiment_id),
        "contextualdfl.experiment_name" => string(settings.contextualdfl_experiment_name),
        "contextualdfl.deterministic_experiment_id" => string(settings.deterministic_experiment_id),
    )
    for (key, value) in experiment_tags
        with_mlflow_retry("set MLflow experiment tag $key") do
            MLFlowClient.setexperimenttag(mlf, string(experiment_id), key, value)
        end
    end

    return merge(settings, (; experiment_id=string(experiment_id)))
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
    machines = [
        (host, count) for (host, count) in remote_worker_specs() if
        !(count isa Integer && count <= 0)
    ]
    isempty(machines) && error("No remote worker machines were enabled.")

    for (host, count) in machines
        println("Adding $count worker(s) on $host")
    end
    addprocs(
        machines;
        exename=remote_julia,
        exeflags="--project=$(remote_project)",
        dir=remote_project,
        tunnel=true,
        max_parallel=length(machines),
    )

    remote_worker_ids = setdiff(workers(), [1])
    isempty(remote_worker_ids) && error("No remote workers were added.")
    return remote_worker_ids
end

function load_worker_stdlibs!()
    tasks = [
        @async remotecall_fetch(process_id) do
            Core.eval(Main, :(using Dates))
            Core.eval(Main, :(using Distributed))
            Core.eval(Main, :(using Pkg))
            Core.eval(Main, :(using Sockets))
            return nothing
        end for process_id in procs()
    ]
    foreach(fetch, tasks)
end

function assert_remote_only_workers!(remote_worker_ids)
    local_hostname = Sockets.gethostname()
    host_tasks = Dict(
        worker => @async remotecall_fetch(() -> Sockets.gethostname(), worker) for
        worker in remote_worker_ids
    )
    worker_hosts = Dict(worker => fetch(task) for (worker, task) in host_tasks)
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

function load_training_project_on_workers!(remote_worker_ids, worker_hosts)
    first_worker_by_host = Dict{String,Int}()
    for worker in sort(remote_worker_ids)
        host = worker_hosts[worker]
        haskey(first_worker_by_host, host) || (first_worker_by_host[host] = worker)
    end

    println("Instantiating ContextualDFLTraining once per remote host")
    instantiate_tasks = [
        @async remotecall_fetch(worker) do
            Pkg.instantiate()
            Core.eval(Main, :(using ContextualDFLTraining))
            return (;
                worker_id=Distributed.myid(),
                hostname=Sockets.gethostname(),
                pid=getpid(),
            )
        end for worker in values(first_worker_by_host)
    ]
    foreach(fetch, instantiate_tasks)

    println("Loading ContextualDFLTraining on remote workers")
    load_tasks = [
        @async remotecall_fetch(worker) do
            Core.eval(Main, :(using ContextualDFLTraining))
            return (;
                worker_id=Distributed.myid(),
                hostname=Sockets.gethostname(),
                pid=getpid(),
            )
        end for worker in remote_worker_ids
    ]
    for task in load_tasks
        metadata = fetch(task)
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

    foreach(fetch, [@async remotecall_fetch(Core.eval, worker, Main, definition) for worker in workers()])
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

function run_grid_on_remote_workers(remote_worker_ids, configs, worker_hosts; on_result=nothing)
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

    function record_result!(index, result)
        results[index] = result
        on_result === nothing && return result

        try
            on_result(index, result)
        catch error
            @warn "Grid result callback failed" run_id=getproperty(result, :run_id) error=sprint(
                showerror,
                error,
                catch_backtrace(),
            )
        end
        return result
    end

    tasks = [
        @async begin
            while true
                item = next_pending!()
                item === nothing && break

                index, config = item
                started = time()
                try
                    if length(remote_worker_ids) > 1
                        sleep(
                            GRID_CANDIDATE_START_STAGGER_SECONDS *
                            mod(index - 1, length(remote_worker_ids)),
                        )
                    end
                    record_result!(
                        index,
                        remotecall_fetch(
                            _contextualdfltraining_remote_eval,
                            worker,
                            config,
                        ),
                    )
                catch error
                    elapsed_seconds = time() - started
                    if transport_failure(error)
                        println(
                            "Worker $worker exited while running $(config.run_id); recording worker_lost and continuing.",
                        )
                        mark_mlflow_run_failed(config, "worker_lost")
                        record_result!(
                            index,
                            coordinator_error_result(
                                config,
                                worker,
                                worker_hosts,
                                "worker_lost",
                                error,
                                catch_backtrace(),
                                elapsed_seconds,
                            ),
                        )
                        break
                    end

                    record_result!(
                        index,
                        coordinator_error_result(
                            config,
                            worker,
                            worker_hosts,
                            "coordinator_error",
                            error,
                            catch_backtrace(),
                            elapsed_seconds,
                        ),
                    )
                end
            end
        end for worker in remote_worker_ids
    ]

    foreach(wait, tasks)

    for (index, config) in enumerate(configs)
        if !isassigned(results, index)
            record_result!(
                index,
                (;
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
                ),
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
        runs, _ = with_mlflow_retry("search failed candidate runs") do
            MLFlowClient.searchruns(
                mlf;
                experiment_ids=[experiment_id],
                filter=filter,
                max_results=100,
            )
        end

        for run in runs
            with_mlflow_retry("set failed candidate tag") do
                MLFlowClient.setruntag(
                    mlf,
                    run,
                    "ContextualDFLTraining.coordinator_status",
                    reason,
                )
            end
            with_mlflow_retry("update failed candidate run") do
                MLFlowClient.updaterun(
                    mlf,
                    run;
                    status=MLFlowClient.RunStatus.FAILED,
                    end_time=unix_milliseconds(),
                )
            end
        end
    catch error
        println("Could not mark MLflow run failed for $(config.run_id): ", sprint(showerror, error))
    end

    return nothing
end

function create_mlflow_grid_parent_run(
    settings,
    grid_id,
    timestamp,
    configs,
    worker_hosts,
    grid_spec::GridSearchSpec,
    ;
    repeat_training_data_seeds=nothing,
)
    settings.enabled || return nothing

    mlf = mlflow_client(settings)
    tags = Dict(
        "gridsearch_id" => grid_id,
        "gridsearch_timestamp" => timestamp,
        "gridsearch_role" => "parent",
        "training_project" => "ContextualDFLTraining",
        "mlflow.experiment.name" => string(settings.experiment_name),
        "contextualdfl.experiment_id" => string(settings.contextualdfl_experiment_id),
        "contextualdfl.experiment_name" => string(settings.contextualdfl_experiment_name),
        "contextualdfl.deterministic_experiment_id" => string(settings.deterministic_experiment_id),
        "mlflow.source.name" => "ContextualDFLTraining/gridsearch.jl",
        "mlflow.source.type" => "LOCAL",
    )
    git_commit = git_commit_or_empty()
    isempty(git_commit) || (tags["mlflow.source.git.commit"] = git_commit)

    parent_params = grid_parent_params(
        grid_id,
        timestamp,
        configs,
        worker_hosts,
        grid_spec;
        repeat_training_data_seeds=repeat_training_data_seeds,
    )

    run = with_mlflow_retry("create grid parent run") do
        MLFlowClient.createrun(
            mlf,
            string(settings.experiment_id);
            run_name=grid_id,
            start_time=unix_milliseconds(),
            tags=tags,
        )
    end

    for (key, value) in parent_params
        with_mlflow_retry("log grid parent param $key") do
            MLFlowClient.logparam(mlf, run, key, value)
        end
    end

    upload_mlflow_grid_config_artifacts!(mlf, grid_spec, configs)

    return (; client=mlf, run=run)
end

function upload_mlflow_grid_config_artifacts!(mlf, grid_spec::GridSearchSpec, configs)
    source_extension = grid_spec.format == :yaml ? ".yaml" : ".json"
    source_data = read(grid_spec.path)
    resolved_json = resolved_grid_json(configs)
    resolved_data = Vector{UInt8}(codeunits(resolved_json))
    digest_data = Vector{UInt8}(codeunits(grid_config_digest(configs) * "\n"))

    with_mlflow_retry("upload grid config source artifact") do
        MLFlowClient.uploadartifact(
            mlf,
            "grid_config/source" * source_extension,
            source_data,
        )
    end
    with_mlflow_retry("upload resolved grid config artifact") do
        MLFlowClient.uploadartifact(mlf, "grid_config/resolved.json", resolved_data)
    end
    with_mlflow_retry("upload grid config digest artifact") do
        MLFlowClient.uploadartifact(mlf, "grid_config/digest.txt", digest_data)
    end

    return nothing
end

function close_mlflow_grid_parent_run(parent, config_parent_results; child_results=Any[])
    parent === nothing && return nothing

    success = all(result -> getproperty(result, :status) == "ok", config_parent_results)
    try
        mark_failed_mlflow_candidates!(child_results)
    catch error
        @warn "Failed to mark failed MLflow child runs" error=sprint(showerror, error)
    end
    try
        log_grid_aggregate_metrics!(parent.client, parent.run, config_parent_results)
    catch error
        @warn "Failed to log grid aggregate metrics" error=sprint(showerror, error)
    end
    with_mlflow_retry("update grid parent run") do
        MLFlowClient.updaterun(
            parent.client,
            parent.run;
            status=success ? MLFlowClient.RunStatus.FINISHED : MLFlowClient.RunStatus.FAILED,
            end_time=unix_milliseconds(),
        )
    end
    return nothing
end

function create_mlflow_config_parent_runs(settings, config_parent_configs)
    settings.enabled || return fill(nothing, length(config_parent_configs))
    return [
        create_mlflow_config_parent_run(settings, config) for
        config in config_parent_configs
    ]
end

function create_mlflow_config_parent_run(settings, config)
    mlf = mlflow_client(settings)
    tags = Dict(
        "gridsearch_id" => string(config.gridsearch_id),
        "gridsearch_timestamp" => string(config.gridsearch_timestamp),
        "gridsearch_role" => "config_parent",
        "training_project" => "ContextualDFLTraining",
        "run_id" => string(config.run_id),
        "base_run_id" => string(config.base_run_id),
        "candidate_index" => string(config.candidate_index),
        "candidate_name" => string(config.candidate_name),
        "repeat_count" => string(config.repeat_count),
        "gridsearch_parent_run_id" => string(config.mlflow_parent_run_id),
        "mlflow.parentRunId" => string(config.mlflow_parent_run_id),
        "mlflow.experiment.name" => string(settings.experiment_name),
        "contextualdfl.experiment_id" => string(settings.contextualdfl_experiment_id),
        "contextualdfl.experiment_name" => string(settings.contextualdfl_experiment_name),
        "contextualdfl.deterministic_experiment_id" => string(settings.deterministic_experiment_id),
        "mlflow.source.name" => "ContextualDFLTraining/gridsearch.jl",
        "mlflow.source.type" => "LOCAL",
    )
    git_commit = git_commit_or_empty()
    isempty(git_commit) || (tags["mlflow.source.git.commit"] = git_commit)

    run = with_mlflow_retry("create config parent run") do
        MLFlowClient.createrun(
            mlf,
            string(settings.experiment_id);
            run_name=string(config.candidate_name),
            start_time=unix_milliseconds(),
            tags=tags,
        )
    end

    log_config_parent_params!(mlf, run, config)
    return (; client=mlf, run=run, config=config)
end

function log_config_parent_params!(mlf, run, config)
    params = Dict{String,String}(
        "gridsearch_id" => string(config.gridsearch_id),
        "candidate_index" => string(config.candidate_index),
        "base_run_id" => string(config.base_run_id),
        "repeat_count" => string(config.repeat_count),
    )

    for key in keys(config)
        value = getproperty(config, key)
        mlflow_scalar_value(value) || continue
        params["config_" * string(key)] = string(value)
    end

    for key in sort!(collect(keys(params)))
        with_mlflow_retry("log config parent param $key") do
            MLFlowClient.logparam(mlf, run, key, params[key])
        end
    end

    return nothing
end

function close_mlflow_config_parent_runs(
    config_parent_runs,
    config_parent_results;
    skip_names=Set{String}(),
)
    for (parent, result) in zip(config_parent_runs, config_parent_results)
        config = getproperty(result, :config)
        parent_name = config isa NamedTuple && hasproperty(config, :candidate_name) ?
            string(config.candidate_name) :
            result_config_parent_name(config)
        parent_name in skip_names && continue

        try
            close_mlflow_config_parent_run(parent, result)
        catch error
            println(
                "Could not close config parent MLflow run for $parent_name: ",
                sprint(showerror, error),
            )
        end
    end
    return nothing
end

function close_mlflow_config_parent_run(parent, result)
    parent === nothing && return nothing

    success = getproperty(result, :status) == "ok"
    try
        log_config_parent_aggregate_metrics!(parent.client, parent.run, result)
    catch error
        @warn "Failed to log config parent aggregate metrics" run_id=getproperty(result, :run_id) error=sprint(
            showerror,
            error,
        )
    end
    with_mlflow_retry("update config parent run") do
        MLFlowClient.updaterun(
            parent.client,
            parent.run;
            status=success ? MLFlowClient.RunStatus.FINISHED : MLFlowClient.RunStatus.FAILED,
            end_time=unix_milliseconds(),
        )
    end
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
        with_mlflow_retry("fail grid parent run") do
            MLFlowClient.updaterun(
                parent.client,
                parent.run;
                status=MLFlowClient.RunStatus.FAILED,
                end_time=unix_milliseconds(),
            )
        end
    catch error
        println("Could not mark parent MLflow run failed: ", sprint(showerror, error))
    end
    return nothing
end

function fail_mlflow_config_parent_runs(parents)
    for parent in parents
        parent === nothing && continue
        try
            with_mlflow_retry("fail config parent run") do
                MLFlowClient.updaterun(
                    parent.client,
                    parent.run;
                    status=MLFlowClient.RunStatus.FAILED,
                    end_time=unix_milliseconds(),
                )
            end
        catch error
            println("Could not mark config parent MLflow run failed: ", sprint(showerror, error))
        end
    end
    return nothing
end

function grid_parent_params(
    grid_id,
    timestamp,
    configs,
    worker_hosts,
    grid_spec::GridSearchSpec;
    repeat_training_data_seeds=nothing,
)
    params = Dict{String,String}(
        "gridsearch_id" => grid_id,
        "gridsearch_timestamp" => timestamp,
        "grid_candidate_count" => string(length(configs)),
        "grid_repeat_run_count" => string(sum(grid_repeat_count(config) for config in configs; init=0)),
        "grid_config_name" => grid_spec.name,
        "grid_config_path" => grid_spec.path,
        "grid_config_digest" => grid_config_digest(configs),
        "grid_config_version" => string(grid_spec.version),
        "grid_config_format" => string(grid_spec.format),
        "grid_worker_count" => string(length(worker_hosts)),
        "grid_worker_hosts" => join(sort!(unique(collect(values(worker_hosts)))), ","),
    )

    if repeat_training_data_seeds !== nothing
        repeat_seeds = normalize_repeat_training_data_seeds(
            repeat_training_data_seeds,
            grid_repeat_seed_count(configs),
        )
        if !isempty(repeat_seeds)
            params["grid_repeat_training_data_seed_count"] = string(length(repeat_seeds))
            params["grid_repeat_training_data_seed_sequence"] =
                repeat_training_data_seed_sequence(repeat_seeds)
            for (index, seed) in enumerate(repeat_seeds)
                params["grid_" * repeat_tag(index) * "_training_data_seed"] = string(seed)
            end
        end
    end

    if !isempty(configs)
        config = first(configs)
        if config isa NamedTuple && hasproperty(config, :experiment_id)
            params["experiment_id"] = string(config.experiment_id)
        end
        if config isa NamedTuple && hasproperty(config, :experiment_name)
            params["experiment_name"] = string(config.experiment_name)
        end
        if config isa NamedTuple && hasproperty(config, :mlflow_deterministic_experiment_id)
            params["mlflow_deterministic_experiment_id"] =
                string(config.mlflow_deterministic_experiment_id)
        end
    end

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
    log_aggregate_metrics!(mlf, run, "grid", aggregate_metric_summaries(results))
    return nothing
end

function log_config_parent_aggregate_metrics!(mlf, run, result)
    summaries = getproperty(result, :aggregate_metrics)
    summaries isa AbstractDict || return nothing
    log_aggregate_metrics!(mlf, run, "config", summaries)
    return nothing
end

function log_aggregate_metrics!(mlf, run, prefix_root::AbstractString, summaries)
    for key in sort!(collect(keys(summaries)); by=String)
        summary = summaries[key]
        prefix = prefix_root * "_" * string(key)
        timestamp = unix_milliseconds()
        for field in (:count, :mean, :median, :min, :max, :std, :stderr)
            value = getproperty(summary, field)
            with_mlflow_retry("log aggregate metric $(prefix)_$(field)") do
                MLFlowClient.logmetric(
                    mlf,
                    run,
                    prefix * "_" * string(field),
                    Float64(value);
                    timestamp=timestamp,
                    step=0,
                )
            end
        end
    end

    return nothing
end

function aggregate_metric_summaries(results)
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

    summaries = Dict{Symbol,NamedTuple}()
    for key in sort!(collect(metric_keys); by=String)
        values = Float64[
            Float64(getproperty(getproperty(result, :final_metrics), key)) for result in results if
            getproperty(result, :status) == "ok" &&
            getproperty(result, :final_metrics) isa NamedTuple &&
            hasproperty(getproperty(result, :final_metrics), key) &&
            mlflow_numeric_metric(getproperty(getproperty(result, :final_metrics), key))
        ]
        isempty(values) && continue
        std_value = length(values) > 1 ? std(values) : 0.0
        summaries[key] = (;
            count=Float64(length(values)),
            mean=mean(values),
            median=median(values),
            min=minimum(values),
            max=maximum(values),
            std=std_value,
            stderr=std_value / sqrt(length(values)),
        )
    end

    return summaries
end

function aggregate_mean_metrics(summaries::AbstractDict)
    keys_sorted = sort!(collect(keys(summaries)); by=String)
    return NamedTuple{Tuple(keys_sorted)}(
        Tuple(getproperty(summaries[key], :mean) for key in keys_sorted),
    )
end

function with_mlflow_retry(callback, operation)
    delay = MLFLOW_RETRY_INITIAL_DELAY_SECONDS
    for attempt in 1:MLFLOW_TRANSIENT_RETRY_ATTEMPTS
        try
            return callback()
        catch error
            max_attempts = mlflow_transient_tracking_error(error) ?
                MLFLOW_TRANSIENT_RETRY_ATTEMPTS :
                MLFLOW_RETRY_ATTEMPTS
            attempt == max_attempts && rethrow()
            @warn "MLflow $operation failed; retrying" attempt error=sprint(showerror, error)
            sleep(delay * (0.75 + 0.5 * rand()))
            delay = min(delay * MLFLOW_RETRY_BACKOFF, MLFLOW_RETRY_MAX_DELAY_SECONDS)
        end
    end
end

function mlflow_transient_tracking_error(error)
    message = sprint(showerror, error)
    return any(
        pattern -> occursin(pattern, message),
        (
            "QueuePool limit",
            "connection timed out",
            "Read timed out",
            "ConnectTimeout",
            "Connection refused",
            "Max retries exceeded",
            "RemoteDisconnected",
            "temporarily unavailable",
            "database is locked",
        ),
    )
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

function experiment_problem_identity_for_grid(experiment)
    experiment_has_function(experiment, :problem_identity_config) || return NamedTuple()
    identity = experiment_call(experiment, :problem_identity_config)
    identity isa NamedTuple ||
        throw(ArgumentError("experiment problem_identity_config() must return a NamedTuple."))
    return identity
end

function validate_grid_does_not_override_problem_identity(experiment, grid_spec::GridSearchSpec)
    identity = experiment_problem_identity_for_grid(experiment)
    isempty(keys(identity)) && return nothing

    identity_keys = Set(Symbol.(keys(identity)))
    provided_keys = Set{Symbol}()
    union!(provided_keys, keys(grid_spec.base))
    union!(provided_keys, keys(grid_spec.fixed))
    union!(provided_keys, keys(grid_spec.grid))
    union!(provided_keys, keys(grid_spec.schedules))
    for schedule_candidate in grid_spec.schedule_grid
        union!(provided_keys, keys(schedule_candidate))
    end

    overridden = sort!(collect(intersect(identity_keys, provided_keys)); by=string)
    isempty(overridden) && return nothing

    throw(
        ArgumentError(
            "grid config $(grid_spec.path) may not set problem-identity key(s) owned by experiment $(experiment.id): $(join(string.(overridden), ", "))",
        ),
    )
end

function selected_grid(experiment, grid_spec::GridSearchSpec)
    validate_grid_does_not_override_problem_identity(experiment, grid_spec)
    return resolve_grid_configs(grid_spec; base_config=experiment_base_config(experiment))
end

function parse_commandline(args=ARGS)
    settings = ArgParseSettings(
        description="Run a ContextualDFLTraining grid search for one experiment.",
    )

    @add_arg_table! settings begin
        "--experiment"
            help = "Experiment id, module name, or config path to run, e.g. resource_allocation/experiment_1"
            required = true
        "--grid-config"
            help = "Path to a YAML or JSON grid-search config file."
            required = true
    end

    return parse_args(args, settings)
end

function gridsearch_id(timestamp::AbstractString)
    return "gridsearch_" * timestamp
end

function candidate_tag(index::Integer)
    return "candidate_" * lpad(string(index), 4, "0")
end

function repeat_tag(index::Integer)
    return "repeat_" * lpad(string(index), 3, "0")
end

function base_run_id(config, index::Integer)
    if config isa NamedTuple && :run_id in keys(config)
        return string(config.run_id)
    end
    return candidate_tag(index)
end

function grid_config_value(config, key::Symbol, default)
    config isa NamedTuple || return default
    return key in keys(config) ? getproperty(config, key) : default
end

function grid_repeat_count(config)
    count = Int(grid_config_value(config, :repeat_count, 1))
    count > 0 || throw(ArgumentError("repeat_count must be positive."))
    return count
end

function grid_repeat_seed_count(configs)
    return maximum((grid_repeat_count(config) for config in configs); init=0)
end

function random_training_data_seeds(count::Integer; rng=Random.default_rng())
    count >= 0 || throw(ArgumentError("repeat seed count must be non-negative."))

    seeds = Int[]
    seen = Set{Int}()
    while length(seeds) < count
        seed = rand(rng, 1:GRID_TRAINING_DATA_SEED_MAX)
        seed in seen && continue
        push!(seeds, seed)
        push!(seen, seed)
    end
    return seeds
end

function generate_repeat_training_data_seeds(configs; rng=Random.default_rng())
    return random_training_data_seeds(grid_repeat_seed_count(configs); rng=rng)
end

function normalize_repeat_training_data_seeds(seeds, required_count::Integer)
    required_count >= 0 ||
        throw(ArgumentError("required repeat seed count must be non-negative."))
    seeds === nothing && return random_training_data_seeds(required_count)

    normalized = Int.(collect(seeds))
    length(normalized) >= required_count || throw(
        ArgumentError(
            "repeat_training_data_seeds must contain at least $required_count seed(s), got $(length(normalized)).",
        ),
    )

    selected = normalized[1:required_count]
    all(seed -> 1 <= seed <= GRID_TRAINING_DATA_SEED_MAX, selected) || throw(
        ArgumentError(
            "repeat_training_data_seeds must be in 1:$(GRID_TRAINING_DATA_SEED_MAX).",
        ),
    )
    length(unique(selected)) == length(selected) || throw(
        ArgumentError("repeat_training_data_seeds must contain distinct seeds."),
    )
    return selected
end

function repeat_training_data_seed_sequence(seeds)
    return join(string.(seeds), ",")
end

function parse_repeat_training_data_seed_sequence(value)
    text = strip(string(value))
    isempty(text) && return Int[]
    return [parse(Int, strip(part)) for part in split(text, ",")]
end

function repeat_training_data_seeds_from_config_parents(config_parent_configs, required_count)
    best_seeds = nothing
    for config_parent in config_parent_configs
        config_parent isa NamedTuple || continue
        hasproperty(config_parent, :repeat_training_data_seed_sequence) || continue
        seeds = parse_repeat_training_data_seed_sequence(
            getproperty(config_parent, :repeat_training_data_seed_sequence),
        )
        length(seeds) >= required_count &&
            return normalize_repeat_training_data_seeds(seeds, required_count)
        if best_seeds === nothing || length(seeds) > length(best_seeds)
            best_seeds = seeds
        end
    end
    return best_seeds === nothing ? nothing : best_seeds
end

function annotate_grid_config_parent(
    config,
    index::Integer,
    timestamp::AbstractString,
    mlflow_settings,
    grid_parent_run_id::AbstractString="",
    coordinator_hostname::AbstractString=Sockets.gethostname(),
    ;
    repeat_training_data_seeds=nothing,
)
    grid_id = gridsearch_id(timestamp)
    candidate = candidate_tag(index)
    previous_run_id = base_run_id(config, index)
    candidate_name = grid_id * "__" * candidate * "__" * previous_run_id
    repeats = grid_repeat_count(config)
    repeat_seeds =
        normalize_repeat_training_data_seeds(repeat_training_data_seeds, repeats)
    seed_sequence = repeat_training_data_seed_sequence(repeat_seeds)

    return merge(
        config,
        (;
            run_id=candidate_name,
            base_run_id=previous_run_id,
            gridsearch_id=grid_id,
            gridsearch_timestamp=timestamp,
            candidate_index=Int(index),
            candidate_name=candidate_name,
            config_parent_name=candidate_name,
            repeat_count=repeats,
            repeat_training_data_seed_sequence=seed_sequence,
            mlflow_enabled=mlflow_settings.enabled,
            mlflow_experiment_id=mlflow_settings.experiment_id,
            mlflow_experiment_name=mlflow_settings.experiment_name,
            mlflow_deterministic_experiment_id=mlflow_settings.deterministic_experiment_id,
            mlflow_tracking_uri=mlflow_settings.tracking_uri,
            mlflow_upload_model_artifact=mlflow_settings.upload_model_artifact,
            checkpoint_upload_mlflow=mlflow_settings.enabled &&
                                     Bool(grid_config_value(config, :checkpoint_upload_mlflow, true)),
            mlflow_parent_run_id=grid_parent_run_id,
            mlflow_run_name=candidate_name,
            coordinator_hostname=coordinator_hostname,
            mlflow_tags=(;
                gridsearch_id=grid_id,
                gridsearch_timestamp=timestamp,
                candidate_index=Int(index),
                base_run_id=previous_run_id,
                candidate_name=candidate_name,
                config_parent_name=candidate_name,
                repeat_count=repeats,
                repeat_training_data_seed_sequence=seed_sequence,
                gridsearch_parent_run_id=grid_parent_run_id,
                mlflow_parentRunId=grid_parent_run_id,
                mlflow_deterministic_experiment_id=mlflow_settings.deterministic_experiment_id,
                mlflow_experiment_name=mlflow_settings.experiment_name,
                gridsearch_role="config_parent",
            ),
        ),
    )
end

function annotate_grid_config_parents(
    configs,
    timestamp::AbstractString,
    mlflow_settings,
    grid_parent_run_id::AbstractString="",
    coordinator_hostname::AbstractString=Sockets.gethostname(),
    ;
    repeat_training_data_seeds=nothing,
)
    shared_repeat_seeds = normalize_repeat_training_data_seeds(
        repeat_training_data_seeds,
        grid_repeat_seed_count(configs),
    )
    return [
        annotate_grid_config_parent(
            config,
            index,
            timestamp,
            mlflow_settings,
            grid_parent_run_id,
            coordinator_hostname;
            repeat_training_data_seeds=shared_repeat_seeds,
        ) for
        (index, config) in enumerate(configs)
    ]
end

function annotate_repeat_config(
    config_parent,
    repeat_index::Integer,
    mlflow_settings,
    config_parent_run_id::AbstractString="",
    ;
    repeat_training_data_seeds=nothing,
)
    child_name = string(config_parent.candidate_name) * "__" * repeat_tag(repeat_index)
    seed_source = if repeat_training_data_seeds === nothing &&
                     config_parent isa NamedTuple &&
                     hasproperty(config_parent, :repeat_training_data_seed_sequence)
        parse_repeat_training_data_seed_sequence(config_parent.repeat_training_data_seed_sequence)
    else
        repeat_training_data_seeds
    end
    repeat_seeds = normalize_repeat_training_data_seeds(
        seed_source,
        Int(config_parent.repeat_count),
    )
    training_seed = repeat_seeds[Int(repeat_index)]
    tags = merge(
        config_parent.mlflow_tags,
        (;
            candidate_name=child_name,
            config_parent_name=config_parent.candidate_name,
            config_parent_run_id=config_parent_run_id,
            repeat_index=Int(repeat_index),
            repeat_count=Int(config_parent.repeat_count),
            training_data_seed=training_seed,
            repeat_training_data_seed=training_seed,
            gridsearch_parent_run_id=config_parent_run_id,
            mlflow_parentRunId=config_parent_run_id,
            gridsearch_role="repeat",
        ),
    )

    return merge(
        config_parent,
        (;
            run_id=child_name,
            candidate_name=child_name,
            config_parent_name=config_parent.candidate_name,
            config_parent_run_id=config_parent_run_id,
            repeat_index=Int(repeat_index),
            repeat_count=Int(config_parent.repeat_count),
            training_data_seed=training_seed,
            repeat_training_data_seed=training_seed,
            training_data_cache=false,
            write_training_data_artifact=false,
            mlflow_enabled=mlflow_settings.enabled,
            mlflow_upload_model_artifact=mlflow_settings.upload_model_artifact,
            checkpoint_upload_mlflow=mlflow_settings.enabled && Bool(
                grid_config_value(config_parent, :checkpoint_upload_mlflow, true),
            ),
            mlflow_parent_run_id=config_parent_run_id,
            mlflow_run_name=child_name,
            mlflow_tags=tags,
        ),
    )
end

function annotate_repeat_configs(
    config_parent_configs,
    config_parent_runs,
    mlflow_settings;
    repeat_training_data_seeds=nothing,
)
    required_count = grid_repeat_seed_count(config_parent_configs)
    seed_source = repeat_training_data_seeds === nothing ?
        repeat_training_data_seeds_from_config_parents(config_parent_configs, required_count) :
        repeat_training_data_seeds
    shared_repeat_seeds = normalize_repeat_training_data_seeds(
        seed_source,
        required_count,
    )
    child_configs = NamedTuple[]
    for (config_parent, config_parent_run) in zip(config_parent_configs, config_parent_runs)
        parent_run_id = mlflow_parent_run_id(config_parent_run)
        for repeat_index in 1:Int(config_parent.repeat_count)
            push!(
                child_configs,
                annotate_repeat_config(
                    config_parent,
                    repeat_index,
                    mlflow_settings,
                    parent_run_id;
                    repeat_training_data_seeds=shared_repeat_seeds,
                ),
            )
        end
    end
    return child_configs
end

function config_parent_results(config_parent_configs, child_results)
    return [
        config_parent_result(config_parent, child_results_for_config(config_parent, child_results)) for
        config_parent in config_parent_configs
    ]
end

function child_results_for_config(config_parent, child_results)
    parent_name = string(config_parent.candidate_name)
    return [
        result for result in child_results if
        result_config_parent_name(getproperty(result, :config)) == parent_name
    ]
end

function result_config_parent_name(config)
    config isa NamedTuple || return ""
    if :config_parent_name in keys(config)
        return string(config.config_parent_name)
    end
    return ""
end

function config_parent_result(config_parent, child_results)
    expected_repeats = Int(config_parent.repeat_count)
    successful_repeats = count(result -> getproperty(result, :status) == "ok", child_results)
    failed_repeats = length(child_results) - successful_repeats
    success = successful_repeats > 0
    summaries = aggregate_metric_summaries(child_results)
    metrics = merge(
        aggregate_mean_metrics(summaries),
        (;
            repeat_count=Float64(expected_repeats),
            repeat_successful_count=Float64(successful_repeats),
            repeat_failed_count=Float64(failed_repeats),
        ),
    )
    started_at = isempty(child_results) ?
        unix_milliseconds() :
        minimum(getproperty(result, :started_at) for result in child_results)
    finished_at = isempty(child_results) ?
        unix_milliseconds() :
        maximum(getproperty(result, :finished_at) for result in child_results)
    elapsed_seconds = sum(
        Float64(getproperty(result, :elapsed_seconds)) for result in child_results;
        init=0.0,
    )
    error = success ? "" : config_parent_error(child_results, expected_repeats)

    return (;
        status=success ? "ok" : "failed",
        run_id=config_parent.run_id,
        config=config_parent,
        worker=NamedTuple(),
        final_metrics=metrics,
        aggregate_metrics=summaries,
        epoch_history=Dict{Symbol,Any}[],
        error=error,
        started_at=started_at,
        finished_at=finished_at,
        elapsed_seconds=elapsed_seconds,
    )
end

function config_parent_error(child_results, expected_repeats)
    parts = String[]
    length(child_results) == expected_repeats || push!(
        parts,
        "Expected $(expected_repeats) repeat(s), recorded $(length(child_results)).",
    )
    for result in child_results
        getproperty(result, :status) == "ok" && continue
        push!(parts, "$(getproperty(result, :run_id)): $(getproperty(result, :status))")
    end
    return join(parts, " ")
end

function main()
    parsed_args = parse_commandline()
    experiment = load_experiment(parsed_args["experiment"])
    grid_spec = load_grid_config(parsed_args["grid-config"])

    ensure_clean_worker_start!()
    sync_code!()
    remote_worker_ids = add_remote_workers!()
    load_worker_stdlibs!()
    worker_hosts = assert_remote_only_workers!(remote_worker_ids)
    load_training_project_on_workers!(remote_worker_ids, worker_hosts)
    define_remote_eval!()

    timestamp = result_timestamp()
    grid_id = gridsearch_id(timestamp)
    mlflow_settings = grid_mlflow_settings(experiment)
    validate_mlflow_settings(mlflow_settings)
    mlflow_settings = ensure_mlflow_grid_experiment(mlflow_settings)
    base_configs = selected_grid(experiment, grid_spec)
    repeat_training_data_seeds = generate_repeat_training_data_seeds(base_configs)
    parent_run = create_mlflow_grid_parent_run(
        mlflow_settings,
        grid_id,
        timestamp,
        base_configs,
        worker_hosts,
        grid_spec;
        repeat_training_data_seeds=repeat_training_data_seeds,
    )
    parent_run_id = mlflow_parent_run_id(parent_run)
    config_parent_configs = annotate_grid_config_parents(
        base_configs,
        timestamp,
        mlflow_settings,
        parent_run_id,
        Sockets.gethostname();
        repeat_training_data_seeds=repeat_training_data_seeds,
    )
    config_parent_runs = Any[]
    configs = NamedTuple[]
    try
        config_parent_runs = create_mlflow_config_parent_runs(mlflow_settings, config_parent_configs)
        configs = annotate_repeat_configs(
            config_parent_configs,
            config_parent_runs,
            mlflow_settings;
            repeat_training_data_seeds=repeat_training_data_seeds,
        )
    catch error
        fail_mlflow_config_parent_runs(config_parent_runs)
        fail_mlflow_grid_parent_run(parent_run)
        rethrow()
    end
    println("Grid search id: $grid_id")
    println(
        "Grid config: $(grid_spec.name) ($(grid_spec.path), $(length(base_configs)) candidate(s), $(length(configs)) repeat run(s))",
    )
    println(
        "Repeat training data seeds: ",
        repeat_training_data_seed_sequence(repeat_training_data_seeds),
    )
    if mlflow_settings.enabled
        println(
            "MLflow experiment id: $(mlflow_settings.experiment_id) ($(mlflow_settings.experiment_name))",
        )
    else
        println("MLflow disabled; writing grid-search results to local CSV/JSONL only.")
    end
    println(
        "Running $(length(configs)) configuration(s) on $(length(remote_worker_ids)) remote worker(s)",
    )

    output_root = joinpath(@__DIR__, "results")
    output_dir = joinpath(output_root, timestamp)
    mkpath(output_dir)
    println("Incremental result JSONL path: $output_dir")

    config_parent_by_name =
        Dict(string(config.candidate_name) => config for config in config_parent_configs)
    config_parent_run_by_name = Dict(
        string(config.candidate_name) => run for
        (config, run) in zip(config_parent_configs, config_parent_runs)
    )
    child_results_by_config_parent = Dict{String,Vector{Any}}()
    closed_config_parent_names = Set{String}()
    config_parent_close_lock = ReentrantLock()
    incremental_result_lock = ReentrantLock()

    function close_config_parent_if_complete!(_, result)
        parent_name = result_config_parent_name(getproperty(result, :config))
        isempty(parent_name) && return nothing
        haskey(config_parent_by_name, parent_name) || return nothing

        parent = nothing
        parent_result = nothing
        lock(config_parent_close_lock)
        try
            parent_name in closed_config_parent_names && return nothing

            child_results = get!(
                () -> Any[],
                child_results_by_config_parent,
                parent_name,
            )
            push!(child_results, result)

            expected_repeats = Int(config_parent_by_name[parent_name].repeat_count)
            length(child_results) < expected_repeats && return nothing

            push!(closed_config_parent_names, parent_name)
            parent = config_parent_run_by_name[parent_name]
            parent_result = config_parent_result(
                config_parent_by_name[parent_name],
                child_results,
            )
        finally
            unlock(config_parent_close_lock)
        end

        try
            lock(incremental_result_lock)
            try
                write_incremental_config_result(output_dir, parent_result)
            catch error
                @warn "Could not write incremental config result" run_id=getproperty(
                    parent_result,
                    :run_id,
                ) error=sprint(showerror, error, catch_backtrace())
            finally
                unlock(incremental_result_lock)
            end
            close_mlflow_config_parent_run(parent, parent_result)
        catch error
            lock(config_parent_close_lock)
            try
                delete!(closed_config_parent_names, parent_name)
            finally
                unlock(config_parent_close_lock)
            end
            println(
                "Could not close completed config parent MLflow run for $parent_name: ",
                sprint(showerror, error),
            )
        end

        return nothing
    end

    function record_completed_repeat!(index, result)
        lock(incremental_result_lock)
        try
            write_incremental_grid_result(output_dir, result)
        catch error
            @warn "Could not write incremental repeat result" run_id=getproperty(
                result,
                :run_id,
            ) error=sprint(showerror, error, catch_backtrace())
        finally
            unlock(incremental_result_lock)
        end
        close_config_parent_if_complete!(index, result)
        return nothing
    end

    results = try
        run_grid_on_remote_workers(
            remote_worker_ids,
            configs,
            worker_hosts;
            on_result=record_completed_repeat!,
        )
    catch error
        fail_mlflow_config_parent_runs(config_parent_runs)
        fail_mlflow_grid_parent_run(parent_run)
        rethrow()
    end
    config_results = config_parent_results(config_parent_configs, results)
    close_mlflow_config_parent_runs(
        config_parent_runs,
        config_results;
        skip_names=closed_config_parent_names,
    )
    close_mlflow_grid_parent_run(parent_run, config_results; child_results=results)

    output_dir = write_grid_results(
        results;
        configs=configs,
        config_results=config_results,
        output_root=output_root,
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
