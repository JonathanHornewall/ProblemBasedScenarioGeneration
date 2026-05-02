#!/usr/bin/env julia

using Dates
using Distributed
import MLFlowClient
using Sockets

include(joinpath(@__DIR__, "src", "grid_config.jl"))
include(joinpath(@__DIR__, "src", "csv_results.jl"))

const DEFAULT_REMOTE_PROJECT =
    "/home/rwl/ProblemBasedScenarioGeneration/src/ContextualDFL/ContextualDFLTraining"
const DEFAULT_REMOTE_JULIA = "/home/rwl/.juliaup/bin/julia"

function _contextualdfltraining_remote_eval(config)
    started_at = string(Dates.now(Dates.UTC))
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
            finished_at=string(Dates.now(Dates.UTC)),
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

function env_flag(name, default=false)
    value = lowercase(get(ENV, name, default ? "1" : "0"))
    return value in ("1", "true", "yes", "y")
end

function grid_mlflow_settings()
    enabled = env_flag("MLFLOW_ENABLED", haskey(ENV, "MLFLOW_EXPERIMENT_ID"))
    return (;
        enabled=enabled,
        experiment_id=get(ENV, "MLFLOW_EXPERIMENT_ID", ""),
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

    remote_specs = [
        ("rwl@gcp-big", env_worker_count("GCP_BIG_WORKERS", :auto)),
        ("rwl@gcp-small", env_worker_count("GCP_SMALL_WORKERS", :auto)),
    ]

    for (host, count) in remote_specs
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
            started_at = string(Dates.now(Dates.UTC))
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
                    finished_at=string(Dates.now(Dates.UTC)),
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
        started_at=string(Dates.now(Dates.UTC)),
        finished_at=string(Dates.now(Dates.UTC)),
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
                started_at=string(Dates.now(Dates.UTC)),
                finished_at=string(Dates.now(Dates.UTC)),
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

function mlflow_http_headers()
    return Dict("Connection" => "close")
end

function mlflow_filter_escape(value)
    return replace(string(value), "\\" => "\\\\", "\"" => "\\\"")
end

function unix_milliseconds()
    return round(Int64, Dates.datetime2unix(Dates.now()) * 1000)
end

function selected_grid()
    if env_flag("GRIDSEARCH_SMOKE", false)
        return smoke_grid()
    end
    return default_grid()
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
            mlflow_tracking_uri=mlflow_settings.tracking_uri,
            mlflow_upload_model_artifact=mlflow_settings.upload_model_artifact,
            mlflow_run_name=candidate_name,
            mlflow_tags=(;
                gridsearch_id=grid_id,
                gridsearch_timestamp=timestamp,
                candidate_index=Int(index),
                base_run_id=previous_run_id,
                candidate_name=candidate_name,
                gridsearch_role="candidate",
            ),
        ),
    )
end

function annotate_grid_configs(
    configs,
    timestamp::AbstractString,
    mlflow_settings=grid_mlflow_settings(),
)
    return [
        annotate_grid_config(config, index, timestamp, mlflow_settings) for
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
    configs = annotate_grid_configs(selected_grid(), timestamp, mlflow_settings)
    println("Grid search id: $grid_id")
    if mlflow_settings.enabled
        println("MLflow experiment id: $(mlflow_settings.experiment_id)")
    end
    println(
        "Running $(length(configs)) configuration(s) on $(length(remote_worker_ids)) remote worker(s)",
    )

    results = run_grid_on_remote_workers(remote_worker_ids, configs, worker_hosts)

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
