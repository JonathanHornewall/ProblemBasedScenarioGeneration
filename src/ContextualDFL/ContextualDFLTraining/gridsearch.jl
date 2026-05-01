#!/usr/bin/env julia

using Dates
using Distributed
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

function selected_grid()
    if env_flag("GRIDSEARCH_SMOKE", false)
        return smoke_grid()
    end
    return default_grid()
end

function main()
    ensure_clean_worker_start!()
    sync_code!()
    remote_worker_ids = add_remote_workers!()
    load_worker_stdlibs!()
    assert_remote_only_workers!(remote_worker_ids)
    load_training_project_on_workers!(remote_worker_ids)
    define_remote_eval!()

    configs = selected_grid()
    println(
        "Running $(length(configs)) configuration(s) on $(length(remote_worker_ids)) remote worker(s)",
    )

    pool = Distributed.WorkerPool(remote_worker_ids)
    results = pmap(
        _contextualdfltraining_remote_eval,
        pool,
        configs;
        batch_size=env_int("PMAP_BATCH_SIZE", 1),
    )

    output_dir = write_grid_results(
        results;
        configs=configs,
        output_root=joinpath(@__DIR__, "results"),
    )
    println("Wrote grid-search CSV results to $output_dir")

    failed_count = count(result -> result.status != "ok", results)
    failed_count > 0 && println("Recorded $failed_count failed configuration(s).")
    return output_dir
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
