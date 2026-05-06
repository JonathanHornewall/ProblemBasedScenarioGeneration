#!/usr/bin/env julia

using Distributed
import Pkg

const DRIVER_SUITE_DIR = @__DIR__
const DRIVER_COMMON_PATH = joinpath(DRIVER_SUITE_DIR, "experiment_common.jl")
const DRIVER_REPO_ROOT = normpath(joinpath(DRIVER_SUITE_DIR, "..", "..", ".."))
const DRIVER_TRAINING_PROJECT_DIR =
    joinpath(DRIVER_REPO_ROOT, "src", "ContextualDFL", "ContextualDFLTraining")

if nworkers() > 0
    @sync for worker in workers()
        @async remotecall_fetch(worker, DRIVER_TRAINING_PROJECT_DIR) do project_dir
            Core.eval(Main, :(import Pkg))
            Base.invokelatest(Main.Pkg.activate, project_dir; io=devnull)
            return nothing
        end
    end
end

include(DRIVER_COMMON_PATH)

function parse_args(args)
    parsed = Dict{String,Any}(
        "smoke" => false,
        "dry-run" => false,
        "force-test-data" => false,
        "jobs" => nothing,
    )

    index = 1
    while index <= length(args)
        arg = args[index]
        if arg == "--smoke"
            parsed["smoke"] = true
        elseif arg == "--dry-run"
            parsed["dry-run"] = true
        elseif arg == "--force-test-data"
            parsed["force-test-data"] = true
        elseif arg == "--jobs"
            index += 1
            index <= length(args) || error("--jobs requires an integer")
            parsed["jobs"] = parse(Int, args[index])
        elseif arg in ("-h", "--help")
            println("""
            Usage:
              julia --machine-file=/home/rwl/gcp-machines.txt --project=<ContextualDFLTraining> run_experiments.jl [--jobs N]
              julia --machine-file=/home/rwl/gcp-machines.txt --project=<ContextualDFLTraining> run_experiments.jl --smoke --jobs 2

            Runs, in order:
              1. 10 standard n100/depth4/GeLU baseline runs
              2. 5 runs each for n500 and n1000 with normalized gradient-step budgets
              3. 6 runs each for depths 5, 6, 10, 20, and 40

            Results and checkpoints stay under:
              $(DRIVER_SUITE_DIR)
            """)
            exit(0)
        else
            error("unknown argument: $arg")
        end
        index += 1
    end

    return parsed
end

function default_job_count(; smoke=false)
    nworkers() > 0 || return 0
    max_wave = smoke ? 2 : maximum(length, (baseline_configs(), data_amount_configs(), depth_configs()))
    return min(nworkers(), max_wave)
end

function selected_workers(jobs)
    nworkers() > 0 || error(
        "No Julia workers are available. Run with --machine-file=/home/rwl/gcp-machines.txt.",
    )
    count = jobs === nothing ? default_job_count() : Int(jobs)
    count = max(1, min(count, nworkers()))
    return workers()[1:count]
end

function initialize_workers!(worker_ids)
    println("Loading experiment code on $(length(worker_ids)) worker(s) asynchronously.")
    flush(stdout)
    tasks = [
        @async remotecall_fetch(
            worker,
            DRIVER_COMMON_PATH,
            DRIVER_TRAINING_PROJECT_DIR,
        ) do common_path, project_dir
            Core.eval(Main, :(import Pkg))
            Base.invokelatest(Main.Pkg.activate, project_dir; io=devnull)
            include(common_path)
            return (;
                worker_id=Distributed.myid(),
                hostname=Sockets.gethostname(),
                pid=getpid(),
                julia_version=string(VERSION),
            )
        end for worker in worker_ids
    ]

    metadata = fetch.(tasks)
    hosts = sort(unique(string(item.hostname) for item in metadata))
    println("Loaded $(length(metadata)) worker(s) on $(length(hosts)) host(s): ", join(hosts, ", "))
    flush(stdout)
    return metadata
end

function start_csv_logger(; smoke=false)
    paths = result_paths(smoke=smoke)
    mkpath(paths.dir)
    mkpath(paths.checkpoints_dir)
    channel = RemoteChannel(() -> Channel{Any}(100_000), 1)

    task = @async begin
        while true
            message = take!(channel)
            if message === :stop ||
               (message isa NamedTuple && :kind in keys(message) && message.kind == :stop)
                break
            end

            kind = message.kind
            row = message.row
            if kind == :attempt
                append_csv_row(paths.attempts, ATTEMPT_HEADERS, row)
            elseif kind == :config
                append_csv_row(paths.configs, CONFIG_HEADERS, row)
            elseif kind == :epoch
                append_csv_row(paths.epochs, EPOCH_HEADERS, row)
            elseif kind == :run
                append_csv_row(paths.runs, RUN_HEADERS, row)
            elseif kind == :test_sample
                append_csv_row(paths.test_samples, TEST_SAMPLE_HEADERS, row)
            elseif kind == :checkpoint
                bytes = row.checkpoint_bytes
                checkpoint_file = joinpath(paths.dir, row.checkpoint_path)
                mkpath(dirname(checkpoint_file))
                write(checkpoint_file, bytes)
                manifest_row = merge(
                    row,
                    (;
                        checkpoint_path=checkpoint_file,
                        checkpoint_bytes=length(bytes),
                    ),
                )
                append_csv_row(paths.checkpoints, CHECKPOINT_HEADERS, manifest_row)
            else
                @warn "Ignoring unknown logger message kind" kind
            end
        end
    end

    return (; channel=channel, task=task)
end

function stop_csv_logger!(logger)
    put!(logger.channel, (; kind=:stop, row=(;)))
    wait(logger.task)
    return nothing
end

function ensure_test_cache!(worker_ids; smoke=false, force=false)
    if !force && test_cache_exists(smoke=smoke)
        println("Loading precomputed test data from CSV.")
        return load_test_cache_from_csv(smoke=smoke)
    end

    worker = first(worker_ids)
    println("Precomputing shared test data and optima on worker $(worker).")
    flush(stdout)
    cache = remotecall_fetch(() -> precompute_test_cache(smoke=smoke), worker)
    write_test_cache_csv!(cache; smoke=smoke)
    println("Wrote shared test cache CSV files to $(test_cache_paths(smoke=smoke).dir).")
    flush(stdout)
    return (; dataset=cache.dataset, optimal_results=cache.optimal_results, metadata=cache.metadata)
end

function initialize_worker_caches!(worker_ids, cache)
    println("Sending shared test cache to $(length(worker_ids)) worker(s).")
    flush(stdout)
    tasks = [
        @async begin
            info = remotecall_fetch(set_worker_test_cache!, worker, cache)
            println(
                "Worker $(info.worker_id) on $(info.hostname) has " *
                "$(info.test_contexts) test contexts.",
            )
            flush(stdout)
            return info
        end for worker in worker_ids
    ]
    return fetch.(tasks)
end

function log_configs!(logger, configs)
    for config in configs
        put_log!(logger.channel, :config, config_row(config))
    end
    return nothing
end

function run_wave!(phase, configs, worker_ids, logger; smoke=false)
    log_configs!(logger, configs)
    pending = pending_configs(configs; smoke=smoke)
    if isempty(pending)
        println("Phase $(phase): all $(length(configs)) run(s) are already complete.")
        return NamedTuple[]
    end

    pool = CachingPool(worker_ids)
    log_channel = logger.channel
    println(
        "Phase $(phase): running $(length(pending)) pending run(s) on " *
        "$(length(worker_ids)) worker(s).",
    )
    flush(stdout)

    heartbeat_done = Ref(false)
    heartbeat = @async begin
        while !heartbeat_done[]
            sleep(60)
            heartbeat_done[] && break
            println("Phase $(phase): still running $(length(pending)) run(s).")
            flush(stdout)
        end
    end

    results = try
        pmap(config -> run_experiment_config(config, log_channel), pool, pending)
    finally
        heartbeat_done[] = true
        wait(heartbeat)
    end

    summary = summarize_phase(phase; smoke=smoke)
    write_csv_file(phase_summary_path(phase; smoke=smoke), SUMMARY_HEADERS, summary)
    println("Phase $(phase): wrote summary to $(phase_summary_path(phase; smoke=smoke)).")
    flush(stdout)
    return results
end

function print_dry_run(; smoke=false)
    waves = (
        :baseline => baseline_configs(smoke=smoke),
        :data_amount => data_amount_configs(smoke=smoke),
        :depth => depth_configs(smoke=smoke),
    )
    for (phase, configs) in waves
        println("Phase $(phase): $(length(configs)) run(s)")
        for config in configs
            println(
                "  ",
                config.run_id,
                " n=",
                config.training_contexts,
                " epochs=",
                config.epochs,
                " depth=",
                config.depth,
                " activation=",
                config.activation,
            )
        end
    end
end

function main()
    args = parse_args(ARGS)
    smoke = Bool(args["smoke"])
    if Bool(args["dry-run"])
        print_dry_run(smoke=smoke)
        return nothing
    end

    jobs = args["jobs"] === nothing ? default_job_count(smoke=smoke) : Int(args["jobs"])
    worker_ids = selected_workers(jobs)
    initialize_workers!(worker_ids)
    cache = ensure_test_cache!(
        worker_ids;
        smoke=smoke,
        force=Bool(args["force-test-data"]),
    )
    initialize_worker_caches!(worker_ids, cache)

    logger = start_csv_logger(smoke=smoke)
    try
        println("Sandbox: $(SUITE_DIR)")
        println("Results: $(result_paths(smoke=smoke).dir)")
        println("Active workers: $(length(worker_ids)) / $(nworkers())")
        flush(stdout)

        run_wave!(:baseline, baseline_configs(smoke=smoke), worker_ids, logger; smoke=smoke)
        run_wave!(:data_amount, data_amount_configs(smoke=smoke), worker_ids, logger; smoke=smoke)
        run_wave!(:depth, depth_configs(smoke=smoke), worker_ids, logger; smoke=smoke)
    finally
        stop_csv_logger!(logger)
    end

    println("Experiment driver complete.")
    flush(stdout)
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
