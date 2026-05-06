#!/usr/bin/env julia

using Distributed
using Base64
import Pkg

const DRIVER_SUITE_DIR = @__DIR__
const DRIVER_COMMON_PATH = joinpath(DRIVER_SUITE_DIR, "suite_common.jl")
const DRIVER_REPO_ROOT = normpath(joinpath(DRIVER_SUITE_DIR, "..", "..", ".."))
const DRIVER_TRAINING_PROJECT_DIR =
    joinpath(DRIVER_REPO_ROOT, "src", "ContextualDFL", "ContextualDFLTraining")

function activate_training_project_on_existing_workers!()
    nworkers() > 0 || return nothing
    @sync for worker in workers()
        @async remotecall_fetch(worker, DRIVER_TRAINING_PROJECT_DIR) do project_dir
            Core.eval(Main, :(import Pkg))
            pkg = Base.invokelatest(getfield, Main, :Pkg)
            Base.invokelatest(pkg.activate, project_dir; io=devnull)
            return nothing
        end
    end
    return nothing
end

Pkg.activate(DRIVER_TRAINING_PROJECT_DIR; io=devnull)
activate_training_project_on_existing_workers!()
include(DRIVER_COMMON_PATH)

function parse_args(args)
    parsed = Dict{String,Any}(
        "smoke" => false,
        "dry-run" => false,
        "jobs" => nothing,
        "test-contexts" => 0,
        "eval-batches" => 0,
    )

    index = 1
    while index <= length(args)
        arg = args[index]
        if arg == "--smoke"
            parsed["smoke"] = true
        elseif arg == "--dry-run"
            parsed["dry-run"] = true
        elseif arg == "--jobs"
            index += 1
            index <= length(args) || error("--jobs requires an integer")
            parsed["jobs"] = parse(Int, args[index])
        elseif arg == "--test-contexts"
            index += 1
            index <= length(args) || error("--test-contexts requires an integer")
            parsed["test-contexts"] = parse(Int, args[index])
        elseif arg == "--eval-batches"
            index += 1
            index <= length(args) || error("--eval-batches requires an integer")
            parsed["eval-batches"] = parse(Int, args[index])
        elseif arg in ("-h", "--help")
            println("""
            Usage:
              julia --machine-file=<sandbox>/machines_no_ibm96c1_10workers.txt --project=$(DRIVER_TRAINING_PROJECT_DIR) run_experiments.jl [--jobs N]
              julia --machine-file=<sandbox>/machines_no_ibm96c1_2workers.txt --project=$(DRIVER_TRAINING_PROJECT_DIR) run_experiments.jl --smoke --jobs 2

            Runs ordered waves:
              1. 10 standard n100/depth4/GeLU/base-schedule runs
              2. 10 standard n100/depth4/GeLU/alternate-schedule runs
              3. 7 n1000 data/compute candidates with 5 base-schedule replicates each
              4. the same 7 n1000 candidates with 5 alternate-schedule replicates each

            All outputs stay under:
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
    return min(nworkers(), smoke ? 2 : 10)
end

function selected_workers(jobs; smoke=false)
    nworkers() > 0 || error(
        "No Julia workers are available. Run with a filtered --machine-file that excludes ibm-96c-1.",
    )
    count = jobs === nothing ? default_job_count(smoke=smoke) : Int(jobs)
    count = max(1, min(count, nworkers()))
    return workers()[1:count]
end

function initialize_workers!(worker_ids)
    println("Loading experiment code on $(length(worker_ids)) worker(s).")
    flush(stdout)
    tasks = [
        @async remotecall_fetch(
            worker,
            DRIVER_COMMON_PATH,
            DRIVER_TRAINING_PROJECT_DIR,
        ) do common_path, project_dir
            Core.eval(Main, :(import Pkg))
            pkg = Base.invokelatest(getfield, Main, :Pkg)
            Base.invokelatest(pkg.activate, project_dir; io=devnull)
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

function initialize_worker_caches!(worker_ids; smoke=false, context_limit=0, evaluation_batch_limit=0)
    println("Loading packaged test artifacts on $(length(worker_ids)) worker(s).")
    flush(stdout)
    tasks = [
        @async begin
            info = remotecall_fetch(
                () -> set_worker_test_cache!(
                    smoke=smoke,
                    context_limit=context_limit,
                    evaluation_batch_limit=evaluation_batch_limit,
                ),
                worker,
            )
            println(
                "Worker $(info.worker_id) on $(info.hostname) loaded " *
                "$(info.test_contexts) test context(s), $(info.evaluation_batches) batch(es).",
            )
            flush(stdout)
            return info
        end for worker in worker_ids
    ]
    return fetch.(tasks)
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
                write_checkpoint_csv_mirror(checkpoint_file, bytes)
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

function write_checkpoint_csv_mirror(checkpoint_file, bytes)
    mirror_file = checkpoint_file * ".csv"
    open(mirror_file, "w") do io
        println(io, "format,base64")
        print(io, "julia_serialized_checkpoint,")
        print(io, base64encode(bytes))
        print(io, "\n")
    end
    return mirror_file
end

function stop_csv_logger!(logger)
    put!(logger.channel, (; kind=:stop, row=(;)))
    wait(logger.task)
    return nothing
end

function log_configs!(logger, configs)
    for config in configs
        put_log!(logger.channel, :config, config_row(config))
    end
    return nothing
end

function run_wave!(wave, worker_ids, logger; smoke=false)
    configs = wave.configs
    log_configs!(logger, configs)
    pending = pending_configs(configs; smoke=smoke)
    if isempty(pending)
        println("Wave $(wave.wave_index) $(wave.wave_name): all $(length(configs)) run(s) are already complete.")
        return NamedTuple[]
    end

    active_worker_count = min(length(worker_ids), length(pending))
    active_workers = worker_ids[1:active_worker_count]
    pool = CachingPool(active_workers)
    log_channel = logger.channel
    println(
        "Wave $(wave.wave_index) $(wave.wave_name): running $(length(pending)) pending run(s) on " *
        "$(length(active_workers)) worker(s).",
    )
    flush(stdout)

    heartbeat_done = Ref(false)
    heartbeat = @async begin
        while !heartbeat_done[]
            sleep(60)
            heartbeat_done[] && break
            completed = length(completed_run_ids(smoke=smoke))
            println(
                "Wave $(wave.wave_index) $(wave.wave_name): still running; " *
                "$(completed) completed run(s) recorded so far.",
            )
            flush(stdout)
        end
    end

    results = try
        pmap(config -> run_experiment_config(config, log_channel), pool, pending)
    finally
        heartbeat_done[] = true
        wait(heartbeat)
    end

    phase_summary = summarize_phase(wave.phase; smoke=smoke)
    write_csv_file(phase_summary_path(wave.phase; smoke=smoke), SUMMARY_HEADERS, phase_summary)
    all_summary = summarize_all(smoke=smoke)
    write_csv_file(result_paths(smoke=smoke).summary, SUMMARY_HEADERS, all_summary)
    println("Wave $(wave.wave_index) $(wave.wave_name): summary updated.")
    flush(stdout)
    return results
end

function print_dry_run(; smoke=false)
    waves = planned_waves(smoke=smoke)
    total = sum(length(wave.configs) for wave in waves)
    println("Dry run matrix: $(length(waves)) wave(s), $(total) run(s)")
    for wave in waves
        println("Wave $(wave.wave_index): $(wave.wave_name) ($(length(wave.configs)) run(s))")
        for config in wave.configs
            println(
                "  ",
                config.run_id,
                " n=",
                config.training_contexts,
                " epochs=",
                config.epochs,
                " batch=",
                config.batch_size,
                " schedule=",
                config.schedule_name,
            )
        end
    end
end

function write_planned_configs!(; smoke=false)
    rows = [config_row(config) for config in all_configs(smoke=smoke)]
    write_csv_file(result_paths(smoke=smoke).planned_configs, CONFIG_HEADERS, rows)
    return rows
end

function main()
    args = parse_args(ARGS)
    smoke = Bool(args["smoke"])
    if Bool(args["dry-run"])
        print_dry_run(smoke=smoke)
        return nothing
    end

    worker_ids = selected_workers(args["jobs"]; smoke=smoke)
    initialize_workers!(worker_ids)
    initialize_worker_caches!(
        worker_ids;
        smoke=smoke,
        context_limit=Int(args["test-contexts"]),
        evaluation_batch_limit=Int(args["eval-batches"]),
    )

    logger = start_csv_logger(smoke=smoke)
    try
        write_planned_configs!(smoke=smoke)
        println("Sandbox: $(SUITE_DIR)")
        println("Results: $(result_paths(smoke=smoke).dir)")
        println("Active workers: $(length(worker_ids)) / $(nworkers())")
        flush(stdout)

        for wave in planned_waves(smoke=smoke)
            run_wave!(wave, worker_ids, logger; smoke=smoke)
        end
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
