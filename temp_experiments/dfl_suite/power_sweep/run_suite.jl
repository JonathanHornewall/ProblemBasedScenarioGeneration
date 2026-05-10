#!/usr/bin/env julia

include(joinpath(@__DIR__, "common.jl"))

function parse_suite_args(args)
    parsed = Dict{String,Any}(
        "smoke" => false,
        "jobs" => 14,
        "force-test-data" => false,
        "precompute-only" => false,
        "assemble-only" => false,
        "dry-run" => false,
    )

    index = 1
    while index <= length(args)
        arg = args[index]
        if arg == "--smoke"
            parsed["smoke"] = true
        elseif arg == "--force-test-data"
            parsed["force-test-data"] = true
        elseif arg == "--precompute-only"
            parsed["precompute-only"] = true
        elseif arg == "--assemble-only"
            parsed["assemble-only"] = true
        elseif arg == "--dry-run"
            parsed["dry-run"] = true
        elseif arg == "--jobs"
            index += 1
            index <= length(args) || error("--jobs requires an integer")
            parsed["jobs"] = parse(Int, args[index])
        elseif arg in ("-h", "--help")
            println("""
            Usage:
              julia run_suite.jl [--jobs 12] [--smoke] [--force-test-data]
              julia run_suite.jl --precompute-only [--smoke]
              julia run_suite.jl --assemble-only [--smoke]
              julia run_suite.jl --dry-run [--smoke]
            """)
            exit(0)
        else
            error("unknown argument: $arg")
        end
        index += 1
    end

    parsed["jobs"] > 0 || error("--jobs must be positive")
    return parsed
end

function write_suite_configs!(configs; smoke=false)
    for config in configs
        write_config!(config)
    end
    root = summary_root(smoke=smoke)
    mkpath(root)
    write_rows_csv(joinpath(root, "all_configs.csv"), [config_summary_row(config) for config in configs])
    for group in sort(unique(config.group for config in configs))
        dir = joinpath(root, group)
        mkpath(dir)
        group_configs = [config for config in configs if config.group == group]
        write_rows_csv(joinpath(dir, "configs.csv"), [config_summary_row(config) for config in group_configs])
    end
    return configs
end

function pending_configs(configs)
    return [config for config in configs if !run_complete(config)]
end

function julia_executable()
    return joinpath(Sys.BINDIR, Base.julia_exename())
end

function run_command(config)
    return Cmd([
        julia_executable(),
        "--project=" * TRAINING_PROJECT_DIR,
        joinpath(SUITE_DIR, "run_single.jl"),
        "--config",
        config_path(config),
    ])
end

function capped_environment()
    env = copy(ENV)
    return env
end

function launch_config(config)
    mkpath(run_dir(config))
    cmd = setenv(run_command(config), capped_environment())
    proc = run(
        pipeline(
            cmd;
            stdout=stdout_path(config),
            stderr=stderr_path(config),
        );
        wait=false,
    )
    println("Launched $(config.run_id) pid=$(getpid(proc))")
    return (;
        config=config,
        process=proc,
        pid=getpid(proc),
        launched_at=unix_milliseconds(),
    )
end

function active_row(job)
    return (;
        run_id=job.config.run_id,
        group=job.config.group,
        candidate_name=job.config.candidate_name,
        replicate=job.config.replicate,
        pid=job.pid,
        launched_at=job.launched_at,
        running=process_running(job.process),
    )
end

function write_controller_status(path, active, completed, queue)
    rows = NamedTuple[]
    for job in active
        push!(rows, merge(active_row(job), (; controller_status="running", exitcode=missing)))
    end
    for row in completed
        push!(rows, row)
    end
    for config in queue
        push!(
            rows,
            (;
                run_id=config.run_id,
                group=config.group,
                candidate_name=config.candidate_name,
                replicate=config.replicate,
                pid=missing,
                launched_at=missing,
                running=false,
                controller_status="queued",
                exitcode=missing,
            ),
        )
    end
    write_rows_csv(path, rows)
end

function run_queue!(configs; jobs, smoke=false)
    queue = pending_configs(configs)
    if isempty(queue)
        println("No pending runs.")
        return nothing
    end

    root = summary_root(smoke=smoke)
    mkpath(root)
    controller_status_path = joinpath(root, "controller_status.csv")
    active = NamedTuple[]
    completed = NamedTuple[]

    while !isempty(queue) || !isempty(active)
        while length(active) < jobs && !isempty(queue)
            config = popfirst!(queue)
            push!(active, launch_config(config))
            sleep(0.25)
        end

        write_controller_status(controller_status_path, active, completed, queue)
        sleep(5)

        still_active = NamedTuple[]
        for job in active
            if process_running(job.process)
                push!(still_active, job)
                continue
            end

            try
                wait(job.process)
            catch
            end
            exitcode = try
                job.process.exitcode
            catch
                missing
            end
            status = exitcode == 0 ? "finished" : "failed"
            println("Completed $(job.config.run_id) status=$(status) exitcode=$(exitcode)")
            push!(
                completed,
                (;
                    run_id=job.config.run_id,
                    group=job.config.group,
                    candidate_name=job.config.candidate_name,
                    replicate=job.config.replicate,
                    pid=job.pid,
                    launched_at=job.launched_at,
                    running=false,
                    controller_status=status,
                    exitcode=exitcode,
                ),
            )
        end
        active = still_active
        write_all_summaries!(configs; smoke=smoke)
    end

    write_controller_status(controller_status_path, active, completed, queue)
    return nothing
end

function print_dry_run(configs; jobs, smoke=false)
    println("Suite dir: ", SUITE_DIR)
    println("Project: ", TRAINING_PROJECT_DIR)
    println("Smoke: ", smoke)
    println("Jobs: ", jobs)
    println("Total configs: ", length(configs))
    for group in sort(unique(config.group for config in configs))
        group_configs = [config for config in configs if config.group == group]
        candidates = sort(unique(config.candidate_name for config in group_configs))
        println(group, ": ", length(group_configs), " runs; candidates=", join(candidates, ", "))
    end
end

function main()
    args = parse_suite_args(ARGS)
    smoke = Bool(args["smoke"])
    configs = write_suite_configs!(experiment_configs(smoke=smoke); smoke=smoke)

    if Bool(args["dry-run"])
        print_dry_run(configs; jobs=Int(args["jobs"]), smoke=smoke)
        return nothing
    end

    if !Bool(args["assemble-only"])
        cache = ensure_test_cache!(
            smoke=smoke,
            force=Bool(args["force-test-data"]),
        )
        println("Using test cache digest $(cache.metadata.dataset_digest)")
    end

    if Bool(args["precompute-only"])
        return nothing
    end

    if !Bool(args["assemble-only"])
        run_queue!(configs; jobs=Int(args["jobs"]), smoke=smoke)
    end

    output = write_all_summaries!(configs; smoke=smoke)
    println("Summaries written to $(output)")
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
