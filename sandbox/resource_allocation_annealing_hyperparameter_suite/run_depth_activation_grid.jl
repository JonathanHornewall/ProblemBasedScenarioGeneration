#!/usr/bin/env julia

include(joinpath(@__DIR__, "suite_common.jl"))

const GRID_PHASE = :depth_activation_grid

function parse_grid_args(args)
    parsed = Dict{String,Any}(
        "dry-run" => false,
        "smoke" => false,
        "jobs" => 4,
        "host" => DEFAULT_REMOTE_HOST,
        "remote-repo" => DEFAULT_REMOTE_REPO,
        "remote-julia" => DEFAULT_REMOTE_JULIA,
        "skip-sync" => false,
        "summarize-only" => false,
        "local-executor" => false,
    )

    index = 1
    while index <= length(args)
        arg = args[index]
        if arg == "--dry-run"
            parsed["dry-run"] = true
        elseif arg == "--smoke"
            parsed["smoke"] = true
        elseif arg == "--skip-sync"
            parsed["skip-sync"] = true
        elseif arg == "--summarize-only"
            parsed["summarize-only"] = true
        elseif arg in ("--local", "--local-executor")
            parsed["local-executor"] = true
        elseif arg == "--jobs"
            index += 1
            index <= length(args) || error("--jobs requires an integer")
            parsed["jobs"] = parse(Int, args[index])
        elseif arg == "--host"
            index += 1
            index <= length(args) || error("--host requires a host")
            parsed["host"] = args[index]
        elseif arg == "--remote-repo"
            index += 1
            index <= length(args) || error("--remote-repo requires a path")
            parsed["remote-repo"] = args[index]
        elseif arg == "--remote-julia"
            index += 1
            index <= length(args) || error("--remote-julia requires a path")
            parsed["remote-julia"] = args[index]
        elseif arg in ("-h", "--help")
            println("""
            Usage:
              julia run_depth_activation_grid.jl --dry-run [--smoke]
              julia run_depth_activation_grid.jl [--jobs N] [--host gcp-big]
              julia run_depth_activation_grid.jl --local-executor [--jobs N]
              julia run_depth_activation_grid.jl --summarize-only

            Runs a sandboxed depth x activation grid:
              depth in 1,2,3,4
              activation in GELU, SiLU

            All other parameters use the original annealing baseline:
              discrete mu schedule, 130 epochs, final 10 epochs with mu_ref = 0,
              rho = 0, batch size = 1, 100 training contexts.

            Use --local-executor when this script itself is already running on the
            target compute host. In that mode it runs run_single.jl directly
            instead of nesting ssh calls.
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

function remote_suite_dir(remote_repo)
    return joinpath(remote_repo, "sandbox", "resource_allocation_annealing_hyperparameter_suite")
end

function remote_project_dir(remote_repo)
    return joinpath(remote_repo, "src", "ContextualDFL", "ContextualDFLTraining")
end

function ssh_cmd(host, remote_shell)
    return Cmd(["ssh", String(host), "--", String(remote_shell)])
end

function rsync_from_remote_cmd(host, remote_suite)
    return Cmd(["rsync", "-avu", String(host) * ":" * String(remote_suite) * "/", SUITE_DIR * "/"])
end

function sync_code!(args)
    if Bool(args["skip-sync"])
        println("Skipping sync because --skip-sync was provided.")
        return nothing
    end
    script = joinpath(homedir(), "sync-julia-code.sh")
    isfile(script) || error("sync script not found: $script")
    println("Syncing local repo to remote machines with $script")
    run(Cmd(Cmd([script]); dir=homedir()))
    return nothing
end

function pull_remote_artifacts!(args)
    host = String(args["host"])
    remote_suite = remote_suite_dir(String(args["remote-repo"]))
    println("Pulling remote sandbox artifacts from $host:$remote_suite")
    run(rsync_from_remote_cmd(host, remote_suite))
    return nothing
end

function remote_run_command(args, config)
    remote_repo = String(args["remote-repo"])
    remote_suite = remote_suite_dir(remote_repo)
    remote_project = remote_project_dir(remote_repo)
    remote_julia = String(args["remote-julia"])
    remote_run = joinpath(remote_suite, config.run_rel_dir)
    remote_config = joinpath(remote_run, "config.jls")

    return join(
        [
            "mkdir -p " * shell_quote(remote_run),
            "cd " * shell_quote(remote_suite),
            shell_quote(remote_julia) *
                " --project=" * shell_quote(remote_project) *
                " " * shell_quote(joinpath(remote_suite, "run_single.jl")) *
                " --config " * shell_quote(remote_config) *
                " > " * shell_quote(joinpath(remote_run, "stdout.log")) *
                " 2> " * shell_quote(joinpath(remote_run, "stderr.log")),
        ],
        " && ",
    )
end

function grid_configs(; smoke=false)
    reps = smoke ? 1 : REPLICATES
    return [
        base_run_config(
            phase=GRID_PHASE,
            candidate_name="depth$(depth)_$(activation)",
            replicate=replicate,
            smoke=smoke,
            depth=depth,
            activation=activation,
            mu_schedule_kind=:discrete,
            rho=0.0,
            batch_size=1,
            epochs=smoke ? 1 : BASE_TOTAL_EPOCHS,
            final_epochs=final_epochs_for(smoke ? 1 : BASE_TOTAL_EPOCHS; smoke=smoke),
        ) for depth in (1, 2, 3, 4), activation in (:gelu, :silu) for
        replicate in 1:reps
    ]
end

function write_grid_configs!(configs; smoke=false)
    paths = phase_paths(GRID_PHASE; smoke=smoke)
    mkpath(paths.dir)
    for config in configs
        write_config!(config)
    end
    write_rows_csv(paths.configs, [config_summary_row(config) for config in configs])
    return configs
end

function pending_configs(configs)
    return [config for config in configs if !run_complete(config)]
end

function run_remote_config(args, config)
    if run_complete(config)
        println("Skipping completed run $(config.run_id)")
        return true
    end

    println("Launching $(config.run_id)")
    try
        run(ssh_cmd(args["host"], remote_run_command(args, config)))
        return true
    catch error
        println("Remote command failed for $(config.run_id): ", sprint(showerror, error))
        return false
    end
end

function run_local_config(config)
    if run_complete(config)
        println("Skipping completed run $(config.run_id)")
        flush(stdout)
        return true
    end

    println("Launching $(config.run_id)")
    flush(stdout)
    mkpath(run_dir(config))
    cmd = Cmd([
        joinpath(Sys.BINDIR, "julia"),
        "--project=" * TRAINING_PROJECT_DIR,
        joinpath(SUITE_DIR, "run_single.jl"),
        "--config",
        config_path(config),
    ])

    try
        run(
            pipeline(
                cmd;
                stdout=joinpath(run_dir(config), "stdout.log"),
                stderr=joinpath(run_dir(config), "stderr.log"),
            ),
        )
        return true
    catch error
        println("Local command failed for $(config.run_id): ", sprint(showerror, error))
        flush(stdout)
        return false
    end
end

function run_config_queue!(configs, run_one!; jobs)
    pending = pending_configs(configs)
    isempty(pending) && begin
        println("All grid runs are already complete.")
        flush(stdout)
        return nothing
    end

    queue = collect(pending)
    queue_lock = ReentrantLock()

    function next_config!()
        lock(queue_lock)
        try
            isempty(queue) && return nothing
            return popfirst!(queue)
        finally
            unlock(queue_lock)
        end
    end

    worker_count = min(Int(jobs), length(pending))
    tasks = [
        @async begin
            while true
                config = next_config!()
                config === nothing && break
                run_one!(config)
            end
        end for _ in 1:worker_count
    ]
    foreach(wait, tasks)
    return nothing
end

function run_remote_grid!(args, configs)
    return run_config_queue!(
        configs,
        config -> run_remote_config(args, config);
        jobs=Int(args["jobs"]),
    )
end

function run_local_grid!(args, configs)
    return run_config_queue!(configs, run_local_config; jobs=Int(args["jobs"]))
end

function summarize_grid!(configs; smoke=false)
    paths = phase_paths(GRID_PHASE; smoke=smoke)
    mkpath(paths.dir)
    results = summarize_phase_results(configs)
    write_rows_csv(paths.configs, [config_summary_row(config) for config in configs])
    write_rows_csv(paths.runs, results.runs)
    write_rows_csv(paths.epochs, results.epochs)
    write_rows_csv(paths.summary, results.summary)

    if any(row.ok_count == row.run_count && row.run_count > 0 for row in results.summary)
        decision = choose_candidate(results.summary)
        write_rows_csv(paths.decision, [decision])
        write(
            paths.decision_md,
            "# Depth x Activation Grid Decision\n\n" *
            "- Selected candidate: `$(decision.candidate_name)`\n" *
            "- Mean test relative regret: `$(decision.mean_test_relative_regret)`\n" *
            "- Successful replicates: `$(decision.ok_count)/$(decision.run_count)`\n",
        )
    else
        write_rows_csv(paths.decision, NamedTuple[])
        write(paths.decision_md, "# Depth x Activation Grid Decision\n\nNo complete candidate yet.\n")
    end

    return results
end

function print_dry_run(args, configs)
    println("Dry run only; no remote commands will be run.")
    println("Host: ", args["host"])
    println("Remote Julia: ", args["remote-julia"])
    println("Remote repo: ", args["remote-repo"])
    println("Jobs: ", args["jobs"])
    println("Runs: ", length(configs))
    println("Candidates: ", join(sort(unique(config.candidate_name for config in configs)), ", "))
    println("Example run id: ", first(configs).run_id)
    println("Example remote command:")
    println(remote_run_command(args, first(configs)))
end

function main()
    args = parse_grid_args(ARGS)
    smoke = Bool(args["smoke"])
    configs = grid_configs(smoke=smoke)

    if Bool(args["dry-run"])
        print_dry_run(args, configs)
        return nothing
    end

    write_grid_configs!(configs; smoke=smoke)

    if Bool(args["summarize-only"])
        summarize_grid!(configs; smoke=smoke)
        return nothing
    end

    test_cache_exists(smoke=smoke) ||
        error("Missing shared test cache. Run run_suite.jl once or run_single.jl --precompute first.")

    if Bool(args["local-executor"])
        run_local_grid!(args, configs)
    else
        sync_code!(args)
        run_remote_grid!(args, configs)
        pull_remote_artifacts!(args)
    end

    summarize_grid!(configs; smoke=smoke)
    println("Depth x activation grid complete.")
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
