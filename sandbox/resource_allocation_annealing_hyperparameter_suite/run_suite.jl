#!/usr/bin/env julia

include(joinpath(@__DIR__, "suite_common.jl"))

function parse_suite_args(args)
    parsed = Dict{String,Any}(
        "dry-run" => false,
        "smoke" => false,
        "jobs" => 8,
        "host" => DEFAULT_REMOTE_HOST,
        "remote-repo" => DEFAULT_REMOTE_REPO,
        "remote-julia" => DEFAULT_REMOTE_JULIA,
        "skip-sync" => false,
        "force-test-data" => false,
        "strict-git-clean" => false,
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
        elseif arg == "--force-test-data"
            parsed["force-test-data"] = true
        elseif arg == "--strict-git-clean"
            parsed["strict-git-clean"] = true
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
              julia run_suite.jl --dry-run [--smoke]
              julia run_suite.jl [--smoke] [--jobs 8] [--host gcp-big]

            Coordinates the sandboxed resource-allocation annealing tuning suite.
            Training runs execute remotely; outputs stay under this sandbox directory.
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

function remote_precompute_command(args)
    remote_repo = String(args["remote-repo"])
    remote_suite = remote_suite_dir(remote_repo)
    remote_project = remote_project_dir(remote_repo)
    remote_julia = String(args["remote-julia"])
    smoke_flag = Bool(args["smoke"]) ? " --smoke" : ""
    force_flag = Bool(args["force-test-data"]) ? " --force-test-data" : ""

    return join(
        [
            "cd " * shell_quote(remote_suite),
            shell_quote(remote_julia) *
                " --project=" * shell_quote(remote_project) *
                " " * shell_quote(joinpath(remote_suite, "run_single.jl")) *
                " --precompute" * smoke_flag * force_flag,
        ],
        " && ",
    )
end

function ensure_remote_test_cache!(args)
    if !Bool(args["force-test-data"]) && test_cache_exists(smoke=Bool(args["smoke"]))
        println("Local test cache already exists; sync will make it available remotely.")
        return nothing
    end

    sync_code!(args)
    println("Precomputing shared test data and SAA optima on $(args["host"])")
    run(ssh_cmd(args["host"], remote_precompute_command(args)))
    pull_remote_artifacts!(args)
    test_cache_exists(smoke=Bool(args["smoke"])) ||
        error("remote precompute finished but local test cache is still missing")
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

function print_dry_run(args)
    println("Dry run only; no files will be written and no remote commands will be run.")
    println("Host: ", args["host"])
    println("Remote Julia: ", args["remote-julia"])
    println("Remote repo: ", args["remote-repo"])
    println("Jobs: ", args["jobs"])
    println("Smoke: ", args["smoke"])

    selection = default_selection()
    for phase in PHASES
        configs = phase_configs(phase, selection; smoke=Bool(args["smoke"]))
        println()
        println("Phase $(phase): $(length(configs)) run(s)")
        candidates = sort(unique(config.candidate_name for config in configs))
        println("Candidates: ", join(candidates, ", "))
        println("Example run id: ", first(configs).run_id)
        println("Example remote command:")
        println(remote_run_command(args, first(configs)))

        if phase == :depth
            selection = merge(selection, (; depth=3))
        elseif phase == :activation
            selection = merge(selection, (; activation=:relu))
        elseif phase == :mu_schedule
            selection = merge(selection, (; mu_schedule_kind=:geometric))
        elseif phase == :rho
            selection = merge(selection, (; rho=0.0, best_nonzero_rho=1e-3))
        end
    end
end

function git_status_lines()
    try
        text = read(Cmd(Cmd(["git", "status", "--short"]); dir=REPO_ROOT), String)
        return filter(!isempty, split(chomp(text), '\n'))
    catch
        return String[]
    end
end

function sandbox_status_line(line)
    stripped = strip(line)
    return occursin(" sandbox/resource_allocation_annealing_hyperparameter_suite", stripped) ||
           startswith(stripped, "?? sandbox/resource_allocation_annealing_hyperparameter_suite")
end

function check_git_scope!(; strict=false)
    outside = [line for line in git_status_lines() if !sandbox_status_line(line)]
    isempty(outside) && return nothing

    message = "Worktree has non-sandbox changes already present:\n" * join(outside, "\n")
    if strict
        error(message)
    end
    println(message)
    println("Continuing; this suite will only create or update files inside the sandbox directory.")
    return nothing
end

function write_phase_configs!(configs)
    for config in configs
        write_config!(config)
    end
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

function run_phase_remote!(args, configs)
    pending = pending_configs(configs)
    isempty(pending) && begin
        println("All runs in this phase are already complete.")
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

    worker_count = min(Int(args["jobs"]), length(pending))
    tasks = [
        @async begin
            while true
                config = next_config!()
                config === nothing && break
                run_remote_config(args, config)
            end
        end for _ in 1:worker_count
    ]
    foreach(wait, tasks)
    return nothing
end

function phase_already_completed(state, phase)
    return phase in Set(Symbol.(state.completed_phases))
end

function run_phase!(args, phase, state)
    smoke = Bool(args["smoke"])
    if phase_already_completed(state, phase)
        println("Skipping completed phase $(phase).")
        return state
    end

    configs = phase_configs(phase, state.selection; smoke=smoke)
    write_phase_configs!(configs)
    paths = phase_paths(phase; smoke=smoke)
    mkpath(paths.dir)
    write_rows_csv(paths.configs, [config_summary_row(config) for config in configs])

    sync_code!(args)
    run_phase_remote!(args, configs)
    pull_remote_artifacts!(args)

    output = write_phase_outputs!(phase, configs; smoke=smoke)
    selection = update_selection(state.selection, phase, output.decision, output.results.summary)
    completed_phases = vcat(Symbol.(state.completed_phases), [phase])
    new_state = (;
        version=CONFIG_VERSION,
        smoke=smoke,
        completed_phases=completed_phases,
        selection=selection,
        updated_at=unix_milliseconds(),
    )
    save_suite_state!(new_state; smoke=smoke)
    println("Phase $(phase) selected $(output.decision.candidate_name)")
    return new_state
end

function main()
    args = parse_suite_args(ARGS)
    check_git_scope!(strict=Bool(args["strict-git-clean"]))

    if Bool(args["dry-run"])
        print_dry_run(args)
        return nothing
    end

    mkpath(SUITE_DIR)
    ensure_remote_test_cache!(args)

    state = load_suite_state(smoke=Bool(args["smoke"]))
    for phase in PHASES
        state = run_phase!(args, phase, state)
    end

    println("Suite complete.")
    println("Final selection: ", state.selection)
    return state
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
