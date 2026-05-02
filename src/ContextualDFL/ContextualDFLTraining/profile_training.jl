#!/usr/bin/env julia

using Dates
using Distributed
using Sockets

include(joinpath(@__DIR__, "src", "grid_config.jl"))
include(joinpath(@__DIR__, "src", "csv_results.jl"))

const DEFAULT_REMOTE_PROJECT =
    "/home/rwl/ProblemBasedScenarioGeneration/src/ContextualDFL/ContextualDFLTraining"
const DEFAULT_REMOTE_JULIA = "/home/rwl/.juliaup/bin/julia"

function env_int(name, default)
    value = get(ENV, name, string(default))
    parsed = tryparse(Int, value)
    parsed === nothing && error("ENV[$name] must be an integer, got: $value")
    return parsed
end

function env_float(name, default)
    value = get(ENV, name, string(default))
    parsed = tryparse(Float64, value)
    parsed === nothing && error("ENV[$name] must be a number, got: $value")
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

function profile_config_from_env()
    cfg = merge(
        DEFAULT_RUN_SETTINGS,
        (;
            epochs=env_int("PROFILE_EPOCHS", 100),
            warmup_epochs=env_int("PROFILE_WARMUP_EPOCHS", 2),
            n_samples=env_int("PROFILE_N_SAMPLES", 2000),
            validation_fraction=env_float("PROFILE_VALIDATION_FRACTION", DEFAULT_RUN_SETTINGS.validation_fraction),
            test_fraction=env_float("PROFILE_TEST_FRACTION", DEFAULT_RUN_SETTINGS.test_fraction),
            sigma=env_float("PROFILE_SIGMA", DEFAULT_RUN_SETTINGS.sigma),
            demand_power=env_float("PROFILE_DEMAND_POWER", DEFAULT_RUN_SETTINGS.demand_power),
            context_terms=env_int("PROFILE_CONTEXT_TERMS", DEFAULT_RUN_SETTINGS.context_terms),
            mu=env_float("PROFILE_MU", DEFAULT_RUN_SETTINGS.mu),
            rho=env_float("PROFILE_RHO", DEFAULT_RUN_SETTINGS.rho),
            tolerance_relative=env_float("PROFILE_TOLERANCE_RELATIVE", DEFAULT_RUN_SETTINGS.tolerance_relative),
            tolerance_absolute_floor=env_float(
                "PROFILE_TOLERANCE_ABSOLUTE_FLOOR",
                DEFAULT_RUN_SETTINGS.tolerance_absolute_floor,
            ),
            learning_rate=env_float("PROFILE_LEARNING_RATE", 1e-3),
            hidden_size=env_int("PROFILE_HIDDEN_SIZE", 128),
            depth=env_int("PROFILE_DEPTH", 2),
            batch_size=env_int("PROFILE_BATCH_SIZE", 64),
            dropout=env_float("PROFILE_DROPOUT", 0.0),
            seed=env_int("PROFILE_SEED", 3),
            run_id=get(ENV, "PROFILE_RUN_ID", "profile_standard_seed3"),
        ),
    )
    return cfg
end

function add_profile_worker!()
    remote_project = get(ENV, "REMOTE_CONTEXTUAL_DFL_TRAINING_PROJECT", DEFAULT_REMOTE_PROJECT)
    remote_julia = get(ENV, "REMOTE_JULIA", DEFAULT_REMOTE_JULIA)
    remote_threads = env_int("PROFILE_REMOTE_THREADS", 2)

    println("Adding one profiling worker on rwl@gcp-big with $remote_threads Julia thread(s)")
    addprocs(
        [("rwl@gcp-big", 1)];
        exename=remote_julia,
        exeflags=["--project=$(remote_project)", "--threads=$(remote_threads)"],
        dir=remote_project,
        tunnel=true,
    )

    remote_worker_ids = setdiff(workers(), [1])
    length(remote_worker_ids) == 1 ||
        error("Expected exactly one remote profiling worker, got $(remote_worker_ids).")
    return only(remote_worker_ids)
end

function load_worker!(worker)
    remotecall_fetch(worker) do
        Core.eval(Main, quote
            using Dates
            using Distributed
            using Pkg
            using Sockets
            Pkg.instantiate()
            using ContextualDFLTraining
        end)
        return Core.eval(Main, quote
            (;
                worker_id=Distributed.myid(),
                hostname=Sockets.gethostname(),
                pid=getpid(),
                thread_count=Threads.nthreads(),
                julia_version=string(VERSION),
            )
        end)
    end
end

function assert_remote_profile_worker!(worker, metadata)
    local_hostname = Sockets.gethostname()
    metadata.hostname == local_hostname &&
        error("Refusing to run profiling on local host $(local_hostname).")
    metadata.thread_count == env_int("PROFILE_REMOTE_THREADS", 2) ||
        error("Remote worker has $(metadata.thread_count) thread(s), expected $(env_int("PROFILE_REMOTE_THREADS", 2)).")
    println(
        "Profiling worker online: id=$(worker), host=$(metadata.hostname), pid=$(metadata.pid), threads=$(metadata.thread_count)",
    )
end

function profile_output_dir()
    stamp = Dates.format(Dates.now(), dateformat"yyyymmdd_HHMMSS")
    output_dir = joinpath(@__DIR__, "results", "profile_" * stamp)
    mkpath(joinpath(output_dir, "assets"))
    return output_dir
end

function result_row(result)
    row = Dict{Symbol,Any}()
    row[:status] = result.status
    row[:run_id] = result.run_id
    row[:started_at] = result.started_at
    row[:finished_at] = result.finished_at
    row[:elapsed_seconds] = result.elapsed_seconds
    row[:error] = result.error
    flatten_to_dict!(row, "config", result.config)
    flatten_to_dict!(row, "", result.worker)
    flatten_to_dict!(row, "", result.final_metrics)
    return row
end

function write_profile_outputs(result, output_dir)
    write_csv(joinpath(output_dir, "profile_metadata.csv"), [result_row(result)])
    write_csv(joinpath(output_dir, "profile_epochs.csv"), epoch_result_rows([result]))

    if result.status == "ok"
        write(joinpath(output_dir, "assets", "prof.svg"), result.profile_svg_bytes)
        write(joinpath(output_dir, "assets", "prof.jlprof"), result.profile_jlprof_bytes)
    else
        open(joinpath(output_dir, "profile_error.txt"), "w") do io
            print(io, result.error)
        end
    end

    return output_dir
end

function main()
    ensure_clean_worker_start!()
    worker = nothing

    try
        sync_code!()
        worker = add_profile_worker!()
        metadata = load_worker!(worker)
        assert_remote_profile_worker!(worker, metadata)

        config = merge(profile_config_from_env(), (; coordinator_hostname=Sockets.gethostname()))
        output_dir = profile_output_dir()
        println("Running remote profile $(config.run_id) with $(config.epochs) profiled epoch(s)")

        result = remotecall_fetch(worker) do
            Main.ContextualDFLTraining.profile_standard_training(config)
        end

        write_profile_outputs(result, output_dir)
        println("Wrote profile outputs to $output_dir")

        if result.status == "ok"
            metrics = result.final_metrics
            println(
                "Train MSE: $(metrics.initial_train_mse) -> $(metrics.final_train_mse) ",
                "(delta=$(metrics.loss_delta))",
            )
            println("Artifacts: assets/prof.svg and assets/prof.jlprof")
        else
            println("Profile failed; see profile_error.txt")
        end

        return output_dir
    finally
        if worker !== nothing && worker in workers()
            try
                rmprocs(worker; waitfor=5)
            catch error
                @warn "Failed to remove profiling worker" worker exception=(error, catch_backtrace())
            end
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
