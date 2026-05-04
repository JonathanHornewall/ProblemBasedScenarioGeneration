#!/usr/bin/env julia

using ArgParse
using Distributed
using Sockets

include(joinpath(@__DIR__, "src", "run_defaults.jl"))
include(joinpath(@__DIR__, "src", "csv_results.jl"))
include(joinpath(@__DIR__, "src", "experiments", "ExperimentAPI.jl"))

const DEFAULT_REMOTE_PROJECT =
    "/home/rwl/ProblemBasedScenarioGeneration/src/ContextualDFL/ContextualDFLTraining"
const DEFAULT_REMOTE_JULIA = "/home/rwl/.juliaup/bin/julia"
const PROFILE_MLFLOW_EXPERIMENT_ID = "3"
const PROFILE_MLFLOW_EXPERIMENT_NAME = "ContextualDFLProfiling"

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

function env_symbol(name, default)
    return Symbol(get(ENV, name, string(default)))
end

function env_flag(name, default=false)
    value = lowercase(get(ENV, name, default ? "1" : "0"))
    return value in ("1", "true", "yes", "y")
end

function parse_commandline(args=ARGS)
    settings = ArgParseSettings(
        description="Run a ContextualDFLTraining profiling job for one experiment.",
    )

    @add_arg_table! settings begin
        "--experiment"
            help = "Experiment id, module name, or config path to profile, e.g. resource_allocation/experiment_1"
            required = true
    end

    return parse_args(args, settings)
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

function profile_config_from_env(experiment)
    run_id = get(ENV, "PROFILE_RUN_ID", "profile_standard_seed3")
    mlflow_enabled = env_flag("PROFILE_MLFLOW_ENABLED", true)
    profile_mlflow_progress = env_flag("PROFILE_MLFLOW_PROGRESS", mlflow_enabled)
    profile_rho = env_float("PROFILE_RHO", DEFAULT_RUN_SETTINGS.rho)
    profile_policy_inference_rho = haskey(ENV, "PROFILE_POLICY_INFERENCE_RHO") ?
        env_float("PROFILE_POLICY_INFERENCE_RHO", profile_rho) :
        nothing
    base = if experiment_has_function(experiment, :profile_config)
        experiment_call(experiment, :profile_config)
    else
        experiment_call(experiment, :base_config)
    end

    cfg = merge(
        base,
        (;
            epochs=env_int("PROFILE_EPOCHS", 10),
            warmup_epochs=env_int("PROFILE_WARMUP_EPOCHS", 2),
            mu=env_float("PROFILE_MU", 1.0),
            mu_start=env_float("PROFILE_MU_START", 1.0),
            mu_end=env_float("PROFILE_MU_END", 1.0),
            mu_schedule=env_symbol("PROFILE_MU_SCHEDULE", :constant),
            rho=profile_rho,
            rho_start=env_float("PROFILE_RHO_START", DEFAULT_RUN_SETTINGS.rho_start),
            rho_end=env_float("PROFILE_RHO_END", DEFAULT_RUN_SETTINGS.rho_end),
            rho_schedule=env_symbol("PROFILE_RHO_SCHEDULE", DEFAULT_RUN_SETTINGS.rho_schedule),
            rho_ref=env_float("PROFILE_RHO_REF", DEFAULT_RUN_SETTINGS.rho_ref),
            rho_ref_start=env_float("PROFILE_RHO_REF_START", DEFAULT_RUN_SETTINGS.rho_ref_start),
            rho_ref_end=env_float("PROFILE_RHO_REF_END", DEFAULT_RUN_SETTINGS.rho_ref_end),
            rho_ref_schedule=env_symbol("PROFILE_RHO_REF_SCHEDULE", DEFAULT_RUN_SETTINGS.rho_ref_schedule),
            tolerance_relative=env_float("PROFILE_TOLERANCE_RELATIVE", DEFAULT_RUN_SETTINGS.tolerance_relative),
            tolerance_absolute_floor=env_float(
                "PROFILE_TOLERANCE_ABSOLUTE_FLOOR",
                DEFAULT_RUN_SETTINGS.tolerance_absolute_floor,
            ),
            optimality_evaluation=env_flag(
                "PROFILE_OPTIMALITY_EVALUATION",
                DEFAULT_RUN_SETTINGS.optimality_evaluation,
            ),
            optimality_test_sample_count=env_int(
                "PROFILE_OPTIMALITY_TEST_SAMPLE_COUNT",
                DEFAULT_RUN_SETTINGS.optimality_test_sample_count,
            ),
            optimality_train_sample_count=env_int(
                "PROFILE_OPTIMALITY_TRAIN_SAMPLE_COUNT",
                DEFAULT_RUN_SETTINGS.optimality_train_sample_count,
            ),
            optimality_validation_sample_count=env_int(
                "PROFILE_OPTIMALITY_VALIDATION_SAMPLE_COUNT",
                DEFAULT_RUN_SETTINGS.optimality_validation_sample_count,
            ),
            optimality_mu=env_float("PROFILE_OPTIMALITY_MU", DEFAULT_RUN_SETTINGS.optimality_mu),
            optimality_rho=env_float("PROFILE_OPTIMALITY_RHO", DEFAULT_RUN_SETTINGS.optimality_rho),
            policy_inference_rho=profile_policy_inference_rho,
            loss=env_symbol("PROFILE_LOSS", :dfl_scen),
            learning_rate=env_float("PROFILE_LEARNING_RATE", 1e-3),
            hidden_size=env_int("PROFILE_HIDDEN_SIZE", 128),
            depth=env_int("PROFILE_DEPTH", 2),
            batch_size=env_int("PROFILE_BATCH_SIZE", 64),
            dropout=env_float("PROFILE_DROPOUT", 0.0),
            seed=env_int("PROFILE_SEED", 3),
            run_id=run_id,
            base_run_id=run_id,
            candidate_name=run_id,
            mlflow_enabled=mlflow_enabled,
            profile_mlflow_progress=profile_mlflow_progress,
            mlflow_experiment_id=PROFILE_MLFLOW_EXPERIMENT_ID,
            mlflow_experiment_name=PROFILE_MLFLOW_EXPERIMENT_NAME,
            mlflow_tracking_uri=get(
                ENV,
                "PROFILE_MLFLOW_TRACKING_URI",
                get(ENV, "MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"),
            ),
            mlflow_run_name=get(ENV, "PROFILE_MLFLOW_RUN_NAME", run_id),
            mlflow_upload_model_artifact=false,
            mlflow_source_name="ContextualDFLTraining/profile_training.jl",
            mlflow_dataset_context="profiling",
            method_variant="profiling",
            mlflow_params=(;
                profile_target="ContextualDFL.train!",
                profile_loss="ContextualDFL.DflScenLoss",
                profile_progress_logged_by="remote_worker",
            ),
            mlflow_tags=(;
                source="ContextualDFLTraining.profile_training",
                run_kind="profiling",
                profile_run=true,
                exclude_from_model_selection=true,
                exclude_from_gridsearch=true,
                mlflow_experiment_name=PROFILE_MLFLOW_EXPERIMENT_NAME,
                profile_target="ContextualDFL.train!",
                profile_loss="ContextualDFL.DflScenLoss",
                profile_progress_logged_by="remote_worker",
                profile_artifacts="local_csv_svg_jlprof",
                coordinator_hostname=Sockets.gethostname(),
            ),
        ),
    )
    return with_experiment_metadata(experiment, cfg)
end

function with_profile_output_config(config, output_dir)
    tags = merge(
        config.mlflow_tags,
        (;
            profile_local_output_dir=output_dir,
            profile_timestamp=basename(output_dir),
        ),
    )
    return merge(config, (; profile_local_output_dir=output_dir, mlflow_tags=tags))
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
    stamp = string(unix_milliseconds())
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
    parsed_args = parse_commandline()
    experiment = load_experiment(parsed_args["experiment"])
    ensure_clean_worker_start!()
    worker = nothing

    try
        output_dir = profile_output_dir()
        config = with_profile_output_config(
            merge(profile_config_from_env(experiment), (; coordinator_hostname=Sockets.gethostname())),
            output_dir,
        )
        println("Running remote profile $(config.run_id) with $(config.epochs) profiled epoch(s)")
        if config.mlflow_enabled && config.profile_mlflow_progress
            println(
                "Remote MLflow profiling progress enabled: ",
                "experiment=$(config.mlflow_experiment_id) ($(config.mlflow_experiment_name)), ",
                "tracking_uri=$(config.mlflow_tracking_uri), run_name=$(config.mlflow_run_name)",
            )
        end

        sync_code!()
        worker = add_profile_worker!()
        metadata = load_worker!(worker)
        assert_remote_profile_worker!(worker, metadata)

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
