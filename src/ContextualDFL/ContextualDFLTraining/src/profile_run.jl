using FileIO
using FlameGraphs
using Profile
using ProfileSVG

function strict_contextualdfl_training(objects, config; profile_mlflow_state=nothing)
    assert_remote_training_worker!(config)
    mu_schedule = mu_schedule_for_config(config)

    if profile_mlflow_active(config)
        profile_mlflow_state === nothing &&
            error("profile MLflow progress requires an active remote MLflow run")
        return profile_train_with_epoch_progress!(objects, config, mu_schedule, profile_mlflow_state)
    end

    return ContextualDFL.train!(
        objects.scenario_generator.neural_net,
        objects.loss,
        mu_schedule,
        objects.data.train;
        learning_rate=config.learning_rate,
        optimizer_type=Flux.Adam,
        epochs=config.epochs,
        batchsize=config.batch_size,
        shuffle=true,
        rng=MersenneTwister(config.seed + 10_000),
        display_plot=false,
        verbose=false,
        nr_scenarios=effective_nr_scenarios(objects, config),
        display_smooth=Bool(config_value(config, :display_smooth, false)),
        display_real=config_value(config, :display_real, nothing),
        display_reference_input=display_reference_input(objects, config),
    )
end

function profile_train_with_epoch_progress!(objects, config, mu_schedule, mlflow_state)
    run_started = time()

    return ContextualDFL.train!(
        objects.scenario_generator.neural_net,
        objects.loss,
        mu_schedule,
        objects.data.train;
        learning_rate=config.learning_rate,
        optimizer_type=Flux.Adam,
        epochs=config.epochs,
        batchsize=config.batch_size,
        shuffle=true,
        rng=MersenneTwister(config.seed + 10_000),
        display_plot=false,
        verbose=false,
        nr_scenarios=effective_nr_scenarios(objects, config),
        display_smooth=Bool(config_value(config, :display_smooth, false)),
        display_real=config_value(config, :display_real, nothing),
        display_reference_input=display_reference_input(objects, config),
        on_epoch_end=(epoch, loss_value, display_loss, metadata) -> begin
            elapsed_seconds = time() - run_started
            profile_mlflow_log_epoch!(
                mlflow_state,
                epoch,
                loss_value,
                display_loss,
                metadata,
                elapsed_seconds,
            )
            println(
                "MLflow profiling progress: epoch=$(epoch)/$(config.epochs) ",
                "loss=$(Float64(loss_value)) elapsed_seconds=$(round(elapsed_seconds; digits=2))",
            )
        end,
    )
end

function standard_profile_config(; overrides...)
    settings = merge(
        DEFAULT_RUN_SETTINGS,
        (;
            epochs=100,
            learning_rate=1e-3,
            hidden_size=128,
            depth=2,
            batch_size=64,
            dropout=0.0,
            seed=3,
            run_id="profile_standard_seed3",
        ),
        NamedTuple(overrides),
    )
    return settings
end

function profile_mlflow_active(config)
    return mlflow_enabled(config) && Bool(config_value(config, :profile_mlflow_progress, false))
end

function profile_mlflow_start!(config)
    profile_mlflow_active(config) || return nothing

    experiment_id = string(config_value(config, :mlflow_experiment_id, ""))
    isempty(experiment_id) &&
        throw(ArgumentError("profile MLflow logging requires config.mlflow_experiment_id"))

    mlf = NamedMLFlowClient(
        tracking_uri=string(config_value(config, :mlflow_tracking_uri, "http://127.0.0.1:5000")),
        run_name=string(
            config_value(
                config,
                :mlflow_run_name,
                config_value(config, :run_id, "profile-run"),
            ),
        ),
        tags=profile_mlflow_tags(config),
        params=profile_mlflow_params(config),
    )
    run = createrun(mlf, experiment_id; start_time=unix_milliseconds())
    return (; mlf=mlf, run=run, run_id=string(config_value(config, :run_id, "")))
end

function profile_mlflow_tags(config)
    tags = Dict{String,String}()
    add_string_pairs!(tags, config_value(config, :mlflow_tags, nothing))
    tags["run_kind"] = "profiling"
    tags["profile_run"] = "true"
    tags["exclude_from_model_selection"] = "true"
    tags["exclude_from_gridsearch"] = "true"
    tags["profile_target"] = "ContextualDFL.train!"
    tags["profile_loss"] = "ContextualDFL.DflScenLoss"
    tags["profile_progress_logged_by"] = "remote_worker"
    tags["run_id"] = string(config_value(config, :run_id, ""))
    tags["training_project"] = "ContextualDFLTraining"
    tags["worker_id"] = string(Distributed.myid())
    tags["worker_hostname"] = Sockets.gethostname()
    tags["worker_pid"] = string(getpid())
    return drop_empty_values(tags)
end

function profile_mlflow_params(config)
    params = Dict{String,String}()
    add_string_pairs!(params, config_value(config, :mlflow_params, nothing))
    params["profile_target"] = "ContextualDFL.train!"
    params["profile_loss"] = "ContextualDFL.DflScenLoss"
    params["profile_progress_logged_by"] = "remote_worker"

    config isa NamedTuple || return params
    for key in keys(config)
        key in (:mlflow_tags, :mlflow_params, :mlflow_tracking_uri) && continue
        value = getproperty(config, key)
        mlflow_param_value(value) || continue
        params["config_" * string(key)] = string(value)
    end

    return drop_empty_values(params)
end

function profile_mlflow_log_epoch!(
    state,
    epoch,
    loss_value,
    display_loss,
    metadata,
    elapsed_seconds,
)
    state === nothing && return nothing

    try
        step = Int(epoch)
        profile_mlflow_logmetric(state, "loss", Float64(loss_value); step=step)
        profile_mlflow_logmetric(state, "display_loss", Float64(display_loss); step=step)
        profile_mlflow_logmetric(
            state,
            "profile_elapsed_seconds",
            Float64(elapsed_seconds);
            step=step,
        )

        if metadata isa NamedTuple
            for (metric_name, field_name) in (
                "epoch_seconds" => :epoch_seconds,
                "epoch_mu" => :mu,
                "epoch_iterations" => :iterations,
                "real_display_loss" => :real_display_loss,
            )
                haskey(metadata, field_name) || continue
                value = getproperty(metadata, field_name)
                value isa Number || continue
                profile_mlflow_logmetric(state, metric_name, Float64(value); step=step)
            end
        end
    catch error
        println("Failed to log MLflow profiling progress: ", sprint(showerror, error))
    end

    return nothing
end

function profile_mlflow_log_final!(state, metrics; status, error="")
    state === nothing && return nothing

    setruntag(state.mlf, state.run, "profile_status", string(status))
    isempty(error) ||
        setruntag(state.mlf, state.run, "profile_error", string(first(split(error, '\n'))))

    metrics isa NamedTuple || return nothing
    for key in keys(metrics)
        value = getproperty(metrics, key)
        value isa Number || continue
        profile_mlflow_logmetric(state, "final_" * string(key), Float64(value); step=0)
    end

    return nothing
end

function profile_mlflow_finish!(state, status::Symbol; error="")
    state === nothing && return nothing

    isempty(error) ||
        setruntag(state.mlf, state.run, "profile_error", string(first(split(error, '\n'))))
    run_status = status == :ok ? RunStatus.FINISHED : RunStatus.FAILED
    updaterun(state.mlf, state.run; status=run_status, end_time=unix_milliseconds())
    return nothing
end

function profile_mlflow_logmetric(state, key, value; step)
    for attempt in 1:3
        try
            return MLFlowClient.logmetric(
                state.mlf.client,
                state.run,
                string(key),
                Float64(value);
                timestamp=unix_milliseconds(),
                step=Int(step),
            )
        catch
            attempt == 3 && rethrow()
            sleep(0.25 * attempt)
        end
    end
end

function profile_standard_training(config::NamedTuple)
    cfg = normalize_config(config)
    assert_remote_training_worker!(cfg)
    started_at = unix_milliseconds()
    elapsed_seconds = 0.0
    remote_output_dir = ""
    mlflow_state = nothing
    mlflow_finished = false

    try
        mlflow_state = profile_mlflow_start!(cfg)

        remote_output_dir = mktempdir(; prefix="contextualdfl_profile_")
        remote_assets_dir = joinpath(remote_output_dir, "assets")
        mkpath(remote_assets_dir)
        svg_path = joinpath(remote_assets_dir, "prof.svg")
        jlprof_path = joinpath(remote_assets_dir, "prof.jlprof")

        profile_result = nothing
        initial_train_mse = NaN
        final_train_mse = NaN
        metrics = NamedTuple()
        history = Dict{Symbol,Any}[]
        svg_bytes = UInt8[]
        jlprof_bytes = UInt8[]

        elapsed_seconds = @elapsed begin
            warmup_epochs = max(Int(hasproperty(cfg, :warmup_epochs) ? cfg.warmup_epochs : 2), 0)
            if warmup_epochs > 0
                warmup_cfg = merge(
                    cfg,
                    (;
                        epochs=warmup_epochs,
                        run_id=string(cfg.run_id, "_warmup"),
                        mlflow_enabled=false,
                        profile_mlflow_progress=false,
                    ),
                )
                warmup_objects = training_objects_for_config(warmup_cfg)
                strict_contextualdfl_training(warmup_objects, warmup_cfg)
                GC.gc()
            end

            objects = training_objects_for_config(cfg)
            model = objects.scenario_generator.neural_net
            initial_train_mse = split_mse(model, objects.data.train, objects)

            Profile.clear()
            profile_result = Profile.@profile strict_contextualdfl_training(
                objects,
                cfg;
                profile_mlflow_state=mlflow_state,
            )

            ProfileSVG.save(svg_path)
            FileIO.save(jlprof_path, Profile.retrieve()...)
            svg_bytes = read(svg_path)
            jlprof_bytes = read(jlprof_path)

            trained_model = extract_model(profile_result, objects.scenario_generator)
            final_train_mse = split_mse(trained_model, objects.data.train, objects)
            metrics = merge(
                evaluate_model_for_reporting(trained_model, objects, cfg),
                (;
                    initial_train_mse=initial_train_mse,
                    final_train_mse=final_train_mse,
                    loss_delta=initial_train_mse - final_train_mse,
                    loss_decreased=final_train_mse < initial_train_mse,
                    training_backend=mlflow_enabled(cfg) && Bool(config_value(cfg, :profile_mlflow_progress, false)) ?
                        "ContextualDFL.train! with MLflow profiling progress" :
                        "ContextualDFL.train!",
                    remote_output_dir=remote_output_dir,
                    thread_count=Threads.nthreads(),
                ),
            )
            history = extract_epoch_history(profile_result)
        end

        require_train_mse_decrease = Bool(
            config_value(cfg, :require_train_mse_decrease, false),
        )
        (!require_train_mse_decrease || final_train_mse < initial_train_mse) ||
            error("profiled training did not reduce train MSE: initial=$(initial_train_mse), final=$(final_train_mse)")

        result = (;
            status="ok",
            run_id=cfg.run_id,
            config=cfg,
            worker=worker_metadata(),
            final_metrics=metrics,
            epoch_history=history,
            profile_svg_bytes=svg_bytes,
            profile_jlprof_bytes=jlprof_bytes,
            error="",
            started_at=started_at,
            finished_at=unix_milliseconds(),
            elapsed_seconds=elapsed_seconds,
        )

        profile_mlflow_log_final!(mlflow_state, result.final_metrics; status=result.status)
        profile_mlflow_finish!(mlflow_state, :ok)
        mlflow_finished = true

        return result
    catch error
        error_text = exception_text(error, catch_backtrace())
        if mlflow_state !== nothing && !mlflow_finished
            try
                profile_mlflow_log_final!(
                    mlflow_state,
                    (;);
                    status="failed",
                    error=error_text,
                )
                profile_mlflow_finish!(mlflow_state, :failed; error=error_text)
                mlflow_finished = true
            catch mlflow_error
                error_text *=
                    "\n\nMLflow failure while marking profile run failed:\n" *
                    exception_text(mlflow_error, catch_backtrace())
            end
        end

        return (;
            status="failed",
            run_id=hasproperty(cfg, :run_id) ? cfg.run_id : "",
            config=cfg,
            worker=worker_metadata(),
            final_metrics=NamedTuple(),
            epoch_history=Dict{Symbol,Any}[],
            profile_svg_bytes=UInt8[],
            profile_jlprof_bytes=UInt8[],
            error=error_text,
            started_at=started_at,
            finished_at=unix_milliseconds(),
            elapsed_seconds=elapsed_seconds,
        )
    finally
        if !isempty(remote_output_dir) && isdir(remote_output_dir)
            try
                rm(remote_output_dir; recursive=true, force=true)
            catch
            end
        end
    end
end
