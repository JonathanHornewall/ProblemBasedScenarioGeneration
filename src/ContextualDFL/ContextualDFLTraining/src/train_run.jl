using Dates
using ContextualDFLExperiments
using Distributed
using Flux
using Random
using Serialization
using SHA
using Sockets
using Statistics

function normalize_config(config::NamedTuple)
    return merge(DEFAULT_RUN_SETTINGS, config)
end

# Explicit Unix epoch milliseconds, independent of the worker timezone.
unix_milliseconds() = round(Int64, time() * 1000)

function worker_metadata()
    return (;
        worker_id=Distributed.myid(),
        hostname=Sockets.gethostname(),
        pid=getpid(),
        julia_version=string(VERSION),
    )
end

function exception_text(error, backtrace)
    return sprint(showerror, error, backtrace)
end

function train_and_evaluate(config::NamedTuple)
    cfg = normalize_config(config)
    assert_remote_training_worker!(cfg)
    started_at = unix_milliseconds()
    elapsed_seconds = 0.0

    try
        train_result = nothing
        training_backend = ""
        fallback_reason = ""
        objects = nothing
        object_build_seconds = 0.0
        training_seconds = 0.0
        evaluation_seconds = 0.0

        object_build_seconds = @elapsed begin
            objects = training_objects_for_config(cfg)
        end
        training_seconds = @elapsed begin
            training = train_with_contextualdfl(objects, cfg)
            train_result = training.result
            training_backend = training.backend
            fallback_reason = training.fallback_reason
        end

        model = extract_model(train_result, objects.scenario_generator)
        split_metrics = if hasproperty(training, :final_metrics) && !isnothing(training.final_metrics)
            training.final_metrics
        else
            measured_metrics = nothing
            evaluation_seconds = @elapsed begin
                measured_metrics = evaluate_model_for_reporting(model, objects, cfg)
            end
            measured_metrics
        end
        elapsed_seconds = object_build_seconds + training_seconds + evaluation_seconds
        metrics = merge(
            split_metrics,
            (;
                training_backend=training_backend,
                fallback_reason=fallback_reason,
                object_build_seconds=object_build_seconds,
                training_seconds=training_seconds,
                evaluation_seconds=evaluation_seconds,
                total_elapsed_seconds=elapsed_seconds,
            ),
        )
        history = extract_epoch_history(train_result)

        return (;
            status="ok",
            run_id=string(config_value(cfg, :run_id, "")),
            config=cfg,
            worker=worker_metadata(),
            final_metrics=metrics,
            epoch_history=history,
            error="",
            started_at=started_at,
            finished_at=unix_milliseconds(),
            elapsed_seconds=elapsed_seconds,
        )
    catch error
        return (;
            status="failed",
            run_id=hasproperty(cfg, :run_id) ? cfg.run_id : "",
            config=cfg,
            worker=worker_metadata(),
            final_metrics=NamedTuple(),
            epoch_history=Dict{Symbol,Any}[],
            error=exception_text(error, catch_backtrace()),
            started_at=started_at,
            finished_at=unix_milliseconds(),
            elapsed_seconds=elapsed_seconds,
        )
    end
end

function train_with_contextualdfl(objects, config)
    if mlflow_enabled(config)
        mlflow_result = train_with_contextualdfl_mlflow(objects, config)
        return (;
            result=mlflow_result.result,
            backend="ContextualDFL.train! with ContextualDFLTraining MLflow",
            fallback_reason="",
            final_metrics=mlflow_result.final_metrics,
        )
    end

    result = ContextualDFL.train!(
        objects.scenario_generator.neural_net,
        objects.loss,
        mu_schedule_for_config(config),
        mu_ref_schedule_for_config(config),
        objects.data.train;
        rho_in_schedule=rho_schedule_for_config(config),
        rho_ref_schedule=rho_ref_schedule_for_config(config),
        learning_rate=config.learning_rate,
        optimizer_type=Flux.Adam,
        epochs=config.epochs,
        batchsize=config.batch_size,
        shuffle=Bool(config_value(config, :shuffle, false)),
        display_smooth=Bool(config_value(config, :display_smooth, false)),
        display_real=config_value(config, :display_real, nothing),
        display_reference_input=display_reference_input(objects, config),
    )
    trained_model = extract_model(result, objects.scenario_generator)
    save_flux_checkpoint_after_training!(trained_model, result, objects, config)

    return (;
        result=result,
        backend="ContextualDFL.train",
        fallback_reason="",
        final_metrics=nothing,
    )
end

function train_with_contextualdfl_mlflow(objects, config)
    mlflow_config = add_worker_mlflow_tags(config)
    mlf, experiment_id = mlflow_client_for_config(mlflow_config)
    loss = contextual_dfl_loss(objects, config)
    upload_model_artifact = Bool(config_value(config, :mlflow_upload_model_artifact, false))
    model_save_path = mlflow_model_save_path(config)
    final_metrics = Ref{Any}(nothing)
    model = objects.scenario_generator.neural_net
    mu_schedule = mu_schedule_for_config(config)
    mu_ref_schedule = mu_ref_schedule_for_config(config, mu_schedule)
    rho_schedule = rho_schedule_for_config(config)
    rho_ref_schedule = rho_ref_schedule_for_config(config, rho_schedule)
    run = createrun(mlf, experiment_id; start_time=unix_milliseconds())
    training_succeeded = false

    try
        log_contextualdfl_training_params!(
            mlf,
            run,
            loss,
            config,
            mu_schedule,
            mu_ref_schedule,
            rho_schedule,
            rho_ref_schedule,
        )
        log_mlflow_params!(mlf, run, "experiment", mlflow_experiment_spec(objects, config))
        log_mlflow_params!(mlf, run, "data", mlflow_data_spec(objects, config))
        log_mlflow_params!(mlf, run, "model", mlflow_model_spec(model, objects, config))
        log_mlflow_params!(mlf, run, "method", mlflow_method_spec(objects, config))
        log_mlflow_source_tags!(
            mlf,
            run;
            source_name=string(
                config_value(
                    config,
                    :mlflow_source_name,
                    "ContextualDFLTraining/gridsearch.jl",
                ),
            ),
            source_type=string(config_value(config, :mlflow_source_type, "LOCAL")),
            source_git_commit=git_commit_or_nothing(),
        )
        log_mlflow_dataset!(
            mlf,
            run;
            dataset_name=mlflow_dataset_name(objects, config),
            dataset_digest=mlflow_dataset_digest(objects, config),
            dataset_source_type=mlflow_dataset_source_type(objects, config),
            dataset_source=mlflow_dataset_source(objects, config),
            dataset_context="training",
        )

        result = ContextualDFL.train!(
            model,
            loss,
            mu_schedule,
            mu_ref_schedule,
            objects.data.train;
            rho_in_schedule=rho_schedule,
            rho_ref_schedule=rho_ref_schedule,
            learning_rate=config.learning_rate,
            optimizer_type=Flux.Adam,
            epochs=config.epochs,
            batchsize=config.batch_size,
            shuffle=Bool(config_value(config, :shuffle, false)),
            reset_optimizer_each_epoch=Bool(
                config_value(config, :reset_optimizer_each_epoch, false),
            ),
            save_model=false,
            model_save_path=model_save_path,
            on_epoch_end=(epoch, loss_value, display_loss, metadata) -> log_mlflow_epoch!(
                mlf,
                run,
                epoch,
                loss_value,
                display_loss,
                metadata,
            ),
            nr_scenarios=effective_nr_scenarios(objects, config),
            display_smooth=Bool(config_value(config, :display_smooth, false)),
            display_real=config_value(config, :display_real, nothing),
            display_reference_input=display_reference_input(objects, config),
        )

        trained_model = extract_model(result, objects.scenario_generator)
        checkpoint = save_flux_checkpoint_after_training!(
            trained_model,
            result,
            objects,
            mlflow_config,
        )
        log_mlflow_checkpoint_artifact!(mlf, run, checkpoint, mlflow_config)

        if upload_model_artifact
            temp_model_save_path = ""
            try
                mkpath(dirname(model_save_path))
                temp_model_save_path = tempname(dirname(model_save_path))
                open(temp_model_save_path, "w") do io
                    Serialization.serialize(io, trained_model)
                end
                mv(temp_model_save_path, model_save_path; force=true)
                println("Model saved to: $model_save_path")
                upload_mlflow_artifact!(
                    mlf,
                    run,
                    model_save_path;
                    artifact_path="models/" * basename(model_save_path),
                )
            catch error
                if !isempty(temp_model_save_path) && isfile(temp_model_save_path)
                    rm(temp_model_save_path; force=true)
                end
                @warn "Failed to save/upload MLflow model artifact" path=model_save_path error=exception_text(
                    error,
                    catch_backtrace(),
                )
            end
        end

        metrics = evaluate_model_for_reporting(trained_model, objects, config)
        final_metrics[] = metrics
        log_mlflow_evaluation_result!(mlf, run, "", metrics)

        training_succeeded = true
        return (; result=result, final_metrics=final_metrics[])
    catch error
        error_text = exception_text(error, catch_backtrace())
        try
            log_mlflow_stacktrace_artifact!(mlf, run, error_text)
        catch mlflow_error
            @warn "Failed to upload MLflow stacktrace artifact" error=exception_text(
                mlflow_error,
                catch_backtrace(),
            )
        end
        rethrow()
    finally
        status = training_succeeded ? RunStatus.FINISHED : RunStatus.FAILED
        try
            updaterun(mlf, run; status=status, end_time=unix_milliseconds())
        catch
            training_succeeded && rethrow()
        end
    end
end

function log_contextualdfl_training_params!(
    mlf,
    run,
    loss,
    config,
    mu_schedule,
    mu_ref_schedule,
    rho_schedule,
    rho_ref_schedule,
)
    logparam(mlf, run, "learning_rate", string(config.learning_rate))
    logparam(mlf, run, "optimizer_type", string(Flux.Adam))
    logparam(mlf, run, "epochs", string(config.epochs))
    logparam(mlf, run, "batchsize", string(config.batch_size))
    logged_scenarios = logged_nr_scenarios(
        loss,
        config_value(config, :nr_scenarios, nothing),
    )
    isnothing(logged_scenarios) ||
        logparam(mlf, run, "nr_scenarios", string(logged_scenarios))
    logparam(mlf, run, "mu_in_schedule", string(collect(mu_schedule)))
    logparam(mlf, run, "mu_ref_schedule", string(collect(mu_ref_schedule)))
    logparam(mlf, run, "rho_in_schedule", string(collect(rho_schedule)))
    logparam(mlf, run, "rho_ref_schedule", string(collect(rho_ref_schedule)))
    logparam(
        mlf,
        run,
        "display_smooth",
        string(Bool(config_value(config, :display_smooth, false))),
    )
    logparam(
        mlf,
        run,
        "display_real",
        string(config_value(config, :display_real, nothing)),
    )
    logparam(mlf, run, "shuffle", string(Bool(config_value(config, :shuffle, false))))
    logparam(
        mlf,
        run,
        "reset_optimizer_each_epoch",
        string(Bool(config_value(config, :reset_optimizer_each_epoch, false))),
    )
    training_seed = config_value(config, :training_data_seed, nothing)
    if training_seed !== nothing && training_seed !== missing
        logparam(mlf, run, "training_data_seed", string(training_seed))
    end
    repeat_training_seed = config_value(config, :repeat_training_data_seed, training_seed)
    if repeat_training_seed !== nothing && repeat_training_seed !== missing
        logparam(mlf, run, "repeat_training_data_seed", string(repeat_training_seed))
    end
    repeat_index = config_value(config, :repeat_index, nothing)
    if repeat_index !== nothing && repeat_index !== missing
        logparam(mlf, run, "repeat_index", string(repeat_index))
    end
    return nothing
end

function display_reference_input(objects, config)
    display_smooth = Bool(config_value(config, :display_smooth, false))
    display_real = config_value(config, :display_real, nothing)
    (display_smooth || !isnothing(display_real)) || return nothing
    hasproperty(objects, :target_extractor) && return objects.target_extractor
    return nothing
end

function logged_nr_scenarios(loss, nr_scenarios)
    if hasproperty(loss, :nr_scenarios)
        return Int(getproperty(loss, :nr_scenarios))
    end
    isnothing(nr_scenarios) || return Int(nr_scenarios)
    return nothing
end

function effective_nr_scenarios(objects, config)
    if hasproperty(objects, :loss)
        value = logged_nr_scenarios(objects.loss, nothing)
        isnothing(value) || return value
    end

    value = config_value(config, :nr_scenarios, nothing)
    isnothing(value) && return nothing
    return Int(value)
end

function add_worker_mlflow_tags(config)
    tags = string_dict(config_value(config, :mlflow_tags, nothing))
    tags["worker_id"] = string(Distributed.myid())
    tags["worker_hostname"] = Sockets.gethostname()
    tags["worker_pid"] = string(getpid())
    parent_run_id = string(config_value(config, :mlflow_parent_run_id, ""))
    if !isempty(parent_run_id)
        tags["gridsearch_parent_run_id"] = parent_run_id
        tags["mlflow.parentRunId"] = parent_run_id
    end
    return merge(config, (; mlflow_tags=tags))
end

function assert_remote_training_worker!(config)
    Bool(config_value(config, :allow_local_training, false)) && return nothing

    Distributed.myid() == 1 &&
        error("Refusing to run training on Distributed worker 1. Use the remote gridsearch/profile entry points.")

    coordinator_hostname = string(config_value(config, :coordinator_hostname, ""))
    if !isempty(coordinator_hostname) && Sockets.gethostname() == coordinator_hostname
        error("Refusing to run training on coordinator host $(coordinator_hostname).")
    end

    return nothing
end

function object_metadata(objects, field::Symbol)
    hasproperty(objects, field) || return NamedTuple()
    value = getproperty(objects, field)
    return value isa NamedTuple ? value : NamedTuple()
end

function metadata_value(metadata::NamedTuple, key::Symbol, default)
    return key in keys(metadata) ? getproperty(metadata, key) : default
end

function mlflow_dataset_name(config)
    return string(
        config_value(
            config,
            :mlflow_dataset_name,
            config_value(config, :experiment_name, "generated_dataset"),
        ),
    )
end

function mlflow_dataset_name(objects, config)
    data_metadata = object_metadata(objects, :data_metadata)
    name = metadata_value(data_metadata, :dataset_name, nothing)
    isnothing(name) || return string(name)
    return mlflow_dataset_name(config)
end

function mlflow_dataset_source(objects, config)
    data_metadata = object_metadata(objects, :data_metadata)
    source = metadata_value(data_metadata, :dataset_source, nothing)
    isnothing(source) || return string(source)

    parts = String["ContextualDFLTraining.experiment"]
    hasproperty(config, :experiment_id) && push!(parts, "experiment_id=$(config.experiment_id)")
    hasproperty(config, :training_data_seed) &&
        push!(parts, "training_data_seed=$(config.training_data_seed)")
    hasproperty(config, :validation_fraction) &&
        push!(parts, "validation_fraction=$(config.validation_fraction)")
    hasproperty(config, :test_fraction) && push!(parts, "test_fraction=$(config.test_fraction)")
    return join(parts, ";")
end

function mlflow_dataset_source_type(objects, config)
    data_metadata = object_metadata(objects, :data_metadata)
    path = metadata_value(data_metadata, :dataset_path, "")
    (path === nothing || path === missing || isempty(string(path))) || return "local"
    return string(config_value(config, :mlflow_dataset_source_type, "generated"))
end

function mlflow_dataset_digest(objects, config)
    data_metadata = object_metadata(objects, :data_metadata)
    digest = metadata_value(data_metadata, :dataset_digest, nothing)
    isnothing(digest) || return string(digest)

    split_summary = (
        "dataset=$(mlflow_dataset_name(objects, config))",
        "training_data_seed=$(config_value(config, :training_data_seed, ""))",
        "train_x=$(size(dataset_context_matrix(objects.data.train)))",
        "train_y=$(size(dataset_target_matrix(objects.data.train, objects)))",
        "validation_x=$(size(dataset_context_matrix(objects.data.validation)))",
        "validation_y=$(size(dataset_target_matrix(objects.data.validation, objects)))",
        "test_x=$(size(dataset_context_matrix(objects.data.test)))",
        "test_y=$(size(dataset_target_matrix(objects.data.test, objects)))",
    )
    return short_mlflow_digest(split_summary)
end

function short_mlflow_digest(values)
    return bytes2hex(sha256(join(values, "\n")))[1:32]
end

function mlflow_model_save_path(config)
    run_id = string(config_value(config, :run_id, "training-run"))
    safe_run_id = replace(run_id, r"[^A-Za-z0-9_.=-]" => "_")
    return joinpath(tempdir(), safe_run_id * ".jls")
end

function checkpoint_enabled(config)
    return Bool(config_value(config, :checkpoint_enabled, true))
end

function checkpoint_upload_mlflow(config)
    return Bool(config_value(config, :checkpoint_upload_mlflow, true))
end

function checkpoint_required(config)
    return Bool(config_value(config, :checkpoint_required, false))
end

function checkpoint_format(config)
    format = Symbol(config_value(config, :checkpoint_format, :jls))
    format == :jls ||
        throw(ArgumentError("unsupported checkpoint_format $(format); use :jls."))
    return format
end

function default_checkpoint_root()
    return joinpath(dirname(@__DIR__), "results", "checkpoints")
end

function checkpoint_directory(config)
    configured_dir = string(config_value(config, :checkpoint_dir, ""))
    if isempty(strip(configured_dir))
        grid_id = string(config_value(config, :gridsearch_id, "standalone"))
        return joinpath(default_checkpoint_root(), safe_checkpoint_identifier(grid_id))
    end
    return abspath(configured_dir)
end

function checkpoint_save_path(config)
    format = checkpoint_format(config)
    run_id = string(config_value(config, :run_id, "training-run"))
    filename = safe_checkpoint_identifier(run_id) * "_checkpoint." * string(format)
    return joinpath(checkpoint_directory(config), filename)
end

function safe_checkpoint_identifier(value)
    text = replace(string(value), r"[^A-Za-z0-9_.=-]+" => "_")
    text = strip(text, ['_'])
    return isempty(text) ? "training-run" : text
end

function save_flux_checkpoint_after_training!(model, train_result, objects, config)
    checkpoint_enabled(config) || return nothing

    path = checkpoint_save_path(config)
    try
        save_flux_checkpoint!(path, model, train_result, objects, config)
        return path
    catch error
        checkpoint_required(config) && rethrow()
        @warn "Failed to save Flux checkpoint" path error=exception_text(error, catch_backtrace())
        return nothing
    end
end

function save_flux_checkpoint!(path::AbstractString, model, train_result, objects, config)
    mkpath(dirname(path))
    payload = flux_checkpoint_payload(model, train_result, objects, config, path)
    temp_path = tempname(dirname(path))
    open(temp_path, "w") do io
        Serialization.serialize(io, payload)
    end
    mv(temp_path, path; force=true)
    return path
end

function flux_checkpoint_payload(model, train_result, objects, config, path)
    return (;
        format_version=1,
        checkpoint_format=:jls,
        saved_at=Dates.format(now(UTC), dateformat"yyyy-mm-ddTHH:MM:SS.sZ"),
        saved_at_unix_ms=unix_milliseconds(),
        checkpoint_path=String(path),
        run_id=string(config_value(config, :run_id, "")),
        gridsearch_id=string(config_value(config, :gridsearch_id, "")),
        candidate_index=config_value(config, :candidate_index, missing),
        repeat_index=config_value(config, :repeat_index, missing),
        worker=worker_metadata(),
        config=config,
        model_state=Flux.state(model),
        optimizer_state=optimizer_flux_state(train_result),
        epoch_history=extract_epoch_history(train_result),
        model_metadata=mlflow_model_spec(model, objects, config),
        data_metadata=mlflow_data_spec(objects, config),
    )
end

function optimizer_flux_state(train_result)
    hasproperty(train_result, :opt_state) || return missing
    opt_state = getproperty(train_result, :opt_state)
    (opt_state === nothing || opt_state === missing) && return missing
    return Flux.state(opt_state)
end

function log_mlflow_checkpoint_artifact!(mlf, run, checkpoint_path, config)
    checkpoint_path === nothing && return nothing
    logparam(mlf, run, "checkpoint_path", string(checkpoint_path))
    logparam(mlf, run, "checkpoint_format", string(checkpoint_format(config)))

    checkpoint_upload_mlflow(config) || return nothing

    artifact_path = "checkpoints/" * basename(checkpoint_path)
    try
        upload_mlflow_artifact!(mlf, run, checkpoint_path; artifact_path=artifact_path)
        logparam(mlf, run, "checkpoint_artifact_path", artifact_path)
    catch error
        checkpoint_required(config) && rethrow()
        @warn "Failed to upload Flux checkpoint artifact" artifact_path error=exception_text(
            error,
            catch_backtrace(),
        )
    end

    return nothing
end

function contextual_dfl_loss(objects, config)
    return objects.loss
end

function mu_schedule_for_config(config)
    epochs = Int(config.epochs)
    epochs >= 0 || throw(ArgumentError("epochs must be non-negative."))
    epochs == 0 && return Float64[]

    raw_schedule = config_value(config, :mu_schedule, :constant)
    if raw_schedule isa AbstractVector
        length(raw_schedule) == epochs ||
            throw(ArgumentError("mu_schedule vector must have one value per epoch."))
        return Float64.(raw_schedule)
    end

    schedule = Symbol(raw_schedule)
    mu_start = Float64(config_value(config, :mu_start, config.mu))
    mu_end = Float64(config_value(config, :mu_end, config.mu))

    if schedule == :constant
        return fill(Float64(config.mu), epochs)
    elseif schedule == :linear
        epochs == 1 && return [mu_start]
        return collect(range(mu_start, mu_end; length=epochs))
    elseif schedule == :geometric || schedule == :exponential
        mu_start > 0 && mu_end > 0 ||
            throw(ArgumentError("$schedule mu annealing requires positive mu_start and mu_end."))
        epochs == 1 && return [mu_start]
        return exp.(range(log(mu_start), log(mu_end); length=epochs))
    end

    throw(ArgumentError("unsupported mu_schedule $(schedule)"))
end

function policy_inference_mu_for_config(config, mu_schedule=nothing)
    if config isa NamedTuple && :policy_inference_mu in keys(config)
        policy_mu = config.policy_inference_mu
        (policy_mu === nothing || policy_mu === missing) || return Float64(policy_mu)
    end

    resolved_mu_schedule =
        isnothing(mu_schedule) ? mu_schedule_for_config(config) : mu_schedule
    isempty(resolved_mu_schedule) && return Float64(config_value(config, :mu, 0.0))
    return Float64(last(resolved_mu_schedule))
end

function policy_inference_rho_for_config(config, rho_schedule=nothing)
    if config isa NamedTuple && :policy_inference_rho in keys(config)
        policy_rho = config.policy_inference_rho
        (policy_rho === nothing || policy_rho === missing) || return Float64(policy_rho)
    end

    resolved_rho_schedule =
        isnothing(rho_schedule) ? rho_schedule_for_config(config) : rho_schedule
    isempty(resolved_rho_schedule) && return Float64(config_value(config, :rho, 0.0))
    return Float64(last(resolved_rho_schedule))
end

function mu_ref_schedule_for_config(config, mu_schedule=mu_schedule_for_config(config))
    epochs = Int(config.epochs)
    raw_schedule = config_value(config, :mu_ref_schedule, :match_input)

    if raw_schedule isa AbstractVector
        length(raw_schedule) == epochs ||
            throw(ArgumentError("mu_ref_schedule vector must have one value per epoch."))
        return Float64.(raw_schedule)
    end

    schedule = Symbol(raw_schedule)
    if schedule in (:match_input, :same, :input)
        length(mu_schedule) == epochs ||
            throw(ArgumentError("mu_schedule must have one value per epoch."))
        return Float64.(mu_schedule)
    elseif schedule in (:zero, :zeros, :none)
        return zeros(Float64, epochs)
    elseif schedule == :constant
        return fill(Float64(config_value(config, :mu_ref, config.mu)), epochs)
    elseif schedule == :linear
        epochs == 1 && return [Float64(config_value(config, :mu_ref_start, config_value(config, :mu_start, config.mu)))]
        return collect(
            range(
                Float64(config_value(config, :mu_ref_start, config_value(config, :mu_start, config.mu))),
                Float64(config_value(config, :mu_ref_end, config_value(config, :mu_end, config.mu)));
                length=epochs,
            ),
        )
    elseif schedule == :geometric || schedule == :exponential
        mu_ref_start = Float64(config_value(config, :mu_ref_start, config_value(config, :mu_start, config.mu)))
        mu_ref_end = Float64(config_value(config, :mu_ref_end, config_value(config, :mu_end, config.mu)))
        mu_ref_start > 0 && mu_ref_end > 0 ||
            throw(ArgumentError("$schedule mu_ref annealing requires positive mu_ref_start and mu_ref_end."))
        epochs == 1 && return [mu_ref_start]
        return exp.(range(log(mu_ref_start), log(mu_ref_end); length=epochs))
    end

    throw(ArgumentError("unsupported mu_ref_schedule $(schedule)"))
end

function rho_schedule_for_config(config)
    epochs = Int(config.epochs)
    epochs >= 0 || throw(ArgumentError("epochs must be non-negative."))
    epochs == 0 && return Float64[]

    raw_schedule = config_value(config, :rho_schedule, :constant)
    if raw_schedule isa AbstractVector
        length(raw_schedule) == epochs ||
            throw(ArgumentError("rho_schedule vector must have one value per epoch."))
        return Float64.(raw_schedule)
    end

    schedule = Symbol(raw_schedule)
    rho_start = Float64(config_value(config, :rho_start, config.rho))
    rho_end = Float64(config_value(config, :rho_end, config.rho))

    if schedule == :constant
        return fill(Float64(config.rho), epochs)
    elseif schedule == :linear
        epochs == 1 && return [rho_start]
        return collect(range(rho_start, rho_end; length=epochs))
    elseif schedule == :geometric || schedule == :exponential
        rho_start > 0 && rho_end > 0 ||
            throw(ArgumentError("$schedule rho annealing requires positive rho_start and rho_end."))
        epochs == 1 && return [rho_start]
        return exp.(range(log(rho_start), log(rho_end); length=epochs))
    end

    throw(ArgumentError("unsupported rho_schedule $(schedule)"))
end

function rho_ref_schedule_for_config(config, rho_schedule=rho_schedule_for_config(config))
    epochs = Int(config.epochs)
    raw_schedule = config_value(config, :rho_ref_schedule, :match_input)

    if raw_schedule isa AbstractVector
        length(raw_schedule) == epochs ||
            throw(ArgumentError("rho_ref_schedule vector must have one value per epoch."))
        return Float64.(raw_schedule)
    end

    schedule = Symbol(raw_schedule)
    if schedule in (:match_input, :same, :input)
        length(rho_schedule) == epochs ||
            throw(ArgumentError("rho_schedule must have one value per epoch."))
        return Float64.(rho_schedule)
    elseif schedule in (:zero, :zeros, :none)
        return zeros(Float64, epochs)
    elseif schedule == :constant
        return fill(Float64(config_value(config, :rho_ref, config.rho)), epochs)
    elseif schedule == :linear
        epochs == 1 && return [Float64(config_value(config, :rho_ref_start, config_value(config, :rho_start, config.rho)))]
        return collect(
            range(
                Float64(config_value(config, :rho_ref_start, config_value(config, :rho_start, config.rho))),
                Float64(config_value(config, :rho_ref_end, config_value(config, :rho_end, config.rho)));
                length=epochs,
            ),
        )
    elseif schedule == :geometric || schedule == :exponential
        rho_ref_start = Float64(config_value(config, :rho_ref_start, config_value(config, :rho_start, config.rho)))
        rho_ref_end = Float64(config_value(config, :rho_ref_end, config_value(config, :rho_end, config.rho)))
        rho_ref_start > 0 && rho_ref_end > 0 ||
            throw(ArgumentError("$schedule rho_ref annealing requires positive rho_ref_start and rho_ref_end."))
        epochs == 1 && return [rho_ref_start]
        return exp.(range(log(rho_ref_start), log(rho_ref_end); length=epochs))
    end

    throw(ArgumentError("unsupported rho_ref_schedule $(schedule)"))
end

function mlflow_experiment_spec(objects, config)
    problem_metadata = object_metadata(objects, :problem_metadata)
    return (;
        problem=string(
            metadata_value(
                problem_metadata,
                :problem,
                config_value(config, :problem, :experiment),
            ),
        ),
        instance_id=metadata_value(problem_metadata, :instance_id, missing),
        method=string(config_value(config, :method, config.loss)),
        variant=string(config_value(config, :method_variant, "default")),
        run_group=string(config_value(config, :gridsearch_id, "")),
        candidate_index=config_value(config, :candidate_index, ""),
        replicate_index=config_value(
            config,
            :repeat_index,
            config_value(config, :replicate_index, missing),
        ),
        base_run_id=string(config_value(config, :base_run_id, "")),
    )
end

function mlflow_data_spec(objects, config)
    train_size = length(objects.data.train)
    validation_size = length(objects.data.validation)
    test_size = length(objects.data.test)
    context_dimension = isempty(objects.data.train) ? 0 : length(first(objects.data.train).context)
    scenario_count =
        isempty(objects.data.train) ? 0 : length(first(objects.data.train).scenario_parameters)
    target_dimension = isempty(objects.data.train) ?
        0 :
        length(target_from_contextual_point(first(objects.data.train), objects))

    defaults = (;
        generator="experiment_config",
        dataset_name=mlflow_dataset_name(objects, config),
        dataset_digest=mlflow_dataset_digest(objects, config),
        train_size=train_size,
        validation_size=validation_size,
        test_size=test_size,
        context_dimension=context_dimension,
        scenario_count=scenario_count,
        target_dimension=target_dimension,
        validation_fraction=config_value(config, :validation_fraction, missing),
        test_fraction=config_value(config, :test_fraction, missing),
        training_data_seed=config_value(config, :training_data_seed, missing),
        repeat_training_data_seed=config_value(
            config,
            :repeat_training_data_seed,
            config_value(config, :training_data_seed, missing),
        ),
        train_context_seed=config_value(config, :training_data_seed, missing),
        train_scenario_seed=config_value(config, :training_data_seed, missing),
        split_seed=config_value(config, :training_data_seed, missing),
        optimization_seed=config_value(config, :optimization_seed, missing),
    )
    return merge(defaults, object_metadata(objects, :data_metadata))
end

function mlflow_model_spec(model, objects, config)
    defaults = (;
        architecture="Flux.Chain",
        depth=config_value(config, :depth, missing),
        width=config_value(config, :hidden_size, missing),
        activation=string(config_value(config, :activation, "relu")),
        output_activation="softplus",
        dropout=config_value(config, :dropout, missing),
        parameter_count=model_parameter_count(model),
        initialization_seed=string(
            config_value(
                config,
                :model_initialization_seed,
                config_value(config, :seed, "global_rng"),
            ),
        ),
        input_dimension=isempty(objects.data.train) ? 0 : length(first(objects.data.train).context),
        output_dimension=model_output_dimension(objects),
    )
    return merge(defaults, object_metadata(objects, :model_metadata))
end

function mlflow_method_spec(objects, config)
    mu_schedule = mu_schedule_for_config(config)
    mu_ref_schedule = mu_ref_schedule_for_config(config, mu_schedule)
    rho_schedule = rho_schedule_for_config(config)
    rho_ref_schedule = rho_ref_schedule_for_config(config, rho_schedule)
    policy_inference_mu = policy_inference_mu_for_config(config, mu_schedule)
    policy_inference_rho = policy_inference_rho_for_config(config, rho_schedule)
    return (;
        loss=string(config.loss),
        solver=string(config.solver),
        decoder=string(typeof(objects.scenario_decoder)),
        reference_decoder=string(typeof(objects.reference_scenario_decoder)),
        learned_components="h",
        nr_scenarios=something(effective_nr_scenarios(objects, config), 1),
        mu=config.mu,
        mu_start=isempty(mu_schedule) ? missing : first(mu_schedule),
        mu_end=isempty(mu_schedule) ? missing : last(mu_schedule),
        mu_schedule=string(config_value(config, :mu_schedule, :constant)),
        mu_ref=Float64(config_value(config, :mu_ref, config.mu)),
        mu_ref_start=isempty(mu_ref_schedule) ? missing : first(mu_ref_schedule),
        mu_ref_end=isempty(mu_ref_schedule) ? missing : last(mu_ref_schedule),
        mu_ref_schedule=string(config_value(config, :mu_ref_schedule, :match_input)),
        rho=config.rho,
        rho_start=isempty(rho_schedule) ? missing : first(rho_schedule),
        rho_end=isempty(rho_schedule) ? missing : last(rho_schedule),
        rho_schedule=string(config_value(config, :rho_schedule, :constant)),
        rho_ref=Float64(config_value(config, :rho_ref, config.rho)),
        rho_ref_start=isempty(rho_ref_schedule) ? missing : first(rho_ref_schedule),
        rho_ref_end=isempty(rho_ref_schedule) ? missing : last(rho_ref_schedule),
        rho_ref_schedule=string(config_value(config, :rho_ref_schedule, :match_input)),
        homotopy_schedule=string(config_value(config, :mu_schedule, :constant)),
        log_barrier_training=any(!iszero, mu_schedule),
        reference_log_barrier_training=any(!iszero, mu_ref_schedule),
        quadratic_smoothing_training=any(!iszero, rho_schedule),
        reference_quadratic_smoothing_training=any(!iszero, rho_ref_schedule),
        log_barrier_inference=Bool(config_value(config, :log_barrier_inference, any(!iszero, mu_schedule))),
        display_smooth=Bool(config_value(config, :display_smooth, false)),
        display_real=config_value(config, :display_real, nothing),
        optimality_evaluation=Bool(config_value(config, :optimality_evaluation, false)),
        optimality_test_sample_count=Int(config_value(config, :optimality_test_sample_count, 0)),
        optimality_train_sample_count=Int(config_value(config, :optimality_train_sample_count, 0)),
        optimality_validation_sample_count=Int(config_value(config, :optimality_validation_sample_count, 0)),
        optimality_mu=Float64(config_value(config, :optimality_mu, 0.0)),
        optimality_rho=Float64(config_value(config, :optimality_rho, 0.0)),
        optimality_evaluation_batches=config_value(config, :optimality_evaluation_batches, nothing),
        policy_inference_mu=policy_inference_mu,
        policy_inference_rho=policy_inference_rho,
        fine_tuning=Bool(config_value(config, :fine_tuning, false)),
        annealing=Bool(config_value(config, :annealing, false)),
        knn_homogenization=Bool(config_value(config, :knn_homogenization, false)),
        rrule_variant=string(config_value(config, :rrule_variant, "default")),
    )
end

function model_parameter_count(model)
    try
        return sum(length, Flux.trainables(model))
    catch
        try
            return sum(length, Flux.params(model))
        catch
            return missing
        end
    end
end

function split_mse(model, dataset, objects)
    target = dataset_target_matrix(dataset, objects)
    prediction = matrix_like(model(dataset_context_matrix(dataset)), target)
    return mean(abs2, prediction .- target)
end

function dataset_context_matrix(dataset)
    isempty(dataset) && return zeros(Float64, 0, 0)
    return reduce(hcat, (point.context for point in dataset))
end

function dataset_target_matrix(dataset, objects)
    isempty(dataset) && return zeros(Float64, 0, 0)
    return reduce(hcat, (target_from_contextual_point(point, objects) for point in dataset))
end

function target_from_contextual_point(point, objects)
    extractor = target_extractor(objects)
    target = extractor(point)
    target isa AbstractVector ||
        throw(ArgumentError("training object target_extractor must return an AbstractVector."))
    return target
end

function target_extractor(objects)
    if hasproperty(objects, :target_extractor)
        extractor = getproperty(objects, :target_extractor)
        extractor isa Function ||
            throw(ArgumentError("training object target_extractor must be a function."))
        return extractor
    end

    throw(
        ArgumentError(
            "training objects must provide target_extractor for reporting and MSE evaluation.",
        ),
    )
end

function model_output_dimension(objects)
    model_metadata = object_metadata(objects, :model_metadata)
    output_dimension = metadata_value(model_metadata, :output_dimension, nothing)
    isnothing(output_dimension) || return output_dimension

    isempty(objects.data.train) && return 0
    return length(target_from_contextual_point(first(objects.data.train), objects))
end

function extract_model(train_result, fallback_generator)
    candidates = Any[train_result, fallback_generator]

    if train_result isa Tuple
        append!(candidates, collect(train_result))
    end

    for candidate in candidates
        candidate === nothing && continue

        if hasproperty(candidate, :scenario_generator)
            scenario_generator = getproperty(candidate, :scenario_generator)
            hasproperty(scenario_generator, :neural_net) &&
                return getproperty(scenario_generator, :neural_net)
        end

        hasproperty(candidate, :neural_net) && return getproperty(candidate, :neural_net)
        hasproperty(candidate, :model) && return getproperty(candidate, :model)
    end

    return fallback_generator.neural_net
end

function extract_epoch_history(train_result)
    raw_history = find_history_payload(train_result)
    return normalize_history(raw_history)
end

function find_history_payload(train_result)
    train_result === nothing && return nothing

    if hasproperty(train_result, :history)
        return getproperty(train_result, :history)
    end
    if hasproperty(train_result, :metrics)
        return getproperty(train_result, :metrics)
    end
    if hasproperty(train_result, :epoch_history)
        return getproperty(train_result, :epoch_history)
    end

    if train_result isa Tuple
        for item in train_result
            payload = find_history_payload(item)
            payload === nothing || return payload
        end
    end

    return train_result isa AbstractVector ? train_result : nothing
end

function normalize_history(raw_history)
    raw_history === nothing && return Dict{Symbol,Any}[]

    if raw_history isa NamedTuple
        return normalize_namedtuple_history(raw_history)
    end

    if raw_history isa AbstractVector
        rows = Dict{Symbol,Any}[]
        for (index, row) in enumerate(raw_history)
            push!(rows, normalize_history_row(row, index))
        end
        return rows
    end

    return [Dict{Symbol,Any}(:epoch => 1, :value => string(raw_history))]
end

function normalize_namedtuple_history(history::NamedTuple)
    vector_lengths = [
        length(value) for value in values(history) if value isa AbstractVector
    ]
    isempty(vector_lengths) && return [Dict{Symbol,Any}(pairs(history))]

    row_count = maximum(vector_lengths)
    rows = Dict{Symbol,Any}[]

    for index in 1:row_count
        row = Dict{Symbol,Any}(:epoch => index)
        for key in keys(history)
            value = getproperty(history, key)
            if value isa AbstractVector
                row[key] = index <= length(value) ? value[index] : missing
            else
                row[key] = value
            end
        end
        push!(rows, row)
    end

    return rows
end

function normalize_history_row(row::NamedTuple, index)
    output = Dict{Symbol,Any}(pairs(row))
    haskey(output, :epoch) || (output[:epoch] = index)
    return output
end

function normalize_history_row(row::AbstractDict, index)
    output = Dict{Symbol,Any}()
    for (key, value) in row
        output[Symbol(key)] = value
    end
    haskey(output, :epoch) || (output[:epoch] = index)
    return output
end

function normalize_history_row(row::Number, index)
    return Dict{Symbol,Any}(:epoch => index, :value => Float64(row))
end

function normalize_history_row(row, index)
    return Dict{Symbol,Any}(:epoch => index, :value => string(row))
end

function evaluate_model_on_splits(model, splits, objects, config)
    try
        Flux.testmode!(model)
    catch
    end

    train_metrics = evaluate_split(model, splits.train, objects, config, "train")
    validation_metrics = evaluate_split(model, splits.validation, objects, config, "validation")
    test_metrics = evaluate_split(model, splits.test, objects, config, "test")
    return merge(train_metrics, validation_metrics, test_metrics)
end

function evaluate_model_for_reporting(model, objects, config)
    metrics = evaluate_model_on_splits(model, objects.data, objects, config)
    Bool(config_value(config, :optimality_evaluation, false)) || return metrics
    GC.gc()
    optimality_metrics = with_optimality_evaluation_slot(config) do
        evaluate_optimality_on_splits(model, objects, config)
    end
    GC.gc()
    return merge(metrics, optimality_metrics)
end

function evaluate_optimality_on_splits(model, objects, config)
    spec = experiment_from_config(config)
    spec === nothing && throw(
        ArgumentError(
            "optimality_evaluation=true requires config.experiment_id so precomputed optimal results can be loaded.",
        ),
    )

    policy = optimality_policy(model, objects, config)
    metrics = NamedTuple()

    for (split_name, dataset) in optimality_splits_for_config(objects, config)
        isempty(dataset) && continue
        optimal_results = load_optimal_results(spec, split_name; dataset=dataset)
        dataset, optimal_results =
            limit_optimality_evaluation_batches(dataset, optimal_results, config)
        result = ContextualDFLExperiments.evaluate_policy_against_optimum(
            policy,
            dataset,
            objects.program,
            objects.reference_scenario_decoder,
            objects.solver;
            optimal_results=optimal_results,
            split_name=split_name,
            mu=Float64(config_value(config, :optimality_mu, 0.0)),
            rho=Float64(config_value(config, :optimality_rho, 0.0)),
        )
        metrics = merge(metrics, result.metrics)
    end

    return metrics
end

function with_optimality_evaluation_slot(callback, config)
    concurrency = optimality_evaluation_concurrency(config)
    concurrency === nothing && return callback()

    slot_path, slot_index = acquire_optimality_evaluation_slot(config, concurrency)
    try
        println(
            "Acquired optimality evaluation slot $slot_index/$concurrency for ",
            config_value(config, :run_id, ""),
        )
        return callback()
    finally
        release_optimality_evaluation_slot(slot_path)
        println(
            "Released optimality evaluation slot $slot_index/$concurrency for ",
            config_value(config, :run_id, ""),
        )
    end
end

function optimality_evaluation_concurrency(config)
    value = config_value(
        config,
        :optimality_evaluation_concurrency,
        get(ENV, "CONTEXTUAL_DFL_OPTIMALITY_EVAL_CONCURRENCY", nothing),
    )
    value === nothing && return nothing
    text = strip(string(value))
    isempty(text) && return nothing
    parsed = tryparse(Int, text)
    parsed === nothing &&
        throw(ArgumentError("optimality_evaluation_concurrency must be an integer."))
    parsed <= 0 && return nothing
    return parsed
end

function optimality_evaluation_lock_root(config)
    root = string(
        config_value(
            config,
            :optimality_evaluation_lock_dir,
            get(
                ENV,
                "CONTEXTUAL_DFL_OPTIMALITY_EVAL_LOCK_DIR",
                joinpath(tempdir(), "contextualdfl_optimality_eval"),
            ),
        ),
    )
    grid_id = string(config_value(config, :gridsearch_id, "default"))
    safe_grid_id = replace(grid_id, r"[^A-Za-z0-9_.-]" => "_")
    return joinpath(root, safe_grid_id)
end

function acquire_optimality_evaluation_slot(config, concurrency::Integer)
    root = optimality_evaluation_lock_root(config)
    mkpath(root)
    owner_text = join(
        [
            "pid=$(getpid())",
            "host=$(Sockets.gethostname())",
            "run_id=$(config_value(config, :run_id, ""))",
            "started_at=$(unix_milliseconds())",
        ],
        "\n",
    ) * "\n"

    while true
        for slot_index in 1:concurrency
            slot_path = joinpath(root, "slot_$slot_index")
            cleanup_stale_optimality_evaluation_slot!(slot_path)
            try
                mkdir(slot_path)
                write(joinpath(slot_path, "owner.txt"), owner_text)
                return slot_path, slot_index
            catch error
                isdir(slot_path) && continue
                rethrow()
            end
        end
        sleep(1.0 + 0.5 * rand())
    end
end

function release_optimality_evaluation_slot(slot_path)
    isempty(string(slot_path)) && return nothing
    rm(slot_path; recursive=true, force=true)
    return nothing
end

function cleanup_stale_optimality_evaluation_slot!(slot_path)
    isdir(slot_path) || return nothing
    owner_path = joinpath(slot_path, "owner.txt")
    isfile(owner_path) || return nothing

    owner = read(owner_path, String)
    match_result = match(r"pid=(\d+)", owner)
    match_result === nothing && return nothing
    pid = parse(Int, match_result.captures[1])
    process_alive(pid) && return nothing

    rm(slot_path; recursive=true, force=true)
    return nothing
end

function process_alive(pid::Integer)
    try
        run(pipeline(`kill -0 $pid`; stdout=devnull, stderr=devnull))
        return true
    catch
        return false
    end
end

function limit_optimality_evaluation_batches(dataset, optimal_results, config)
    batch_limit = config_value(config, :optimality_evaluation_batches, nothing)
    batch_limit === nothing && return dataset, optimal_results

    batch_limit = Int(batch_limit)
    batch_limit > 0 ||
        throw(ArgumentError("optimality_evaluation_batches must be positive."))

    limited_dataset = similar(dataset, 0)
    limited_results = NamedTuple[]
    for (data_point, result) in zip(dataset, optimal_results)
        objective_values = optimality_objective_values(result)
        source_batch_count = length(objective_values)
        batch_limit <= source_batch_count || throw(
            ArgumentError(
                "optimality_evaluation_batches=$batch_limit exceeds available optimality batches $source_batch_count.",
            ),
        )

        scenario_count = length(data_point.scenario_parameters)
        scenario_count % source_batch_count == 0 || throw(
            ArgumentError(
                "scenario count $scenario_count is not divisible by stored optimality batches $source_batch_count.",
            ),
        )

        scenarios_per_batch = scenario_count ÷ source_batch_count
        scenario_limit = batch_limit * scenarios_per_batch
        selected_objective_values = objective_values[1:batch_limit]
        push!(
            limited_dataset,
            ContextualDFL.ContextualDataPoint(
                data_point.context,
                data_point.scenario_parameters[1:scenario_limit],
            ),
        )
        push!(
            limited_results,
            merge(
                result,
                (;
                    evaluation_batches=batch_limit,
                    objective_values=selected_objective_values,
                    objective_value=mean(Float64.(selected_objective_values)),
                ),
            ),
        )
    end
    return limited_dataset, limited_results
end

function optimality_objective_values(result)
    if hasproperty(result, :objective_values)
        values = Float64.(collect(result.objective_values))
        isempty(values) &&
            throw(ArgumentError("optimality objective_values must not be empty."))
        return values
    elseif hasproperty(result, :objective_value)
        return [Float64(result.objective_value)]
    end

    throw(ArgumentError("optimality results must contain objective_values."))
end

function optimality_policy(model, objects, config)
    scenario_generator = ContextualDFL.ScenarioGenerator(
        neural_net=model,
        scenario_decoder=objects.scenario_decoder,
    )
    policy_mu = policy_inference_mu_for_config(config)
    policy_rho = policy_inference_rho_for_config(config)
    return ContextualDFLExperiments.ScenarioGenerationPolicy(
        scenario_generator,
        objects.solver,
        objects.program;
        mu=policy_mu,
        rho=policy_rho,
    )
end

function optimality_evaluation_datasets(objects, config)
    datasets = Pair{Symbol,Any}[]

    push!(
        datasets,
        :test => limited_dataset(
            objects.data.test,
            Int(config_value(config, :optimality_test_sample_count, 0)),
        ),
    )

    train_count = Int(config_value(config, :optimality_train_sample_count, 0))
    train_count > 0 && push!(
        datasets,
        :train_subset => limited_dataset(objects.data.train, train_count),
    )

    validation_count = Int(config_value(config, :optimality_validation_sample_count, 0))
    validation_count > 0 && push!(
        datasets,
        :validation_subset => limited_dataset(objects.data.validation, validation_count),
    )

    return datasets
end

function limited_dataset(dataset, limit::Integer)
    limit <= 0 && return dataset
    return dataset[1:min(Int(limit), length(dataset))]
end

function evaluate_split(model, dataset, objects, config, prefix)
    x_data = dataset_context_matrix(dataset)
    target = dataset_target_matrix(dataset, objects)
    predictions, inference_timings = timed_model_prediction(model, x_data, config)
    target = reporting_target_for_prediction(target, predictions)
    prediction_matrix = matrix_like(predictions, target)

    errors = prediction_matrix .- target
    absolute_errors = abs.(errors)
    denominator = max.(abs.(target), config.tolerance_absolute_floor)
    tolerance = max.(abs.(target) .* config.tolerance_relative, config.tolerance_absolute_floor)

    metrics = (;
        mse=mean(abs2, errors),
        mae=mean(absolute_errors),
        rmse=sqrt(mean(abs2, errors)),
        relative_mae=mean(absolute_errors ./ denominator),
        tolerance_accuracy=mean(absolute_errors .<= tolerance),
        sample_count=size(target, 2),
        inference_seconds_mean=mean(inference_timings),
        inference_seconds_p95=percentile_95(inference_timings),
        inference_seconds_total=sum(inference_timings),
    )

    return prefix_named_tuple(Symbol(prefix), metrics)
end

function timed_model_prediction(model, x_data, config)
    repetitions = max(Int(config_value(config, :inference_repetitions, 1)), 1)
    timings = Float64[]
    predictions = nothing

    for _ in 1:repetitions
        elapsed = @elapsed begin
            predictions = model(x_data)
        end
        push!(timings, elapsed)
    end

    return predictions, timings
end

function reporting_target_for_prediction(target, prediction)
    output_dimension = reporting_prediction_output_dimension(prediction, target)
    output_dimension === nothing && return target
    size(target, 1) == output_dimension && return target
    size(target, 1) % output_dimension == 0 || return target

    scenario_count = size(target, 1) ÷ output_dimension
    scenario_count > 1 || return target

    scenario_target = reshape(target, output_dimension, scenario_count, size(target, 2))
    return dropdims(mean(scenario_target; dims=2); dims=2)
end

function reporting_prediction_output_dimension(prediction, target)
    prediction_matrix = Array(prediction)
    if ndims(prediction_matrix) == 2 && size(prediction_matrix, 2) == size(target, 2)
        return size(prediction_matrix, 1)
    elseif ndims(prediction_matrix) == 1 && size(target, 2) == 1
        return length(prediction_matrix)
    end
    return nothing
end

function percentile_95(values::AbstractVector{<:Real})
    isempty(values) && return NaN
    sorted = sort!(collect(Float64.(values)))
    index = clamp(ceil(Int, 0.95 * length(sorted)), 1, length(sorted))
    return sorted[index]
end

function matrix_like(value, target)
    matrix = Array(value)
    size(matrix) == size(target) && return matrix
    length(matrix) == length(target) && return reshape(matrix, size(target))

    throw(
        DimensionMismatch(
            "prediction size $(size(matrix)) cannot be compared with target size $(size(target))",
        ),
    )
end

function prefix_named_tuple(prefix::Symbol, values::NamedTuple)
    prefixed_pairs = Pair{Symbol,Any}[]
    for key in keys(values)
        push!(prefixed_pairs, Symbol(prefix, "_", key) => getproperty(values, key))
    end
    return (; prefixed_pairs...)
end
