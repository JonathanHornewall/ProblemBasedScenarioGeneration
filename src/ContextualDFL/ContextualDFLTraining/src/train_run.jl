using Dates
using Distributed
using Flux
using Random
using SHA
using Sockets
using Statistics

function normalize_config(config::NamedTuple)
    return merge(DEFAULT_RUN_SETTINGS, config)
end

function utc_timestamp()
    return string(Dates.now(Dates.UTC))
end

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
    started_at = utc_timestamp()
    elapsed_seconds = 0.0

    try
        train_result = nothing
        training_backend = ""
        fallback_reason = ""
        objects = nothing

        elapsed_seconds = @elapsed begin
            objects = resource_allocation_training_objects(cfg)
            training = train_with_contextualdfl_or_fallback(objects, cfg)
            train_result = training.result
            training_backend = training.backend
            fallback_reason = training.fallback_reason
        end

        model = extract_model(train_result, objects.scenario_generator)
        metrics = merge(
            evaluate_model_on_splits(model, objects.data, cfg),
            (;
                training_backend=training_backend,
                fallback_reason=fallback_reason,
            ),
        )
        history = extract_epoch_history(train_result)

        return (;
            status="ok",
            run_id=cfg.run_id,
            config=cfg,
            worker=worker_metadata(),
            final_metrics=metrics,
            epoch_history=history,
            error="",
            started_at=started_at,
            finished_at=utc_timestamp(),
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
            finished_at=utc_timestamp(),
            elapsed_seconds=elapsed_seconds,
        )
    end
end

function train_with_contextualdfl_or_fallback(objects, config)
    if mlflow_enabled(config)
        result = train_with_contextualdfl_mlflow(objects, config)
        return (;
            result=result,
            backend="ContextualDFL.train_with_mlflow!",
            fallback_reason="",
        )
    end

    try
        result = ContextualDFL.train(
            objects.scenario_generator,
            objects.loss,
            objects.data.train,
            objects.scenario_decoder,
            objects.schedules.mu,
            objects.schedules.rho,
            objects.schedules.batch_size,
            objects.schedules.step_size;
            epochs=config.epochs,
            validation_data=objects.data.validation,
            test_data=objects.data.test,
            config=config,
        )

        return (;
            result=result,
            backend="ContextualDFL.train",
            fallback_reason="",
        )
    catch error
        message = sprint(showerror, error)
        package_training_placeholder_error(message) || rethrow()

        fallback = supervised_mse_train!(
            objects.scenario_generator.neural_net,
            objects.data,
            config,
        )
        return (;
            result=fallback,
            backend="ContextualDFLTraining.supervised_mse_fallback",
            fallback_reason=first_line(message),
        )
    end
end

function train_with_contextualdfl_mlflow(objects, config)
    mlflow_config = add_worker_mlflow_tags(config)
    mlf, experiment_id = mlflow_client_for_config(mlflow_config)
    data = ContextualDFL._contextual_training_samples(objects.data.train, objects.scenario_decoder)
    loss = contextual_dfl_loss(objects, config)
    upload_model_artifact = Bool(config_value(config, :mlflow_upload_model_artifact, false))
    model_save_path = mlflow_model_save_path(config)

    return ContextualDFL.train_with_mlflow!(
        mlf,
        experiment_id,
        loss,
        objects.scenario_generator.neural_net,
        data;
        learning_rate=config.learning_rate,
        optimizer_type=Flux.Adam,
        epochs=config.epochs,
        batchsize=config.batch_size,
        shuffle=Bool(config_value(config, :shuffle, false)),
        reset_optimizer_each_epoch=Bool(
            config_value(config, :reset_optimizer_each_epoch, false),
        ),
        source_name=string(
            config_value(
                config,
                :mlflow_source_name,
                "ContextualDFLTraining/gridsearch.jl",
            ),
        ),
        source_type=string(config_value(config, :mlflow_source_type, "LOCAL")),
        dataset_name=mlflow_dataset_name(config),
        dataset_digest=mlflow_dataset_digest(objects, config),
        dataset_source_type="generated",
        dataset_source=mlflow_dataset_source(config),
        dataset_context="training",
        save_model=upload_model_artifact,
        model_save_path=model_save_path,
        upload_model_artifact=upload_model_artifact,
        model_artifact_path="models/" * basename(model_save_path),
    )
end

function add_worker_mlflow_tags(config)
    tags = string_dict(config_value(config, :mlflow_tags, nothing))
    tags["worker_id"] = string(Distributed.myid())
    tags["worker_hostname"] = Sockets.gethostname()
    tags["worker_pid"] = string(getpid())
    return merge(config, (; mlflow_tags=tags))
end

function mlflow_dataset_name(config)
    return string(config_value(config, :mlflow_dataset_name, "resource_allocation_generated"))
end

function mlflow_dataset_source(config)
    parameter_path = resource_parameter_path()
    parameter_source = isfile(parameter_path) ? parameter_path : "generated-default-parameters"
    return join(
        (
            "resource_allocation",
            "parameter_source=$(parameter_source)",
            "seed=$(config.seed)",
            "n_samples=$(config.n_samples)",
            "validation_fraction=$(config.validation_fraction)",
            "test_fraction=$(config.test_fraction)",
        ),
        ";",
    )
end

function mlflow_dataset_digest(objects, config)
    split_summary = (
        "dataset=$(mlflow_dataset_name(config))",
        "seed=$(config.seed)",
        "n_samples=$(config.n_samples)",
        "sigma=$(config.sigma)",
        "demand_power=$(config.demand_power)",
        "context_terms=$(config.context_terms)",
        "train_x=$(size(objects.data.train.x_data))",
        "train_y=$(size(objects.data.train.xi_h_data))",
        "validation_x=$(size(objects.data.validation.x_data))",
        "validation_y=$(size(objects.data.validation.xi_h_data))",
        "test_x=$(size(objects.data.test.x_data))",
        "test_y=$(size(objects.data.test.xi_h_data))",
    )
    return "sha256:" * bytes2hex(sha256(join(split_summary, "\n")))
end

function mlflow_model_save_path(config)
    run_id = string(config_value(config, :run_id, "training-run"))
    safe_run_id = replace(run_id, r"[^A-Za-z0-9_.=-]" => "_")
    return joinpath(tempdir(), safe_run_id * ".jls")
end

function contextual_dfl_loss(objects, config)
    mu = objects.schedules.mu(1)
    rho = objects.schedules.rho(1)

    return (predicted, reference) -> objects.loss(
        objects.program,
        predicted,
        reference,
        mu;
        rho=rho,
        config=config,
        validation_data=objects.data.validation,
        test_data=objects.data.test,
    )
end

function package_training_placeholder_error(message::AbstractString)
    return any(
        needle -> occursin(needle, message),
        (
            "Training has not been implemented yet",
            "MSE scenario loss has not been implemented yet",
            "Scenario generation from context has not been implemented yet",
            "Data-set scenario decoding has not been implemented yet",
        ),
    )
end

function first_line(message::AbstractString)
    lines = split(message, '\n'; limit=2)
    return isempty(lines) ? message : first(lines)
end

function supervised_mse_train!(model, splits, config)
    Random.seed!(config.seed)

    optimizer_state = Flux.setup(Adam(config.learning_rate), model)
    history = NamedTuple[]

    for epoch in 1:config.epochs
        Flux.trainmode!(model)
        loader = Flux.DataLoader(
            (splits.train.x_data, splits.train.xi_h_data);
            batchsize=config.batch_size,
            shuffle=true,
        )
        minibatch_losses = Float64[]

        for (x_batch, y_batch) in loader
            loss_value, gradients = Flux.withgradient(model) do trainable_model
                y_pred = matrix_like(trainable_model(x_batch), y_batch)
                mean(abs2, y_pred .- y_batch)
            end

            Flux.update!(optimizer_state, model, gradients[1])
            push!(minibatch_losses, Float64(loss_value))
        end

        Flux.testmode!(model)
        epoch_row = (;
            epoch=epoch,
            minibatch_mse=isempty(minibatch_losses) ? NaN : mean(minibatch_losses),
            train_mse=split_mse(model, splits.train),
            validation_mse=split_mse(model, splits.validation),
            test_mse=split_mse(model, splits.test),
        )

        push!(history, epoch_row)
    end

    Flux.testmode!(model)
    return (;
        model=model,
        history=history,
        backend="ContextualDFLTraining.supervised_mse_fallback",
    )
end

function split_mse(model, dataset)
    target = dataset.xi_h_data
    prediction = matrix_like(model(dataset.x_data), target)
    return mean(abs2, prediction .- target)
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

function evaluate_model_on_splits(model, splits, config)
    try
        Flux.testmode!(model)
    catch
    end

    train_metrics = evaluate_split(model, splits.train, config, "train")
    validation_metrics = evaluate_split(model, splits.validation, config, "validation")
    test_metrics = evaluate_split(model, splits.test, config, "test")
    return merge(train_metrics, validation_metrics, test_metrics)
end

function evaluate_split(model, dataset, config, prefix)
    predictions = model(dataset.x_data)
    target = dataset.xi_h_data
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
    )

    return prefix_named_tuple(Symbol(prefix), metrics)
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
