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
            objects = resource_allocation_training_objects(cfg)
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
            run_id=cfg.run_id,
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
            backend="ContextualDFL.train_with_mlflow!",
            fallback_reason="",
            final_metrics=mlflow_result.final_metrics,
        )
    end

    result = ContextualDFL.train!(
        objects.scenario_generator.neural_net,
        objects.loss,
        mu_schedule_for_config(config),
        objects.data.train;
        learning_rate=config.learning_rate,
        optimizer_type=Flux.Adam,
        epochs=config.epochs,
        batchsize=config.batch_size,
        shuffle=Bool(config_value(config, :shuffle, false)),
    )

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

    result = ContextualDFL.train_with_mlflow!(
        mlf,
        experiment_id,
        model,
        loss,
        mu_schedule,
        objects.data.train;
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
        experiment_spec=mlflow_experiment_spec(objects, config),
        data_spec=mlflow_data_spec(objects, config),
        model_spec=mlflow_model_spec(model, objects, config),
        method_spec=mlflow_method_spec(objects, config),
        evaluation_callbacks=Dict(
            "" => (train_result) -> begin
                trained_model = extract_model(train_result, objects.scenario_generator)
                metrics = evaluate_model_for_reporting(trained_model, objects, config)
                final_metrics[] = metrics
                return metrics
            end,
        ),
    )

    return (; result=result, final_metrics=final_metrics[])
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

function mlflow_dataset_name(config)
    return string(config_value(config, :mlflow_dataset_name, "resource_allocation_generated"))
end

function mlflow_dataset_source(config)
    return join(
        (
            "ContextualDFLExperiments.resource_allocation",
            "problem_data=default_resource_allocation_problem_data",
            "context_generator=ResourceAllocationContextDataGenerator",
            "scenario_generator=ResourceAllocationScenarioDataGenerator",
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
        "train_x=$(size(dataset_context_matrix(objects.data.train)))",
        "train_y=$(size(dataset_demand_matrix(objects.data.train)))",
        "validation_x=$(size(dataset_context_matrix(objects.data.validation)))",
        "validation_y=$(size(dataset_demand_matrix(objects.data.validation)))",
        "test_x=$(size(dataset_context_matrix(objects.data.test)))",
        "test_y=$(size(dataset_demand_matrix(objects.data.test)))",
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

function contextual_dfl_loss(objects, config)
    return objects.loss
end

function mu_schedule_for_config(config)
    epochs = Int(config.epochs)
    epochs >= 0 || throw(ArgumentError("epochs must be non-negative."))
    epochs == 0 && return Float64[]

    schedule = Symbol(config_value(config, :mu_schedule, :constant))
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

function mlflow_experiment_spec(objects, config)
    return (;
        problem=string(config_value(config, :problem, :resource_allocation)),
        instance_id=resource_problem_digest(objects.problem),
        method=string(config_value(config, :method, config.loss)),
        variant=string(config_value(config, :method_variant, "default")),
        run_group=string(config_value(config, :gridsearch_id, "")),
        candidate_index=config_value(config, :candidate_index, ""),
        replicate_index=config_value(config, :replicate_index, config.seed),
        base_run_id=string(config_value(config, :base_run_id, "")),
    )
end

function mlflow_data_spec(objects, config)
    train_size = length(objects.data.train)
    validation_size = length(objects.data.validation)
    test_size = length(objects.data.test)
    context_dimension = isempty(objects.data.train) ? 0 : length(first(objects.data.train).context)
    scenario_count = isempty(objects.data.train) ? 0 : length(first(objects.data.train).scenario_parameters)

    return (;
        generator="ContextualDFLExperiments.resource_allocation",
        dataset_name=mlflow_dataset_name(config),
        dataset_digest=mlflow_dataset_digest(objects, config),
        train_size=train_size,
        validation_size=validation_size,
        test_size=test_size,
        context_dimension=context_dimension,
        scenario_count=scenario_count,
        n_samples=config.n_samples,
        validation_fraction=config.validation_fraction,
        test_fraction=config.test_fraction,
        noise_generator="normal",
        sigma=config.sigma,
        demand_power=config.demand_power,
        context_terms=config.context_terms,
        train_context_seed=config.seed,
        test_context_seed=config.seed,
        train_scenario_seed=config.seed,
        test_scenario_seed=config.seed,
        split_seed=config.seed,
        optimization_seed=config_value(config, :optimization_seed, config.seed),
    )
end

function mlflow_model_spec(model, objects, config)
    return (;
        architecture="Flux.Chain",
        depth=config.depth,
        width=config.hidden_size,
        activation=string(config_value(config, :activation, "relu")),
        output_activation="softplus",
        dropout=config.dropout,
        parameter_count=model_parameter_count(model),
        initialization_seed=string(config_value(config, :model_initialization_seed, "global_rng")),
        input_dimension=isempty(objects.data.train) ? 0 : length(first(objects.data.train).context),
        output_dimension=resource_allocation_demand_count(objects.problem),
    )
end

function mlflow_method_spec(objects, config)
    mu_schedule = mu_schedule_for_config(config)
    return (;
        loss=string(config.loss),
        solver=string(config.solver),
        decoder=string(typeof(objects.scenario_decoder)),
        reference_decoder=string(typeof(objects.reference_scenario_decoder)),
        learned_components="h",
        nr_scenarios=Int(config_value(config, :nr_scenarios, 1)),
        mu=config.mu,
        mu_start=isempty(mu_schedule) ? missing : first(mu_schedule),
        mu_end=isempty(mu_schedule) ? missing : last(mu_schedule),
        mu_schedule=string(config_value(config, :mu_schedule, :constant)),
        rho=config.rho,
        homotopy_schedule=string(config_value(config, :mu_schedule, :constant)),
        log_barrier_training=any(!iszero, mu_schedule),
        log_barrier_inference=Bool(config_value(config, :log_barrier_inference, any(!iszero, mu_schedule))),
        optimality_evaluation=Bool(config_value(config, :optimality_evaluation, false)),
        optimality_test_sample_count=Int(config_value(config, :optimality_test_sample_count, 0)),
        optimality_train_sample_count=Int(config_value(config, :optimality_train_sample_count, 0)),
        optimality_validation_sample_count=Int(config_value(config, :optimality_validation_sample_count, 0)),
        optimality_mu=Float64(config_value(config, :optimality_mu, 0.0)),
        policy_inference_mu=Float64(config_value(config, :policy_inference_mu, config.mu)),
        fine_tuning=Bool(config_value(config, :fine_tuning, false)),
        annealing=Bool(config_value(config, :annealing, false)),
        knn_homogenization=Bool(config_value(config, :knn_homogenization, false)),
        rrule_variant=string(config_value(config, :rrule_variant, "default")),
    )
end

function resource_problem_digest(problem)
    values = (
        "service_rate=$(vec(problem.problem_data.service_rate_parameters))",
        "first_stage=$(problem.problem_data.first_stage_costs)",
        "second_stage=$(problem.problem_data.second_stage_costs)",
        "yield=$(problem.problem_data.yield_parameters)",
    )
    return "sha256:" * bytes2hex(sha256(join(values, "\n")))
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

function split_mse(model, dataset)
    target = dataset_demand_matrix(dataset)
    prediction = matrix_like(model(dataset_context_matrix(dataset)), target)
    return mean(abs2, prediction .- target)
end

function dataset_context_matrix(dataset)
    isempty(dataset) && return zeros(Float64, 0, 0)
    return reduce(hcat, (point.context for point in dataset))
end

function dataset_demand_matrix(dataset)
    isempty(dataset) && return zeros(Float64, 0, 0)
    return reduce(hcat, (demand_from_contextual_point(point) for point in dataset))
end

function demand_from_contextual_point(point)
    scenario_parameters = point.scenario_parameters
    length(scenario_parameters) == 1 ||
        throw(ArgumentError("expected exactly one scenario per contextual data point"))
    return resource_allocation_demand_from_scenario(only(scenario_parameters))
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

function evaluate_model_for_reporting(model, objects, config)
    metrics = evaluate_model_on_splits(model, objects.data, config)
    Bool(config_value(config, :optimality_evaluation, false)) || return metrics
    return merge(metrics, evaluate_optimality_on_splits(model, objects, config))
end

function evaluate_optimality_on_splits(model, objects, config)
    policy = optimality_policy(model, objects, config)
    metrics = NamedTuple()

    for (split_name, dataset) in optimality_evaluation_datasets(objects, config)
        isempty(dataset) && continue
        result = ContextualDFLExperiments.evaluate_policy_against_optimum(
            policy,
            dataset,
            objects.program,
            objects.reference_scenario_decoder,
            objects.solver;
            split_name=split_name,
            mu=Float64(config_value(config, :optimality_mu, 0.0)),
        )
        metrics = merge(metrics, result.metrics)
    end

    return metrics
end

function optimality_policy(model, objects, config)
    scenario_generator = ContextualDFL.ScenarioGenerator(
        neural_net=model,
        scenario_decoder=objects.scenario_decoder,
    )
    policy_mu = Float64(config_value(config, :policy_inference_mu, config.mu))
    return ContextualDFLExperiments.ScenarioGenerationPolicy(
        scenario_generator,
        objects.solver,
        objects.program;
        mu=policy_mu,
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

function evaluate_split(model, dataset, config, prefix)
    x_data = dataset_context_matrix(dataset)
    target = dataset_demand_matrix(dataset)
    predictions, inference_timings = timed_model_prediction(model, x_data, config)
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
