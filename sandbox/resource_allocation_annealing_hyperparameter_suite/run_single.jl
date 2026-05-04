#!/usr/bin/env julia

include(joinpath(@__DIR__, "suite_common.jl"))

function parse_single_args(args)
    parsed = Dict{String,Any}(
        "config" => "",
        "precompute" => false,
        "smoke" => false,
        "force-test-data" => false,
    )

    index = 1
    while index <= length(args)
        arg = args[index]
        if arg == "--config"
            index += 1
            index <= length(args) || error("--config requires a path")
            parsed["config"] = args[index]
        elseif arg == "--precompute"
            parsed["precompute"] = true
        elseif arg == "--smoke"
            parsed["smoke"] = true
        elseif arg == "--force-test-data"
            parsed["force-test-data"] = true
        elseif arg in ("-h", "--help")
            println("""
            Usage:
              julia run_single.jl --precompute [--smoke] [--force-test-data]
              julia run_single.jl --config PATH

            Runs one sandboxed resource-allocation annealing config or precomputes
            the shared test-data cache. All outputs stay under this sandbox directory.
            """)
            exit(0)
        else
            error("unknown argument: $arg")
        end
        index += 1
    end

    return parsed
end

function display_reference_input(point)
    return reduce(vcat, (scenario.h_eq_xi for scenario in point.scenario_parameters))
end

function build_loss(objects, config)
    return ContextualDFL.DflScenLoss(
        objects.scenario_decoder,
        objects.reference_scenario_decoder,
        objects.solver,
        objects.program;
        nr_scenarios=Int(config.nr_scenarios),
    )
end

function write_checkpoint!(config, model, epoch_rows, completed_epochs)
    Serialization.serialize(
        checkpoint_path(config),
        (;
            version=CONFIG_VERSION,
            config=config,
            model=model,
            epoch_rows=copy(epoch_rows),
            completed_epochs=Int(completed_epochs),
            updated_at=unix_milliseconds(),
        ),
    )
    return nothing
end

function load_checkpoint(config)
    path = checkpoint_path(config)
    isfile(path) || return nothing
    checkpoint = Serialization.deserialize(path)
    hasproperty(checkpoint, :config) &&
        checkpoint.config.run_id == config.run_id ||
        error("checkpoint at $path belongs to a different run")
    return checkpoint
end

function epoch_row(config, global_epoch, loss_value, display_loss, metadata)
    return (;
        run_id=config.run_id,
        phase=String(config.phase),
        candidate_name=config.candidate_name,
        replicate=Int(config.replicate),
        epoch=Int(global_epoch),
        mu=Float64(metadata.mu),
        mu_in=Float64(metadata.mu_in),
        mu_ref=Float64(metadata.mu_ref),
        rho_in=Float64(metadata.rho_in),
        rho_ref=Float64(metadata.rho_ref),
        iterations=Int(metadata.iterations),
        epoch_seconds=Float64(metadata.epoch_seconds),
        training_loss=Float64(loss_value),
        display_loss=Float64(display_loss),
        depth=Int(config.depth),
        activation=String(config.activation),
        mu_schedule_kind=String(config.mu_schedule_kind),
        rho=Float64(config.rho),
        batch_size=Int(config.batch_size),
        seed=Int(config.seed),
    )
end

function train_or_resume!(config)
    mkpath(run_dir(config))
    objects = problem_objects()
    checkpoint = load_checkpoint(config)
    epoch_rows = NamedTuple[]
    model = nothing
    completed_epochs = 0

    if checkpoint === nothing
        model = build_model(config, objects.problem)
        write_rows_csv(epochs_csv_path(config), epoch_rows)
        write_checkpoint!(config, model, epoch_rows, completed_epochs)
    else
        model = checkpoint.model
        epoch_rows = collect(checkpoint.epoch_rows)
        completed_epochs = Int(checkpoint.completed_epochs)
        write_rows_csv(epochs_csv_path(config), epoch_rows)
    end

    total_epochs = Int(config.epochs)
    completed_epochs >= total_epochs && return (; model=model, epoch_rows=epoch_rows)

    data_set_training = generate_training_dataset(config)
    loss = build_loss(objects, config)
    mu_in, mu_ref = mu_schedules_for_config(config)
    rho_in, rho_ref = rho_schedules_for_config(config)
    remaining = (completed_epochs + 1):total_epochs

    result = ContextualDFL.train!(
        model,
        loss,
        mu_in[remaining],
        mu_ref[remaining],
        data_set_training;
        rho_in_schedule=rho_in[remaining],
        rho_ref_schedule=rho_ref[remaining],
        opt=Flux.Adam(Float64(config.learning_rate)),
        epochs=length(remaining),
        batchsize=Int(config.batch_size),
        shuffle=false,
        display_iterations=false,
        verbose=false,
        display_plot=false,
        save_model=false,
        reset_optimizer_each_epoch=Bool(config.reset_optimizer_each_epoch),
        nr_scenarios=Int(config.nr_scenarios),
        display_smooth=false,
        display_reference_input=display_reference_input,
        on_epoch_end=(local_epoch, loss_value, display_loss, metadata) -> begin
            global_epoch = completed_epochs + Int(local_epoch)
            push!(epoch_rows, epoch_row(config, global_epoch, loss_value, display_loss, metadata))
            write_rows_csv(epochs_csv_path(config), epoch_rows)
            write_checkpoint!(config, model, epoch_rows, global_epoch)
        end,
    )

    model = result.model
    write_checkpoint!(config, model, epoch_rows, total_epochs)
    return (; model=model, epoch_rows=epoch_rows)
end

function metric_value(metrics, key, default=NaN)
    hasproperty(metrics, key) || return default
    return Float64(getproperty(metrics, key))
end

function evaluation_vector_field(row, preferred_key, fallback_key)
    if hasproperty(row, preferred_key)
        return getproperty(row, preferred_key)
    elseif hasproperty(row, fallback_key)
        return getproperty(row, fallback_key)
    end
    return Float64[]
end

function evaluate_trained_model(config, model)
    cache = load_test_cache(smoke=Bool(config.smoke))
    objects = problem_objects()
    try
        Flux.testmode!(model)
    catch
    end

    scenario_generator = ContextualDFL.ScenarioGenerator(
        neural_net=model,
        scenario_decoder=objects.scenario_decoder,
    )
    mu_in, _ = mu_schedules_for_config(config)
    rho_in, _ = rho_schedules_for_config(config)
    policy = ContextualDFLExperiments.ScenarioGenerationPolicy(
        scenario_generator,
        objects.solver,
        objects.program;
        mu=isempty(mu_in) ? 0.0 : Float64(last(mu_in)),
        rho=isempty(rho_in) ? 0.0 : Float64(last(rho_in)),
    )

    evaluation = ContextualDFLExperiments.evaluate_policy_against_optimum(
        policy,
        cache.dataset,
        objects.program,
        objects.reference_scenario_decoder,
        objects.solver;
        optimal_results=cache.optimal_results,
        split_name=:test,
        mu=0.0,
        rho=0.0,
    )

    per_sample_rows = [
        (;
            run_id=config.run_id,
            phase=String(config.phase),
            candidate_name=config.candidate_name,
            replicate=Int(config.replicate),
            sample_index=row.sample_index,
            policy_value=row.policy_value,
            optimal_value=row.optimal_value,
            regret=row.regret,
            relative_regret=row.relative_regret,
            policy_batch_values=evaluation_vector_field(
                row,
                :policy_batch_values,
                :policy_split_values,
            ),
            optimal_batch_values=evaluation_vector_field(
                row,
                :optimal_batch_values,
                :optimal_split_values,
            ),
        ) for row in evaluation.per_sample
    ]
    write_rows_csv(test_per_sample_csv_path(config), per_sample_rows)

    metrics = evaluation.metrics
    mean_relative_regret = metric_value(metrics, :test_relative_regret_mean)
    return (;
        status="ok",
        run_id=config.run_id,
        phase=config.phase,
        candidate_name=config.candidate_name,
        replicate=Int(config.replicate),
        seed=Int(config.seed),
        depth=Int(config.depth),
        activation=String(config.activation),
        mu_schedule_kind=String(config.mu_schedule_kind),
        rho=Float64(config.rho),
        batch_size=Int(config.batch_size),
        epochs=Int(config.epochs),
        final_epochs=Int(config.final_epochs),
        average_test_loss=mean_relative_regret,
        mean_test_relative_regret=mean_relative_regret,
        mean_test_regret=metric_value(metrics, :test_regret_mean),
        mean_test_policy_value=metric_value(metrics, :test_policy_value_mean),
        mean_test_optimal_value=metric_value(metrics, :test_optimal_value_mean),
        test_sample_count=metric_value(metrics, :test_sample_count),
        test_policy_eval_seconds=metric_value(metrics, :test_policy_eval_seconds),
        error="",
        finished_at=unix_milliseconds(),
    )
end

function failed_result(config, error, backtrace)
    mkpath(run_dir(config))
    text = sprint(showerror, error, backtrace)
    write(error_path(config), text)
    return (;
        status="failed",
        run_id=config.run_id,
        phase=config.phase,
        candidate_name=config.candidate_name,
        replicate=Int(config.replicate),
        seed=Int(config.seed),
        depth=Int(config.depth),
        activation=String(config.activation),
        mu_schedule_kind=String(config.mu_schedule_kind),
        rho=Float64(config.rho),
        batch_size=Int(config.batch_size),
        epochs=Int(config.epochs),
        final_epochs=Int(config.final_epochs),
        average_test_loss=Inf,
        mean_test_relative_regret=Inf,
        mean_test_regret=Inf,
        mean_test_policy_value=NaN,
        mean_test_optimal_value=NaN,
        test_sample_count=0,
        test_policy_eval_seconds=NaN,
        error=text,
        finished_at=unix_milliseconds(),
    )
end

function write_run_result!(config, result)
    write_rows_csv(run_result_csv_path(config), [result])
    Serialization.serialize(run_result_jls_path(config), result)
    return result
end

function run_config(config)
    if run_complete(config)
        println("Run $(config.run_id) already completed; skipping.")
        return Serialization.deserialize(run_result_jls_path(config))
    end

    started_at = unix_milliseconds()
    try
        training = train_or_resume!(config)
        result = evaluate_trained_model(config, training.model)
        result = merge(result, (; started_at=started_at))
        return write_run_result!(config, result)
    catch error
        result = failed_result(config, error, catch_backtrace())
        result = merge(result, (; started_at=started_at))
        write_run_result!(config, result)
        rethrow()
    end
end

function main()
    args = parse_single_args(ARGS)

    if args["precompute"]
        cache = ensure_test_cache!(
            smoke=Bool(args["smoke"]),
            force=Bool(args["force-test-data"]),
        )
        println("Precomputed test data: ", cache.metadata)
        return nothing
    end

    config_file = String(args["config"])
    isempty(config_file) && error("missing --config PATH")
    config = read_config(config_file)
    println("Starting run $(config.run_id)")
    result = run_config(config)
    println("Finished run $(config.run_id) with status $(result.status)")
    return result
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
