#!/usr/bin/env julia

include(joinpath(@__DIR__, "common.jl"))

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
            """)
            exit(0)
        else
            error("unknown argument: $arg")
        end
        index += 1
    end

    return parsed
end

function write_checkpoint!(config, model, opt_state, epoch_rows, completed_epochs; snapshot=false)
    payload = (;
        version=CONFIG_VERSION,
        config=config,
        model=model,
        opt_state=opt_state,
        epoch_rows=copy(epoch_rows),
        completed_epochs=Int(completed_epochs),
        updated_at=unix_milliseconds(),
    )
    mkpath(run_dir(config))
    Serialization.serialize(checkpoint_path(config), payload)
    if snapshot
        mkpath(checkpoint_dir(config))
        path = joinpath(checkpoint_dir(config), "epoch_" * lpad(string(Int(completed_epochs)), 4, "0") * ".jls")
        Serialization.serialize(path, payload)
    end
    return payload
end

function load_checkpoint(config)
    path = checkpoint_path(config)
    isfile(path) || return nothing
    checkpoint = Serialization.deserialize(path)
    hasproperty(checkpoint, :config) &&
        checkpoint.config.run_id == config.run_id ||
        error("checkpoint at $(path) belongs to a different run")
    return checkpoint
end

function epoch_row(config, epoch, loss_value, display_loss, metadata)
    return (;
        run_id=config.run_id,
        group=config.group,
        candidate_name=config.candidate_name,
        replicate=Int(config.replicate),
        epoch=Int(epoch),
        mu=Float64(metadata.mu_in),
        mu_in=Float64(metadata.mu_in),
        mu_ref=Float64(metadata.mu_ref),
        rho_in=Float64(metadata.rho_in),
        rho_ref=Float64(metadata.rho_ref),
        iterations=Int(metadata.iterations),
        epoch_seconds=Float64(metadata.epoch_seconds),
        training_loss=Float64(loss_value),
        display_loss=Float64(display_loss),
        hidden_size=Int(config.hidden_size),
        depth=Int(config.depth),
        activation=String(config.activation),
        nr_scenarios=Int(config.nr_scenarios),
        loss_kind=String(config.loss_kind),
        schedule_kind=String(config.schedule_kind),
        rho=Float64(config.rho),
        batch_size=Int(config.batch_size),
        data_seed=Int(config.data_seed),
        model_seed=Int(config.model_seed),
        training_seed=Int(config.training_seed),
        reset_optimizer_each_epoch=Bool(config.reset_optimizer_each_epoch),
        knn_k=Int(config.knn_k),
    )
end

function train_dfl_one_epoch!(model, loss, data_set, config, opt_state, epoch, mu_in, mu_ref, rho_in, rho_ref)
    result = ContextualDFL.train!(
        model,
        loss,
        [Float64(mu_in)],
        [Float64(mu_ref)],
        data_set;
        rho_in_schedule=[Float64(rho_in)],
        rho_ref_schedule=[Float64(rho_ref)],
        opt=Flux.Adam(Float64(config.learning_rate)),
        opt_state=opt_state,
        epochs=1,
        batchsize=Int(config.batch_size),
        shuffle=false,
        display_iterations=false,
        verbose=false,
        display_plot=false,
        save_model=false,
        reset_optimizer_each_epoch=Bool(config.reset_optimizer_each_epoch),
        nr_scenarios=Int(config.nr_scenarios),
        display_smooth=false,
    )
    history_row = only(result.history)
    return (;
        model=result.model,
        opt_state=result.opt_state,
        loss=Float64(history_row.loss),
        display_loss=Float64(history_row.display_loss),
        iterations=Int(history_row.iterations),
        epoch_seconds=Float64(history_row.epoch_seconds),
    )
end

function train_or_resume!(config)
    mkpath(run_dir(config))
    checkpoint = load_checkpoint(config)
    objects = problem_objects()
    model = nothing
    opt_state = nothing
    epoch_rows = NamedTuple[]
    completed_epochs = 0

    if checkpoint === nothing
        model = build_model(config, objects.problem)
        write_rows_csv(epochs_csv_path(config), epoch_rows)
        write_checkpoint!(config, model, opt_state, epoch_rows, completed_epochs)
    else
        model = checkpoint.model
        opt_state = hasproperty(checkpoint, :opt_state) ? checkpoint.opt_state : nothing
        epoch_rows = collect(checkpoint.epoch_rows)
        completed_epochs = Int(checkpoint.completed_epochs)
        write_rows_csv(epochs_csv_path(config), epoch_rows)
    end

    mu_in, mu_ref = schedule_values_for_config(config)
    rho_in, rho_ref = rho_values_for_config(config)
    total_epochs = length(mu_in)
    length(mu_ref) == total_epochs || error("mu_ref length mismatch")
    length(rho_in) == total_epochs || error("rho_in length mismatch")
    length(rho_ref) == total_epochs || error("rho_ref length mismatch")
    completed_epochs >= total_epochs && return (; model=model, opt_state=opt_state, epoch_rows=epoch_rows)

    data_set_training = generate_training_dataset(config)
    dfl_loss = Symbol(config.loss_kind) == :dfl ? build_dfl_loss(objects, config) : nothing

    for epoch in (completed_epochs + 1):total_epochs
        epoch_started = time()
        trained = if Symbol(config.loss_kind) == :mse
            train_mse_one_epoch!(model, data_set_training, config, opt_state)
        else
            train_dfl_one_epoch!(
                model,
                dfl_loss,
                data_set_training,
                config,
                opt_state,
                epoch,
                mu_in[epoch],
                mu_ref[epoch],
                rho_in[epoch],
                rho_ref[epoch],
            )
        end
        model = trained.model
        opt_state = trained.opt_state
        metadata = (;
            mu_in=mu_in[epoch],
            mu_ref=mu_ref[epoch],
            rho_in=rho_in[epoch],
            rho_ref=rho_ref[epoch],
            iterations=trained.iterations,
            epoch_seconds=haskey(Dict(pairs(trained)), :epoch_seconds) ?
                trained.epoch_seconds :
                time() - epoch_started,
        )
        push!(epoch_rows, epoch_row(config, epoch, trained.loss, trained.display_loss, metadata))
        write_rows_csv(epochs_csv_path(config), epoch_rows)

        snapshot = epoch % Int(config.checkpoint_interval) == 0 || epoch == total_epochs
        write_checkpoint!(config, model, opt_state, epoch_rows, epoch; snapshot=snapshot)
        println("$(config.run_id): epoch $(epoch)/$(total_epochs), loss=$(trained.loss)")
    end

    return (; model=model, opt_state=opt_state, epoch_rows=epoch_rows)
end

function run_result(config, training, metrics, started_at)
    final_loss = final_training_loss_from_epochs(training.epoch_rows)
    return merge(
        (;
            status="ok",
            run_id=config.run_id,
            group=config.group,
            candidate_name=config.candidate_name,
            replicate=Int(config.replicate),
            data_seed=Int(config.data_seed),
            model_seed=Int(config.model_seed),
            training_seed=Int(config.training_seed),
            test_seed=Int(config.test_seed),
            hidden_size=Int(config.hidden_size),
            depth=Int(config.depth),
            activation=String(config.activation),
            nr_scenarios=Int(config.nr_scenarios),
            loss_kind=String(config.loss_kind),
            schedule_kind=String(config.schedule_kind),
            rho=Float64(config.rho),
            batch_size=Int(config.batch_size),
            total_epochs=sum(Int.(config.stage_epochs)),
            learning_rate=Float64(config.learning_rate),
            reset_optimizer_each_epoch=Bool(config.reset_optimizer_each_epoch),
            knn_k=Int(config.knn_k),
            final_training_loss=final_loss,
            error="",
            started_at=started_at,
            finished_at=unix_milliseconds(),
        ),
        metrics,
    )
end

function failed_result(config, error, backtrace, started_at)
    mkpath(run_dir(config))
    text = sprint(showerror, error, backtrace)
    write(error_path(config), text)
    return (;
        status="failed",
        run_id=config.run_id,
        group=config.group,
        candidate_name=config.candidate_name,
        replicate=Int(config.replicate),
        data_seed=Int(config.data_seed),
        model_seed=Int(config.model_seed),
        training_seed=Int(config.training_seed),
        test_seed=Int(config.test_seed),
        hidden_size=Int(config.hidden_size),
        depth=Int(config.depth),
        activation=String(config.activation),
        nr_scenarios=Int(config.nr_scenarios),
        loss_kind=String(config.loss_kind),
        schedule_kind=String(config.schedule_kind),
        rho=Float64(config.rho),
        batch_size=Int(config.batch_size),
        total_epochs=sum(Int.(config.stage_epochs)),
        learning_rate=Float64(config.learning_rate),
        reset_optimizer_each_epoch=Bool(config.reset_optimizer_each_epoch),
        knn_k=Int(config.knn_k),
        final_training_loss=Inf,
        mean_test_relative_regret=Inf,
        optimality_gap_percent=Inf,
        mean_test_regret=Inf,
        mean_test_policy_value=NaN,
        mean_test_optimal_value=NaN,
        test_sample_count=0,
        test_policy_eval_seconds=NaN,
        error=text,
        started_at=started_at,
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
        metrics = evaluate_model(config, training.model)
        result = run_result(config, training, metrics, started_at)
        return write_run_result!(config, result)
    catch error
        result = failed_result(config, error, catch_backtrace(), started_at)
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
        println("Precomputed test cache: ", cache.metadata)
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
