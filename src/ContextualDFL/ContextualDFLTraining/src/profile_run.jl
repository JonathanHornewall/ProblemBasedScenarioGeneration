using FileIO
using FlameGraphs
using Profile
using ProfileSVG

function strict_contextualdfl_training(objects, config)
    assert_remote_training_worker!(config)
    return ContextualDFL.train!(
        objects.loss,
        objects.program,
        fill(config.mu, Int(config.epochs)),
        Int(config_value(config, :nr_scenarios, 1)),
        objects.scenario_generator.neural_net,
        objects.data.train;
        learning_rate=config.learning_rate,
        optimizer_type=Flux.Adam,
        epochs=config.epochs,
        batchsize=config.batch_size,
        shuffle=true,
        rng=MersenneTwister(config.seed + 10_000),
        display_plot=false,
        verbose=false,
    )
end

function standard_profile_config(; overrides...)
    settings = merge(
        DEFAULT_RUN_SETTINGS,
        (;
            epochs=100,
            n_samples=2000,
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

function profile_standard_training(config::NamedTuple)
    cfg = normalize_config(config)
    assert_remote_training_worker!(cfg)
    started_at = utc_timestamp()
    elapsed_seconds = 0.0
    remote_output_dir = ""

    try
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
                warmup_cfg = merge(cfg, (; epochs=warmup_epochs, run_id=string(cfg.run_id, "_warmup")))
                warmup_objects = resource_allocation_training_objects(warmup_cfg)
                strict_contextualdfl_training(warmup_objects, warmup_cfg)
                GC.gc()
            end

            objects = resource_allocation_training_objects(cfg)
            model = objects.scenario_generator.neural_net
            initial_train_mse = split_mse(model, objects.data.train)

            Profile.clear()
            profile_result = Profile.@profile strict_contextualdfl_training(objects, cfg)

            ProfileSVG.save(svg_path)
            FileIO.save(jlprof_path, Profile.retrieve()...)
            svg_bytes = read(svg_path)
            jlprof_bytes = read(jlprof_path)

            trained_model = extract_model(profile_result, objects.scenario_generator)
            final_train_mse = split_mse(trained_model, objects.data.train)
            metrics = merge(
                evaluate_model_on_splits(trained_model, objects.data, cfg),
                (;
                    initial_train_mse=initial_train_mse,
                    final_train_mse=final_train_mse,
                    loss_delta=initial_train_mse - final_train_mse,
                    loss_decreased=final_train_mse < initial_train_mse,
                    training_backend="ContextualDFL.train",
                    remote_output_dir=remote_output_dir,
                    thread_count=Threads.nthreads(),
                ),
            )
            history = extract_epoch_history(profile_result)
        end

        final_train_mse < initial_train_mse ||
            error("profiled training did not reduce train MSE: initial=$(initial_train_mse), final=$(final_train_mse)")

        return (;
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
            profile_svg_bytes=UInt8[],
            profile_jlprof_bytes=UInt8[],
            error=exception_text(error, catch_backtrace()),
            started_at=started_at,
            finished_at=utc_timestamp(),
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
