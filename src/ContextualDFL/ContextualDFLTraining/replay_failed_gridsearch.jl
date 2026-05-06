include(joinpath(@__DIR__, "gridsearch.jl"))

using MLFlowClient
using Sockets

function parse_replay_args(args)
    values = Dict{String,String}()
    index = 1
    while index <= length(args)
        key = args[index]
        startswith(key, "--") || error("Unexpected argument: $key")
        index == length(args) && error("Missing value for $key")
        values[key[3:end]] = args[index + 1]
        index += 2
    end
    return values
end

function run_status(run)
    return replace(string(run.info.status), "MLFlowClient.RunStatus." => "")
end

function tag_value(run, key; default="")
    for tag in run.data.tags
        tag.key == key && return String(tag.value)
    end
    return default
end

function param_value(run, key; default="")
    for param in run.data.params
        param.key == key && return String(param.value)
    end
    return default
end

function metric_value(run, key)
    for metric in run.data.metrics
        metric.key == key && return metric.value
    end
    return nothing
end

function int_run_value(run, key)
    value = tag_value(run, key)
    isempty(value) && (value = param_value(run, "config_" * key))
    isempty(value) && error("Run $(run.info.run_name) is missing $key")
    return parse(Int, value)
end

function run_seed_sequence(run)
    value = tag_value(run, "repeat_training_data_seed_sequence")
    isempty(value) && (value = param_value(run, "config_repeat_training_data_seed_sequence"))
    isempty(value) && error("Run $(run.info.run_name) is missing repeat_training_data_seed_sequence")
    return parse_repeat_training_data_seed_sequence(value)
end

function cleanly_finished(run; expected_test_count=30.0, expected_batches=1.0)
    run_status(run) == "FINISHED" || return false
    metric_value(run, "test_sample_count") == expected_test_count || return false
    metric_value(run, "test_evaluation_batches") == expected_batches || return false
    metric_value(run, "test_policy_value_count") == expected_test_count || return false
    metric_value(run, "test_optimal_value_count") == expected_test_count || return false
    return true
end

function dict_from_pairs(pairs_iter)
    result = Dict{String,String}()
    for (key, value) in pairs(pairs_iter)
        result[string(key)] = string(value)
    end
    return result
end

function replay_manifest_items(path)
    items = NamedTuple[]
    open(path, "r") do io
        for line in eachline(io)
            text = strip(line)
            isempty(text) && continue
            startswith(text, "candidate_index\t") && continue
            parts = split(text, '\t')
            length(parts) >= 5 ||
                error("Replay manifest line must have at least 5 tab-separated fields: $text")
            push!(
                items,
                (;
                    candidate_index=parse(Int, parts[1]),
                    repeat_index=parse(Int, parts[2]),
                    replay_original_status=String(parts[3]),
                    replay_of_run_uuid=String(parts[4]),
                    replay_of_run_name=String(parts[5]),
                ),
            )
        end
    end
    return items
end

function build_replay_configs(
    source_grid_id,
    experiment_selector,
    grid_config_path,
    repeat_seed_sequence,
    replay_items,
)
    experiment = load_experiment(experiment_selector)
    grid_spec = load_grid_config(grid_config_path)
    base_mlflow_settings = grid_mlflow_settings(experiment)
    replay_experiment_id = strip(get(ENV, "MLFLOW_REPLAY_EXPERIMENT_ID", ""))
    mlflow_settings = isempty(replay_experiment_id) ?
        ensure_mlflow_grid_experiment(base_mlflow_settings) :
        merge(base_mlflow_settings, (; experiment_id=replay_experiment_id))

    base_configs = selected_grid(experiment, grid_spec)
    timestamp = "replay_" * replace(source_grid_id, "gridsearch_" => "") * "_" * result_timestamp()
    replay_grid_id = gridsearch_id(timestamp)
    config_parents = annotate_grid_config_parents(
        base_configs,
        timestamp,
        mlflow_settings,
        "",
        Sockets.gethostname();
        repeat_training_data_seeds=repeat_seed_sequence,
    )

    configs = NamedTuple[]
    seen = Set{Tuple{Int,Int}}()
    for item in sort(replay_items; by=item -> (item.candidate_index, item.repeat_index))
        key = (item.candidate_index, item.repeat_index)
        key in seen && continue
        push!(seen, key)

        item.candidate_index in eachindex(config_parents) ||
            error("candidate_index $(item.candidate_index) is outside replay grid")
        cfg = annotate_repeat_config(
            config_parents[item.candidate_index],
            item.repeat_index,
            mlflow_settings,
            "";
            repeat_training_data_seeds=repeat_seed_sequence,
        )

        tags = dict_from_pairs(cfg.mlflow_tags)
        tags["replay_of_gridsearch_id"] = source_grid_id
        tags["replay_of_run_name"] = item.replay_of_run_name
        tags["replay_of_run_uuid"] = item.replay_of_run_uuid
        tags["replay_original_status"] = item.replay_original_status

        push!(
            configs,
            merge(
                cfg,
                (;
                    mlflow_tags=tags,
                    replay_of_gridsearch_id=source_grid_id,
                    replay_of_run_name=item.replay_of_run_name,
                    replay_of_run_uuid=item.replay_of_run_uuid,
                    replay_original_status=item.replay_original_status,
                ),
            ),
        )
    end

    return replay_grid_id, timestamp, configs
end

function replay_configs(source_grid_id, experiment_selector, grid_config_path)
    experiment = load_experiment(experiment_selector)
    grid_spec = load_grid_config(grid_config_path)
    mlflow_settings = ensure_mlflow_grid_experiment(grid_mlflow_settings(experiment))
    mlf = mlflow_client(mlflow_settings)

    runs, _ = with_mlflow_retry("search source grid runs") do
        MLFlowClient.searchruns(
            mlf;
            experiment_ids=[string(mlflow_settings.experiment_id)],
            filter="tags.gridsearch_id = \"$(mlflow_filter_escape(source_grid_id))\"",
            max_results=1000,
        )
    end

    repeat_runs = filter(run -> tag_value(run, "gridsearch_role") == "repeat", runs)
    isempty(repeat_runs) && error("No repeat runs found for $source_grid_id")

    repeat_seed_sequence = run_seed_sequence(first(repeat_runs))

    items = NamedTuple[]
    seen = Set{Tuple{Int,Int}}()
    for run in sort(repeat_runs; by=run -> string(run.info.run_name))
        cleanly_finished(run) && continue
        candidate_index = int_run_value(run, "candidate_index")
        repeat_index = int_run_value(run, "repeat_index")
        key = (candidate_index, repeat_index)
        key in seen && continue
        push!(seen, key)

        push!(
            items,
            (;
                candidate_index=candidate_index,
                repeat_index=repeat_index,
                replay_of_run_name=string(run.info.run_name),
                replay_of_run_uuid=string(run.info.run_id),
                replay_original_status=run_status(run),
            ),
        )
    end

    return build_replay_configs(
        source_grid_id,
        experiment_selector,
        grid_config_path,
        repeat_seed_sequence,
        items,
    )
end

function main(args=ARGS)
    parsed = parse_replay_args(args)
    source_grid_id = get(parsed, "source-grid", "")
    isempty(source_grid_id) && error("--source-grid is required")
    experiment_selector = get(parsed, "experiment", "resource_allocation/experiment_1_tiny")
    grid_config_path = get(
        parsed,
        "grid-config",
        joinpath(
            @__DIR__,
            "src/experiments/resource_allocation/experiment_1_tiny/grid_configs/small_mus_eval_1batch.yaml",
        ),
    )

    replay_grid_id, timestamp, configs = if haskey(parsed, "manifest-tsv")
        seed_sequence = get(parsed, "seed-sequence", "")
        isempty(seed_sequence) &&
            error("--seed-sequence is required when --manifest-tsv is used")
        build_replay_configs(
            source_grid_id,
            experiment_selector,
            grid_config_path,
            parse_repeat_training_data_seed_sequence(seed_sequence),
            replay_manifest_items(parsed["manifest-tsv"]),
        )
    else
        replay_configs(source_grid_id, experiment_selector, grid_config_path)
    end

    println("Replay grid search id: $replay_grid_id")
    println("Source grid search id: $source_grid_id")
    println("Replay configuration count: $(length(configs))")
    for config in configs
        println(
            "Replay ",
            config.replay_of_run_name,
            " -> ",
            config.run_id,
            " seed=",
            config.repeat_training_data_seed,
        )
    end

    isempty(configs) && return nothing

    ensure_clean_worker_start!()
    sync_code!()
    remote_worker_ids = add_remote_workers!()
    load_worker_stdlibs!()
    worker_hosts = assert_remote_only_workers!(remote_worker_ids)
    load_training_project_on_workers!(remote_worker_ids, worker_hosts)
    define_remote_eval!()

    println("Running replay on $(length(remote_worker_ids)) remote worker(s)")
    results = run_grid_on_remote_workers(remote_worker_ids, configs, worker_hosts)
    output_dir = write_grid_results(
        results;
        configs=configs,
        output_root=joinpath(@__DIR__, "results"),
        timestamp=timestamp,
    )
    println("Wrote replay CSV results to $output_dir")

    failed_count = count(result -> result.status != "ok", results)
    failed_count > 0 && println("Recorded $failed_count failed replay configuration(s).")
    failed_count == 0 || exit(2)
    return output_dir
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
