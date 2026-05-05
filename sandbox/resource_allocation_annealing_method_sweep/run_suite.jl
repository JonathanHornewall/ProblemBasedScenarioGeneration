#!/usr/bin/env julia

import Pkg
using Distributed

const DRIVER_SUITE_DIR = @__DIR__
const DRIVER_REPO_ROOT = normpath(joinpath(DRIVER_SUITE_DIR, "..", ".."))
const DRIVER_TRAINING_PROJECT_DIR =
    joinpath(DRIVER_REPO_ROOT, "src", "ContextualDFL", "ContextualDFLTraining")
const DRIVER_COMMON_PATH = joinpath(DRIVER_SUITE_DIR, "suite_common.jl")

@everywhere begin
    import Pkg
    Pkg.activate($DRIVER_TRAINING_PROJECT_DIR; io=devnull)
    include($DRIVER_COMMON_PATH)
end

function parse_args(args)
    parsed = Dict{String,Any}(
        "smoke" => false,
        "dry-run" => false,
        "force-test-data" => false,
        "jobs" => nworkers(),
    )

    index = 1
    while index <= length(args)
        arg = args[index]
        if arg == "--smoke"
            parsed["smoke"] = true
        elseif arg == "--dry-run"
            parsed["dry-run"] = true
        elseif arg == "--force-test-data"
            parsed["force-test-data"] = true
        elseif arg == "--jobs"
            index += 1
            index <= length(args) || error("--jobs requires an integer")
            parsed["jobs"] = parse(Int, args[index])
        elseif arg in ("-h", "--help")
            println("""
            Usage:
              julia --machine-file=/home/rwl/gcp-machines.txt --project=... run_suite.jl [--jobs N]
              julia --machine-file=/home/rwl/gcp-machines.txt --project=... run_suite.jl --smoke

            Runs the resource-allocation annealing hyperparameter suite:
              0. precompute 100 test contexts once
              1. data amount: 100, 500, 1000
              2. depth: 3, 4, 5, 10, 20 hidden layers
              3. width: 32, 64, 128, 256, 512 hidden units
              4. schedule shape
              5. piecewise-linear schedule parameters

            CSV artifacts stay under:
              $(DRIVER_SUITE_DIR)
            """)
            exit(0)
        else
            error("unknown argument: $(arg)")
        end
        index += 1
    end

    parsed["jobs"] = max(1, min(Int(parsed["jobs"]), nworkers()))
    return parsed
end

function ensure_machine_workers!()
    nworkers() > 0 || error(
        "No Julia workers are available. Run this script with " *
        "--machine-file=/home/rwl/gcp-machines.txt.",
    )
    return workers()
end

function start_csv_logger(; smoke=false)
    paths = result_paths(smoke=smoke)
    mkpath(paths.dir)
    channel = RemoteChannel(() -> Channel{Any}(100_000), 1)

    task = @async begin
        while true
            message = take!(channel)
            if message === :stop ||
               (message isa NamedTuple && :kind in keys(message) && message.kind == :stop)
                break
            end

            kind = message.kind
            row = message.row
            if kind == :attempt
                append_csv_row(paths.attempts, ATTEMPT_HEADERS, row)
            elseif kind == :epoch
                append_csv_row(paths.epochs, EPOCH_HEADERS, row)
            elseif kind == :run
                append_csv_row(paths.runs, RUN_HEADERS, row)
            elseif kind == :test_sample
                append_csv_row(paths.test_samples, TEST_SAMPLE_HEADERS, row)
            else
                @warn "Ignoring unknown logger message kind" kind
            end
        end
    end

    return (; channel=channel, task=task)
end

function stop_csv_logger!(logger)
    put!(logger.channel, (; kind=:stop, row=(;)))
    wait(logger.task)
    return nothing
end

function ensure_test_cache!(; smoke=false, force=false)
    if !force && test_cache_exists(smoke=smoke)
        println("Loading precomputed test data from CSV.")
        return load_test_cache_from_csv(smoke=smoke)
    end

    worker = first(workers())
    println("Precomputing shared test data and optima on worker $(worker).")
    cache = remotecall_fetch(() -> precompute_test_cache(smoke=smoke), worker)
    write_test_cache_csv!(cache; smoke=smoke)
    println("Wrote test cache CSV files to $(test_cache_paths(smoke=smoke).dir).")
    return (; dataset=cache.dataset, optimal_results=cache.optimal_results, metadata=cache.metadata)
end

function initialize_worker_caches!(worker_ids, cache)
    println("Sending precomputed test cache to $(length(worker_ids)) worker(s).")
    @sync for worker in worker_ids
        @async begin
            info = remotecall_fetch(set_worker_test_cache!, worker, cache)
            println(
                "Worker $(info.worker_id) on $(info.hostname) has " *
                "$(info.test_contexts) test contexts.",
            )
        end
    end
    return nothing
end

function write_phase_config_csv!(phase, configs; smoke=false)
    path = phase_path(phase, "configs.csv"; smoke=smoke)
    write_csv_file(path, CONFIG_HEADERS, [config_row(config) for config in configs])
    return path
end

function pending_configs(configs; smoke=false)
    completed = completed_run_ids(smoke=smoke)
    return [config for config in configs if !(config.run_id in completed)]
end

function run_configs!(configs, logger; smoke=false, jobs=nworkers())
    pending = pending_configs(configs; smoke=smoke)
    if isempty(pending)
        println("All runs for this phase are already complete.")
        return NamedTuple[]
    end

    worker_ids = workers()[1:min(Int(jobs), nworkers())]
    pool = CachingPool(worker_ids)
    println("Running $(length(pending)) pending config(s) on $(length(worker_ids)) worker(s).")
    flush(stdout)
    results = pmap(config -> run_sweep_config(config, logger.channel), pool, pending)
    flush(stdout)
    return results
end

function write_phase_outputs!(phase, configs, decision_rows; smoke=false)
    summary_rows = summarize_phase(configs; smoke=smoke)
    write_csv_file(phase_path(phase, "summary.csv"; smoke=smoke), SUMMARY_HEADERS, summary_rows)

    decision = choose_candidate(summary_rows)
    selected_config = config_for_candidate(configs, decision.candidate_name)
    row = decision_row(phase, decision, selected_config)
    push!(decision_rows, row)
    write_csv_file(phase_path(phase, "decision.csv"; smoke=smoke), DECISION_HEADERS, [row])
    write_csv_file(result_paths(smoke=smoke).decisions, DECISION_HEADERS, decision_rows)

    return (; decision=decision, selected_config=selected_config, row=row)
end

function print_dry_run(; smoke=false)
    selection = default_selection(smoke=smoke)
    for phase in PHASES
        configs = phase_configs(phase, selection; smoke=smoke)
        println("Phase $(phase): $(length(configs)) run(s)")
        println("Candidates: ", join(sort(unique(config.candidate_name for config in configs)), ", "))
        write_phase_config_csv!(phase, configs; smoke=smoke)
        summary = summarize_phase(configs; smoke=smoke)
        complete = [row for row in summary if row.ok_count == row.run_count && row.run_count > 0]
        if !isempty(complete)
            decision = choose_candidate(summary)
            selected_config = config_for_candidate(configs, decision.candidate_name)
            selection = update_selection(selection, phase, decision, selected_config)
        end
    end
    return nothing
end

function main()
    args = parse_args(ARGS)
    ensure_machine_workers!()

    smoke = Bool(args["smoke"])
    if Bool(args["dry-run"])
        print_dry_run(smoke=smoke)
        return nothing
    end

    cache = ensure_test_cache!(smoke=smoke, force=Bool(args["force-test-data"]))
    active_workers = workers()[1:min(Int(args["jobs"]), nworkers())]
    initialize_worker_caches!(active_workers, cache)

    logger = start_csv_logger(smoke=smoke)
    selection = default_selection(smoke=smoke)
    decision_rows = NamedTuple[]

    try
        for phase in PHASES
            println()
            println("Starting phase $(phase).")
            configs = phase_configs(phase, selection; smoke=smoke)
            write_phase_config_csv!(phase, configs; smoke=smoke)
            run_configs!(configs, logger; smoke=smoke, jobs=Int(args["jobs"]))
            output = write_phase_outputs!(phase, configs, decision_rows; smoke=smoke)
            selection = update_selection(
                selection,
                phase,
                output.decision,
                output.selected_config,
            )
            println(
                "Phase $(phase) selected $(output.decision.candidate_name) " *
                "with mean average_test_loss $(output.decision.mean_average_test_loss).",
            )
            flush(stdout)
        end

        final_row = (;
            version=SUITE_VERSION,
            smoke=smoke,
            training_contexts=selection.training_contexts,
            epochs=selection.epochs,
            final_epochs=selection.final_epochs,
            depth=selection.depth,
            hidden_size=selection.hidden_size,
            activation=String(selection.activation),
            schedule_kind=String(selection.schedule_kind),
            starting_mu=selection.starting_mu,
            ending_mu=selection.ending_mu,
            piece_length=selection.piece_length,
            nr_pieces=selection.nr_pieces,
            starting_phase_length=selection.starting_phase_length,
            fine_tuning_phase_length=selection.fine_tuning_phase_length,
            finished_at=unix_milliseconds(),
        )
        write_csv_file(
            result_paths(smoke=smoke).final_selection,
            (
                :version,
                :smoke,
                :training_contexts,
                :epochs,
                :final_epochs,
                :depth,
                :hidden_size,
                :activation,
                :schedule_kind,
                :starting_mu,
                :ending_mu,
                :piece_length,
                :nr_pieces,
                :starting_phase_length,
                :fine_tuning_phase_length,
                :finished_at,
            ),
            [final_row],
        )
        println("Suite complete. Final selection written to $(result_paths(smoke=smoke).final_selection).")
    finally
        stop_csv_logger!(logger)
    end

    return selection
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
