#!/usr/bin/env julia

const DFL_SOURCE_DIR = normpath(
    get(ENV, "CDFL_SOURCE_DIR", "/home/rwl/ProblemBasedScenarioGeneration/src/ContextualDFL"),
)
const DFL_SANDBOX_DIR = @__DIR__
const DFL_BASELINE_SCRIPT = joinpath(
    DFL_SOURCE_DIR,
    "ContextualDFLExperiments",
    "experiments",
    "baseline_benchmarks",
    "run_baselines.jl",
)

isfile(DFL_BASELINE_SCRIPT) || error("baseline helper script not found: $(DFL_BASELINE_SCRIPT)")
include(DFL_BASELINE_SCRIPT)

using Dates
using Distributed
using Printf
using Random
using Serialization
using SHA
using Sockets
using Statistics

const DFL_SCRIPT_PATH = abspath(@__FILE__)
const DFL_PROBLEMS = (
    "transshipment_h",
    "transshipment_h_and_q",
    "random_yield",
    "resource_allocation",
)
const DFL_OUTPUTS = ("q", "h")
const DFL_SEEDS = (20260505, 20260506, 20260507)
const DFL_DEFAULT_ARTIFACT_DIR = joinpath(
    DFL_SOURCE_DIR,
    "ContextualDFLExperiments",
    "experiments",
    "baseline_benchmarks",
    "artifacts",
    "tiny_30ctx_5x100_seed20260505",
)
const DFL_DEFAULT_OUTPUT_DIR = joinpath(DFL_SANDBOX_DIR, "results_final_eval_once")
const DFL_VERSION = "dflscenloss_tiny_qh_v3_latest_checkpoint"
const DFL_FLUX = ContextualDFL.Flux

function main(args=ARGS)
    options = dfl_parse_options(args)
    mkpath(options.output_dir)
    dfl_write_manifest(joinpath(options.output_dir, "manifest.txt"), options, args)
    dfl_configure_workers!(options)

    jobs = dfl_jobs(options)
    println("DflScenLoss tiny q/h jobs: $(length(jobs))")
    println("Problems: $(join(options.problems, ", "))")
    println("Outputs: $(join(options.outputs, ", "))")
    println("Seeds: $(join(options.seeds, ", "))")
    println("Max epochs: $(options.max_epochs)")
    println("Workers: $(workers())")
    println("Artifact dir: $(options.artifact_dir)")
    println("Output dir: $(options.output_dir)")

    rows = NamedTuple[]
    for batch in Iterators.partition(jobs, max(1, options.job_batch_size))
        batch_rows = dfl_pmap_or_map(run_dfl_job, collect(batch))
        append!(rows, batch_rows)
        dfl_write_outputs(options.output_dir, rows)
        for row in batch_rows
            dfl_print_final_row(row)
        end
    end

    timestamp = Dates.format(now(), dateformat"yyyymmdd_HHMMSS")
    dfl_write_namedtuple_csv(joinpath(options.output_dir, "individual_results_$(timestamp).csv"), rows)
    dfl_write_outputs(options.output_dir, rows)
    any(row -> hasproperty(row, :status) && row.status != "ok", rows) && exit(1)
    return rows
end

function dfl_parse_options(args)
    artifact_dir = DFL_DEFAULT_ARTIFACT_DIR
    output_dir = DFL_DEFAULT_OUTPUT_DIR
    local_workers = 0
    job_batch_size = 0
    problems = collect(DFL_PROBLEMS)
    outputs = collect(DFL_OUTPUTS)
    seeds = collect(DFL_SEEDS)
    max_epochs = 0
    hidden_dim = 128
    depth = 3
    learning_rate = 1e-3
    batchsize = 1

    index = 1
    while index <= length(args)
        arg = args[index]
        if arg == "--artifact-dir"
            index += 1
            artifact_dir = abspath(args[index])
        elseif startswith(arg, "--artifact-dir=")
            artifact_dir = abspath(split(arg, "=", limit=2)[2])
        elseif arg == "--output-dir"
            index += 1
            output_dir = abspath(args[index])
        elseif startswith(arg, "--output-dir=")
            output_dir = abspath(split(arg, "=", limit=2)[2])
        elseif arg == "--local-workers"
            index += 1
            local_workers = parse(Int, args[index])
        elseif startswith(arg, "--local-workers=")
            local_workers = parse(Int, split(arg, "=", limit=2)[2])
        elseif arg == "--job-batch-size"
            index += 1
            job_batch_size = parse(Int, args[index])
        elseif startswith(arg, "--job-batch-size=")
            job_batch_size = parse(Int, split(arg, "=", limit=2)[2])
        elseif arg == "--problems"
            index += 1
            problems = dfl_selected_names(DFL_PROBLEMS, dfl_split_names(args[index]), "problem")
        elseif startswith(arg, "--problems=")
            problems = dfl_selected_names(
                DFL_PROBLEMS,
                dfl_split_names(split(arg, "=", limit=2)[2]),
                "problem",
            )
        elseif arg == "--outputs"
            index += 1
            outputs = dfl_selected_names(DFL_OUTPUTS, dfl_split_names(args[index]), "output")
        elseif startswith(arg, "--outputs=")
            outputs = dfl_selected_names(
                DFL_OUTPUTS,
                dfl_split_names(split(arg, "=", limit=2)[2]),
                "output",
            )
        elseif arg == "--seeds"
            index += 1
            seeds = dfl_parse_ints(args[index])
        elseif startswith(arg, "--seeds=")
            seeds = dfl_parse_ints(split(arg, "=", limit=2)[2])
        elseif arg == "--max-epochs"
            index += 1
            max_epochs = parse(Int, args[index])
        elseif startswith(arg, "--max-epochs=")
            max_epochs = parse(Int, split(arg, "=", limit=2)[2])
        elseif arg == "--hidden-dim"
            index += 1
            hidden_dim = parse(Int, args[index])
        elseif startswith(arg, "--hidden-dim=")
            hidden_dim = parse(Int, split(arg, "=", limit=2)[2])
        elseif arg == "--depth"
            index += 1
            depth = parse(Int, args[index])
        elseif startswith(arg, "--depth=")
            depth = parse(Int, split(arg, "=", limit=2)[2])
        elseif arg == "--learning-rate"
            index += 1
            learning_rate = parse(Float64, args[index])
        elseif startswith(arg, "--learning-rate=")
            learning_rate = parse(Float64, split(arg, "=", limit=2)[2])
        elseif arg == "--batchsize"
            index += 1
            batchsize = parse(Int, args[index])
        elseif startswith(arg, "--batchsize=")
            batchsize = parse(Int, split(arg, "=", limit=2)[2])
        else
            throw(ArgumentError("unknown argument: $(arg)"))
        end
        index += 1
    end

    local_workers >= 0 || throw(ArgumentError("local_workers must be non-negative."))
    hidden_dim > 0 || throw(ArgumentError("hidden_dim must be positive."))
    depth > 0 || throw(ArgumentError("depth must be positive."))
    learning_rate > 0 || throw(ArgumentError("learning_rate must be positive."))
    batchsize > 0 || throw(ArgumentError("batchsize must be positive."))
    if job_batch_size <= 0
        job_batch_size = local_workers > 0 ? local_workers : 1
    end
    isempty(seeds) && throw(ArgumentError("seeds must not be empty."))

    full_schedule = dfl_training_schedule()
    if max_epochs <= 0
        max_epochs = length(full_schedule)
    end
    max_epochs > 0 || throw(ArgumentError("max_epochs must be positive."))
    max_epochs <= length(full_schedule) ||
        throw(ArgumentError("max_epochs=$(max_epochs) exceeds schedule length $(length(full_schedule))."))

    return (;
        artifact_dir=abspath(artifact_dir),
        output_dir=abspath(output_dir),
        local_workers=local_workers,
        job_batch_size=job_batch_size,
        problems=problems,
        outputs=outputs,
        seeds=Int.(seeds),
        max_epochs=Int(max_epochs),
        hidden_dim=Int(hidden_dim),
        depth=Int(depth),
        learning_rate=Float64(learning_rate),
        batchsize=Int(batchsize),
    )
end

dfl_split_names(value) = [strip(item) for item in split(value, ",") if !isempty(strip(item))]

function dfl_selected_names(valid_names, requested_names, label)
    isempty(requested_names) && return collect(valid_names)
    unknown = setdiff(requested_names, valid_names)
    isempty(unknown) ||
        throw(ArgumentError("unknown $(label) name(s): $(join(unknown, ", "))"))
    return [name for name in valid_names if name in requested_names]
end

function dfl_parse_ints(value)
    parsed = Int[]
    for raw in split(value, ",")
        text = strip(raw)
        isempty(text) || push!(parsed, parse(Int, text))
    end
    isempty(parsed) && throw(ArgumentError("integer list must not be empty."))
    return unique!(parsed)
end

function dfl_configure_workers!(options)
    options.local_workers == 0 && return nothing
    addprocs(
        options.local_workers;
        exeflags="--project=$(PROJECT_DIR)",
        dir=PROJECT_DIR,
    )
    for worker in workers()
        worker == 1 && continue
        remotecall_wait(worker, DFL_SCRIPT_PATH) do script
            include(script)
            return nothing
        end
    end
    return nothing
end

dfl_pmap_or_map(f, jobs) = nworkers() == 0 ? map(f, jobs) : pmap(f, jobs)

function dfl_jobs(options)
    jobs = NamedTuple[]
    for problem in options.problems
        for learned_output in options.outputs
            for (replica_index, seed) in enumerate(options.seeds)
                push!(
                    jobs,
                    (;
                        problem=problem,
                        learned_output=learned_output,
                        seed=Int(seed),
                        replica_index=Int(replica_index),
                        options=options,
                    ),
                )
            end
        end
    end
    return jobs
end

function dfl_training_schedule()
    anneal_values = exp.(range(log(1e-1), log(1e-4); length=11))
    epoch_counts = vcat(20, fill(10, length(anneal_values) - 1), 10)
    rows = NamedTuple[]
    global_epoch = 0
    for (stage_index, mu_value) in enumerate(anneal_values)
        for stage_epoch in 1:epoch_counts[stage_index]
            global_epoch += 1
            push!(
                rows,
                (;
                    epoch=global_epoch,
                    stage=stage_index,
                    stage_epoch=stage_epoch,
                    stage_epochs=epoch_counts[stage_index],
                    mu_in=Float64(mu_value),
                    mu_ref=Float64(mu_value),
                    rho_in=0.0,
                    rho_ref=0.0,
                ),
            )
        end
    end
    final_stage = length(anneal_values) + 1
    final_mu = Float64(last(anneal_values))
    for stage_epoch in 1:last(epoch_counts)
        global_epoch += 1
        push!(
            rows,
            (;
                epoch=global_epoch,
                stage=final_stage,
                stage_epoch=stage_epoch,
                stage_epochs=last(epoch_counts),
                mu_in=final_mu,
                mu_ref=0.0,
                rho_in=0.0,
                rho_ref=0.0,
            ),
        )
    end
    return rows
end

function run_dfl_job(job)
    worker = worker_metadata()
    out_dir = dfl_run_dir(job)
    result_path = joinpath(out_dir, "final_result.jls")
    if isfile(result_path)
        try
            row = Serialization.deserialize(result_path)
            hasproperty(row, :status) && row.status == "ok" && return row
        catch error
            @warn "ignoring unreadable final_result; job will resume" result_path error
        end
    end

    mkpath(out_dir)
    started = time()
    try
        row = run_dfl_job_inner(job, out_dir, worker, started)
        Serialization.serialize(result_path, row)
        return row
    catch error
        row = merge(
            worker,
            (;
                status="error",
                problem=job.problem,
                learned_output=job.learned_output,
                seed=job.seed,
                replica_index=job.replica_index,
                total_seconds=time() - started,
                output_dir=out_dir,
                error=sprint(showerror, error, catch_backtrace()),
            ),
        )
        Serialization.serialize(result_path, row)
        return row
    finally
        GC.gc()
    end
end

function run_dfl_job_inner(job, out_dir, worker, started)
    options = job.options
    schedule = dfl_training_schedule()[1:options.max_epochs]
    state_path = joinpath(out_dir, "state.jls")
    epoch_history_path = joinpath(out_dir, "epoch_history.csv")
    test_metrics_path = joinpath(out_dir, "test_metrics_final.csv")
    final_model_path = joinpath(out_dir, "model_final.jls")
    checkpoint_model_path = joinpath(out_dir, "model_checkpoint_latest.jls")
    final_per_sample_path = joinpath(out_dir, "per_sample_final.csv")

    problem = make_problem(job.problem)
    reference_decoder = make_decoder(job.problem, problem)
    vector_spec = dfl_vector_decoder(job.problem, job.learned_output, problem)
    vector_decoder = vector_spec.decoder
    scenario_width = vector_spec.width
    solver = benchmark_solver()
    program = stochastic_program(problem)
    bundle = dfl_load_bundle(options.artifact_dir, job.problem)
    context_dim = context_dimension_per_point(bundle.train_data)

    model = nothing
    epoch_rows = NamedTuple[]
    if isfile(state_path)
        saved = Serialization.deserialize(state_path)
        if hasproperty(saved, :model)
            model = saved.model
        end
        if hasproperty(saved, :epoch_rows)
            epoch_rows = collect(saved.epoch_rows)
        end
        println(
            "resume $(job.problem)/$(job.learned_output)/seed$(job.seed): " *
            "$(length(epoch_rows)) completed epoch(s)",
        )
    end

    if model === nothing
        Random.seed!(job.seed)
        model = build_generic_nn(
            context_dim,
            scenario_width;
            hidden_dim=options.hidden_dim,
            depth=options.depth,
        )
    end

    loss = ContextualDFL.DflScenLoss(
        vector_decoder,
        reference_decoder,
        solver,
        program;
        nr_scenarios=1,
    )

    completed = length(epoch_rows)
    if completed > length(schedule)
        resize!(epoch_rows, length(schedule))
        completed = length(epoch_rows)
    end

    for epoch_index in (completed + 1):length(schedule)
        item = schedule[epoch_index]
        println(
            "train $(job.problem)/$(job.learned_output)/seed$(job.seed) " *
            "epoch $(item.epoch)/$(length(schedule)) mu_in=$(item.mu_in) mu_ref=$(item.mu_ref)",
        )
        train_result = nothing
        train_seconds = @elapsed train_result = ContextualDFL.train!(
            model,
            loss,
            nothing,
            [item.mu_in],
            [item.mu_ref],
            bundle.train_data;
            optimizer_type=DFL_FLUX.Adam,
            learning_rate=options.learning_rate,
            epochs=1,
            batchsize=options.batchsize,
            display_iterations=false,
            verbose=false,
            display_plot=false,
            shuffle=false,
            rng=Random.MersenneTwister(job.seed),
            reset_optimizer_each_epoch=true,
            nr_scenarios=1,
            rho_in_schedule=[item.rho_in],
            rho_ref_schedule=[item.rho_ref],
        )
        model = train_result.model
        train_history = only(train_result.history)

        Serialization.serialize(checkpoint_model_path, model)

        row = merge(
            worker,
            (;
                status="ok",
                version=DFL_VERSION,
                problem=job.problem,
                learned_output=job.learned_output,
                seed=job.seed,
                replica_index=job.replica_index,
                epoch=item.epoch,
                stage=item.stage,
                stage_epoch=item.stage_epoch,
                stage_epochs=item.stage_epochs,
                mu_in=item.mu_in,
                mu_ref=item.mu_ref,
                rho_in=item.rho_in,
                rho_ref=item.rho_ref,
                policy_mu=item.mu_in,
                policy_rho=0.0,
                train_loss=train_history.loss,
                train_display_loss=train_history.display_loss,
                train_real_display_loss=something(train_history.real_display_loss, ""),
                train_iterations=train_history.iterations,
                train_epoch_seconds=train_history.epoch_seconds,
                train_total_seconds=train_seconds,
                source_artifact_path=data_bundle_artifact_path(options.artifact_dir, job.problem),
                model_checkpoint_path=checkpoint_model_path,
                state_path=state_path,
                epoch_history_path=epoch_history_path,
                test_metrics_path=test_metrics_path,
                output_dir=out_dir,
                error="",
            ),
        )
        push!(epoch_rows, row)
        dfl_write_namedtuple_csv(epoch_history_path, epoch_rows)
        Serialization.serialize(
            state_path,
            (;
                status="partial",
                version=DFL_VERSION,
                model=model,
                epoch_rows=epoch_rows,
                schedule=collect(schedule),
                settings=dfl_job_settings(job),
                updated_at=Dates.format(now(), dateformat"yyyy-mm-ddTHH:MM:SS"),
            ),
        )
        println(
            @sprintf(
                "epoch done %s/%s seed=%d epoch=%d train_loss=%g",
                job.problem,
                job.learned_output,
                job.seed,
                item.epoch,
                Float64(train_history.loss),
            ),
        )
        GC.gc()
    end

    isempty(epoch_rows) && throw(ArgumentError("no epochs were completed."))
    final_epoch = last(epoch_rows)
    comparison = nothing
    test_seconds = @elapsed comparison = dfl_evaluate_model(
        model,
        vector_decoder,
        problem,
        solver,
        program,
        reference_decoder,
        bundle;
        policy_mu=final_epoch.policy_mu,
    )
    metrics = comparison.metrics
    dfl_write_per_sample_csv(final_per_sample_path, comparison.per_sample)
    Serialization.serialize(joinpath(out_dir, "comparison_final.jls"), comparison)
    Serialization.serialize(final_model_path, model)
    final_row = merge(
        final_epoch,
        (;
            test_eval_seconds=test_seconds,
            test_contexts=length(bundle.test_data),
            test_scenarios_per_context=scenario_count_per_context(bundle.test_data),
            evaluation_batches=optimal_results_batch_count(bundle.optimal_results),
            sample_count=metrics.test_sample_count,
            policy_value_mean=metrics.test_policy_value_mean,
            optimal_value_mean=metrics.test_optimal_value_mean,
            regret_mean=metrics.test_regret_mean,
            relative_regret_mean=metrics.test_relative_regret_mean,
            gap_stderr_mean=metrics.test_gap_stderr_mean,
            policy_eval_seconds=metrics.test_policy_eval_seconds,
            epochs_completed=length(epoch_rows),
            model_final_path=final_model_path,
            per_sample_path=final_per_sample_path,
            total_seconds=time() - started,
        ),
    )
    dfl_write_namedtuple_csv(test_metrics_path, [final_row])
    Serialization.serialize(
        state_path,
        (;
            status="ok",
            version=DFL_VERSION,
            model=model,
            epoch_rows=epoch_rows,
            schedule=collect(schedule),
            settings=dfl_job_settings(job),
            final_result_row=final_row,
            final_per_sample_path=final_per_sample_path,
            final_model_path=final_model_path,
            updated_at=Dates.format(now(), dateformat"yyyy-mm-ddTHH:MM:SS"),
        ),
    )

    return final_row
end

function dfl_evaluate_model(
    model,
    vector_decoder,
    problem,
    solver,
    program,
    reference_decoder,
    bundle;
    policy_mu,
)
    policy = ScenarioGenerationPolicy(
        ContextualDFL.ScenarioGenerator(
            neural_net=model,
            scenario_decoder=vector_decoder,
        ),
        solver,
        program;
        mu=policy_mu,
        rho=0.0,
        nr_scenarios=1,
    )
    return evaluate_policy_against_optimum(
        policy,
        bundle.test_data,
        program,
        reference_decoder,
        solver;
        optimal_results=bundle.optimal_results,
        mu=0.0,
        rho=0.0,
    )
end

function dfl_vector_decoder(problem_name, learned_output, problem)
    if startswith(problem_name, "transshipment_")
        if learned_output == "q"
            decoder = TransShipmentPositiveQVectorDecoder(problem)
            return (; decoder=decoder, width=length(ContextualDFL.transshipment_mean_parameters(problem.core_problem).q))
        elseif learned_output == "h"
            decoder = TransShipmentPositiveHVectorDecoder(problem)
            return (; decoder=decoder, width=length(ContextualDFL.transshipment_mean_parameters(problem.core_problem).rhs))
        end
    elseif problem_name == "random_yield"
        base = base_scenario(problem)
        learned_output == "q" && return (; decoder=RandomYieldPositiveQVectorDecoder(problem), width=length(base.q))
        learned_output == "h" && return (; decoder=RandomYieldHVectorDecoder(problem), width=length(base.h_eq))
    elseif problem_name == "resource_allocation"
        base = base_scenario(problem)
        if learned_output == "q"
            return (; decoder=ResourceAllocationFullCostVectorDecoder(problem), width=length(base.q))
        elseif learned_output == "h"
            demand_count = size(problem.problem_data.service_rate_parameters, 2)
            return (; decoder=ResourceAllocationDemandVectorDecoder(problem), width=demand_count)
        end
    end
    throw(ArgumentError("unsupported learned output $(repr(learned_output)) for $(problem_name)."))
end

function dfl_load_bundle(artifact_dir, problem_name)
    path = data_bundle_artifact_path(artifact_dir, problem_name)
    isfile(path) || throw(ArgumentError("missing artifact: $(path)"))
    payload = open(Serialization.deserialize, path)
    return ensure_bundle_metadata(artifact_payload_bundle(payload, path))
end

dfl_run_dir(job) = joinpath(
    job.options.output_dir,
    job.problem,
    job.learned_output,
    "seed_$(job.seed)",
)

function dfl_job_settings(job)
    options = job.options
    return (;
        version=DFL_VERSION,
        problem=job.problem,
        learned_output=job.learned_output,
        seed=job.seed,
        replica_index=job.replica_index,
        max_epochs=options.max_epochs,
        hidden_dim=options.hidden_dim,
        depth=options.depth,
        learning_rate=options.learning_rate,
        batchsize=options.batchsize,
        artifact_dir=options.artifact_dir,
        output_dir=options.output_dir,
    )
end

function dfl_write_outputs(output_dir, rows)
    dfl_write_namedtuple_csv(joinpath(output_dir, "individual_results.csv"), rows)
    dfl_write_namedtuple_csv(joinpath(output_dir, "epoch_results.csv"), dfl_collect_epoch_rows(output_dir))
    dfl_write_namedtuple_csv(joinpath(output_dir, "summary_by_config.csv"), dfl_summary_rows(rows))
    dfl_write_summary_md(joinpath(output_dir, "summary.md"), rows)
    return nothing
end

function dfl_collect_epoch_rows(output_dir)
    rows = NamedTuple[]
    isdir(output_dir) || return rows
    state_paths = String[]
    for (root, _, files) in walkdir(output_dir)
        "state.jls" in files && push!(state_paths, joinpath(root, "state.jls"))
    end
    for state_path in sort(state_paths)
        try
            state = Serialization.deserialize(state_path)
            hasproperty(state, :epoch_rows) && append!(rows, collect(state.epoch_rows))
        catch error
            @warn "could not read state file" state_path error
        end
    end
    return rows
end

function dfl_summary_rows(rows)
    ok_rows = [
        row for row in rows
        if hasproperty(row, :status) && row.status == "ok"
    ]
    groups = sort!(collect(Set((row.problem, row.learned_output) for row in ok_rows)))
    return [
        dfl_summary_row(problem, learned_output, [
            row for row in ok_rows
            if row.problem == problem && row.learned_output == learned_output
        ])
        for (problem, learned_output) in groups
    ]
end

function dfl_summary_row(problem, learned_output, rows)
    regrets = Float64[row.regret_mean for row in rows]
    relative_regrets = Float64[row.relative_regret_mean for row in rows]
    policy_values = Float64[row.policy_value_mean for row in rows]
    optimal_values = Float64[row.optimal_value_mean for row in rows]
    return (;
        problem=problem,
        learned_output=learned_output,
        n=length(rows),
        seeds=join(sort(Int[row.seed for row in rows]), ";"),
        regret_mean=Statistics.mean(regrets),
        regret_std=dfl_std(regrets),
        relative_regret_mean=Statistics.mean(relative_regrets),
        relative_regret_std=dfl_std(relative_regrets),
        policy_value_mean=Statistics.mean(policy_values),
        optimal_value_mean=Statistics.mean(optimal_values),
    )
end

dfl_std(values) = length(values) > 1 ? Statistics.std(values) : 0.0

function dfl_write_summary_md(path, rows)
    ok_rows = [row for row in rows if hasproperty(row, :status) && row.status == "ok"]
    error_rows = [row for row in rows if !hasproperty(row, :status) || row.status != "ok"]
    summaries = dfl_summary_rows(rows)
    open(path, "w") do io
        println(io, "# DflScenLoss Tiny q/h Benchmark")
        println(io)
        println(io, "- rows: $(length(rows))")
        println(io, "- ok rows: $(length(ok_rows))")
        println(io, "- error rows: $(length(error_rows))")
        println(io, "- generated_at: $(Dates.format(now(), dateformat"yyyy-mm-ddTHH:MM:SS"))")
        println(io)
        println(io, "| problem | learned_output | n | regret_mean | regret_std | rel_regret_mean | rel_regret_std |")
        println(io, "|---|---:|---:|---:|---:|---:|---:|")
        for row in summaries
            println(
                io,
                "| $(row.problem) | $(row.learned_output) | $(row.n) | " *
                "$(row.regret_mean) | $(row.regret_std) | " *
                "$(row.relative_regret_mean) | $(row.relative_regret_std) |",
            )
        end
        if !isempty(error_rows)
            println(io)
            println(io, "## Errors")
            for row in error_rows
                println(
                    io,
                    "- $(dfl_get(row, :problem, ""))/$(dfl_get(row, :learned_output, ""))/seed$(dfl_get(row, :seed, "")): " *
                    "$(first(split(String(dfl_get(row, :error, "")), '\n')))",
                )
            end
        end
    end
    return path
end

function dfl_write_namedtuple_csv(path, rows)
    mkpath(dirname(path))
    rows = collect(rows)
    if isempty(rows)
        open(path, "w") do io
            println(io)
        end
        return path
    end
    columns = Symbol[]
    for row in rows
        for column in keys(row)
            column in columns || push!(columns, column)
        end
    end
    open(path, "w") do io
        println(io, join(string.(columns), ","))
        for row in rows
            println(
                io,
                join(
                    (
                        csv_cell(hasproperty(row, column) ? getproperty(row, column) : "")
                        for column in columns
                    ),
                    ",",
                ),
            )
        end
    end
    return path
end

function dfl_write_per_sample_csv(path, rows)
    dfl_write_namedtuple_csv(
        path,
        [
            (;
                sample_index=row.sample_index,
                policy_value=row.policy_value,
                optimal_value=row.optimal_value,
                regret=row.regret,
                relative_regret=row.relative_regret,
                gap_std=row.gap_std,
                gap_stderr=row.gap_stderr,
                policy_collection_values=join(row.policy_collection_values, ";"),
                optimal_collection_values=join(row.optimal_collection_values, ";"),
                gap_values=join(row.gap_values, ";"),
            )
            for row in rows
        ],
    )
end

function dfl_write_manifest(path, options, args)
    schedule = dfl_training_schedule()[1:options.max_epochs]
    open(path, "w") do io
        println(io, "created_at=$(Dates.format(now(), dateformat"yyyy-mm-ddTHH:MM:SS"))")
        println(io, "hostname=$(Sockets.gethostname())")
        println(io, "julia=$(VERSION)")
        println(io, "version=$(DFL_VERSION)")
        println(io, "source_dir=$(DFL_SOURCE_DIR)")
        println(io, "sandbox_dir=$(DFL_SANDBOX_DIR)")
        println(io, "args=$(join(args, " "))")
        println(io, "artifact_dir=$(options.artifact_dir)")
        println(io, "output_dir=$(options.output_dir)")
        println(io, "problems=$(join(options.problems, ","))")
        println(io, "outputs=$(join(options.outputs, ","))")
        println(io, "seeds=$(join(options.seeds, ","))")
        println(io, "workers=$(options.local_workers)")
        println(io, "hidden_dim=$(options.hidden_dim)")
        println(io, "depth=$(options.depth)")
        println(io, "learning_rate=$(options.learning_rate)")
        println(io, "batchsize=$(options.batchsize)")
        println(io)
        println(io, "[schedule]")
        for row in schedule
            println(
                io,
                "epoch=$(row.epoch) stage=$(row.stage) stage_epoch=$(row.stage_epoch)/$(row.stage_epochs) " *
                "mu_in=$(row.mu_in) mu_ref=$(row.mu_ref) rho_in=$(row.rho_in) rho_ref=$(row.rho_ref)",
            )
        end
        println(io)
        println(io, "[artifacts]")
        for problem in options.problems
            artifact = data_bundle_artifact_path(options.artifact_dir, problem)
            digest = isfile(artifact) ? bytes2hex(open(SHA.sha256, artifact)) : "MISSING"
            println(io, "$(problem)=$(digest)  $(artifact)")
        end
    end
    return path
end

function dfl_print_final_row(row)
    if hasproperty(row, :status) && row.status == "ok"
        @printf(
            "ok %-22s learned=%s seed=%d regret=%12.6g rel=%12.6g epochs=%d worker=%s\n",
            row.problem,
            row.learned_output,
            Int(row.seed),
            Float64(row.regret_mean),
            Float64(row.relative_regret_mean),
            Int(row.epochs_completed),
            row.hostname,
        )
    else
        println("error $(dfl_get(row, :problem, "")) learned=$(dfl_get(row, :learned_output, "")) seed=$(dfl_get(row, :seed, "")): $(dfl_get(row, :error, ""))")
    end
end

dfl_get(row, field::Symbol, default) = hasproperty(row, field) ? getproperty(row, field) : default

if abspath(PROGRAM_FILE) == abspath(@__FILE__) && Distributed.myid() == 1
    main()
end
