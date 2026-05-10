#!/usr/bin/env julia

const DQ_REPO_ROOT = normpath(joinpath(@__DIR__, "..", "..", ".."))
const DQ_SOURCE_DIR = normpath(
    get(ENV, "CDFL_SOURCE_DIR", joinpath(DQ_REPO_ROOT, "src", "ContextualDFL")),
)
const DQ_BASELINE_SCRIPT = joinpath(
    DQ_SOURCE_DIR,
    "ContextualDFLExperiments",
    "experiments",
    "baseline_benchmarks",
    "run_baselines.jl",
)

isfile(DQ_BASELINE_SCRIPT) || error("baseline helper script not found: $(DQ_BASELINE_SCRIPT)")
include(DQ_BASELINE_SCRIPT)

using Dates
using Distributed
using Printf
using Random
using Serialization
using SHA
using Sockets
using Statistics

const DQ_SCRIPT_PATH = abspath(@__FILE__)
const DQ_PROBLEM = "random_yield"
const DQ_METHODS = ("spoplus_q_conversion", "dflscen_q_conversion")
const DQ_SEEDS = (20260505, 20260506)
const DQ_EPOCHS = 50
const DQ_DEFAULT_ARTIFACT_DIR = joinpath(
    DQ_SOURCE_DIR,
    "ContextualDFLExperiments",
    "experiments",
    "baseline_benchmarks",
    "artifacts",
    "tiny_30ctx_5x100_seed20260505",
)
const DQ_DEFAULT_OUTPUT_DIR = joinpath(@__DIR__, "results")
const DQ_VERSION = "random_yield_q_conversion_spoplus_dfl_v1"
const DQ_FLUX = ContextualDFL.Flux

function main(args=ARGS)
    options = dq_parse_options(args)
    mkpath(options.output_dir)
    dq_write_manifest(joinpath(options.output_dir, "manifest.txt"), options, args)
    dq_configure_workers!(options)

    prepared = dq_prepare_conversion_cache(options)
    println("Random-yield q-conversion jobs: $(length(dq_jobs(options)))")
    println("Methods: $(join(options.methods, ", "))")
    println("Seeds: $(join(options.seeds, ", "))")
    println("Epochs: $(options.epochs)")
    println("Workers: $(workers())")
    println("Threads: $(Threads.nthreads())")
    println("Artifact dir: $(options.artifact_dir)")
    println("Output dir: $(options.output_dir)")
    println("Converted q width: $(length(prepared.q_lower_bound))")

    jobs = dq_jobs(options)
    rows = NamedTuple[]
    for batch in Iterators.partition(jobs, max(1, options.job_batch_size))
        batch_rows = dq_pmap_or_map(run_dq_job, collect(batch))
        append!(rows, batch_rows)
        dq_write_outputs(options.output_dir, rows)
        for row in batch_rows
            dq_print_final_row(row)
        end
    end

    timestamp = Dates.format(now(), dateformat"yyyymmdd_HHMMSS")
    dq_write_namedtuple_csv(joinpath(options.output_dir, "individual_results_$(timestamp).csv"), rows)
    dq_write_outputs(options.output_dir, rows)
    any(row -> hasproperty(row, :status) && row.status != "ok", rows) && exit(1)
    return rows
end

function dq_parse_options(args)
    artifact_dir = DQ_DEFAULT_ARTIFACT_DIR
    output_dir = DQ_DEFAULT_OUTPUT_DIR
    local_workers = 0
    job_batch_size = 0
    methods = collect(DQ_METHODS)
    seeds = collect(DQ_SEEDS)
    epochs = DQ_EPOCHS
    hidden_dim = 128
    depth = 3
    learning_rate = 1e-3
    batchsize = 1
    lower_bound_margin = 1e-6
    constraint_tolerance = 1e-8
    spoplus_rho = 0.1

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
        elseif arg == "--methods"
            index += 1
            methods = dq_selected_names(DQ_METHODS, dq_split_names(args[index]), "method")
        elseif startswith(arg, "--methods=")
            methods = dq_selected_names(
                DQ_METHODS,
                dq_split_names(split(arg, "=", limit=2)[2]),
                "method",
            )
        elseif arg == "--seeds"
            index += 1
            seeds = dq_parse_ints(args[index])
        elseif startswith(arg, "--seeds=")
            seeds = dq_parse_ints(split(arg, "=", limit=2)[2])
        elseif arg == "--epochs"
            index += 1
            epochs = parse(Int, args[index])
        elseif startswith(arg, "--epochs=")
            epochs = parse(Int, split(arg, "=", limit=2)[2])
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
        elseif arg == "--lower-bound-margin"
            index += 1
            lower_bound_margin = parse(Float64, args[index])
        elseif startswith(arg, "--lower-bound-margin=")
            lower_bound_margin = parse(Float64, split(arg, "=", limit=2)[2])
        elseif arg == "--constraint-tolerance"
            index += 1
            constraint_tolerance = parse(Float64, args[index])
        elseif startswith(arg, "--constraint-tolerance=")
            constraint_tolerance = parse(Float64, split(arg, "=", limit=2)[2])
        elseif arg == "--spoplus-rho"
            index += 1
            spoplus_rho = parse(Float64, args[index])
        elseif startswith(arg, "--spoplus-rho=")
            spoplus_rho = parse(Float64, split(arg, "=", limit=2)[2])
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
    epochs > 0 || throw(ArgumentError("epochs must be positive."))
    lower_bound_margin >= 0 || throw(ArgumentError("lower_bound_margin must be non-negative."))
    constraint_tolerance >= 0 || throw(ArgumentError("constraint_tolerance must be non-negative."))
    spoplus_rho >= 0 || throw(ArgumentError("spoplus_rho must be non-negative."))
    isempty(seeds) && throw(ArgumentError("seeds must not be empty."))
    isempty(methods) && throw(ArgumentError("methods must not be empty."))
    if job_batch_size <= 0
        job_batch_size = local_workers > 0 ? local_workers : length(methods) * length(seeds)
    end

    return (;
        artifact_dir=abspath(artifact_dir),
        output_dir=abspath(output_dir),
        local_workers=local_workers,
        job_batch_size=job_batch_size,
        methods=methods,
        seeds=Int.(seeds),
        epochs=Int(epochs),
        hidden_dim=Int(hidden_dim),
        depth=Int(depth),
        learning_rate=Float64(learning_rate),
        batchsize=Int(batchsize),
        lower_bound_margin=Float64(lower_bound_margin),
        constraint_tolerance=Float64(constraint_tolerance),
        spoplus_rho=Float64(spoplus_rho),
    )
end

dq_split_names(value) = [strip(item) for item in split(value, ",") if !isempty(strip(item))]

function dq_selected_names(valid_names, requested_names, label)
    isempty(requested_names) && return collect(valid_names)
    unknown = setdiff(requested_names, valid_names)
    isempty(unknown) ||
        throw(ArgumentError("unknown $(label) name(s): $(join(unknown, ", "))"))
    return [name for name in valid_names if name in requested_names]
end

function dq_parse_ints(value)
    parsed = Int[]
    for raw in split(value, ",")
        text = strip(raw)
        isempty(text) || push!(parsed, parse(Int, text))
    end
    isempty(parsed) && throw(ArgumentError("integer list must not be empty."))
    return unique!(parsed)
end

function dq_configure_workers!(options)
    options.local_workers == 0 && return nothing
    addprocs(
        options.local_workers;
        exeflags="--project=$(PROJECT_DIR) --threads=$(Threads.nthreads())",
        dir=PROJECT_DIR,
    )
    for worker in workers()
        worker == 1 && continue
        remotecall_wait(worker, DQ_SCRIPT_PATH) do script
            include(script)
            return nothing
        end
    end
    return nothing
end

dq_pmap_or_map(f, jobs) = nworkers() == 0 ? map(f, jobs) : pmap(f, jobs)

function dq_jobs(options)
    jobs = NamedTuple[]
    for method in options.methods
        for (replica_index, seed) in enumerate(options.seeds)
            push!(
                jobs,
                (;
                    problem=DQ_PROBLEM,
                    method=method,
                    seed=Int(seed),
                    replica_index=Int(replica_index),
                    options=options,
                ),
            )
        end
    end
    return jobs
end

function dq_training_schedule(method, epochs; spoplus_rho=0.1)
    if method == "spoplus_q_conversion"
        return [
            (;
                epoch=epoch,
                stage=1,
                stage_epoch=epoch,
                stage_epochs=epochs,
                mu_in=0.0,
                mu_ref=0.0,
                rho_in=Float64(spoplus_rho),
                rho_ref=Float64(spoplus_rho),
            )
            for epoch in 1:epochs
        ]
    elseif method == "dflscen_q_conversion"
        values = exp.(range(log(1e-1), log(1e-4); length=epochs))
        return [
            (;
                epoch=epoch,
                stage=epoch,
                stage_epoch=1,
                stage_epochs=1,
                mu_in=Float64(values[epoch]),
                mu_ref=Float64(values[epoch]),
                rho_in=0.0,
                rho_ref=0.0,
            )
            for epoch in 1:epochs
        ]
    end
    throw(ArgumentError("unsupported method $(repr(method))."))
end

function dq_prepare_conversion_cache(options)
    cache_dir = joinpath(options.output_dir, "conversion_cache")
    cache_path = joinpath(cache_dir, "random_yield_converted_q.jls")
    if isfile(cache_path)
        return Serialization.deserialize(cache_path)
    end

    mkpath(cache_dir)
    problem = make_problem(DQ_PROBLEM)
    solver = benchmark_solver()
    program = stochastic_program(problem)
    original_decoder = make_decoder(DQ_PROBLEM, problem)
    bundle = dq_load_bundle(options.artifact_dir, DQ_PROBLEM)
    base = dq_full_parametric_scenario(base_scenario(problem))
    prepared = ContextualDFLExperiments.prepare_spoplus_q_dataset(
        bundle.train_data,
        solver,
        program,
        original_decoder,
        base;
        lower_bound_margin=options.lower_bound_margin,
        constraint_tolerance=options.constraint_tolerance,
    )
    result = (;
        status="ok",
        version=DQ_VERSION,
        artifact_path=data_bundle_artifact_path(options.artifact_dir, DQ_PROBLEM),
        converted_dataset=prepared.converted_dataset,
        q_lower_bound=prepared.q_lower_bound,
        created_at=Dates.format(now(), dateformat"yyyy-mm-ddTHH:MM:SS"),
        lower_bound_margin=options.lower_bound_margin,
        constraint_tolerance=options.constraint_tolerance,
    )
    Serialization.serialize(cache_path, result)
    return result
end

function dq_load_bundle(artifact_dir, problem_name)
    path = data_bundle_artifact_path(artifact_dir, problem_name)
    isfile(path) || throw(ArgumentError("missing artifact: $(path)"))
    payload = open(Serialization.deserialize, path)
    return ensure_bundle_metadata(artifact_payload_bundle(payload, path))
end

function dq_full_parametric_scenario(base)
    return ContextualDFL.ParametricScenario(;
        W_eq_xi=copy(base.W_eq),
        W_ineq_xi=copy(base.W_ineq),
        T_eq_xi=copy(base.T_eq),
        T_ineq_xi=copy(base.T_ineq),
        h_eq_xi=copy(base.h_eq),
        h_ineq_xi=copy(base.h_ineq),
        q_xi=copy(base.q),
    )
end

dq_run_dir(job) = joinpath(job.options.output_dir, job.problem, job.method, "seed_$(job.seed)")

function run_dq_job(job)
    worker = dq_worker_metadata()
    out_dir = dq_run_dir(job)
    result_path = joinpath(out_dir, "final_result.jls")
    if isfile(result_path)
        row = Serialization.deserialize(result_path)
        hasproperty(row, :status) && row.status == "ok" && return row
    end

    mkpath(out_dir)
    started = time()
    try
        row = run_dq_job_inner(job, out_dir, worker, started)
        Serialization.serialize(result_path, row)
        return row
    catch error
        row = merge(
            worker,
            (;
                status="error",
                version=DQ_VERSION,
                problem=job.problem,
                method=job.method,
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

function run_dq_job_inner(job, out_dir, worker, started)
    options = job.options
    schedule = dq_training_schedule(
        job.method,
        options.epochs;
        spoplus_rho=options.spoplus_rho,
    )
    state_path = joinpath(out_dir, "state.jls")
    epoch_history_path = joinpath(out_dir, "epoch_history.csv")
    test_metrics_path = joinpath(out_dir, "test_metrics_by_epoch.csv")
    final_model_path = joinpath(out_dir, "model_final.jls")
    final_per_sample_path = joinpath(out_dir, "per_sample_final.csv")

    problem = make_problem(job.problem)
    solver = benchmark_solver()
    program = stochastic_program(problem)
    reference_decoder = make_decoder(job.problem, problem)
    bundle = dq_load_bundle(options.artifact_dir, job.problem)
    prepared = dq_prepare_conversion_cache(options)
    converted_data = prepared.converted_dataset
    base = dq_full_parametric_scenario(base_scenario(problem))
    q_decoder = ContextualDFLExperiments.LowerBoundedQDecoder(
        base,
        prepared.q_lower_bound,
    )
    context_dim = context_dimension_per_point(converted_data)
    scenario_width = length(prepared.q_lower_bound)

    model = nothing
    epoch_rows = NamedTuple[]
    if isfile(state_path)
        saved = Serialization.deserialize(state_path)
        hasproperty(saved, :model) && (model = saved.model)
        hasproperty(saved, :epoch_rows) && (epoch_rows = collect(saved.epoch_rows))
        println(
            "resume $(job.method)/seed$(job.seed): $(length(epoch_rows)) completed epoch(s)",
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

    loss = dq_training_loss(job.method, q_decoder, solver, program)
    completed = length(epoch_rows)
    if completed > length(schedule)
        resize!(epoch_rows, length(schedule))
        completed = length(epoch_rows)
    end

    for epoch_index in (completed + 1):length(schedule)
        item = schedule[epoch_index]
        println(
            "train $(job.method)/seed$(job.seed) epoch $(item.epoch)/$(length(schedule)) " *
            "mu_in=$(item.mu_in) mu_ref=$(item.mu_ref)",
        )
        train_result = nothing
        train_seconds = @elapsed train_result = ContextualDFL.train!(
            model,
            loss,
            nothing,
            [item.mu_in],
            [item.mu_ref],
            converted_data;
            optimizer_type=DQ_FLUX.Adam,
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

        is_final_epoch = epoch_index == length(schedule)
        comparison = nothing
        test_seconds = ""
        metrics = nothing
        if is_final_epoch
            test_seconds = @elapsed comparison = dq_evaluate_model(
                model,
                q_decoder,
                problem,
                solver,
                program,
                reference_decoder,
                bundle;
                policy_mu=dq_policy_mu(job.method, item),
            )
            metrics = comparison.metrics
        end

        model_epoch_path = joinpath(out_dir, "model_epoch_$(lpad(item.epoch, 3, '0')).jls")
        Serialization.serialize(model_epoch_path, model)
        if is_final_epoch
            Serialization.serialize(final_model_path, model)
            dq_write_per_sample_csv(final_per_sample_path, comparison.per_sample)
            Serialization.serialize(joinpath(out_dir, "comparison_final.jls"), comparison)
        end

        row = merge(
            worker,
            (;
                status="ok",
                version=DQ_VERSION,
                problem=job.problem,
                method=job.method,
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
                policy_mu=dq_policy_mu(job.method, item),
                policy_rho=0.0,
                train_loss=train_history.loss,
                train_display_loss=train_history.display_loss,
                train_real_display_loss=something(train_history.real_display_loss, ""),
                train_iterations=train_history.iterations,
                train_epoch_seconds=train_history.epoch_seconds,
                train_total_seconds=train_seconds,
                dq_epoch_test_fields(bundle, metrics, test_seconds)...,
                source_artifact_path=data_bundle_artifact_path(options.artifact_dir, job.problem),
                converted_cache_path=joinpath(options.output_dir, "conversion_cache", "random_yield_converted_q.jls"),
                model_epoch_path=model_epoch_path,
                state_path=state_path,
                epoch_history_path=epoch_history_path,
                test_metrics_path=test_metrics_path,
                output_dir=out_dir,
                error="",
            ),
        )
        push!(epoch_rows, row)
        dq_write_namedtuple_csv(epoch_history_path, epoch_rows)
        dq_write_namedtuple_csv(test_metrics_path, epoch_rows)
        Serialization.serialize(
            state_path,
            (;
                status="partial",
                version=DQ_VERSION,
                model=model,
                epoch_rows=epoch_rows,
                schedule=collect(schedule),
                settings=dq_job_settings(job),
                updated_at=Dates.format(now(), dateformat"yyyy-mm-ddTHH:MM:SS"),
            ),
        )

        if is_final_epoch
            println(
                @sprintf(
                    "epoch done %s seed=%d epoch=%d final_regret=%g final_rel=%g",
                    job.method,
                    job.seed,
                    item.epoch,
                    Float64(metrics.test_regret_mean),
                    Float64(metrics.test_relative_regret_mean),
                ),
            )
        else
            println(
                @sprintf(
                    "epoch done %s seed=%d epoch=%d train_loss=%g",
                    job.method,
                    job.seed,
                    item.epoch,
                    Float64(train_history.loss),
                ),
            )
        end
        GC.gc()
    end

    isempty(epoch_rows) && throw(ArgumentError("no epochs were completed."))
    final_epoch = last(epoch_rows)
    if !isfile(final_per_sample_path)
        comparison = dq_evaluate_model(
            model,
            q_decoder,
            problem,
            solver,
            program,
            reference_decoder,
            bundle;
            policy_mu=final_epoch.policy_mu,
        )
        dq_write_per_sample_csv(final_per_sample_path, comparison.per_sample)
        Serialization.serialize(joinpath(out_dir, "comparison_final.jls"), comparison)
    end
    Serialization.serialize(final_model_path, model)
    Serialization.serialize(
        state_path,
        (;
            status="ok",
            version=DQ_VERSION,
            model=model,
            epoch_rows=epoch_rows,
            schedule=collect(schedule),
            settings=dq_job_settings(job),
            final_per_sample_path=final_per_sample_path,
            final_model_path=final_model_path,
            updated_at=Dates.format(now(), dateformat"yyyy-mm-ddTHH:MM:SS"),
        ),
    )

    return merge(
        final_epoch,
        (;
            epochs_completed=length(epoch_rows),
            model_final_path=final_model_path,
            per_sample_path=final_per_sample_path,
            total_seconds=time() - started,
        ),
    )
end

function dq_training_loss(method, q_decoder, solver, program)
    reference_decoder = ContextualDFL.ParametricDecoder()
    method == "spoplus_q_conversion" && return ContextualDFL.SPOPlusLoss(
        q_decoder,
        reference_decoder,
        solver,
        program;
        nr_scenarios=1,
    )
    method == "dflscen_q_conversion" && return ContextualDFL.DflScenLoss(
        q_decoder,
        reference_decoder,
        solver,
        program;
        nr_scenarios=1,
    )
    throw(ArgumentError("unsupported method $(repr(method))."))
end

dq_policy_mu(method, item) = method == "dflscen_q_conversion" ? item.mu_in : 0.0

function dq_evaluate_model(
    model,
    q_decoder,
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
            scenario_decoder=q_decoder,
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

function dq_epoch_test_fields(bundle, metrics, test_seconds)
    if metrics === nothing
        return (;
            test_eval_seconds="",
            test_contexts=length(bundle.test_data),
            test_scenarios_per_context=scenario_count_per_context(bundle.test_data),
            evaluation_batches=optimal_results_batch_count(bundle.optimal_results),
            sample_count="",
            policy_value_mean="",
            optimal_value_mean="",
            regret_mean="",
            relative_regret_mean="",
            gap_stderr_mean="",
            policy_eval_seconds="",
        )
    end

    return (;
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
    )
end

function dq_job_settings(job)
    options = job.options
    return (;
        version=DQ_VERSION,
        problem=job.problem,
        method=job.method,
        seed=job.seed,
        replica_index=job.replica_index,
        epochs=options.epochs,
        hidden_dim=options.hidden_dim,
        depth=options.depth,
        learning_rate=options.learning_rate,
        batchsize=options.batchsize,
        artifact_dir=options.artifact_dir,
        output_dir=options.output_dir,
        lower_bound_margin=options.lower_bound_margin,
        constraint_tolerance=options.constraint_tolerance,
        spoplus_rho=options.spoplus_rho,
    )
end

function dq_worker_metadata()
    return (;
        hostname=Sockets.gethostname(),
        pid=getpid(),
        worker_id=Distributed.myid(),
        julia_version=string(VERSION),
        julia_threads=Threads.nthreads(),
    )
end

function dq_write_outputs(output_dir, rows)
    dq_write_namedtuple_csv(joinpath(output_dir, "individual_results.csv"), rows)
    dq_write_namedtuple_csv(joinpath(output_dir, "epoch_results.csv"), dq_collect_epoch_rows(output_dir))
    dq_write_namedtuple_csv(joinpath(output_dir, "summary_by_method.csv"), dq_summary_rows(rows))
    dq_write_summary_md(joinpath(output_dir, "summary.md"), rows)
    return nothing
end

function dq_collect_epoch_rows(output_dir)
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

function dq_summary_rows(rows)
    ok_rows = [
        row for row in rows
        if hasproperty(row, :status) && row.status == "ok"
    ]
    groups = sort!(collect(Set(row.method for row in ok_rows)))
    return [
        dq_summary_row(method, [row for row in ok_rows if row.method == method])
        for method in groups
    ]
end

function dq_summary_row(method, rows)
    regrets = Float64[row.regret_mean for row in rows]
    relative_regrets = Float64[row.relative_regret_mean for row in rows]
    policy_values = Float64[row.policy_value_mean for row in rows]
    optimal_values = Float64[row.optimal_value_mean for row in rows]
    return (;
        problem=DQ_PROBLEM,
        method=method,
        n=length(rows),
        seeds=join(sort(Int[row.seed for row in rows]), ";"),
        regret_mean=Statistics.mean(regrets),
        regret_std=dq_std(regrets),
        relative_regret_mean=Statistics.mean(relative_regrets),
        relative_regret_std=dq_std(relative_regrets),
        policy_value_mean=Statistics.mean(policy_values),
        optimal_value_mean=Statistics.mean(optimal_values),
    )
end

dq_std(values) = length(values) > 1 ? Statistics.std(values) : 0.0

function dq_write_summary_md(path, rows)
    ok_rows = [row for row in rows if hasproperty(row, :status) && row.status == "ok"]
    error_rows = [row for row in rows if !hasproperty(row, :status) || row.status != "ok"]
    summaries = dq_summary_rows(rows)
    open(path, "w") do io
        println(io, "# Random Yield q-Conversion Benchmark")
        println(io)
        println(io, "- rows: $(length(rows))")
        println(io, "- ok rows: $(length(ok_rows))")
        println(io, "- error rows: $(length(error_rows))")
        println(io, "- generated_at: $(Dates.format(now(), dateformat"yyyy-mm-ddTHH:MM:SS"))")
        println(io)
        println(io, "| problem | method | n | regret_mean | regret_std | rel_regret_mean | rel_regret_std |")
        println(io, "|---|---:|---:|---:|---:|---:|---:|")
        for row in summaries
            println(
                io,
                "| $(row.problem) | $(row.method) | $(row.n) | " *
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
                    "- $(dq_get(row, :method, ""))/seed$(dq_get(row, :seed, "")): " *
                    "$(first(split(String(dq_get(row, :error, "")), '\n')))",
                )
            end
        end
    end
    return path
end

function dq_write_namedtuple_csv(path, rows)
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
                        dq_csv_cell(hasproperty(row, column) ? getproperty(row, column) : "")
                        for column in columns
                    ),
                    ",",
                ),
            )
        end
    end
    return path
end

function dq_write_per_sample_csv(path, rows)
    dq_write_namedtuple_csv(
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

function dq_csv_cell(value)
    text = string(value)
    if occursin(',', text) || occursin('"', text) || occursin('\n', text) || occursin('\r', text)
        return "\"" * replace(text, "\"" => "\"\"") * "\""
    end
    return text
end

function dq_write_manifest(path, options, args)
    artifact = data_bundle_artifact_path(options.artifact_dir, DQ_PROBLEM)
    digest = isfile(artifact) ? bytes2hex(open(SHA.sha256, artifact)) : "MISSING"
    open(path, "w") do io
        println(io, "created_at=$(Dates.format(now(), dateformat"yyyy-mm-ddTHH:MM:SS"))")
        println(io, "hostname=$(Sockets.gethostname())")
        println(io, "julia=$(VERSION)")
        println(io, "version=$(DQ_VERSION)")
        println(io, "repo_root=$(DQ_REPO_ROOT)")
        println(io, "source_dir=$(DQ_SOURCE_DIR)")
        println(io, "script_path=$(DQ_SCRIPT_PATH)")
        println(io, "args=$(join(args, " "))")
        println(io, "artifact_dir=$(options.artifact_dir)")
        println(io, "output_dir=$(options.output_dir)")
        println(io, "problem=$(DQ_PROBLEM)")
        println(io, "methods=$(join(options.methods, ","))")
        println(io, "seeds=$(join(options.seeds, ","))")
        println(io, "epochs=$(options.epochs)")
        println(io, "workers=$(options.local_workers)")
        println(io, "julia_threads=$(Threads.nthreads())")
        println(io, "hidden_dim=$(options.hidden_dim)")
        println(io, "depth=$(options.depth)")
        println(io, "learning_rate=$(options.learning_rate)")
        println(io, "batchsize=$(options.batchsize)")
        println(io, "lower_bound_margin=$(options.lower_bound_margin)")
        println(io, "constraint_tolerance=$(options.constraint_tolerance)")
        println(io, "spoplus_rho=$(options.spoplus_rho)")
        println(io)
        println(io, "[schedules]")
        for method in options.methods
            println(io, "method=$(method)")
            for row in dq_training_schedule(
                method,
                options.epochs;
                spoplus_rho=options.spoplus_rho,
            )
                println(
                    io,
                    "epoch=$(row.epoch) mu_in=$(row.mu_in) mu_ref=$(row.mu_ref) " *
                    "rho_in=$(row.rho_in) rho_ref=$(row.rho_ref)",
                )
            end
        end
        println(io)
        println(io, "[artifacts]")
        println(io, "$(DQ_PROBLEM)=$(digest)  $(artifact)")
    end
    return path
end

function dq_print_final_row(row)
    if hasproperty(row, :status) && row.status == "ok"
        @printf(
            "ok %-24s seed=%d regret=%12.6g rel=%12.6g epochs=%d worker=%s:%s\n",
            row.method,
            Int(row.seed),
            Float64(row.regret_mean),
            Float64(row.relative_regret_mean),
            Int(row.epochs_completed),
            row.hostname,
            row.worker_id,
        )
    else
        println(
            "error $(dq_get(row, :method, "")) seed=$(dq_get(row, :seed, "")): " *
            "$(dq_get(row, :error, ""))",
        )
    end
end

dq_get(row, field::Symbol, default) = hasproperty(row, field) ? getproperty(row, field) : default

if abspath(PROGRAM_FILE) == abspath(@__FILE__) && Distributed.myid() == 1
    main()
end
