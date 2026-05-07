#!/usr/bin/env julia

const CDFL_SOURCE_DIR = normpath(
    get(ENV, "CDFL_SOURCE_DIR", "/home/rwl/ProblemBasedScenarioGeneration/src/ContextualDFL"),
)
const EVAL_SANDBOX_DIR = @__DIR__
const BASELINE_SCRIPT = joinpath(
    CDFL_SOURCE_DIR,
    "ContextualDFLExperiments",
    "experiments",
    "baseline_benchmarks",
    "run_baselines.jl",
)

isfile(BASELINE_SCRIPT) || error("baseline helper script not found: $(BASELINE_SCRIPT)")
include(BASELINE_SCRIPT)

using Statistics

const DEFAULT_MODEL_ROOT = joinpath(EVAL_SANDBOX_DIR, "trained_models")
const DEFAULT_ARTIFACT_DIR = joinpath(
    CDFL_SOURCE_DIR,
    "ContextualDFLExperiments",
    "experiments",
    "baseline_benchmarks",
    "artifacts",
    "realworld_30ctx_20x1000_20260507",
)
const DEFAULT_OUTPUT_DIR = joinpath(EVAL_SANDBOX_DIR, "results")
const EVALUATED_PROBLEMS = ("transshipment_h_and_q", "random_yield", "resource_allocation")
const EVALUATED_OUTPUTS = ("q", "h")
const FINAL_POLICY_MU = 0.01

function main(args=ARGS)
    options = parse_options_realworld(args)
    mkpath(options.output_dir)
    write_manifest(joinpath(options.output_dir, "manifest.txt"), options, args)
    configure_workers_realworld!(options)

    jobs = evaluation_jobs(options)
    println("Realworld model evaluation jobs: $(length(jobs))")
    println("Problems: $(join(options.problems, ", "))")
    println("Outputs: $(join(options.outputs, ", "))")
    println("Repeats: $(join(options.repeats, ", "))")
    println("Workers: $(workers())")
    println("Output dir: $(options.output_dir)")

    rows = NamedTuple[]
    skipped = skipped_model_rows(options)
    for batch in Iterators.partition(jobs, max(1, options.job_batch_size))
        batch_rows = pmap_or_map(run_eval_job, collect(batch))
        append!(rows, batch_rows)
        write_namedtuple_csv(joinpath(options.output_dir, "realworld_eval_latest.csv"), rows)
        write_summary(joinpath(options.output_dir, "summary.md"), rows, skipped, options)
        for row in batch_rows
            print_eval_row(row)
        end
    end

    timestamp = Dates.format(now(), dateformat"yyyymmdd_HHMMSS")
    write_namedtuple_csv(joinpath(options.output_dir, "realworld_eval_$(timestamp).csv"), rows)
    write_namedtuple_csv(joinpath(options.output_dir, "realworld_eval_latest.csv"), rows)
    write_namedtuple_csv(joinpath(options.output_dir, "skipped_models.csv"), skipped)
    write_summary(joinpath(options.output_dir, "summary.md"), rows, skipped, options)
    any(row -> row.status != "ok", rows) && exit(1)
    return rows
end

function parse_options_realworld(args)
    model_root = DEFAULT_MODEL_ROOT
    artifact_dir = DEFAULT_ARTIFACT_DIR
    output_dir = DEFAULT_OUTPUT_DIR
    local_workers = 0
    job_batch_size = 0
    problems = collect(EVALUATED_PROBLEMS)
    outputs = collect(EVALUATED_OUTPUTS)
    repeats = collect(1:2)

    index = 1
    while index <= length(args)
        arg = args[index]
        if arg == "--model-root"
            index += 1
            model_root = abspath(args[index])
        elseif startswith(arg, "--model-root=")
            model_root = abspath(split(arg, "=", limit=2)[2])
        elseif arg == "--artifact-dir"
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
            problems = selected_names_realworld(EVALUATED_PROBLEMS, split_names_realworld(args[index]), "problem")
        elseif startswith(arg, "--problems=")
            problems = selected_names_realworld(
                EVALUATED_PROBLEMS,
                split_names_realworld(split(arg, "=", limit=2)[2]),
                "problem",
            )
        elseif arg == "--outputs"
            index += 1
            outputs = selected_names_realworld(EVALUATED_OUTPUTS, split_names_realworld(args[index]), "output")
        elseif startswith(arg, "--outputs=")
            outputs = selected_names_realworld(
                EVALUATED_OUTPUTS,
                split_names_realworld(split(arg, "=", limit=2)[2]),
                "output",
            )
        elseif arg == "--repeats"
            index += 1
            repeats = parse_repeats_realworld(args[index])
        elseif startswith(arg, "--repeats=")
            repeats = parse_repeats_realworld(split(arg, "=", limit=2)[2])
        else
            throw(ArgumentError("unknown argument: $(arg)"))
        end
        index += 1
    end

    local_workers >= 0 || throw(ArgumentError("local_workers must be non-negative."))
    if job_batch_size <= 0
        job_batch_size = local_workers > 0 ? local_workers : 1
    end
    return (;
        model_root=abspath(model_root),
        artifact_dir=abspath(artifact_dir),
        output_dir=abspath(output_dir),
        local_workers=local_workers,
        job_batch_size=job_batch_size,
        problems=problems,
        outputs=outputs,
        repeats=repeats,
    )
end

split_names_realworld(value) = [strip(item) for item in split(value, ",") if !isempty(strip(item))]

function selected_names_realworld(valid_names, requested_names, label)
    isempty(requested_names) && return collect(valid_names)
    unknown = setdiff(requested_names, valid_names)
    isempty(unknown) ||
        throw(ArgumentError("unknown $(label) name(s): $(join(unknown, ", "))"))
    return collect(requested_names)
end

function parse_repeats_realworld(value)
    repeats = Int[]
    for raw in split(value, ",")
        text = strip(raw)
        isempty(text) && continue
        if occursin(":", text)
            endpoints = split(text, ":")
            length(endpoints) == 2 ||
                throw(ArgumentError("repeat ranges must have form a:b, got $(text)."))
            append!(repeats, parse(Int, endpoints[1]):parse(Int, endpoints[2]))
        else
            push!(repeats, parse(Int, text))
        end
    end
    isempty(repeats) && throw(ArgumentError("repeat filter must not be empty."))
    any(<=(0), repeats) && throw(ArgumentError("repeat ids must be positive."))
    return unique!(repeats)
end

function configure_workers_realworld!(options)
    options.local_workers == 0 && return nothing
    addprocs(
        options.local_workers;
        exeflags="--project=$(PROJECT_DIR)",
        dir=PROJECT_DIR,
    )
    script_path = abspath(@__FILE__)
    for worker in workers()
        worker == 1 && continue
        remotecall_wait(worker, script_path) do script
            include(script)
            return nothing
        end
    end
    return nothing
end

function evaluation_jobs(options)
    return [
        (;
            problem=problem,
            learned_output=learned_output,
            repeat=repeat,
            options=options,
        )
        for problem in options.problems
        for learned_output in options.outputs
        for repeat in options.repeats
        if isfile(model_path(options.model_root, problem, learned_output, repeat)) &&
           isfile(artifact_path(options.artifact_dir, problem))
    ]
end

function skipped_model_rows(options)
    rows = NamedTuple[]
    all_problem_dirs = filter(isdir, readdir(options.model_root; join=true))
    for problem_dir in all_problem_dirs
        problem = basename(problem_dir)
        for output_dir in filter(isdir, readdir(problem_dir; join=true))
            learned_output = basename(output_dir)
            for repeat_dir in filter(isdir, readdir(output_dir; join=true))
                startswith(basename(repeat_dir), "repeat_") || continue
                repeat = parse(Int, replace(basename(repeat_dir), "repeat_" => ""))
                problem in options.problems || continue
                learned_output in options.outputs || continue
                isfile(joinpath(repeat_dir, "model.jls")) || continue
                if !isfile(artifact_path(options.artifact_dir, problem))
                    push!(
                        rows,
                        (;
                            problem=problem,
                            learned_output=learned_output,
                            repeat=repeat,
                            status="skipped",
                            reason="no matching realworld artifact",
                        ),
                    )
                end
            end
        end
    end
    return rows
end

model_path(root, problem, learned_output, repeat) =
    joinpath(root, problem, learned_output, "repeat_$(lpad(repeat, 2, '0'))", "model.jls")

artifact_path(root, problem) = joinpath(root, "$(problem).jls")

function run_eval_job(job)
    out_dir = joinpath(
        job.options.output_dir,
        job.problem,
        job.learned_output,
        "repeat_$(lpad(job.repeat, 2, '0'))",
    )
    result_path = joinpath(out_dir, "evaluation_result.jls")
    if isfile(result_path)
        row = Serialization.deserialize(result_path)
        hasproperty(row, :status) && row.status == "ok" && return row
    end

    mkpath(out_dir)
    started = time()
    try
        row = run_eval_job_inner(job, out_dir, started)
        Serialization.serialize(result_path, row)
        return row
    catch error
        row = (;
            status="error",
            problem=job.problem,
            learned_output=job.learned_output,
            repeat=job.repeat,
            worker_id=Distributed.myid(),
            hostname=Sockets.gethostname(),
            eval_seconds=time() - started,
            error=sprint(showerror, error, catch_backtrace()),
        )
        Serialization.serialize(result_path, row)
        return row
    finally
        GC.gc()
    end
end

function run_eval_job_inner(job, out_dir, started)
    bundle = load_bundle(artifact_path(job.options.artifact_dir, job.problem))
    problem = make_problem(job.problem)
    reference_decoder = make_decoder(job.problem, problem)
    vector_decoder = first(learned_vector_decoder(job.problem, job.learned_output, problem))
    solver = benchmark_solver()
    program = stochastic_program(problem)
    model_file = model_path(job.options.model_root, job.problem, job.learned_output, job.repeat)
    model = Serialization.deserialize(model_file)
    policy = ScenarioGenerationPolicy(
        ContextualDFL.ScenarioGenerator(
            neural_net=model,
            scenario_decoder=vector_decoder,
        ),
        solver,
        program;
        mu=FINAL_POLICY_MU,
        rho=0.0,
        nr_scenarios=1,
    )

    comparison = nothing
    eval_seconds = @elapsed comparison = evaluate_policy_against_optimum(
        policy,
        bundle.test_data,
        program,
        reference_decoder,
        solver;
        optimal_results=bundle.optimal_results,
    )
    Serialization.serialize(joinpath(out_dir, "comparison.jls"), comparison)
    write_per_sample_csv(joinpath(out_dir, "per_sample.csv"), comparison.per_sample)

    metrics = comparison.metrics
    return (;
        status="ok",
        problem=job.problem,
        learned_output=job.learned_output,
        repeat=job.repeat,
        worker_id=Distributed.myid(),
        hostname=Sockets.gethostname(),
        model_path=model_file,
        artifact_path=artifact_path(job.options.artifact_dir, job.problem),
        policy_mu=FINAL_POLICY_MU,
        eval_seconds=eval_seconds,
        total_seconds=time() - started,
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
        per_sample_path=joinpath(out_dir, "per_sample.csv"),
        error="",
    )
end

function load_bundle(path)
    isfile(path) || throw(ArgumentError("missing artifact: $(path)"))
    payload = open(Serialization.deserialize, path)
    bundle = ensure_bundle_metadata(artifact_payload_bundle(payload, path))
    return bundle
end

function learned_vector_decoder(problem_name, learned_output, problem)
    if problem_name == "transshipment_h_and_q"
        if learned_output == "q"
            decoder = TransShipmentPositiveQVectorDecoder(problem)
            return decoder, length(ContextualDFL.transshipment_mean_parameters(problem.core_problem).q)
        elseif learned_output == "h"
            decoder = TransShipmentPositiveHVectorDecoder(problem)
            return decoder, length(ContextualDFL.transshipment_mean_parameters(problem.core_problem).rhs)
        end
    elseif problem_name == "random_yield"
        base = base_scenario(problem)
        learned_output == "q" && return RandomYieldPositiveQVectorDecoder(problem), length(base.q)
        learned_output == "h" && return RandomYieldHVectorDecoder(problem), length(base.h_eq)
    end
    throw(ArgumentError("unsupported learned output $(repr(learned_output)) for $(problem_name)."))
end

function write_per_sample_csv(path, rows)
    write_namedtuple_csv(
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

function write_namedtuple_csv(path, rows)
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
                    (csv_cell(hasproperty(row, column) ? getproperty(row, column) : "") for column in columns),
                    ",",
                ),
            )
        end
    end
    return path
end

function write_summary(path, rows, skipped, options)
    ok_rows = [row for row in rows if hasproperty(row, :status) && row.status == "ok"]
    groups = sort!(collect(Set((row.problem, row.learned_output) for row in ok_rows)))
    open(path, "w") do io
        println(io, "# Realworld DFL q/h Model Evaluation")
        println(io)
        println(io, "- rows: $(length(rows))")
        println(io, "- ok rows: $(length(ok_rows))")
        println(io, "- skipped rows: $(length(skipped))")
        println(io, "- model_root: $(options.model_root)")
        println(io, "- artifact_dir: $(options.artifact_dir)")
        println(io, "- policy_mu: $(FINAL_POLICY_MU)")
        println(io)
        println(io, "| problem | learned_output | n | regret_mean | regret_std | rel_regret_mean | rel_regret_std |")
        println(io, "|---|---:|---:|---:|---:|---:|---:|")
        for (problem, learned_output) in groups
            subset = [row for row in ok_rows if row.problem == problem && row.learned_output == learned_output]
            regrets = Float64[row.regret_mean for row in subset]
            rels = Float64[row.relative_regret_mean for row in subset]
            println(
                io,
                "| $(problem) | $(learned_output) | $(length(subset)) | " *
                "$(Statistics.mean(regrets)) | $(length(regrets) > 1 ? Statistics.std(regrets) : 0.0) | " *
                "$(Statistics.mean(rels)) | $(length(rels) > 1 ? Statistics.std(rels) : 0.0) |",
            )
        end
    end
    return path
end

function write_manifest(path, options, args)
    open(path, "w") do io
        created_at = Dates.format(now(), dateformat"yyyy-mm-ddTHH:MM:SS")
        println(io, "created_at=$(created_at)")
        println(io, "hostname=$(Sockets.gethostname())")
        println(io, "julia=$(VERSION)")
        println(io, "source_dir=$(CDFL_SOURCE_DIR)")
        println(io, "sandbox_dir=$(EVAL_SANDBOX_DIR)")
        println(io, "args=$(join(args, " "))")
        println(io, "model_root=$(options.model_root)")
        println(io, "artifact_dir=$(options.artifact_dir)")
        println(io, "output_dir=$(options.output_dir)")
        println(io)
        println(io, "[artifacts]")
        for problem in options.problems
            path = artifact_path(options.artifact_dir, problem)
            println(io, "$(problem)=$(isfile(path) ? bytes2hex(open(SHA.sha256, path)) : "MISSING")  $(path)")
        end
    end
    return path
end

function print_eval_row(row)
    if row.status == "ok"
        @printf(
            "ok %-22s learned=%s repeat=%d regret=%12.6g rel=%12.6g eval=%8.1fs worker=%s\n",
            row.problem,
            row.learned_output,
            Int(row.repeat),
            Float64(row.regret_mean),
            Float64(row.relative_regret_mean),
            Float64(row.eval_seconds),
            row.hostname,
        )
    else
        println("error $(row.problem) learned=$(row.learned_output) repeat=$(row.repeat): $(row.error)")
    end
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__) && Distributed.myid() == 1
    main()
end
