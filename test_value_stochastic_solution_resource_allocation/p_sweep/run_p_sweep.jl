#!/usr/bin/env julia

import Pkg

const SCRIPT_DIR = @__DIR__
const EXPERIMENT_ROOT = dirname(SCRIPT_DIR)
const REPO_ROOT = dirname(EXPERIMENT_ROOT)
const TRAINING_PROJECT_DIR =
    joinpath(REPO_ROOT, "src", "ContextualDFL", "ContextualDFLTraining")

Pkg.activate(TRAINING_PROJECT_DIR; io=devnull)

using CSV
using ContextualDFL
using ContextualDFLExperiments
using Dates
using Random
using Serialization
using SHA

const DEFAULT_P_VALUES = (0.5, 1.0, 1.5)
const DEFAULT_SEEDS = (1, 2, 3)
const DEFAULT_CONTEXTS_PER_SEED = 10
const DEFAULT_SCENARIOS_PER_CONTEXT = 1000
const DEFAULT_EVALUATION_BATCHES = 20
const DEFAULT_SIGMA = 5.0
const DEFAULT_CONTEXT_TERMS = 3
const DEFAULT_MU = 0.0
const DEFAULT_RHO = 0.0

const REFERENCE_DIR = EXPERIMENT_ROOT
const REFERENCE_SUMMARY_PATH = joinpath(REFERENCE_DIR, "summary.csv")
const REFERENCE_PER_CONTEXT_PATH = joinpath(REFERENCE_DIR, "per_context.csv")
const REFERENCE_REPORT_PATH = joinpath(REFERENCE_DIR, "report.md")

function parse_args(args)
    options = Dict{Symbol,Any}(
        :smoke => false,
        :force => false,
        :include_reference => true,
        :output_dir => nothing,
    )

    index = 1
    while index <= length(args)
        arg = args[index]
        if arg == "--smoke"
            options[:smoke] = true
        elseif arg == "--force"
            options[:force] = true
        elseif arg == "--no-reference"
            options[:include_reference] = false
        elseif arg == "--output-dir"
            index += 1
            index <= length(args) ||
                throw(ArgumentError("--output-dir requires a path argument."))
            options[:output_dir] = args[index]
        elseif startswith(arg, "--output-dir=")
            options[:output_dir] = split(arg, "=", limit=2)[2]
        else
            throw(ArgumentError("unsupported argument: $arg"))
        end
        index += 1
    end

    return (; (key => options[key] for key in keys(options))...)
end

function run_config(args)
    parsed = parse_args(args)
    output_dir = if parsed.output_dir === nothing
        parsed.smoke ? joinpath(EXPERIMENT_ROOT, "p_sweep_smoke") : SCRIPT_DIR
    else
        abspath(String(parsed.output_dir))
    end

    if parsed.smoke
        return (;
            smoke=true,
            force=Bool(parsed.force),
            include_reference=false,
            output_dir=output_dir,
            p_values=(1.0,),
            seeds=(1,),
            contexts_per_seed=2,
            scenarios_per_context=20,
            evaluation_batches=2,
            sigma=DEFAULT_SIGMA,
            context_terms=DEFAULT_CONTEXT_TERMS,
            mu=DEFAULT_MU,
            rho=DEFAULT_RHO,
        )
    end

    return (;
        smoke=false,
        force=Bool(parsed.force),
        include_reference=Bool(parsed.include_reference),
        output_dir=output_dir,
        p_values=DEFAULT_P_VALUES,
        seeds=DEFAULT_SEEDS,
        contexts_per_seed=DEFAULT_CONTEXTS_PER_SEED,
        scenarios_per_context=DEFAULT_SCENARIOS_PER_CONTEXT,
        evaluation_batches=DEFAULT_EVALUATION_BATCHES,
        sigma=DEFAULT_SIGMA,
        context_terms=DEFAULT_CONTEXT_TERMS,
        mu=DEFAULT_MU,
        rho=DEFAULT_RHO,
    )
end

function log_message(io, message)
    timestamp = Dates.format(now(UTC), dateformat"yyyy-mm-ddTHH:MM:SS.sZ")
    line = "[$timestamp] $message"
    println(line)
    println(io, line)
    flush(stdout)
    flush(io)
end

function p_token(p)
    return replace(string(Float64(p)), "." => "p", "-" => "m")
end

p_label(p) = "p_" * p_token(p)

p_dir(config, p) = joinpath(config.output_dir, p_label(p))
seed_dir(config, p, seed) = joinpath(p_dir(config, p), "seed_$(Int(seed))")

function dataset_path(config, p, seed)
    return joinpath(seed_dir(config, p, seed), "test_dataset.jls")
end

function optimal_path(config, p, seed)
    return joinpath(seed_dir(config, p, seed), "optimal_results.jls")
end

function seed_summary_path(config, p, seed)
    return joinpath(seed_dir(config, p, seed), "summary.csv")
end

function seed_per_context_path(config, p, seed)
    return joinpath(seed_dir(config, p, seed), "per_context.csv")
end

function serialized_digest(value)
    io = IOBuffer()
    Serialization.serialize(io, value)
    return "sha1:" * bytes2hex(sha1(take!(io)))
end

function atomic_serialize(path, payload)
    mkpath(dirname(path))
    temp_path = tempname(dirname(path))
    open(temp_path, "w") do io
        Serialization.serialize(io, payload)
    end
    mv(temp_path, path; force=true)
    return path
end

payload_value(payload, property::Symbol) =
    hasproperty(payload, property) ? getproperty(payload, property) : payload

function problem_objects()
    problem = ResourceAllocationProblem(default_resource_allocation_problem_data())
    solver = ContextualDFL.Solver(ContextualDFL.IpoptSolver(), ContextualDFL.HiGHSSolver())
    return (;
        problem=problem,
        solver=solver,
        program=stochastic_program(problem),
        decoder=ResourceAllocationDemandParametricDecoder(problem),
    )
end

function generate_resource_allocation_dataset(objects, config, p, seed)
    rng = Random.MersenneTwister(Int(seed))
    context_generator = ResourceAllocationContextDataGenerator(rng=rng)
    scenario_generator = ResourceAllocationScenarioDataGenerator(
        objects.problem;
        sigma=config.sigma,
        p=Float64(p),
        L=config.context_terms,
        rng=rng,
    )

    contexts = Vector{Vector{Float64}}()
    scenario_collections = Vector{Vector{ContextualDFL.ParametricScenario}}()

    for _ in 1:Int(config.contexts_per_seed)
        context = Vector{Float64}(context_generator())
        push!(contexts, context)
        push!(
            scenario_collections,
            [scenario_generator(context) for _ in 1:Int(config.scenarios_per_context)],
        )
    end

    return generate_contextual_data_set(contexts, scenario_collections)
end

function validate_dataset(dataset, config; p, seed)
    length(dataset) == Int(config.contexts_per_seed) ||
        throw(ArgumentError(
            "p=$p seed=$seed dataset has $(length(dataset)) contexts; expected $(config.contexts_per_seed).",
        ))
    all(data_point -> length(data_point.scenario_parameters) == Int(config.scenarios_per_context), dataset) ||
        throw(ArgumentError(
            "p=$p seed=$seed dataset does not have $(config.scenarios_per_context) scenarios per context.",
        ))
    return dataset
end

function validate_optimal_results(results, dataset, config; p, seed)
    length(results) == length(dataset) ||
        throw(ArgumentError(
            "p=$p seed=$seed optimal results length $(length(results)) does not match dataset length $(length(dataset)).",
        ))
    for result in results
        hasproperty(result, :objective_values) ||
            throw(ArgumentError("p=$p seed=$seed optimal result is not in objective_values format."))
        length(result.objective_values) == Int(config.evaluation_batches) ||
            throw(ArgumentError(
                "p=$p seed=$seed optimal result has $(length(result.objective_values)) objective values; expected $(config.evaluation_batches).",
            ))
    end
    return results
end

function ensure_dataset!(objects, config, p, seed, log_io)
    path = dataset_path(config, p, seed)
    if isfile(path) && !config.force
        payload = Serialization.deserialize(path)
        dataset = payload_value(payload, :dataset)
        validate_dataset(dataset, config; p=p, seed=seed)
        log_message(log_io, "loaded dataset p=$p seed=$seed from $path")
        return payload, dataset
    end

    dataset = generate_resource_allocation_dataset(objects, config, p, seed)
    validate_dataset(dataset, config; p=p, seed=seed)
    payload = (;
        format_version=1,
        artifact_type=:resource_allocation_vss_p_sweep_dataset,
        generated_at=Dates.format(now(UTC), dateformat"yyyy-mm-ddTHH:MM:SS.sZ"),
        seed=Int(seed),
        demand_power=Float64(p),
        demand_sigma=Float64(config.sigma),
        context_terms=Int(config.context_terms),
        contexts_per_seed=Int(config.contexts_per_seed),
        scenarios_per_context=Int(config.scenarios_per_context),
        dataset_digest=serialized_digest(dataset),
        dataset=dataset,
    )
    atomic_serialize(path, payload)
    log_message(log_io, "wrote dataset p=$p seed=$seed to $path")
    return payload, dataset
end

function ensure_optimal_results!(objects, config, p, seed, dataset_payload, dataset, log_io)
    path = optimal_path(config, p, seed)
    if isfile(path) && !config.force
        payload = Serialization.deserialize(path)
        results = payload_value(payload, :optimal_results)
        validate_optimal_results(results, dataset, config; p=p, seed=seed)
        log_message(log_io, "loaded optimal results p=$p seed=$seed from $path")
        return payload, results
    end

    results = nothing
    solve_seconds = @elapsed begin
        results = solve_dataset_to_optimality(
            dataset,
            objects.program,
            objects.decoder,
            objects.solver;
            mu=config.mu,
            rho=config.rho,
            evaluation_batches=config.evaluation_batches,
        )
    end
    validate_optimal_results(results, dataset, config; p=p, seed=seed)
    payload = (;
        format_version=1,
        artifact_type=:resource_allocation_vss_p_sweep_optimal_results,
        generated_at=Dates.format(now(UTC), dateformat"yyyy-mm-ddTHH:MM:SS.sZ"),
        seed=Int(seed),
        demand_power=Float64(p),
        demand_sigma=Float64(config.sigma),
        context_terms=Int(config.context_terms),
        contexts_per_seed=Int(config.contexts_per_seed),
        scenarios_per_context=Int(config.scenarios_per_context),
        evaluation_batches=Int(config.evaluation_batches),
        mu=Float64(config.mu),
        rho=Float64(config.rho),
        solve_seconds=solve_seconds,
        dataset_digest=dataset_payload.dataset_digest,
        optimal_results_digest=serialized_digest(results),
        optimal_results=results,
    )
    atomic_serialize(path, payload)
    log_message(
        log_io,
        "wrote optimal results p=$p seed=$seed to $path in $(round(solve_seconds; digits=3)) seconds",
    )
    return payload, results
end

function average_component(scenarios, field::Symbol)
    first_value = getproperty(first(scenarios), field)

    if first_value isa Number
        return sum(Float64(getproperty(scenario, field)) for scenario in scenarios) /
               length(scenarios)
    end

    first_value isa AbstractArray ||
        throw(ArgumentError("cannot average non-numeric scenario component $field."))

    accumulator = zeros(Float64, size(first_value))
    for scenario in scenarios
        value = getproperty(scenario, field)
        size(value) == size(first_value) ||
            throw(DimensionMismatch("scenario component $field has inconsistent sizes."))
        accumulator .+= Float64.(value)
    end
    return accumulator ./ length(scenarios)
end

function average_scenario(scenarios)
    isempty(scenarios) && throw(ArgumentError("cannot average an empty scenario collection."))
    return ContextualDFL.ParametricScenario(;
        W_eq_xi=average_component(scenarios, :W_eq_xi),
        W_ineq_xi=average_component(scenarios, :W_ineq_xi),
        T_eq_xi=average_component(scenarios, :T_eq_xi),
        T_ineq_xi=average_component(scenarios, :T_ineq_xi),
        h_eq_xi=average_component(scenarios, :h_eq_xi),
        h_ineq_xi=average_component(scenarios, :h_ineq_xi),
        q_xi=average_component(scenarios, :q_xi),
    )
end

function solve_average_scenario_decision(data_point, objects, config)
    scenario = average_scenario(data_point.scenario_parameters)
    arrays = ContextualDFL.decode_scenario_collection(objects.decoder, [scenario])
    solution = ContextualDFL.solve(
        objects.solver,
        objects.program,
        arrays...;
        μ=config.mu,
        ρ=config.rho,
    )
    return collect(solution[1]), scenario
end

joined(values) = join(string.(Float64.(collect(values))), "|")

function context_row(; p, seed, local_index, data_point, average_scenario, decision, comparison_row)
    return (;
        demand_power=Float64(p),
        p_label=p_label(p),
        seed=string(seed),
        sample_index=Int(local_index),
        source_seed=string(seed),
        source_index=Int(local_index),
        context=joined(data_point.context),
        average_demand=joined(average_scenario.h_eq_xi),
        average_scenario_decision=joined(decision),
        average_scenario_policy_value=Float64(comparison_row.policy_value),
        stochastic_optimal_value=Float64(comparison_row.optimal_value),
        gap=Float64(comparison_row.regret),
        relative_gap=Float64(comparison_row.relative_regret),
        policy_collection_values=joined(comparison_row.policy_collection_values),
        optimal_collection_values=joined(comparison_row.optimal_collection_values),
        gap_values=joined(comparison_row.gap_values),
        gap_std=Float64(comparison_row.gap_std),
        gap_stderr=Float64(comparison_row.gap_stderr),
    )
end

function numeric_field(row, field::Symbol)
    return Float64(getproperty(row, field))
end

function percentile_95(values)
    isempty(values) && return NaN
    sorted = sort(Float64.(values))
    index = clamp(ceil(Int, 0.95 * length(sorted)), 1, length(sorted))
    return sorted[index]
end

function mean_value(values)
    isempty(values) && return NaN
    return sum(Float64.(values)) / length(values)
end

function median_value(values)
    isempty(values) && return NaN
    sorted = sort(Float64.(values))
    midpoint = length(sorted) ÷ 2
    isodd(length(sorted)) && return sorted[midpoint + 1]
    return (sorted[midpoint] + sorted[midpoint + 1]) / 2
end

function std_value(values)
    count = length(values)
    count == 0 && return NaN
    count == 1 && return 0.0
    mean = mean_value(values)
    return sqrt(sum((value - mean)^2 for value in Float64.(values)) / (count - 1))
end

function summarize_numeric(values; prefix)
    values = Float64.(values)
    pairs = Pair{Symbol,Any}[
        Symbol(prefix, :_mean) => mean_value(values),
        Symbol(prefix, :_median) => median_value(values),
        Symbol(prefix, :_std) => std_value(values),
        Symbol(prefix, :_min) => isempty(values) ? NaN : minimum(values),
        Symbol(prefix, :_max) => isempty(values) ? NaN : maximum(values),
        Symbol(prefix, :_p95) => percentile_95(values),
    ]
    return NamedTuple{Tuple(first.(pairs))}(Tuple(last.(pairs)))
end

function summary_row(;
    p,
    summary_level,
    seed,
    seeds,
    config,
    rows,
    decision_solve_seconds,
    policy_eval_seconds,
    artifact_dir,
    dataset_digest,
    optimal_results_digest,
    source,
)
    policy_values = [numeric_field(row, :average_scenario_policy_value) for row in rows]
    optimal_values = [numeric_field(row, :stochastic_optimal_value) for row in rows]
    gaps = [numeric_field(row, :gap) for row in rows]
    relative_gaps = [numeric_field(row, :relative_gap) for row in rows]

    return merge(
        (;
            demand_power=Float64(p),
            p_label=p_label(p),
            summary_level=String(summary_level),
            seed=String(seed),
            seeds=String(seeds),
            source=String(source),
            demand_sigma=Float64(config.sigma),
            context_terms=Int(config.context_terms),
            contexts=length(rows),
            contexts_per_seed=Int(config.contexts_per_seed),
            scenarios_per_context=Int(config.scenarios_per_context),
            evaluation_batches=Int(config.evaluation_batches),
            mu=Float64(config.mu),
            rho=Float64(config.rho),
            decision_solve_seconds=Float64(decision_solve_seconds),
            test_policy_eval_seconds=Float64(policy_eval_seconds),
            total_eval_seconds=Float64(decision_solve_seconds + policy_eval_seconds),
            artifact_dir=String(artifact_dir),
            dataset_digest=String(dataset_digest),
            optimal_results_digest=String(optimal_results_digest),
            generated_at=Dates.format(now(UTC), dateformat"yyyy-mm-ddTHH:MM:SS.sZ"),
        ),
        summarize_numeric(policy_values; prefix=:test_policy_value),
        summarize_numeric(optimal_values; prefix=:test_optimal_value),
        summarize_numeric(gaps; prefix=:test_regret),
        summarize_numeric(relative_gaps; prefix=:test_relative_regret),
    )
end

function read_namedtuple_rows(path)
    return [NamedTuple(row) for row in CSV.File(path)]
end

function ensure_evaluation!(
    objects,
    config,
    p,
    seed,
    dataset_payload,
    dataset,
    optimal_payload,
    optimal_results,
    log_io,
)
    summary_path = seed_summary_path(config, p, seed)
    per_context_path = seed_per_context_path(config, p, seed)
    if isfile(summary_path) && isfile(per_context_path) && !config.force
        log_message(log_io, "loaded evaluation p=$p seed=$seed from $(seed_dir(config, p, seed))")
        return only(read_namedtuple_rows(summary_path)), read_namedtuple_rows(per_context_path)
    end

    average_scenarios = ContextualDFL.ParametricScenario[]
    decision_columns = Vector{Float64}[]
    decision_solve_seconds = @elapsed begin
        for data_point in dataset
            decision, scenario = solve_average_scenario_decision(data_point, objects, config)
            push!(decision_columns, decision)
            push!(average_scenarios, scenario)
        end
    end

    decision_set = hcat(decision_columns...)
    comparison = evaluate_policy_against_optimum(
        decision_set,
        dataset,
        objects.program,
        objects.decoder,
        objects.solver;
        optimal_results=optimal_results,
        split_name=:test,
        mu=config.mu,
        rho=config.rho,
    )

    rows = [
        context_row(
            p=p,
            seed=seed,
            local_index=index,
            data_point=dataset[index],
            average_scenario=average_scenarios[index],
            decision=view(decision_set, :, index),
            comparison_row=comparison.per_sample[index],
        ) for index in eachindex(dataset)
    ]

    summary = summary_row(
        p=p,
        summary_level="seed",
        seed=string(seed),
        seeds=string(seed),
        config=config,
        rows=rows,
        decision_solve_seconds=decision_solve_seconds,
        policy_eval_seconds=comparison.metrics.test_policy_eval_seconds,
        artifact_dir=seed_dir(config, p, seed),
        dataset_digest=dataset_payload.dataset_digest,
        optimal_results_digest=optimal_payload.optimal_results_digest,
        source="generated",
    )

    CSV.write(per_context_path, rows)
    CSV.write(summary_path, [summary])
    log_message(
        log_io,
        "wrote evaluation p=$p seed=$seed: mean relative gap $(summary.test_relative_regret_mean)",
    )
    return summary, rows
end

function aggregate_p_outputs!(config, p, seed_summaries, context_rows)
    summaries = NamedTuple.(seed_summaries)
    decision_seconds = sum(numeric_field(row, :decision_solve_seconds) for row in summaries)
    policy_seconds = sum(numeric_field(row, :test_policy_eval_seconds) for row in summaries)
    dataset_digest = join([getproperty(row, :dataset_digest) for row in summaries], "|")
    optimal_digest = join([getproperty(row, :optimal_results_digest) for row in summaries], "|")
    summary = summary_row(
        p=p,
        summary_level="p",
        seed="all",
        seeds=join(config.seeds, "|"),
        config=config,
        rows=context_rows,
        decision_solve_seconds=decision_seconds,
        policy_eval_seconds=policy_seconds,
        artifact_dir=p_dir(config, p),
        dataset_digest=dataset_digest,
        optimal_results_digest=optimal_digest,
        source="generated",
    )

    CSV.write(joinpath(p_dir(config, p), "summary.csv"), [summary])
    CSV.write(joinpath(p_dir(config, p), "seed_summary.csv"), summaries)
    CSV.write(joinpath(p_dir(config, p), "per_context.csv"), context_rows)
    return summary
end

function reference_config_from_row(row)
    return (;
        sigma=DEFAULT_SIGMA,
        context_terms=DEFAULT_CONTEXT_TERMS,
        contexts_per_seed=max(1, Int(getproperty(row, :contexts)) ÷ 3),
        scenarios_per_context=Int(getproperty(row, :scenarios_per_context)),
        evaluation_batches=Int(getproperty(row, :test_evaluation_batches)),
        mu=DEFAULT_MU,
        rho=DEFAULT_RHO,
    )
end

function reference_context_row(row)
    return (;
        demand_power=2.0,
        p_label="p_2_reference",
        seed=string(getproperty(row, :source_seed)),
        sample_index=Int(getproperty(row, :sample_index)),
        source_seed=string(getproperty(row, :source_seed)),
        source_index=Int(getproperty(row, :source_index)),
        context=String(getproperty(row, :context)),
        average_demand=String(getproperty(row, :average_demand)),
        average_scenario_decision=String(getproperty(row, :average_scenario_decision)),
        average_scenario_policy_value=Float64(getproperty(row, :average_scenario_policy_value)),
        stochastic_optimal_value=Float64(getproperty(row, :stochastic_optimal_value)),
        gap=Float64(getproperty(row, :gap)),
        relative_gap=Float64(getproperty(row, :relative_gap)),
        policy_collection_values=String(getproperty(row, :policy_collection_values)),
        optimal_collection_values=String(getproperty(row, :optimal_collection_values)),
        gap_values=String(getproperty(row, :gap_values)),
        gap_std=Float64(getproperty(row, :gap_std)),
        gap_stderr=Float64(getproperty(row, :gap_stderr)),
    )
end

function reference_summary_row(row, context_rows, output_dir)
    config = reference_config_from_row(row)
    return summary_row(
        p=2.0,
        summary_level="reference",
        seed="reference",
        seeds=String(getproperty(row, :seeds)),
        config=config,
        rows=context_rows,
        decision_solve_seconds=Float64(getproperty(row, :decision_solve_seconds)),
        policy_eval_seconds=Float64(getproperty(row, :test_policy_eval_seconds)),
        artifact_dir=output_dir,
        dataset_digest=String(getproperty(row, :source_dataset_digests)),
        optimal_results_digest=String(getproperty(row, :source_optimal_digests)),
        source="reference_p2",
    )
end

function include_reference!(config, all_summaries, all_context_rows, log_io)
    if !(isfile(REFERENCE_SUMMARY_PATH) && isfile(REFERENCE_PER_CONTEXT_PATH))
        log_message(log_io, "skipping p=2 reference because prior summary/per_context files were not found")
        return
    end

    reference_dir = joinpath(config.output_dir, "p_2_reference")
    mkpath(reference_dir)
    cp(REFERENCE_SUMMARY_PATH, joinpath(reference_dir, "summary_source.csv"); force=true)
    cp(REFERENCE_PER_CONTEXT_PATH, joinpath(reference_dir, "per_context_source.csv"); force=true)
    isfile(REFERENCE_REPORT_PATH) &&
        cp(REFERENCE_REPORT_PATH, joinpath(reference_dir, "report_source.md"); force=true)

    source_summary = only(read_namedtuple_rows(REFERENCE_SUMMARY_PATH))
    reference_rows = [reference_context_row(row) for row in CSV.File(REFERENCE_PER_CONTEXT_PATH)]
    reference_summary = reference_summary_row(source_summary, reference_rows, reference_dir)

    CSV.write(joinpath(reference_dir, "summary.csv"), [reference_summary])
    CSV.write(joinpath(reference_dir, "per_context.csv"), reference_rows)

    push!(all_summaries, reference_summary)
    append!(all_context_rows, reference_rows)
    log_message(log_io, "included p=2 reference from $REFERENCE_DIR")
end

function write_report(config, summaries)
    path = joinpath(config.output_dir, "report.md")
    generated_at = Dates.format(now(UTC), dateformat"yyyy-mm-ddTHH:MM:SS.sZ")
    open(path, "w") do io
        println(io, "# Resource-Allocation Average-Scenario VSS p Sweep")
        println(io)
        println(io, "This run regenerates resource-allocation test data for demand powers `p = $(join(config.p_values, ", "))`, solves benchmark optima, and evaluates one average-scenario decision per context.")
        println(io)
        println(io, "| p | source | contexts | scenarios/context | batches | mean gap | mean relative gap | eval seconds |")
        println(io, "|---:|:---|---:|---:|---:|---:|---:|---:|")
        for row in summaries
            println(
                io,
                "| $(row.demand_power) | $(row.source) | $(row.contexts) | $(row.scenarios_per_context) | $(row.evaluation_batches) | $(row.test_regret_mean) | $(row.test_relative_regret_mean) | $(row.total_eval_seconds) |",
            )
        end
        println(io)
        println(io, "Generated at $generated_at.")
    end
    return path
end

function write_config_csv(config)
    row = (;
        smoke=Bool(config.smoke),
        output_dir=String(config.output_dir),
        p_values=join(config.p_values, "|"),
        seeds=join(config.seeds, "|"),
        contexts_per_seed=Int(config.contexts_per_seed),
        scenarios_per_context=Int(config.scenarios_per_context),
        evaluation_batches=Int(config.evaluation_batches),
        demand_sigma=Float64(config.sigma),
        context_terms=Int(config.context_terms),
        mu=Float64(config.mu),
        rho=Float64(config.rho),
        include_reference=Bool(config.include_reference),
    )
    CSV.write(joinpath(config.output_dir, "run_config.csv"), [row])
end

function main(args=ARGS)
    config = run_config(args)
    mkpath(config.output_dir)
    write_config_csv(config)

    log_path = joinpath(config.output_dir, "run.log")
    open(log_path, config.force ? "w" : "a") do log_io
        log_message(log_io, "starting sweep output_dir=$(config.output_dir)")
        objects = problem_objects()
        all_summaries = NamedTuple[]
        all_seed_summaries = NamedTuple[]
        all_context_rows = NamedTuple[]

        for p in config.p_values
            log_message(log_io, "starting p=$p")
            seed_summaries = NamedTuple[]
            p_context_rows = NamedTuple[]
            mkpath(p_dir(config, p))

            for seed in config.seeds
                log_message(log_io, "starting p=$p seed=$seed")
                mkpath(seed_dir(config, p, seed))
                dataset_payload, dataset = ensure_dataset!(objects, config, p, seed, log_io)
                optimal_payload, optimal_results = ensure_optimal_results!(
                    objects,
                    config,
                    p,
                    seed,
                    dataset_payload,
                    dataset,
                    log_io,
                )
                seed_summary, seed_rows = ensure_evaluation!(
                    objects,
                    config,
                    p,
                    seed,
                    dataset_payload,
                    dataset,
                    optimal_payload,
                    optimal_results,
                    log_io,
                )
                push!(seed_summaries, seed_summary)
                push!(all_seed_summaries, seed_summary)
                append!(p_context_rows, seed_rows)
            end

            p_summary = aggregate_p_outputs!(config, p, seed_summaries, p_context_rows)
            push!(all_summaries, p_summary)
            append!(all_context_rows, p_context_rows)
            log_message(log_io, "finished p=$p mean relative gap $(p_summary.test_relative_regret_mean)")
        end

        !config.smoke && config.include_reference &&
            include_reference!(config, all_summaries, all_context_rows, log_io)

        all_summary_path = joinpath(config.output_dir, "all_summary.csv")
        all_seed_summary_path = joinpath(config.output_dir, "all_seed_summary.csv")
        all_per_context_path = joinpath(config.output_dir, "all_per_context.csv")
        CSV.write(all_summary_path, all_summaries)
        CSV.write(all_seed_summary_path, all_seed_summaries)
        CSV.write(all_per_context_path, all_context_rows)
        report_path = write_report(config, all_summaries)

        log_message(log_io, "wrote $all_summary_path")
        log_message(log_io, "wrote $all_per_context_path")
        log_message(log_io, "wrote $report_path")
        log_message(log_io, "finished sweep")
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
