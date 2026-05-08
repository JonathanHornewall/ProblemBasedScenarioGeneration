using ContextualDFL
using ContextualDFLExperiments
using Dates
using Random
using Serialization
using Sockets
using Statistics

const Flux = ContextualDFL.Flux
const DEFAULT_OUTPUT_DIR = joinpath(@__DIR__, "results_dflscen_q")

function parse_args(args)
    options = Dict{String,String}()
    index = 1
    while index <= length(args)
        arg = args[index]
        startswith(arg, "--") || throw(ArgumentError("unexpected argument $(repr(arg))"))
        key_value = split(arg[3:end], "="; limit=2)
        if length(key_value) == 2
            options[key_value[1]] = key_value[2]
        else
            index += 1
            index <= length(args) ||
                throw(ArgumentError("missing value for argument $(repr(arg))"))
            options[key_value[1]] = args[index]
        end
        index += 1
    end

    seed = parse(Int, get(options, "seed", "20260507"))
    run_id = get(options, "run-id", "seed$(seed)")
    train_source = get(options, "train-source", "train")
    train_source in ("train", "test") ||
        throw(ArgumentError("--train-source must be train or test."))

    return (;
        run_id=run_id,
        seed=seed,
        train_source=train_source,
        train_contexts=parse(Int, get(options, "train-contexts", "100")),
        train_scenarios_per_context=parse(Int, get(options, "train-scenarios-per-context", "1")),
        test_contexts=parse(Int, get(options, "test-contexts", "10")),
        test_scenarios_per_context=parse(Int, get(options, "test-scenarios-per-context", "100")),
        epochs=parse(Int, get(options, "epochs", "50")),
        batchsize=parse(Int, get(options, "batchsize", "1")),
        hidden_dim=parse(Int, get(options, "hidden-dim", "128")),
        depth=parse(Int, get(options, "depth", "3")),
        learning_rate=parse(Float64, get(options, "learning-rate", "1e-3")),
        train_schedule=get(options, "train-schedule", "constant"),
        train_mu=parse(Float64, get(options, "train-mu", "0.0")),
        train_rho=parse(Float64, get(options, "train-rho", "0.0")),
        policy_mu=parse(Float64, get(options, "policy-mu", "0.0")),
        policy_rho=parse(Float64, get(options, "policy-rho", "0.0")),
        conversion_mu=parse(Float64, get(options, "conversion-mu", "1e-4")),
        conversion_rho=parse(Float64, get(options, "conversion-rho", "0.0")),
        ipopt_max_iter=parse(Int, get(options, "ipopt-max-iter", "10000")),
        lower_bound_margin=parse(Float64, get(options, "lower-bound-margin", "1e-4")),
        constraint_tolerance=parse(Float64, get(options, "constraint-tolerance", "1e-8")),
        output_dir=abspath(get(options, "output-dir", DEFAULT_OUTPUT_DIR)),
    )
end

struct LossWithSolverKwargs{TLoss,TKwargs}
    loss::TLoss
    solver_kwargs::TKwargs
end

function (wrapped::LossWithSolverKwargs)(
    input_scenario_parameter_collection,
    reference_scenario_parameter_collection,
    mu_in=0,
    mu_ref=mu_in;
    kwargs...,
)
    return wrapped.loss(
        input_scenario_parameter_collection,
        reference_scenario_parameter_collection,
        mu_in,
        mu_ref;
        merge((; kwargs...), wrapped.solver_kwargs)...,
    )
end

function dfl_standard_schedule(epochs::Integer)
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
                    mu_in=Float64(mu_value),
                    mu_ref=Float64(mu_value),
                    rho_in=0.0,
                    rho_ref=0.0,
                ),
            )
        end
    end

    final_mu = Float64(last(anneal_values))
    for _ in 1:last(epoch_counts)
        global_epoch += 1
        push!(
            rows,
            (;
                epoch=global_epoch,
                mu_in=final_mu,
                mu_ref=0.0,
                rho_in=0.0,
                rho_ref=0.0,
            ),
        )
    end

    epochs <= length(rows) ||
        throw(ArgumentError("standard DFL schedule supports at most $(length(rows)) epochs."))
    return rows[1:Int(epochs)]
end

function training_schedule(options)
    if options.train_schedule == "constant"
        return [
            (;
                epoch=epoch,
                mu_in=options.train_mu,
                mu_ref=options.train_mu,
                rho_in=options.train_rho,
                rho_ref=options.train_rho,
            ) for epoch in 1:options.epochs
        ]
    elseif options.train_schedule == "standard"
        return dfl_standard_schedule(options.epochs)
    end

    throw(ArgumentError("--train-schedule must be constant or standard."))
end

function unit_demand_base_scenario(problem, base)
    h_eq = copy(base.h_eq)
    resource_count = length(problem.problem_data.first_stage_costs)
    demand_count = length(problem.problem_data.second_stage_costs)
    h_eq[1:resource_count] .= 0.0
    h_eq[(resource_count + 1):(resource_count + demand_count)] .= 1.0

    return ContextualDFL.ParametricScenario(;
        W_eq_xi=copy(base.W_eq),
        W_ineq_xi=copy(base.W_ineq),
        T_eq_xi=copy(base.T_eq),
        T_ineq_xi=copy(base.T_ineq),
        h_eq_xi=h_eq,
        h_ineq_xi=copy(base.h_ineq),
        q_xi=copy(base.q),
    )
end

function build_model(input_dim::Integer, output_dim::Integer; hidden_dim::Integer, depth::Integer)
    input_dim > 0 || throw(ArgumentError("input_dim must be positive."))
    output_dim > 0 || throw(ArgumentError("output_dim must be positive."))
    depth > 0 || throw(ArgumentError("depth must be positive."))

    layers = Any[Flux.Dense(input_dim => hidden_dim, Flux.relu)]
    for _ in 2:depth
        push!(layers, Flux.Dense(hidden_dim => hidden_dim, Flux.relu))
    end
    push!(layers, Flux.Dense(hidden_dim => output_dim))
    return Flux.Chain(layers...) |> Flux.f64
end

function mean_dfl_loss(loss, model, data; mu, rho, constraint_tolerance)
    isempty(data) && throw(ArgumentError("data must not be empty."))
    return mean(
        loss(
            model(dp.context),
            dp.scenario_parameters,
            mu,
            mu;
            rho_in=rho,
            rho_ref=rho,
            constraint_tolerance=constraint_tolerance,
        ) for dp in data
    )
end

function q_prediction_error(model, decoder, converted_dataset)
    errors = Float64[]
    for dp in converted_dataset
        predicted = vec(decoder(vec(model(dp.context)))[7])
        reference = only(dp.scenario_parameters).q_xi
        length(predicted) == length(reference) ||
            throw(DimensionMismatch("model output and q label have different lengths."))
        push!(errors, sqrt(mean(abs2, predicted .- reference)))
    end
    return (;
        rmse_mean=mean(errors),
        rmse_min=minimum(errors),
        rmse_max=maximum(errors),
    )
end

function flatten_named_tuple(nt)
    rows = Pair{String,String}[]
    for name in propertynames(nt)
        push!(rows, string(name) => string(getproperty(nt, name)))
    end
    return rows
end

function write_key_value_csv(path, pairs)
    open(path, "w") do io
        println(io, "key,value")
        for (key, value) in pairs
            println(io, key, ",", replace(value, "," => ";"))
        end
    end
end

function write_history_csv(path, rows)
    open(path, "w") do io
        println(io, "epoch,train_epoch_loss,train_full_loss,q_rmse_mean,epoch_seconds")
        for row in rows
            println(
                io,
                join(
                    (
                        row.epoch,
                        row.train_epoch_loss,
                        row.train_full_loss,
                        row.q_rmse_mean,
                        row.epoch_seconds,
                    ),
                    ",",
                ),
            )
        end
    end
end

function write_per_sample_csv(path, rows)
    open(path, "w") do io
        println(io, "sample_index,policy_value,optimal_value,regret,relative_regret,gap_std,gap_stderr")
        for row in rows
            println(
                io,
                join(
                    (
                        row.sample_index,
                        row.policy_value,
                        row.optimal_value,
                        row.regret,
                        row.relative_regret,
                        row.gap_std,
                        row.gap_stderr,
                    ),
                    ",",
                ),
            )
        end
    end
end

function logmsg(message)
    println(Dates.format(now(Dates.UTC), dateformat"yyyy-mm-ddTHH:MM:SS"), "Z ", message)
    flush(stdout)
end

function main(args=ARGS)
    options = parse_args(args)
    run_dir = joinpath(options.output_dir, options.run_id)
    mkpath(run_dir)

    Random.seed!(options.seed)
    solver = ContextualDFL.Solver(ContextualDFL.IpoptSolver(), ContextualDFL.HiGHSSolver())
    problem = ContextualDFLExperiments.ResourceAllocationProblem(
        ContextualDFLExperiments.default_resource_allocation_problem_data(),
    )
    program = ContextualDFLExperiments.stochastic_program(problem)
    base = unit_demand_base_scenario(
        problem,
        ContextualDFLExperiments.base_scenario(problem),
    )
    original_decoder = ContextualDFLExperiments.ResourceAllocationDemandParametricDecoder(problem)

    logmsg("run_id=$(options.run_id) host=$(Sockets.gethostname()) pid=$(getpid())")
    logmsg(
        "conversion_mu=$(options.conversion_mu) conversion_rho=$(options.conversion_rho) " *
        "train_mu=$(options.train_mu) train_rho=$(options.train_rho) " *
        "policy_mu=$(options.policy_mu) policy_rho=$(options.policy_rho) train_source=$(options.train_source)",
    )

    logmsg("generating train data contexts=$(options.train_contexts) scenarios_per_context=$(options.train_scenarios_per_context)")
    train_data = ContextualDFLExperiments.generate_benchmark_dataset(
        problem;
        n_contexts=options.train_contexts,
        scenarios_per_context=options.train_scenarios_per_context,
        seed=options.seed,
    )
    logmsg("generating test data contexts=$(options.test_contexts) scenarios_per_context=$(options.test_scenarios_per_context)")
    test_data = ContextualDFLExperiments.generate_benchmark_dataset(
        problem;
        n_contexts=options.test_contexts,
        scenarios_per_context=options.test_scenarios_per_context,
        seed=options.seed + 10_000,
    )

    source_data = options.train_source == "test" ? test_data : train_data
    logmsg("converting $(length(source_data)) $(options.train_source) datapoints to decision-optimal q labels")
    conversion_seconds = @elapsed converted_dataset = ContextualDFLExperiments.convert_dataset_to_q(
        source_data,
        solver,
        program,
        original_decoder,
        base;
        μ=options.conversion_mu,
        ρ=options.conversion_rho,
        constraint_tolerance=options.constraint_tolerance,
    )
    q_lower_bound = ContextualDFLExperiments.q_lower_bound_from_converted_dataset(
        converted_dataset;
        margin=options.lower_bound_margin,
    )
    q_dim = length(q_lower_bound)
    decision_decoder = ContextualDFLExperiments.LowerBoundedQDecoder(base, q_lower_bound)
    reference_decoder = ContextualDFL.ParametricDecoder()
    base_loss = ContextualDFL.DflScenLoss(
        decision_decoder,
        reference_decoder,
        solver,
        program;
        nr_scenarios=1,
    )
    loss = LossWithSolverKwargs(
        base_loss,
        (;
            max_iter=options.ipopt_max_iter,
            constraint_tolerance=options.constraint_tolerance,
        ),
    )
    logmsg(
        "converted q dataset count=$(length(converted_dataset)) q_dim=$(q_dim) " *
        "q_lb_min=$(minimum(q_lower_bound)) q_lb_max=$(maximum(q_lower_bound)) " *
        "seconds=$(round(conversion_seconds; digits=3))",
    )

    model = build_model(
        length(first(source_data).context),
        q_dim;
        hidden_dim=options.hidden_dim,
        depth=options.depth,
    )
    schedule = training_schedule(options)
    mu_in_schedule = Float64[row.mu_in for row in schedule]
    mu_ref_schedule = Float64[row.mu_ref for row in schedule]
    rho_in_schedule = Float64[row.rho_in for row in schedule]
    rho_ref_schedule = Float64[row.rho_ref for row in schedule]
    logmsg(
        "train_schedule=$(options.train_schedule) first_mu_in=$(first(mu_in_schedule)) " *
        "last_mu_in=$(last(mu_in_schedule)) first_mu_ref=$(first(mu_ref_schedule)) " *
        "last_mu_ref=$(last(mu_ref_schedule)) ipopt_max_iter=$(options.ipopt_max_iter)",
    )

    initial_loss = mean_dfl_loss(
        loss,
        model,
        converted_dataset;
        mu=first(mu_in_schedule),
        rho=first(rho_in_schedule),
        constraint_tolerance=options.constraint_tolerance,
    )
    initial_q_error = q_prediction_error(model, decision_decoder, converted_dataset)
    logmsg("initial_train_loss=$(initial_loss) initial_q_rmse_mean=$(initial_q_error.rmse_mean)")

    epoch_rows = NamedTuple[]
    train_seconds = @elapsed result = ContextualDFL.train!(
        model,
        loss,
        nothing,
        mu_in_schedule,
        converted_dataset;
        mu_ref_schedule=mu_ref_schedule,
        epochs=options.epochs,
        batchsize=options.batchsize,
        learning_rate=options.learning_rate,
        display_iterations=true,
        display_plot=false,
        nr_scenarios=1,
        rho_in_schedule=rho_in_schedule,
        rho_ref_schedule=rho_ref_schedule,
        on_epoch_end=(epoch, train_epoch_loss, _, metadata) -> begin
            full_loss = mean_dfl_loss(
                loss,
                model,
                converted_dataset;
                mu=mu_in_schedule[epoch],
                rho=rho_in_schedule[epoch],
                constraint_tolerance=options.constraint_tolerance,
            )
            q_error = q_prediction_error(model, decision_decoder, converted_dataset)
            push!(
                epoch_rows,
                (;
                    epoch=epoch,
                    train_epoch_loss=Float64(train_epoch_loss),
                    train_full_loss=Float64(full_loss),
                    q_rmse_mean=Float64(q_error.rmse_mean),
                    epoch_seconds=Float64(metadata.epoch_seconds),
                ),
            )
            logmsg(
                "epoch=$(epoch) train_epoch_loss=$(train_epoch_loss) " *
                "train_full_loss=$(full_loss) q_rmse_mean=$(q_error.rmse_mean)",
            )
        end,
    )
    final_loss = mean_dfl_loss(
        loss,
        model,
        converted_dataset;
        mu=last(mu_in_schedule),
        rho=last(rho_in_schedule),
        constraint_tolerance=options.constraint_tolerance,
    )
    final_q_error = q_prediction_error(model, decision_decoder, converted_dataset)
    logmsg("final_train_loss=$(final_loss) final_q_rmse_mean=$(final_q_error.rmse_mean) train_seconds=$(round(train_seconds; digits=3))")

    policy = ContextualDFLExperiments.ScenarioGenerationPolicy(
        ContextualDFL.ScenarioGenerator(
            neural_net=model,
            scenario_decoder=decision_decoder,
        ),
        solver,
        program;
        mu=options.policy_mu,
        rho=options.policy_rho,
        nr_scenarios=1,
    )

    logmsg("solving test optima")
    optimal_seconds = @elapsed optimal_results =
        ContextualDFLExperiments.solve_dataset_to_optimality(
            test_data,
            program,
            original_decoder,
            solver;
            evaluation_batches=1,
            progress_io=stdout,
            progress_label=options.run_id,
            constraint_tolerance=options.constraint_tolerance,
        )

    logmsg("evaluating q-DFLScen policy against test optima")
    evaluation_seconds = @elapsed comparison =
        ContextualDFLExperiments.evaluate_policy_against_optimum(
            policy,
            test_data,
            program,
            original_decoder,
            solver;
            optimal_results=optimal_results,
            split_name=:test,
            mu=0.0,
            rho=0.0,
            constraint_tolerance=options.constraint_tolerance,
        )

    metrics = comparison.metrics
    logmsg("test_policy_value_mean=$(metrics.test_policy_value_mean)")
    logmsg("test_optimal_value_mean=$(metrics.test_optimal_value_mean)")
    logmsg("test_regret_mean=$(metrics.test_regret_mean)")
    logmsg("test_relative_regret_mean=$(metrics.test_relative_regret_mean)")

    write_history_csv(joinpath(run_dir, "history.csv"), epoch_rows)
    write_per_sample_csv(joinpath(run_dir, "per_sample.csv"), comparison.per_sample)
    write_key_value_csv(
        joinpath(run_dir, "summary.csv"),
        vcat(
            [
                "run_id" => options.run_id,
                "host" => Sockets.gethostname(),
                "pid" => string(getpid()),
                "seed" => string(options.seed),
                "train_source" => options.train_source,
                "train_contexts" => string(options.train_contexts),
                "train_scenarios_per_context" => string(options.train_scenarios_per_context),
                "test_contexts" => string(options.test_contexts),
                "test_scenarios_per_context" => string(options.test_scenarios_per_context),
                "epochs" => string(options.epochs),
                "batchsize" => string(options.batchsize),
                "hidden_dim" => string(options.hidden_dim),
                "depth" => string(options.depth),
                "learning_rate" => string(options.learning_rate),
                "train_schedule" => options.train_schedule,
                "train_mu" => string(options.train_mu),
                "train_rho" => string(options.train_rho),
                "policy_mu" => string(options.policy_mu),
                "policy_rho" => string(options.policy_rho),
                "conversion_mu" => string(options.conversion_mu),
                "conversion_rho" => string(options.conversion_rho),
                "first_mu_in" => string(first(mu_in_schedule)),
                "last_mu_in" => string(last(mu_in_schedule)),
                "first_mu_ref" => string(first(mu_ref_schedule)),
                "last_mu_ref" => string(last(mu_ref_schedule)),
                "ipopt_max_iter" => string(options.ipopt_max_iter),
                "q_dim" => string(q_dim),
                "q_lower_bound_min" => string(minimum(q_lower_bound)),
                "q_lower_bound_max" => string(maximum(q_lower_bound)),
                "initial_train_loss" => string(initial_loss),
                "final_train_loss" => string(final_loss),
                "initial_q_rmse_mean" => string(initial_q_error.rmse_mean),
                "final_q_rmse_mean" => string(final_q_error.rmse_mean),
                "conversion_seconds" => string(conversion_seconds),
                "train_seconds" => string(train_seconds),
                "optimal_seconds" => string(optimal_seconds),
                "evaluation_seconds" => string(evaluation_seconds),
            ],
            flatten_named_tuple(metrics),
        ),
    )
    Serialization.serialize(joinpath(run_dir, "result.jls"), (; options, result, metrics, epoch_rows))
    logmsg("wrote results to $(run_dir)")
    return nothing
end

main()
