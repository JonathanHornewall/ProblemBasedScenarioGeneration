using ContextualDFL
using ContextualDFLExperiments
using Dates
using Flux
using Random
using Serialization
using Sockets
using Statistics

const DEFAULT_OUTPUT_DIR = joinpath(@__DIR__, "results")

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
    return (;
        run_id=run_id,
        seed=seed,
        train_contexts=parse(Int, get(options, "train-contexts", "20")),
        train_scenarios_per_context=parse(Int, get(options, "train-scenarios-per-context", "5")),
        test_contexts=parse(Int, get(options, "test-contexts", "10")),
        test_scenarios_per_context=parse(Int, get(options, "test-scenarios-per-context", "100")),
        epochs=parse(Int, get(options, "epochs", "5")),
        batchsize=parse(Int, get(options, "batchsize", "1")),
        hidden_dim=parse(Int, get(options, "hidden-dim", "128")),
        depth=parse(Int, get(options, "depth", "3")),
        learning_rate=parse(Float64, get(options, "learning-rate", "1e-3")),
        rho=parse(Float64, get(options, "rho", "0.0")),
        lower_bound_margin=parse(Float64, get(options, "lower-bound-margin", "1e-4")),
        constraint_tolerance=parse(Float64, get(options, "constraint-tolerance", "1e-8")),
        output_dir=abspath(get(options, "output-dir", DEFAULT_OUTPUT_DIR)),
    )
end

function full_parametric_scenario(base)
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

function mean_spoplus_loss(loss, model, data; constraint_tolerance, rho)
    isempty(data) && throw(ArgumentError("data must not be empty."))
    return mean(
        loss(
            model(dp.context),
            dp.scenario_parameters,
            0.0,
            0.0;
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

function generate_policy_decisions(model, decoder, solver, program, test_data; constraint_tolerance, rho)
    decisions = Vector{Vector{Float64}}(undef, length(test_data))
    for (index, dp) in enumerate(test_data)
        raw_q = vec(model(dp.context))
        W_eq, W_ineq, T_eq, T_ineq, h_eq, h_ineq, q =
            ContextualDFL.decode_scenario_collection(decoder, raw_q; nr_scenarios=1)
        z, _, _, _, _, _ = ContextualDFL.solve(
            solver,
            program,
            W_eq,
            W_ineq,
            T_eq,
            T_ineq,
            h_eq,
            h_ineq,
            q;
            ρ=rho,
            constraint_tolerance=constraint_tolerance,
        )
        decisions[index] = Float64.(z)
    end

    decision_dim = length(first(decisions))
    decision_set = zeros(Float64, decision_dim, length(decisions))
    for index in eachindex(decisions)
        decision_set[:, index] = decisions[index]
    end
    return decision_set
end

function flatten_named_tuple(nt)
    rows = Pair{String,String}[]
    for name in propertynames(nt)
        value = getproperty(nt, name)
        push!(rows, string(name) => string(value))
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
        println(io, "epoch,train_epoch_loss,train_full_loss,epoch_seconds")
        for row in rows
            println(
                io,
                join(
                    (
                        row.epoch,
                        row.train_epoch_loss,
                        row.train_full_loss,
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
    base = full_parametric_scenario(ContextualDFLExperiments.base_scenario(problem))
    original_decoder = ContextualDFLExperiments.ResourceAllocationDemandParametricDecoder(problem)

    logmsg("run_id=$(options.run_id) host=$(Sockets.gethostname()) pid=$(getpid())")
    logmsg("generating train data contexts=$(options.train_contexts) scenarios_per_context=$(options.train_scenarios_per_context)")
    train_data = ContextualDFLExperiments.generate_benchmark_dataset(
        problem;
        n_contexts=options.train_contexts,
        scenarios_per_context=options.train_scenarios_per_context,
        seed=options.seed,
    )

    logmsg("converting train data to decision-optimal q labels")
    prepare_seconds = @elapsed prepared = ContextualDFLExperiments.prepare_spoplus_q_dataset(
        train_data,
        solver,
        program,
        original_decoder,
        base;
        lower_bound_margin=options.lower_bound_margin,
        constraint_tolerance=options.constraint_tolerance,
    )
    q_dim = length(prepared.q_lower_bound)
    decision_decoder = ContextualDFLExperiments.LowerBoundedQDecoder(base, prepared.q_lower_bound)
    logmsg("prepared q dataset count=$(length(prepared.converted_dataset)) q_dim=$(q_dim) seconds=$(round(prepare_seconds; digits=3))")

    model = build_model(
        length(first(train_data).context),
        q_dim;
        hidden_dim=options.hidden_dim,
        depth=options.depth,
    )
    initial_loss = mean_spoplus_loss(
        prepared.spo_loss,
        model,
        prepared.converted_dataset;
        constraint_tolerance=options.constraint_tolerance,
        rho=options.rho,
    )
    initial_q_error = q_prediction_error(model, decision_decoder, prepared.converted_dataset)
    logmsg("initial_train_loss=$(initial_loss) initial_q_rmse_mean=$(initial_q_error.rmse_mean)")

    epoch_rows = NamedTuple[]
    train_seconds = @elapsed result = ContextualDFL.train!(
        model,
        prepared.spo_loss,
        nothing,
        fill(0.0, options.epochs),
        prepared.converted_dataset;
        epochs=options.epochs,
        batchsize=options.batchsize,
        learning_rate=options.learning_rate,
        display_iterations=true,
        display_plot=false,
        nr_scenarios=1,
        rho_in_schedule=fill(options.rho, options.epochs),
        rho_ref_schedule=fill(options.rho, options.epochs),
        on_epoch_end=(epoch, train_epoch_loss, _, metadata) -> begin
            full_loss = mean_spoplus_loss(
                prepared.spo_loss,
                model,
                prepared.converted_dataset;
                constraint_tolerance=options.constraint_tolerance,
                rho=options.rho,
            )
            push!(
                epoch_rows,
                (;
                    epoch=epoch,
                    train_epoch_loss=Float64(train_epoch_loss),
                    train_full_loss=Float64(full_loss),
                    epoch_seconds=Float64(metadata.epoch_seconds),
                ),
            )
            logmsg("epoch=$(epoch) train_epoch_loss=$(train_epoch_loss) train_full_loss=$(full_loss)")
        end,
    )
    final_loss = mean_spoplus_loss(
        prepared.spo_loss,
        model,
        prepared.converted_dataset;
        constraint_tolerance=options.constraint_tolerance,
        rho=options.rho,
    )
    final_q_error = q_prediction_error(model, decision_decoder, prepared.converted_dataset)
    logmsg("final_train_loss=$(final_loss) final_q_rmse_mean=$(final_q_error.rmse_mean) train_seconds=$(round(train_seconds; digits=3))")

    logmsg("generating test data contexts=$(options.test_contexts) scenarios_per_context=$(options.test_scenarios_per_context)")
    test_data = ContextualDFLExperiments.generate_benchmark_dataset(
        problem;
        n_contexts=options.test_contexts,
        scenarios_per_context=options.test_scenarios_per_context,
        seed=options.seed + 10_000,
    )

    logmsg("solving policy decisions for test contexts")
    decision_seconds = @elapsed decision_set = generate_policy_decisions(
        model,
        decision_decoder,
        solver,
        program,
        test_data;
        constraint_tolerance=options.constraint_tolerance,
        rho=options.rho,
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

    logmsg("evaluating policy against optima")
    evaluation_seconds = @elapsed comparison =
        ContextualDFLExperiments.evaluate_policy_against_optimum(
            decision_set,
            test_data,
            program,
            original_decoder,
            solver;
            optimal_results=optimal_results,
            split_name=:test,
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
                "train_contexts" => string(options.train_contexts),
                "train_scenarios_per_context" => string(options.train_scenarios_per_context),
                "test_contexts" => string(options.test_contexts),
                "test_scenarios_per_context" => string(options.test_scenarios_per_context),
                "epochs" => string(options.epochs),
                "batchsize" => string(options.batchsize),
                "hidden_dim" => string(options.hidden_dim),
                "depth" => string(options.depth),
                "learning_rate" => string(options.learning_rate),
                "rho" => string(options.rho),
                "q_dim" => string(q_dim),
                "q_lower_bound_min" => string(minimum(prepared.q_lower_bound)),
                "q_lower_bound_max" => string(maximum(prepared.q_lower_bound)),
                "initial_train_loss" => string(initial_loss),
                "final_train_loss" => string(final_loss),
                "initial_q_rmse_mean" => string(initial_q_error.rmse_mean),
                "final_q_rmse_mean" => string(final_q_error.rmse_mean),
                "prepare_seconds" => string(prepare_seconds),
                "train_seconds" => string(train_seconds),
                "decision_seconds" => string(decision_seconds),
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
