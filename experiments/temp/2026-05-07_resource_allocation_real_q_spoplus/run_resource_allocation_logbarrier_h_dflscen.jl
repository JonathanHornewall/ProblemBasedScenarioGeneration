using ContextualDFL
using ContextualDFLExperiments
using Dates
using LinearAlgebra
using Random
using Serialization
using Sockets
using Statistics

const Flux = ContextualDFL.Flux
const DEFAULT_OUTPUT_DIR = joinpath(@__DIR__, "results_logbarrier_h_dflscen")

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
        conversion_delta=parse(Float64, get(options, "conversion-delta", "1e-3")),
        ipopt_max_iter=parse(Int, get(options, "ipopt-max-iter", "10000")),
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

function dfl_standard_schedule(epochs::Integer; rho::Real=0.0)
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
                    rho_in=Float64(rho),
                    rho_ref=Float64(rho),
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
                rho_in=Float64(rho),
                rho_ref=Float64(rho),
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
        return dfl_standard_schedule(options.epochs; rho=options.train_rho)
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

function transformed_cost_base_scenario(base, qmax_delta::AbstractVector)
    length(qmax_delta) == length(base.q) ||
        throw(DimensionMismatch("qmax_delta has the wrong length."))
    return ContextualDFL.ParametricScenario(;
        W_eq_xi=copy(base.W_eq),
        W_ineq_xi=copy(base.W_ineq),
        T_eq_xi=copy(base.T_eq),
        T_ineq_xi=copy(base.T_ineq),
        h_eq_xi=copy(base.h_eq),
        h_ineq_xi=copy(base.h_ineq),
        q_xi=Float64.(qmax_delta),
    )
end

function normalized_probabilities(K::Integer, probabilities)
    K > 0 || throw(ArgumentError("K must be positive."))
    probabilities === nothing && return fill(1.0 / K, K)

    length(probabilities) == K ||
        throw(DimensionMismatch("probabilities must have one entry per scenario."))
    p = Float64.(collect(probabilities))
    all(isfinite, p) || throw(ArgumentError("probabilities must be finite."))
    all(>=(0.0), p) || throw(ArgumentError("probabilities must be nonnegative."))
    total = sum(p)
    total > 0.0 || throw(ArgumentError("probabilities must have positive sum."))
    return p ./ total
end

function convert_datapoint_to_logbarrier_h(
    datapoint::ContextualDFL.ContextualDataPoint,
    solver,
    program,
    original_decoder,
    transformed_base_scenario::ContextualDFL.ParametricScenario;
    probabilities=nothing,
    conversion_mu::Real,
    conversion_rho::Real,
    conversion_delta::Real,
    atol::Real=1e-10,
    solve_kwargs...,
)
    conversion_mu > 0.0 ||
        throw(ArgumentError("conversion_mu must be positive for the log-barrier h transform."))
    conversion_delta > 0.0 ||
        throw(ArgumentError("conversion_delta must be positive."))

    arrays = ContextualDFLExperiments.decode_q_conversion_arrays(
        original_decoder,
        datapoint.scenario_parameters;
        atol=atol,
    )
    K = size(arrays.q_array, 2)
    p = normalized_probabilities(K, probabilities)

    z_star,
    _,
    _,
    _,
    lambda_h_eq_array,
    _ = ContextualDFL.solve(
        solver,
        program,
        arrays.W_eq_array,
        arrays.W_ineq_array,
        arrays.T_eq_array,
        arrays.T_ineq_array,
        arrays.h_eq_array,
        arrays.h_ineq_array,
        arrays.q_array;
        probabilities=p,
        μ=conversion_mu,
        ρ=conversion_rho,
        solve_kwargs...,
    )

    base = ContextualDFLExperiments.full_base_scenario_arrays(
        transformed_base_scenario;
        atol=atol,
    )
    q_support_max = vec(maximum(arrays.q_array; dims=2))
    dominance_margin = minimum(base.q .- q_support_max)
    dominance_margin >= conversion_delta - 1e-8 ||
        throw(ArgumentError(
            "transformed base q does not dominate support q by conversion_delta; " *
            "margin=$(dominance_margin), delta=$(conversion_delta).",
        ))

    lambda_bar = vec(sum(lambda_h_eq_array; dims=2))
    residual = vec(base.q .- transpose(base.W_eq) * lambda_bar)
    min_residual = minimum(residual)
    min_residual > 0.0 ||
        throw(ArgumentError("log-barrier h residual must be positive; min=$(min_residual)."))

    y_star = Float64(conversion_mu) ./ residual
    h_star = vec(base.T_eq * z_star .+ base.W_eq * y_star)
    all(isfinite, h_star) || throw(ArgumentError("converted h label must be finite."))

    scenario_star = ContextualDFL.ParametricScenario(;
        W_eq_xi=copy(base.W_eq),
        W_ineq_xi=copy(base.W_ineq),
        T_eq_xi=copy(base.T_eq),
        T_ineq_xi=copy(base.T_ineq),
        h_eq_xi=copy(h_star),
        h_ineq_xi=copy(base.h_ineq),
        q_xi=copy(base.q),
    )

    diagnostics = (;
        z_star=Float64.(z_star),
        residual_min=Float64(min_residual),
        residual_max=Float64(maximum(residual)),
        y_min=Float64(minimum(y_star)),
        y_max=Float64(maximum(y_star)),
        h_min=Float64(minimum(h_star)),
        h_max=Float64(maximum(h_star)),
        dominance_margin=Float64(dominance_margin),
    )
    return ContextualDFL.ContextualDataPoint(datapoint.context, [scenario_star]), diagnostics
end

function convert_dataset_to_logbarrier_h(
    dataset,
    solver,
    program,
    original_decoder,
    transformed_base_scenario;
    probabilities_by_datapoint=nothing,
    kwargs...,
)
    converted = ContextualDFL.ContextualDataPoint[]
    diagnostics = NamedTuple[]
    for (index, dp) in enumerate(dataset)
        probs =
            probabilities_by_datapoint === nothing ? nothing :
            probabilities_by_datapoint isa Function ? probabilities_by_datapoint(dp) :
            probabilities_by_datapoint[index]
        converted_dp, diag = convert_datapoint_to_logbarrier_h(
            dp,
            solver,
            program,
            original_decoder,
            transformed_base_scenario;
            probabilities=probs,
            kwargs...,
        )
        push!(converted, converted_dp)
        push!(diagnostics, diag)
    end
    return converted, diagnostics
end

function reproduction_errors(
    converted_dataset,
    diagnostics,
    solver,
    program;
    conversion_mu::Real,
    conversion_rho::Real,
    checks::Integer,
    constraint_tolerance::Real,
    ipopt_max_iter::Integer,
)
    check_count = min(Int(checks), length(converted_dataset))
    errors = Float64[]
    for index in 1:check_count
        scenario = only(converted_dataset[index].scenario_parameters)
        W_eq, W_ineq, T_eq, T_ineq, h_eq, h_ineq, q =
            ContextualDFL.decode_scenario_collection(ContextualDFL.ParametricDecoder(), [scenario])
        z_converted, _, _, _, _, _ = ContextualDFL.solve(
            solver,
            program,
            W_eq,
            W_ineq,
            T_eq,
            T_ineq,
            h_eq,
            h_ineq,
            q;
            μ=conversion_mu,
            ρ=conversion_rho,
            constraint_tolerance=constraint_tolerance,
            max_iter=ipopt_max_iter,
        )
        push!(errors, norm(Float64.(z_converted) .- diagnostics[index].z_star))
    end
    isempty(errors) && return (; count=0, mean=NaN, max=NaN)
    return (; count=length(errors), mean=mean(errors), max=maximum(errors))
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

function h_prediction_error(model, decoder, converted_dataset)
    errors = Float64[]
    for dp in converted_dataset
        predicted = vec(decoder(vec(model(dp.context)))[5])
        reference = only(dp.scenario_parameters).h_eq_xi
        length(predicted) == length(reference) ||
            throw(DimensionMismatch("model output and h label have different lengths."))
        push!(errors, sqrt(mean(abs2, predicted .- reference)))
    end
    return (;
        rmse_mean=mean(errors),
        rmse_min=minimum(errors),
        rmse_max=maximum(errors),
    )
end

function generate_policy_decisions(
    model,
    decoder,
    solver,
    program,
    test_data;
    mu::Real,
    rho::Real,
    constraint_tolerance::Real,
    ipopt_max_iter::Integer,
)
    decisions = Vector{Vector{Float64}}(undef, length(test_data))
    for (index, dp) in enumerate(test_data)
        raw_h = vec(model(dp.context))
        W_eq, W_ineq, T_eq, T_ineq, h_eq, h_ineq, q =
            ContextualDFL.decode_scenario_collection(decoder, raw_h; nr_scenarios=1)
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
            μ=mu,
            ρ=rho,
            constraint_tolerance=constraint_tolerance,
            max_iter=ipopt_max_iter,
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
        println(io, "epoch,train_epoch_loss,train_full_loss,h_rmse_mean,epoch_seconds")
        for row in rows
            println(
                io,
                join(
                    (
                        row.epoch,
                        row.train_epoch_loss,
                        row.train_full_loss,
                        row.h_rmse_mean,
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
    original_base = ContextualDFLExperiments.base_scenario(problem)
    qmax_delta = Float64.(original_base.q .+ options.conversion_delta)
    transformed_base = transformed_cost_base_scenario(original_base, qmax_delta)
    original_decoder = ContextualDFLExperiments.ResourceAllocationDemandParametricDecoder(problem)

    logmsg("run_id=$(options.run_id) host=$(Sockets.gethostname()) pid=$(getpid())")
    logmsg(
        "conversion_mu=$(options.conversion_mu) conversion_rho=$(options.conversion_rho) " *
        "conversion_delta=$(options.conversion_delta) " *
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
    logmsg("converting $(length(source_data)) $(options.train_source) datapoints to log-barrier decision-optimal h labels")
    local converted_dataset, conversion_diagnostics
    conversion_seconds = @elapsed begin
        converted_dataset, conversion_diagnostics = convert_dataset_to_logbarrier_h(
            source_data,
            solver,
            program,
            original_decoder,
            transformed_base;
            conversion_mu=options.conversion_mu,
            conversion_rho=options.conversion_rho,
            conversion_delta=options.conversion_delta,
            constraint_tolerance=options.constraint_tolerance,
            max_iter=options.ipopt_max_iter,
        )
    end
    h_dim = length(first(converted_dataset).scenario_parameters[1].h_eq_xi)
    decision_decoder = ContextualDFLExperiments.DecisionOptimalHDecoder(transformed_base)
    reference_decoder = ContextualDFL.ParametricDecoder()
    reproduction = reproduction_errors(
        converted_dataset,
        conversion_diagnostics,
        solver,
        program;
        conversion_mu=options.conversion_mu,
        conversion_rho=options.conversion_rho,
        checks=min(5, length(converted_dataset)),
        constraint_tolerance=options.constraint_tolerance,
        ipopt_max_iter=options.ipopt_max_iter,
    )
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
    residual_mins = Float64[diag.residual_min for diag in conversion_diagnostics]
    residual_maxs = Float64[diag.residual_max for diag in conversion_diagnostics]
    y_mins = Float64[diag.y_min for diag in conversion_diagnostics]
    y_maxs = Float64[diag.y_max for diag in conversion_diagnostics]
    h_mins = Float64[diag.h_min for diag in conversion_diagnostics]
    h_maxs = Float64[diag.h_max for diag in conversion_diagnostics]
    logmsg(
        "converted h dataset count=$(length(converted_dataset)) h_dim=$(h_dim) " *
        "residual_min=$(minimum(residual_mins)) residual_max=$(maximum(residual_maxs)) " *
        "y_min=$(minimum(y_mins)) y_max=$(maximum(y_maxs)) " *
        "h_min=$(minimum(h_mins)) h_max=$(maximum(h_maxs)) " *
        "reproduction_checks=$(reproduction.count) reproduction_z_norm_mean=$(reproduction.mean) " *
        "reproduction_z_norm_max=$(reproduction.max) " *
        "seconds=$(round(conversion_seconds; digits=3))",
    )
    write_key_value_csv(
        joinpath(run_dir, "conversion_summary.csv"),
        [
            "run_id" => options.run_id,
            "seed" => string(options.seed),
            "train_contexts" => string(options.train_contexts),
            "train_scenarios_per_context" => string(options.train_scenarios_per_context),
            "conversion_mu" => string(options.conversion_mu),
            "conversion_rho" => string(options.conversion_rho),
            "conversion_delta" => string(options.conversion_delta),
            "h_dim" => string(h_dim),
            "qmax_delta_min" => string(minimum(qmax_delta)),
            "qmax_delta_max" => string(maximum(qmax_delta)),
            "residual_min" => string(minimum(residual_mins)),
            "residual_max" => string(maximum(residual_maxs)),
            "y_min" => string(minimum(y_mins)),
            "y_max" => string(maximum(y_maxs)),
            "h_label_min" => string(minimum(h_mins)),
            "h_label_max" => string(maximum(h_maxs)),
            "reproduction_check_count" => string(reproduction.count),
            "reproduction_z_norm_mean" => string(reproduction.mean),
            "reproduction_z_norm_max" => string(reproduction.max),
            "conversion_seconds" => string(conversion_seconds),
        ],
    )

    model = build_model(
        length(first(source_data).context),
        h_dim;
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
    initial_h_error = h_prediction_error(model, decision_decoder, converted_dataset)
    logmsg("initial_train_loss=$(initial_loss) initial_h_rmse_mean=$(initial_h_error.rmse_mean)")

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
            h_error = h_prediction_error(model, decision_decoder, converted_dataset)
            push!(
                epoch_rows,
                (;
                    epoch=epoch,
                    train_epoch_loss=Float64(train_epoch_loss),
                    train_full_loss=Float64(full_loss),
                    h_rmse_mean=Float64(h_error.rmse_mean),
                    epoch_seconds=Float64(metadata.epoch_seconds),
                ),
            )
            logmsg(
                "epoch=$(epoch) train_epoch_loss=$(train_epoch_loss) " *
                "train_full_loss=$(full_loss) h_rmse_mean=$(h_error.rmse_mean)",
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
    final_h_error = h_prediction_error(model, decision_decoder, converted_dataset)
    logmsg("final_train_loss=$(final_loss) final_h_rmse_mean=$(final_h_error.rmse_mean) train_seconds=$(round(train_seconds; digits=3))")

    logmsg("solving policy decisions for test contexts")
    decision_seconds = @elapsed decision_set = generate_policy_decisions(
        model,
        decision_decoder,
        solver,
        program,
        test_data;
        mu=options.policy_mu,
        rho=options.policy_rho,
        constraint_tolerance=options.constraint_tolerance,
        ipopt_max_iter=options.ipopt_max_iter,
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

    logmsg("evaluating log-barrier-h DFLScen policy against test optima")
    evaluation_seconds = @elapsed comparison =
        ContextualDFLExperiments.evaluate_policy_against_optimum(
            decision_set,
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
                "conversion_delta" => string(options.conversion_delta),
                "first_mu_in" => string(first(mu_in_schedule)),
                "last_mu_in" => string(last(mu_in_schedule)),
                "first_mu_ref" => string(first(mu_ref_schedule)),
                "last_mu_ref" => string(last(mu_ref_schedule)),
                "ipopt_max_iter" => string(options.ipopt_max_iter),
                "h_dim" => string(h_dim),
                "qmax_delta_min" => string(minimum(qmax_delta)),
                "qmax_delta_max" => string(maximum(qmax_delta)),
                "residual_min" => string(minimum(residual_mins)),
                "residual_max" => string(maximum(residual_maxs)),
                "y_min" => string(minimum(y_mins)),
                "y_max" => string(maximum(y_maxs)),
                "h_label_min" => string(minimum(h_mins)),
                "h_label_max" => string(maximum(h_maxs)),
                "reproduction_check_count" => string(reproduction.count),
                "reproduction_z_norm_mean" => string(reproduction.mean),
                "reproduction_z_norm_max" => string(reproduction.max),
                "initial_train_loss" => string(initial_loss),
                "final_train_loss" => string(final_loss),
                "initial_h_rmse_mean" => string(initial_h_error.rmse_mean),
                "final_h_rmse_mean" => string(final_h_error.rmse_mean),
                "conversion_seconds" => string(conversion_seconds),
                "train_seconds" => string(train_seconds),
                "decision_seconds" => string(decision_seconds),
                "optimal_seconds" => string(optimal_seconds),
                "evaluation_seconds" => string(evaluation_seconds),
            ],
            flatten_named_tuple(metrics),
        ),
    )
    Serialization.serialize(
        joinpath(run_dir, "result.jls"),
        (; options, result, metrics, epoch_rows, conversion_diagnostics),
    )
    logmsg("wrote results to $(run_dir)")
    return nothing
end

main()
