import Pkg

const REPO_ROOT = abspath(joinpath(@__DIR__, "..", ".."))
const EXPERIMENT_PROJECT = joinpath(REPO_ROOT, "src", "ContextualDFL", "ContextualDFLExperiments")
const ANNEALING_EXPERIMENT_DIR = joinpath(
    EXPERIMENT_PROJECT,
    "experiments",
    "resource_allocation_annealing",
)

Pkg.activate(EXPERIMENT_PROJECT)
Pkg.instantiate()

using ContextualDFL
using ContextualDFLExperiments

import Dates
import Random
import Serialization
import SparseArrays
import Statistics

const Flux = ContextualDFL.Flux

env_int(name, default) = parse(Int, get(ENV, name, string(default)))
env_float(name, default) = parse(Float64, get(ENV, name, string(default)))
env_bool(name, default=false) = get(ENV, name, default ? "1" : "0") == "1"

function env_float_list(name, default)
    value = strip(get(ENV, name, ""))
    isempty(value) && return collect(default)
    return [parse(Float64, strip(part)) for part in split(value, ",") if !isempty(strip(part))]
end

csv_value(value) = isnothing(value) ? "" : string(value)
field_or_default(value, name::Symbol, default=nothing) =
    hasproperty(value, name) ? getproperty(value, name) : default

function write_testing_csv(path, rows)
    open(path, "w") do io
        println(io, "sample_index,policy_value,optimal_value,regret,relative_regret,ucb_percent")
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
                        row.ucb_percent,
                    ),
                    ",",
                ),
            )
        end
    end
end

function write_training_history_csv(path, stage_histories)
    open(path, "w") do io
        println(
            io,
            join(
                (
                    "stage",
                    "local_epoch",
                    "global_epoch",
                    "reg_param_surr",
                    "reg_param_prim",
                    "rho_in",
                    "rho_ref",
                    "loss",
                    "display_loss",
                    "real_display_loss",
                    "iterations",
                    "epoch_seconds",
                ),
                ",",
            ),
        )

        global_epoch = 0
        for stage in stage_histories
            for history_row in field_or_default(stage, :history, ())
                global_epoch += 1
                println(
                    io,
                    join(
                        csv_value.(
                            (
                                field_or_default(stage, :stage),
                                field_or_default(history_row, :epoch),
                                global_epoch,
                                field_or_default(stage, :reg_param_surr),
                                field_or_default(stage, :reg_param_prim),
                                field_or_default(history_row, :rho_in, field_or_default(stage, :rho_in)),
                                field_or_default(history_row, :rho_ref, field_or_default(stage, :rho_ref)),
                                field_or_default(history_row, :loss),
                                field_or_default(history_row, :display_loss),
                                field_or_default(history_row, :real_display_loss),
                                field_or_default(history_row, :iterations),
                                field_or_default(history_row, :epoch_seconds),
                            ),
                        ),
                        ",",
                    ),
                )
            end
        end
    end
end

function write_run_summary_csv(path, summary)
    keys = collect(propertynames(summary))
    open(path, "w") do io
        println(io, join(String.(keys), ","))
        println(io, join((csv_value(getproperty(summary, key)) for key in keys), ","))
    end
end

function normalized_gap_ucb_percent(policy_split_values, optimal_split_values)
    gaps = Float64.(policy_split_values) .- Float64.(optimal_split_values)
    optimal_mean = Statistics.mean(Float64.(optimal_split_values))
    gap_variance = length(gaps) > 1 ? Statistics.var(gaps) : 0.0
    ucb_gap = Statistics.mean(gaps) + 2.462 * sqrt(gap_variance / length(gaps))
    return 100 * ucb_gap / max(abs(optimal_mean), eps(Float64))
end

function testing_split_ranges(data_point, split_count)
    scenario_count = length(data_point.scenario_parameters)
    scenario_count % split_count == 0 ||
        throw(ArgumentError(
            "scenario count $scenario_count is not divisible by splits=$split_count.",
        ))

    split_size = scenario_count ÷ split_count
    return [
        ((split_index - 1) * split_size + 1):(split_index * split_size) for
        split_index in 1:split_count
    ]
end

function write_testing_partial(
    path,
    rows,
    testing_count,
    testing_splits,
    csv_path;
    active_context_index=nothing,
    optimal_split_values=Float64[],
    policy_split_values=Float64[],
)
    Serialization.serialize(
        path,
        (;
            ucb_rows=rows,
            completed_contexts=length(rows),
            testing_context_limit=testing_count,
            testing_splits=testing_splits,
            csv_path=csv_path,
            active_context_index=active_context_index,
            optimal_split_values=Float64.(collect(optimal_split_values)),
            policy_split_values=Float64.(collect(policy_split_values)),
        ),
    )
end

function resource_allocation_h_eq_matrix(problem, scenario_parameters)
    scenario = base_scenario(problem)
    resource_count = size(scenario.T_eq, 2)
    demand_count = length(scenario.h_eq) - resource_count
    h_eq_array = zeros(Float64, length(scenario.h_eq), length(scenario_parameters))

    for (index, scenario_parameter) in enumerate(scenario_parameters)
        length(scenario_parameter.h_eq_xi) == demand_count ||
            throw(DimensionMismatch("scenario demand vector must have length $demand_count."))
        h_eq_array[(resource_count + 1):end, index] = scenario_parameter.h_eq_xi
    end

    return h_eq_array
end

function resource_allocation_h_eq_vector(problem, scenario_parameter)
    h_eq_array = resource_allocation_h_eq_matrix(problem, (scenario_parameter,))
    return vec(h_eq_array)
end

function resource_allocation_extensive_lp(problem, scenario_parameters)
    program = stochastic_program(problem)
    first_stage_lp = program.first_stage_lp
    scenario = base_scenario(problem)
    h_eq_array = resource_allocation_h_eq_matrix(problem, scenario_parameters)

    split_count = size(h_eq_array, 2)
    z_count = length(first_stage_lp.c)
    y_count = length(scenario.q)
    variable_count = z_count + split_count * y_count
    first_ineq_count = length(first_stage_lp.b_ineq)
    recourse_eq_count = size(scenario.W_eq, 1)
    recourse_ineq_count = size(scenario.W_ineq, 1)

    A_eq = SparseArrays.spzeros(Float64, split_count * recourse_eq_count, variable_count)
    A_ineq = SparseArrays.spzeros(
        Float64,
        first_ineq_count + split_count * recourse_ineq_count,
        variable_count,
    )
    b_eq = zeros(Float64, split_count * recourse_eq_count)
    b_ineq = zeros(Float64, first_ineq_count + split_count * recourse_ineq_count)
    c = zeros(Float64, variable_count)

    z_cols = 1:z_count
    c[z_cols] = first_stage_lp.c
    if first_ineq_count > 0
        A_ineq[1:first_ineq_count, z_cols] = SparseArrays.sparse(first_stage_lp.A_ineq)
        b_ineq[1:first_ineq_count] = first_stage_lp.b_ineq
    end

    W_eq = SparseArrays.sparse(scenario.W_eq)
    W_ineq = SparseArrays.sparse(scenario.W_ineq)
    T_eq = SparseArrays.sparse(scenario.T_eq)
    T_ineq = SparseArrays.sparse(scenario.T_ineq)
    probability = 1.0 / split_count

    for split_index in 1:split_count
        y_cols = (z_count + (split_index - 1) * y_count + 1):(z_count + split_index * y_count)
        eq_rows = ((split_index - 1) * recourse_eq_count + 1):(split_index * recourse_eq_count)
        ineq_rows = (
            first_ineq_count + (split_index - 1) * recourse_ineq_count + 1
        ):(first_ineq_count + split_index * recourse_ineq_count)

        A_eq[eq_rows, z_cols] = T_eq
        A_eq[eq_rows, y_cols] = W_eq
        b_eq[eq_rows] = view(h_eq_array, :, split_index)

        A_ineq[ineq_rows, z_cols] = T_ineq
        A_ineq[ineq_rows, y_cols] = W_ineq
        b_ineq[ineq_rows] = scenario.h_ineq

        c[y_cols] = probability .* scenario.q
    end

    return ContextualDFL.LP(A_eq, A_ineq, b_eq, b_ineq, c)
end

function resource_allocation_extensive_rho(problem, scenario_parameters, rho)
    rho isa Number ||
        throw(ArgumentError("resource-allocation SAA testing expects scalar rho."))
    rho >= 0 || throw(ArgumentError("rho must be non-negative."))

    program = stochastic_program(problem)
    scenario = base_scenario(problem)
    split_count = length(scenario_parameters)
    z_count = length(program.first_stage_lp.c)
    y_count = length(scenario.q)
    probability = 1.0 / split_count

    rho_vector = zeros(Float64, z_count + split_count * y_count)
    rho_vector[1:z_count] .= rho
    for split_index in 1:split_count
        y_cols = (z_count + (split_index - 1) * y_count + 1):(z_count + split_index * y_count)
        rho_vector[y_cols] .= probability * rho
    end
    return rho_vector
end

function resource_allocation_optimal_split_value(problem, solver, scenario_parameters; mu, rho)
    iszero(mu) || throw(ArgumentError("resource-allocation SAA testing expects mu=0."))

    lp = resource_allocation_extensive_lp(problem, scenario_parameters)
    result = ContextualDFL.solve(
        solver,
        lp;
        μ=mu,
        ρ=resource_allocation_extensive_rho(problem, scenario_parameters, rho),
    )
    return result.objective_value
end

function resource_allocation_policy_split_value(problem, solver, z, scenario_parameters; mu, rho)
    iszero(mu) || throw(ArgumentError("resource-allocation SAA testing expects mu=0."))
    rho isa Number ||
        throw(ArgumentError("resource-allocation SAA testing expects scalar rho."))
    rho >= 0 || throw(ArgumentError("rho must be non-negative."))

    program = stochastic_program(problem)
    scenario = base_scenario(problem)
    probability = 1.0 / length(scenario_parameters)
    value = sum(program.first_stage_lp.c .* z) + 0.5 * rho * sum(abs2, z)

    for scenario_parameter in scenario_parameters
        h_eq = resource_allocation_h_eq_vector(problem, scenario_parameter)
        value += probability * ContextualDFL.G_hat(
            solver,
            z,
            scenario.W_eq,
            scenario.W_ineq,
            scenario.T_eq,
            scenario.T_ineq,
            h_eq,
            scenario.h_ineq,
            scenario.q;
            μ=mu,
            ρ=rho,
        )
    end

    return value
end

function solve_data_point_to_optimality_with_progress(
    data_point,
    problem,
    solver;
    mu,
    rho=0,
    splits,
    context_index,
    testing_count,
    partial_path,
    rows,
    csv_path,
    resume_values=Float64[],
)
    split_ranges = testing_split_ranges(data_point, splits)
    objective_values = Float64.(collect(resume_values))

    for split_index in (length(objective_values) + 1):splits
        println("Testing context $(context_index)/$(testing_count): optimal split $(split_index)/$(splits)...")
        scenario_range = split_ranges[split_index]
        objective_value = resource_allocation_optimal_split_value(
            problem,
            solver,
            view(data_point.scenario_parameters, scenario_range);
            mu=mu,
            rho=rho,
        )

        push!(objective_values, objective_value)
        write_testing_partial(
            partial_path,
            rows,
            testing_count,
            splits,
            csv_path;
            active_context_index=context_index,
            optimal_split_values=objective_values,
        )
    end

    return objective_values
end

function evaluate_policy_on_data_point_with_progress(
    policy,
    data_point,
    problem,
    solver;
    mu,
    rho=0,
    splits,
    context_index,
    testing_count,
    partial_path,
    rows,
    csv_path,
    optimal_split_values,
    resume_values=Float64[],
)
    split_ranges = testing_split_ranges(data_point, splits)
    policy_split_values = Float64.(collect(resume_values))
    decision_set = generate_decision_set(policy, [data_point])
    z = view(decision_set, :, 1)

    for split_index in (length(policy_split_values) + 1):splits
        println("Testing context $(context_index)/$(testing_count): policy split $(split_index)/$(splits)...")
        scenario_range = split_ranges[split_index]
        policy_value = resource_allocation_policy_split_value(
            problem,
            solver,
            z,
            view(data_point.scenario_parameters, scenario_range);
            mu=mu,
            rho=rho,
        )

        push!(policy_split_values, policy_value)
        write_testing_partial(
            partial_path,
            rows,
            testing_count,
            splits,
            csv_path;
            active_context_index=context_index,
            optimal_split_values=optimal_split_values,
            policy_split_values=policy_split_values,
        )
    end

    return policy_split_values
end

function testing_sample_row(index, policy_split_values, optimal_split_values)
    policy_value = Statistics.mean(Float64.(policy_split_values))
    optimal_value = Statistics.mean(Float64.(optimal_split_values))
    regret = policy_value - optimal_value
    relative_regret = regret / max(abs(optimal_value), eps(Float64))

    return (;
        sample_index=index,
        policy_value=policy_value,
        optimal_value=optimal_value,
        regret=regret,
        relative_regret=relative_regret,
        policy_split_values=Float64.(collect(policy_split_values)),
        optimal_split_values=Float64.(collect(optimal_split_values)),
        ucb_percent=normalized_gap_ucb_percent(policy_split_values, optimal_split_values),
    )
end

function run_saa_testing(
    model,
    problem,
    solver,
    data_set_testing;
    output_dir,
    reg_param_surr,
    reg_param_ref,
    rho_surr=0.0,
    rho_ref=0.0,
    testing_splits,
    testing_context_limit=length(data_set_testing),
)
    program = stochastic_program(problem)
    policy = ScenarioGenerationPolicy(
        ContextualDFL.ScenarioGenerator(;
            neural_net=model,
            scenario_decoder=ResourceAllocationDemandVectorDecoder(problem),
        ),
        solver,
        program;
        mu=reg_param_surr,
        rho=rho_surr,
    )

    csv_path = joinpath(output_dir, "testing_saa_results.csv")
    partial_path = joinpath(output_dir, "testing_saa_partial.jls")
    testing_count = min(Int(testing_context_limit), length(data_set_testing))
    rows = NamedTuple[]
    active_context_index = nothing
    active_optimal_split_values = Float64[]
    active_policy_split_values = Float64[]

    if isfile(partial_path)
        partial = Serialization.deserialize(partial_path)
        if hasproperty(partial, :ucb_rows)
            rows = collect(partial.ucb_rows)
            println("Resuming testing SAA from $(length(rows)) completed contexts.")
        end
        if hasproperty(partial, :active_context_index)
            active_context_index = partial.active_context_index
            active_optimal_split_values = hasproperty(partial, :optimal_split_values) ?
                Float64.(collect(partial.optimal_split_values)) : Float64[]
            active_policy_split_values = hasproperty(partial, :policy_split_values) ?
                Float64.(collect(partial.policy_split_values)) : Float64[]
        end
    end

    length(rows) > testing_count && resize!(rows, testing_count)
    write_testing_csv(csv_path, rows)

    for index in (length(rows) + 1):testing_count
        resume_current_context = active_context_index == index
        optimal_resume_values =
            resume_current_context ? active_optimal_split_values : Float64[]
        policy_resume_values =
            resume_current_context ? active_policy_split_values : Float64[]

        println("Testing context $(index)/$(testing_count): solving SAA optima...")
        data_point = data_set_testing[index]
        optimal_split_values = solve_data_point_to_optimality_with_progress(
            data_point,
            problem,
            solver;
            mu=reg_param_ref,
            rho=rho_ref,
            splits=testing_splits,
            context_index=index,
            testing_count=testing_count,
            partial_path=partial_path,
            rows=rows,
            csv_path=csv_path,
            resume_values=optimal_resume_values,
        )

        println("Testing context $(index)/$(testing_count): evaluating policy...")
        policy_split_values = evaluate_policy_on_data_point_with_progress(
            policy,
            data_point,
            problem,
            solver;
            mu=reg_param_ref,
            rho=rho_ref,
            splits=testing_splits,
            context_index=index,
            testing_count=testing_count,
            partial_path=partial_path,
            rows=rows,
            csv_path=csv_path,
            optimal_split_values=optimal_split_values,
            resume_values=policy_resume_values,
        )

        row = testing_sample_row(index, policy_split_values, optimal_split_values)
        push!(rows, row)
        write_testing_csv(csv_path, rows)
        write_testing_partial(
            partial_path,
            rows,
            testing_count,
            testing_splits,
            csv_path,
        )
        active_context_index = nothing
        empty!(active_optimal_split_values)
        empty!(active_policy_split_values)
        println(
            "Testing context $(index)/$(testing_count): UCB = $(row.ucb_percent), " *
            "relative regret = $(row.relative_regret)",
        )
    end

    clean_ucbs = [row.ucb_percent for row in rows if isfinite(row.ucb_percent)]
    mean_ucb = isempty(clean_ucbs) ? NaN : Statistics.mean(clean_ucbs)
    println("mean UCB: ", mean_ucb)
    println("Testing SAA results saved to: $(csv_path)")

    return (;
        ucb_rows=rows,
        mean_ucb=mean_ucb,
        tested_contexts=length(rows),
        testing_splits=testing_splits,
        csv_path=csv_path,
        partial_path=partial_path,
    )
end

function testing_metrics(testing_result)
    rows = collect(field_or_default(testing_result, :ucb_rows, ()))
    clean_ucbs = [row.ucb_percent for row in rows if isfinite(row.ucb_percent)]

    mean_field(name) = isempty(rows) ? NaN : Statistics.mean(
        Float64(field_or_default(row, name, NaN)) for row in rows
    )

    return (;
        tested_contexts=length(rows),
        mean_ucb=isempty(clean_ucbs) ? NaN : Statistics.mean(clean_ucbs),
        mean_policy_value=mean_field(:policy_value),
        mean_optimal_value=mean_field(:optimal_value),
        mean_regret=mean_field(:regret),
        mean_relative_regret=mean_field(:relative_regret),
    )
end

function final_history_row(stage_histories)
    isempty(stage_histories) && return nothing
    final_stage = stage_histories[end]
    history = collect(field_or_default(final_stage, :history, ()))
    isempty(history) && return nothing
    return history[end]
end

function run_experiment()
    seed = env_int("CDFL_SANDBOX_SEED", 20260504)
    Random.seed!(seed)
    rng = Random.default_rng()

    problem = ResourceAllocationProblem(default_resource_allocation_problem_data())
    n_training_samples = env_int("CDFL_SANDBOX_TRAINING_SAMPLES", 100)
    n_testing_samples = env_int("CDFL_SANDBOX_TESTING_SAMPLES", 30)
    sigma = env_float("CDFL_SANDBOX_SIGMA", 5)
    p = env_int("CDFL_SANDBOX_P", 2)
    l = env_int("CDFL_SANDBOX_L", 3)
    n_xi_per_x = env_int("CDFL_SANDBOX_XI_PER_X", 100)
    testing_splits = env_int("CDFL_SANDBOX_TESTING_SPLITS", 30)
    nr_scenarios = 1

    context_generator = ResourceAllocationContextDataGenerator(rng=rng)
    scenario_generator = ResourceAllocationScenarioDataGenerator(
        problem;
        sigma=sigma,
        p=p,
        L=l,
        rng=rng,
    )

    contexts = [Vector{Float64}(context_generator()) for _ in 1:n_training_samples]
    scenarios = [scenario_generator(context) for context in contexts]
    data_set_training = generate_contextual_data_set(contexts, scenarios)

    testing_contexts = [Vector{Float64}(context_generator()) for _ in 1:n_testing_samples]
    testing_scenarios = [
        [scenario_generator(context) for _ in 1:(testing_splits * n_xi_per_x)]
        for context in testing_contexts
    ]
    data_set_testing = generate_contextual_data_set(testing_contexts, testing_scenarios)

    demand_count = size(problem.problem_data.service_rate_parameters, 2)
    model = Flux.Chain(
        Flux.Dense(3 => 128, Flux.relu),
        Flux.Dense(128 => 128, Flux.relu),
        Flux.Dense(128 => 128, Flux.relu),
        Flux.Dense(128 => demand_count * nr_scenarios, Flux.relu),
    ) |> Flux.f64

    solver = ContextualDFL.Solver(ContextualDFL.IpoptSolver(), ContextualDFL.HiGHSSolver())
    loss = ContextualDFL.DflScenLoss(
        ResourceAllocationDemandVectorDecoder(problem),
        ResourceAllocationDemandParametricDecoder(problem),
        solver,
        stochastic_program(problem);
        nr_scenarios=nr_scenarios,
    )

    display_reference_input(point) =
        reduce(vcat, (scenario.h_eq_xi for scenario in point.scenario_parameters))

    reg_param_ref = env_float("CDFL_SANDBOX_MU_REF", 0.0)
    rho_in = env_float("CDFL_SANDBOX_RHO", 0.0)
    rho_ref = env_float("CDFL_SANDBOX_RHO_REF", rho_in)
    optimality_rho = env_float("CDFL_SANDBOX_OPTIMALITY_RHO", 0.0)
    policy_inference_rho = env_float("CDFL_SANDBOX_POLICY_INFERENCE_RHO", rho_in)
    batchsize = env_int("CDFL_SANDBOX_BATCHSIZE", 1)
    default_epochs = env_int("CDFL_SANDBOX_DEFAULT_EPOCHS", 10)
    first_stage_epochs = env_int("CDFL_SANDBOX_FIRST_STAGE_EPOCHS", 20)
    step_size = env_float("CDFL_SANDBOX_STEP_SIZE", 1e-3)
    save_model_training = env_bool("CDFL_SANDBOX_SAVE_MODEL", true)
    skip_training = env_bool("CDFL_SANDBOX_SKIP_TRAINING")
    resume_training = env_bool("CDFL_SANDBOX_RESUME_TRAINING", true)
    skip_testing = env_bool("CDFL_SANDBOX_SKIP_TESTING")
    testing_context_limit = env_int("CDFL_SANDBOX_TEST_CONTEXTS", n_testing_samples)

    param_list = env_float_list(
        "CDFL_SANDBOX_PARAM_LIST",
        [1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.08, 0.06, 0.04, 0.02, 0.01],
    )
    epoch_list = fill(default_epochs, length(param_list) + 1)
    epoch_list[1] = first_stage_epochs

    default_output_dir = joinpath(
        ANNEALING_EXPERIMENT_DIR,
        "results",
        "quadratic_smoothing_sandbox",
        Dates.format(Dates.now(), Dates.DateFormat("yyyymmddTHHMMSS")),
    )
    output_dir = get(ENV, "CDFL_SANDBOX_OUTPUT_DIR", default_output_dir)
    mkpath(output_dir)

    model_save_path = joinpath(output_dir, "trained_model_annealing.jls")
    state_save_path = joinpath(output_dir, "experiment_state_annealing.jls")
    training_history_csv_path = joinpath(output_dir, "training_history.csv")
    run_summary_csv_path = joinpath(output_dir, "run_summary.csv")
    run_summary_path = joinpath(output_dir, "run_summary.jls")

    experiment_started_at = Dates.now()
    experiment_parameters = (;
        seed=seed,
        n_training_samples=n_training_samples,
        n_testing_samples=n_testing_samples,
        sigma=sigma,
        p=p,
        L=l,
        n_xi_per_x=n_xi_per_x,
        testing_splits=testing_splits,
        reg_param_ref=reg_param_ref,
        rho_in=rho_in,
        rho_ref=rho_ref,
        optimality_rho=optimality_rho,
        policy_inference_rho=policy_inference_rho,
        batchsize=batchsize,
        default_epochs=default_epochs,
        first_stage_epochs=first_stage_epochs,
        step_size=step_size,
        param_list=param_list,
        epoch_list=epoch_list,
        nr_scenarios=nr_scenarios,
        output_dir=output_dir,
        started_at=experiment_started_at,
    )

    final_reg_param_surr = param_list[end]
    final_stage_epochs = epoch_list[end]
    final_reg_param_prim = 0.0
    final_stage_index = length(param_list) + 1
    stage_histories = NamedTuple[]
    completed_stage_numbers = Set{Int}()

    if skip_training
        isfile(state_save_path) ||
            error("CDFL_SANDBOX_SKIP_TRAINING=1 requires saved state at $(state_save_path).")
        saved_state = Serialization.deserialize(state_save_path)
        model = saved_state.model
        data_set_training = saved_state.data_set_training
        data_set_testing = saved_state.data_set_testing
        stage_histories = collect(saved_state.stage_histories)
        completed_stage_numbers = Set(Int(stage.stage) for stage in stage_histories)
        println("Skipping training and resuming from: $(state_save_path)")
    else
        if resume_training && isfile(state_save_path)
            saved_state = Serialization.deserialize(state_save_path)
            model = saved_state.model
            data_set_training = saved_state.data_set_training
            data_set_testing = saved_state.data_set_testing
            stage_histories = collect(saved_state.stage_histories)
            completed_stage_numbers = Set(Int(stage.stage) for stage in stage_histories)
            println(
                "Resuming training from $(state_save_path) with " *
                "$(length(completed_stage_numbers)) completed stages.",
            )
        else
            println("Starting sandbox training with rho=$(rho_in)...")
        end

        for (idx, reg_param_surr) in enumerate(param_list)
            if idx in completed_stage_numbers
                println("Skipping completed annealing stage $(idx).")
                continue
            end

            stage_epochs = epoch_list[idx]
            reg_param_prim = reg_param_surr
            println(
                "Starting stage $(idx): mu_in=$(reg_param_surr), " *
                "mu_ref=$(reg_param_prim), rho=$(rho_in), epochs=$(stage_epochs)",
            )

            result = ContextualDFL.train!(
                model,
                loss,
                fill(reg_param_surr, stage_epochs),
                fill(reg_param_prim, stage_epochs),
                data_set_training;
                opt=Flux.Adam(step_size),
                epochs=stage_epochs,
                batchsize=batchsize,
                display_iterations=true,
                display_plot=false,
                save_model=save_model_training,
                model_save_path=model_save_path,
                reset_optimizer_each_epoch=true,
                nr_scenarios=nr_scenarios,
                display_smooth=true,
                display_reference_input=display_reference_input,
                rho_in_schedule=fill(rho_in, stage_epochs),
                rho_ref_schedule=fill(rho_ref, stage_epochs),
            )

            push!(
                stage_histories,
                (;
                    stage=idx,
                    reg_param_surr=reg_param_surr,
                    reg_param_prim=reg_param_prim,
                    rho_in=rho_in,
                    rho_ref=rho_ref,
                    epochs=stage_epochs,
                    history=result.history,
                ),
            )
            push!(completed_stage_numbers, idx)
            write_training_history_csv(training_history_csv_path, stage_histories)

            Serialization.serialize(
                state_save_path,
                (;
                    model=model,
                    data_set_training=data_set_training,
                    data_set_testing=data_set_testing,
                    problem=problem,
                    stage_histories=stage_histories,
                    parameters=experiment_parameters,
                ),
            )
        end

        if final_stage_index in completed_stage_numbers
            println("Skipping completed final annealing stage.")
        else
            println(
                "Starting final stage: mu_in=$(final_reg_param_surr), " *
                "mu_ref=$(final_reg_param_prim), rho=$(rho_in), epochs=$(final_stage_epochs)",
            )

            final_result = ContextualDFL.train!(
                model,
                loss,
                fill(final_reg_param_surr, final_stage_epochs),
                fill(final_reg_param_prim, final_stage_epochs),
                data_set_training;
                opt=Flux.Adam(step_size),
                epochs=final_stage_epochs,
                batchsize=batchsize,
                display_iterations=true,
                display_plot=false,
                save_model=save_model_training,
                model_save_path=model_save_path,
                reset_optimizer_each_epoch=true,
                nr_scenarios=nr_scenarios,
                display_smooth=true,
                display_reference_input=display_reference_input,
                rho_in_schedule=fill(rho_in, final_stage_epochs),
                rho_ref_schedule=fill(rho_ref, final_stage_epochs),
            )

            push!(
                stage_histories,
                (;
                    stage=final_stage_index,
                    reg_param_surr=final_reg_param_surr,
                    reg_param_prim=final_reg_param_prim,
                    rho_in=rho_in,
                    rho_ref=rho_ref,
                    epochs=final_stage_epochs,
                    history=final_result.history,
                ),
            )
            push!(completed_stage_numbers, final_stage_index)
            write_training_history_csv(training_history_csv_path, stage_histories)

            Serialization.serialize(
                state_save_path,
                (;
                    model=model,
                    data_set_training=data_set_training,
                    data_set_testing=data_set_testing,
                    problem=problem,
                    stage_histories=stage_histories,
                    parameters=experiment_parameters,
                ),
            )
        end

        println("Training completed.")
    end

    testing_result = nothing
    if skip_testing
        println("Skipping testing because CDFL_SANDBOX_SKIP_TESTING=1.")
    else
        println("Testing the trained model...")
        write_training_history_csv(training_history_csv_path, stage_histories)
        testing_result = run_saa_testing(
            model,
            problem,
            solver,
            data_set_testing;
            output_dir=output_dir,
            reg_param_surr=final_reg_param_surr,
            reg_param_ref=reg_param_ref,
            rho_surr=policy_inference_rho,
            rho_ref=optimality_rho,
            testing_splits=testing_splits,
            testing_context_limit=testing_context_limit,
        )
    end

    metrics = isnothing(testing_result) ? nothing : testing_metrics(testing_result)
    final_training_row = final_history_row(stage_histories)
    summary = (;
        status="completed",
        seed=seed,
        rho=rho_in,
        rho_ref=rho_ref,
        policy_inference_rho=policy_inference_rho,
        optimality_rho=optimality_rho,
        final_reg_param_surr=final_reg_param_surr,
        final_reg_param_prim=final_reg_param_prim,
        training_epochs=sum(Int(field_or_default(stage, :epochs, 0)) for stage in stage_histories),
        final_training_loss=field_or_default(final_training_row, :loss, NaN),
        final_display_loss=field_or_default(final_training_row, :display_loss, NaN),
        final_real_display_loss=field_or_default(final_training_row, :real_display_loss, NaN),
        tested_contexts=isnothing(metrics) ? 0 : metrics.tested_contexts,
        testing_splits=testing_splits,
        mean_ucb=isnothing(metrics) ? NaN : metrics.mean_ucb,
        mean_policy_value=isnothing(metrics) ? NaN : metrics.mean_policy_value,
        mean_optimal_value=isnothing(metrics) ? NaN : metrics.mean_optimal_value,
        mean_regret=isnothing(metrics) ? NaN : metrics.mean_regret,
        mean_relative_regret=isnothing(metrics) ? NaN : metrics.mean_relative_regret,
        output_dir=output_dir,
        training_history_csv=training_history_csv_path,
        testing_csv=isnothing(testing_result) ? "" : testing_result.csv_path,
        started_at=string(experiment_started_at),
        finished_at=string(Dates.now()),
    )

    Serialization.serialize(run_summary_path, summary)
    write_run_summary_csv(run_summary_csv_path, summary)
    Serialization.serialize(
        state_save_path,
        (;
            model=model,
            data_set_training=data_set_training,
            data_set_testing=data_set_testing,
            problem=problem,
            stage_histories=stage_histories,
            testing_result=testing_result,
            summary=summary,
            parameters=experiment_parameters,
        ),
    )

    println("Sandbox output: $(output_dir)")
    println("Training history: $(training_history_csv_path)")
    println("Testing CSV: $(summary.testing_csv)")
    println("Run summary: $(run_summary_csv_path)")
end

run_experiment()
