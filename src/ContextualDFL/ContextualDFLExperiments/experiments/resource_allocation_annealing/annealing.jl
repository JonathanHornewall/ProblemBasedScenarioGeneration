import Pkg

Pkg.activate(joinpath(@__DIR__, "..", ".."))
Pkg.instantiate()

using ContextualDFL
using ContextualDFLExperiments

import Random
import Serialization
import Statistics

const Flux = ContextualDFL.Flux

rng = Random.default_rng()

problem = ResourceAllocationProblem(default_resource_allocation_problem_data())

# Match the prototype training and SAA testing data settings.
Ntraining_samples = 100
Ntesting_samples = 30
sigma = 5
p = 2
L = 3
N_xi_per_x = 100

context_generator = ResourceAllocationContextDataGenerator(rng=rng)
scenario_generator = ResourceAllocationScenarioDataGenerator(
    problem;
    sigma=sigma,
    p=p,
    L=L,
    rng=rng,
)

contexts = [Vector{Float64}(context_generator()) for _ in 1:Ntraining_samples]
scenarios = [scenario_generator(context) for context in contexts]
data_set_training = generate_contextual_data_set(contexts, scenarios)

testing_splits = 30
testing_contexts = [Vector{Float64}(context_generator()) for _ in 1:Ntesting_samples]
testing_scenarios = [
    [scenario_generator(context) for _ in 1:(testing_splits * N_xi_per_x)]
    for context in testing_contexts
]
data_set_testing = generate_contextual_data_set(testing_contexts, testing_scenarios)

nr_scenarios = 1
demand_count = size(problem.problem_data.service_rate_parameters, 2)

# Same hidden architecture as the old resource-allocation prototype.
# The new VectorDecoder expects a flat demand vector, so we leave out the old final reshape.
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

function normalized_gap_ucb_percent(policy_split_values, optimal_split_values)
    gaps = Float64.(policy_split_values) .- Float64.(optimal_split_values)
    optimal_mean = Statistics.mean(Float64.(optimal_split_values))
    gap_variance = length(gaps) > 1 ? Statistics.var(gaps) : 0.0
    ucb_gap = Statistics.mean(gaps) + 2.462 * sqrt(gap_variance / length(gaps))
    return 100 * ucb_gap / max(abs(optimal_mean), eps(Float64))
end

function run_saa_testing(
    model,
    problem,
    solver,
    data_set_testing;
    output_dir,
    reg_param_surr,
    reg_param_ref,
    testing_splits,
    testing_context_limit=length(data_set_testing),
)
    program = stochastic_program(problem)
    parametric_decoder = ResourceAllocationDemandParametricDecoder(problem)
    policy = ScenarioGenerationPolicy(
        ContextualDFL.ScenarioGenerator(;
            neural_net=model,
            scenario_decoder=ResourceAllocationDemandVectorDecoder(problem),
        ),
        solver,
        program;
        mu=reg_param_surr,
    )

    csv_path = joinpath(output_dir, "testing_saa_results.csv")
    partial_path = joinpath(output_dir, "testing_saa_partial.jls")
    testing_count = min(Int(testing_context_limit), length(data_set_testing))
    rows = NamedTuple[]

    if isfile(partial_path)
        partial = Serialization.deserialize(partial_path)
        if hasproperty(partial, :ucb_rows)
            rows = collect(partial.ucb_rows)
            println("Resuming testing SAA from $(length(rows)) completed contexts.")
        end
    end

    length(rows) > testing_count && resize!(rows, testing_count)
    write_testing_csv(csv_path, rows)

    for index in (length(rows) + 1):testing_count
        println("Testing context $(index)/$(testing_count): solving SAA optima...")
        single_data_set = [data_set_testing[index]]
        optimal_results = solve_dataset_to_optimality(
            single_data_set,
            program,
            parametric_decoder,
            solver;
            mu=reg_param_ref,
            splits=testing_splits,
        )

        println("Testing context $(index)/$(testing_count): evaluating policy...")
        comparison = evaluate_policy_against_optimum(
            policy,
            single_data_set,
            program,
            parametric_decoder,
            solver;
            optimal_results=optimal_results,
            split_name=:test,
            mu=reg_param_ref,
            splits=testing_splits,
        )

        sample = only(comparison.per_sample)
        row = merge(
            sample,
            (;
                sample_index=index,
                ucb_percent=normalized_gap_ucb_percent(
                    sample.policy_split_values,
                    sample.optimal_split_values,
                ),
            ),
        )
        push!(rows, row)
        write_testing_csv(csv_path, rows)
        Serialization.serialize(
            partial_path,
            (;
                ucb_rows=rows,
                completed_contexts=length(rows),
                testing_context_limit=testing_count,
                testing_splits=testing_splits,
                csv_path=csv_path,
            ),
        )
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

reg_param_ref = 0.0
batchsize = 1
default_epochs = 10
step_size = 1e-3
save_model_training = true

param_list = [1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.08, 0.06, 0.04, 0.02, 0.01]
epoch_list = fill(default_epochs, length(param_list) + 1)
epoch_list[1] = 20
@assert length(epoch_list) == length(param_list) + 1

output_dir = joinpath(@__DIR__, "results")
mkpath(output_dir)
model_save_path = joinpath(output_dir, "trained_model_annealing.jls")
state_save_path = joinpath(output_dir, "experiment_state_annealing.jls")
skip_training = get(ENV, "CDFL_ANNEALING_SKIP_TRAINING", "0") == "1"
resume_training = get(ENV, "CDFL_ANNEALING_RESUME_TRAINING", "0") == "1"
skip_testing = get(ENV, "CDFL_ANNEALING_SKIP_TESTING", "0") == "1"
testing_context_limit = parse(
    Int,
    get(ENV, "CDFL_ANNEALING_TEST_CONTEXTS", string(Ntesting_samples)),
)

experiment_parameters = (;
    Ntraining_samples=Ntraining_samples,
    Ntesting_samples=Ntesting_samples,
    sigma=sigma,
    p=p,
    L=L,
    N_xi_per_x=N_xi_per_x,
    testing_splits=testing_splits,
    reg_param_ref=reg_param_ref,
    batchsize=batchsize,
    default_epochs=default_epochs,
    step_size=step_size,
    param_list=param_list,
    epoch_list=epoch_list,
    nr_scenarios=nr_scenarios,
)

final_reg_param_surr = param_list[end]
final_stage_epochs = epoch_list[end]
final_reg_param_prim = 0.0
final_stage_index = length(param_list) + 1
stage_histories = NamedTuple[]
completed_stage_numbers = Set{Int}()

if skip_training
    isfile(state_save_path) ||
        error("CDFL_ANNEALING_SKIP_TRAINING=1 requires saved state at $(state_save_path).")
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
        println("Starting training with annealing...")
    end

    for (idx, reg_param_surr) in enumerate(param_list)
        if idx in completed_stage_numbers
            println("Skipping completed annealing stage $(idx).")
            continue
        end

        stage_epochs = epoch_list[idx]
        reg_param_prim = reg_param_surr
        println(
            "Starting annealing stage $(idx) with reg_param_surr = $(reg_param_surr), " *
            "reg_param_prim = $(reg_param_prim), epochs = $(stage_epochs)",
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
        )

        push!(
            stage_histories,
            (;
                stage=idx,
                reg_param_surr=reg_param_surr,
                reg_param_prim=reg_param_prim,
                epochs=stage_epochs,
                history=result.history,
            ),
        )
        push!(completed_stage_numbers, idx)

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
            "Starting final annealing stage with reg_param_surr = $(final_reg_param_surr), " *
            "reg_param_prim = $(final_reg_param_prim), epochs = $(final_stage_epochs)",
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
        )

        push!(
            stage_histories,
            (;
                stage=final_stage_index,
                reg_param_surr=final_reg_param_surr,
                reg_param_prim=final_reg_param_prim,
                epochs=final_stage_epochs,
                history=final_result.history,
            ),
        )
        push!(completed_stage_numbers, final_stage_index)

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

    println("Training completed!")
end

if skip_testing
    println("Skipping testing because CDFL_ANNEALING_SKIP_TESTING=1.")
else
    println("Testing the trained model...")
    testing_result = run_saa_testing(
        model,
        problem,
        solver,
        data_set_testing;
        output_dir=output_dir,
        reg_param_surr=final_reg_param_surr,
        reg_param_ref=reg_param_ref,
        testing_splits=testing_splits,
        testing_context_limit=testing_context_limit,
    )

    Serialization.serialize(
        state_save_path,
        (;
            model=model,
            data_set_training=data_set_training,
            data_set_testing=data_set_testing,
            problem=problem,
            stage_histories=stage_histories,
            testing_result=testing_result,
            parameters=experiment_parameters,
        ),
    )
end

println("Model saved to: $(model_save_path)")
println("Experiment state saved to: $(state_save_path)")
