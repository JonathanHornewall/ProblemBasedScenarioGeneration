import Pkg

Pkg.activate(joinpath(@__DIR__, "..", ".."))
Pkg.instantiate()

using ContextualDFL
using ContextualDFLExperiments

import Random
import Serialization

const Flux = ContextualDFL.Flux

rng = Random.default_rng()

problem = ResourceAllocationProblem(default_resource_allocation_problem_data())

# Match the prototype training data settings. The testing/SAA portion is intentionally skipped.
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

# Display the same kind of relative loss as the prototype: compare the current
# model-generated scenarios against using the realized demand as the input scenario.
function relative_loss(
    input_scenario_parameter_collection,
    reference_scenario_parameter_collection,
    mu_in=0,
    mu_ref=0;
    kwargs...,
)
    evaluated_value = loss(
        input_scenario_parameter_collection,
        reference_scenario_parameter_collection,
        mu_in,
        mu_ref;
        kwargs...,
    )
    reference_input = reduce(
        vcat,
        (scenario.h_eq_xi for scenario in reference_scenario_parameter_collection),
    )
    reference_value = loss(
        reference_input,
        reference_scenario_parameter_collection,
        mu_ref,
        mu_ref;
        kwargs...,
    )
    return (evaluated_value - reference_value) / abs(reference_value)
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

experiment_parameters = (;
    Ntraining_samples=Ntraining_samples,
    Ntesting_samples=Ntesting_samples,
    sigma=sigma,
    p=p,
    L=L,
    N_xi_per_x=N_xi_per_x,
    reg_param_ref=reg_param_ref,
    batchsize=batchsize,
    default_epochs=default_epochs,
    step_size=step_size,
    param_list=param_list,
    epoch_list=epoch_list,
    nr_scenarios=nr_scenarios,
)

stage_histories = NamedTuple[]

println("Starting training with annealing...")

for (idx, reg_param_surr) in enumerate(param_list)
    stage_epochs = epoch_list[idx]
    reg_param_prim = reg_param_surr
    println(
        "Starting annealing stage $(idx) with reg_param_surr = $(reg_param_surr), " *
        "reg_param_prim = $(reg_param_prim), epochs = $(stage_epochs)",
    )

    result = ContextualDFL.train!(
        model,
        loss,
        relative_loss,
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

    Serialization.serialize(
        state_save_path,
        (;
            model=model,
            data_set_training=data_set_training,
            problem=problem,
            stage_histories=stage_histories,
            parameters=experiment_parameters,
        ),
    )
end

final_reg_param_surr = param_list[end]
final_stage_epochs = epoch_list[end]
final_reg_param_prim = 0.0

println(
    "Starting final annealing stage with reg_param_surr = $(final_reg_param_surr), " *
    "reg_param_prim = $(final_reg_param_prim), epochs = $(final_stage_epochs)",
)

final_result = ContextualDFL.train!(
    model,
    loss,
    relative_loss,
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
)

push!(
    stage_histories,
    (;
        stage=length(param_list) + 1,
        reg_param_surr=final_reg_param_surr,
        reg_param_prim=final_reg_param_prim,
        epochs=final_stage_epochs,
        history=final_result.history,
    ),
)

Serialization.serialize(
    state_save_path,
    (;
        model=model,
        data_set_training=data_set_training,
        problem=problem,
        stage_histories=stage_histories,
        parameters=experiment_parameters,
    ),
)

println("Training completed!")
println("Model saved to: $(model_save_path)")
println("Experiment state saved to: $(state_save_path)")
