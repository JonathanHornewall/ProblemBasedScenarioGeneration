using ContextualDFL
using ContextualDFLExperiments
using Flux
using Random
using Statistics

function small_resource_allocation_problem()
    data = ContextualDFLExperiments.ResourceAllocationProblemData(
        [1.0 0.8 1.2; 0.7 1.1 0.9],
        [1.0, 1.2],
        [3.0, 4.0, 5.0],
        [1.0, 1.0],
    )
    return ContextualDFLExperiments.ResourceAllocationProblem(data)
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

function mean_loss(loss, model, data; constraint_tolerance=1e-8)
    return mean(
        loss(model(dp.context), dp.scenario_parameters, 0.0; constraint_tolerance=constraint_tolerance)
        for dp in data
    )
end

Random.seed!(20260507)

solver = ContextualDFL.Solver(ContextualDFL.IpoptSolver(), ContextualDFL.HiGHSSolver())
problem = small_resource_allocation_problem()
program = ContextualDFLExperiments.stochastic_program(problem)
base_scenario = full_parametric_scenario(ContextualDFLExperiments.base_scenario(problem))
original_decoder = ContextualDFLExperiments.ResourceAllocationDemandParametricDecoder(problem)

train_data = ContextualDFLExperiments.generate_benchmark_dataset(
    problem;
    n_contexts=6,
    scenarios_per_context=2,
    seed=20260507,
)

prepared = ContextualDFLExperiments.prepare_spoplus_q_dataset(
    train_data,
    solver,
    program,
    original_decoder,
    base_scenario;
    lower_bound_margin=1e-4,
    constraint_tolerance=1e-8,
)

model = Flux.Chain(
    Flux.Dense(length(first(train_data).context) => 16, Flux.relu),
    Flux.Dense(16 => length(prepared.q_lower_bound)),
) |> Flux.f64

initial_loss = mean_loss(prepared.spo_loss, model, prepared.converted_dataset)

result = ContextualDFL.train!(
    model,
    prepared.spo_loss,
    [0.0, 0.0],
    prepared.converted_dataset;
    epochs=2,
    batchsize=1,
    learning_rate=1e-3,
    display_iterations=true,
    nr_scenarios=1,
)

final_loss = mean_loss(prepared.spo_loss, model, prepared.converted_dataset)

println("converted_count=", length(prepared.converted_dataset))
println("q_dimension=", length(prepared.q_lower_bound))
println("q_lower_bound_min=", minimum(prepared.q_lower_bound))
println("q_lower_bound_max=", maximum(prepared.q_lower_bound))
println("initial_loss=", initial_loss)
println("final_loss=", final_loss)
println("history=", result.history)
