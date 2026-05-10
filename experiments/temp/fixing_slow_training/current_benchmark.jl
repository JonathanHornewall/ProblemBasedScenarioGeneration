using ContextualDFL
using ContextualDFLExperiments

import Random

const Flux = ContextualDFL.Flux

const N_TRAIN = parse(Int, get(ENV, "FIX_SLOW_N_TRAIN", "3"))
const EPOCHS = parse(Int, get(ENV, "FIX_SLOW_EPOCHS", "1"))
const BATCHSIZE = parse(Int, get(ENV, "FIX_SLOW_BATCHSIZE", "1"))
const MU_SURR = parse(Float64, get(ENV, "FIX_SLOW_MU_SURR", "1.0"))
const MU_REF = parse(Float64, get(ENV, "FIX_SLOW_MU_REF", "1.0"))
const STEP_SIZE = parse(Float64, get(ENV, "FIX_SLOW_STEP_SIZE", "1e-3"))
const DISPLAY_ITERATIONS = get(ENV, "FIX_SLOW_DISPLAY", "0") == "1"
const DISPLAY_SMOOTH = get(ENV, "FIX_SLOW_DISPLAY_SMOOTH", "0") == "1"

function seconds(f)
    GC.gc()
    started = time_ns()
    result = f()
    return (time_ns() - started) / 1e9, result
end

display_reference_input(point) =
    reduce(vcat, (scenario.h_eq_xi for scenario in point.scenario_parameters))

function make_current_case(n_train)
    rng = Random.MersenneTwister(1234)
    problem = ResourceAllocationProblem(default_resource_allocation_problem_data())

    context_generator = ResourceAllocationContextDataGenerator(rng=rng)
    scenario_generator = ResourceAllocationScenarioDataGenerator(
        problem;
        sigma=5,
        p=2,
        L=3,
        rng=rng,
    )

    contexts = [Vector{Float64}(context_generator()) for _ in 1:n_train]
    scenarios = [scenario_generator(context) for context in contexts]
    training_data = generate_contextual_data_set(contexts, scenarios)

    nr_scenarios = 1
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

    return (; training_data, model, loss, nr_scenarios)
end

function run_training!(case; epochs)
    kwargs = (;
        opt=Flux.Adam(STEP_SIZE),
        epochs=epochs,
        batchsize=BATCHSIZE,
        display_iterations=DISPLAY_ITERATIONS,
        verbose=DISPLAY_ITERATIONS,
        display_plot=false,
        save_model=false,
        reset_optimizer_each_epoch=true,
        nr_scenarios=case.nr_scenarios,
        display_smooth=DISPLAY_SMOOTH,
    )

    if DISPLAY_SMOOTH
        return ContextualDFL.train!(
            case.model,
            case.loss,
            fill(MU_SURR, epochs),
            fill(MU_REF, epochs),
            case.training_data;
            kwargs...,
            display_reference_input=display_reference_input,
        )
    end

    return ContextualDFL.train!(
        case.model,
        case.loss,
        fill(MU_SURR, epochs),
        fill(MU_REF, epochs),
        case.training_data;
        kwargs...,
    )
end

setup_seconds, case = seconds(() -> make_current_case(N_TRAIN))
warmup_case = make_current_case(1)
warmup_seconds, _ = seconds(() -> run_training!(warmup_case; epochs=1))
training_seconds, result = seconds(() -> run_training!(case; epochs=EPOCHS))

iterations = sum(row.iterations for row in result.history)
seconds_per_iteration = training_seconds / iterations

result_path = joinpath(@__DIR__, DISPLAY_SMOOTH ? "current_smooth_results.csv" : "current_results.csv")
open(result_path, "w") do io
    println(io, "implementation,n_train,epochs,batchsize,display,display_smooth,setup_seconds,warmup_seconds,training_seconds,iterations,seconds_per_iteration")
    println(
        io,
        join(
            (
                DISPLAY_SMOOTH ? "current_contextualdfl_smooth" : "current_contextualdfl",
                N_TRAIN,
                EPOCHS,
                BATCHSIZE,
                DISPLAY_ITERATIONS,
                DISPLAY_SMOOTH,
                setup_seconds,
                warmup_seconds,
                training_seconds,
                iterations,
                seconds_per_iteration,
            ),
            ",",
        ),
    )
end

println("implementation=$(DISPLAY_SMOOTH ? "current_contextualdfl_smooth" : "current_contextualdfl")")
println("n_train=$(N_TRAIN)")
println("epochs=$(EPOCHS)")
println("batchsize=$(BATCHSIZE)")
println("display=$(DISPLAY_ITERATIONS)")
println("display_smooth=$(DISPLAY_SMOOTH)")
println("setup_seconds=$(setup_seconds)")
println("warmup_seconds=$(warmup_seconds)")
println("training_seconds=$(training_seconds)")
println("iterations=$(iterations)")
println("seconds_per_iteration=$(seconds_per_iteration)")
