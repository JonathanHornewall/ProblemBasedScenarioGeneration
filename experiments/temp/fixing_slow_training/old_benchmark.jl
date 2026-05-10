using ProblemBasedScenarioGeneration

import ChainRulesCore
import Flux
import Random
import Statistics

using ChainRulesCore: NoTangent
import ProblemBasedScenarioGeneration: loss, relative_loss, surrogate_solution

const ROOT = abspath(joinpath(@__DIR__, "..", "..", ".."))
include(joinpath(ROOT, "scripts", "resource_allocation_prototype", "parameters.jl"))
include(joinpath(ROOT, "scripts", "resource_allocation_prototype", "custom_code", "neural_net.jl"))

const N_TRAIN = parse(Int, get(ENV, "FIX_SLOW_N_TRAIN", "3"))
const EPOCHS = parse(Int, get(ENV, "FIX_SLOW_EPOCHS", "1"))
const BATCHSIZE = parse(Int, get(ENV, "FIX_SLOW_BATCHSIZE", "1"))
const MU_SURR = parse(Float64, get(ENV, "FIX_SLOW_MU_SURR", "1.0"))
const MU_REF = parse(Float64, get(ENV, "FIX_SLOW_MU_REF", "1.0"))
const STEP_SIZE = parse(Float64, get(ENV, "FIX_SLOW_STEP_SIZE", "1e-3"))
const DISPLAY_ITERATIONS = get(ENV, "FIX_SLOW_DISPLAY", "0") == "1"

function seconds(f)
    GC.gc()
    started = time_ns()
    result = f()
    return (time_ns() - started) / 1e9, result
end

function make_old_case(n_train)
    Random.seed!(1234)
    problem_data = ResourceAllocationProblemData(μᵢⱼ, vec(cz), vec(qw), vec(ρᵢ))
    problem_instance = ResourceAllocationProblem(problem_data)
    training_data, _, _, _ =
        ProblemBasedScenarioGeneration.dataGeneration(problem_instance, n_train, 0, 1, 5, 2, 3)
    model = construct_neural_network(problem_instance; nr_of_scenarios=1)

    input_loss(y_hat, y) = loss(problem_instance, MU_SURR, MU_REF, y_hat, y)
    input_relative_loss(y_hat, y) =
        relative_loss(problem_instance, MU_SURR, MU_REF, y_hat, y)

    return (; problem_instance, training_data, model, input_loss, input_relative_loss)
end

function run_training!(case; epochs)
    train!(
        case.input_loss,
        case.input_relative_loss,
        case.model,
        case.training_data;
        opt=Flux.Adam(STEP_SIZE),
        epochs=epochs,
        batchsize=BATCHSIZE,
        display_iterations=DISPLAY_ITERATIONS,
        save_model=false,
    )
end

setup_seconds, case = seconds(() -> make_old_case(N_TRAIN))
warmup_case = make_old_case(1)
warmup_seconds, _ = seconds(() -> run_training!(warmup_case; epochs=1))
training_seconds, _ = seconds(() -> run_training!(case; epochs=EPOCHS))

iterations = EPOCHS * length(case.training_data)
seconds_per_iteration = training_seconds / iterations

result_path = joinpath(@__DIR__, DISPLAY_ITERATIONS ? "old_display_results.csv" : "old_results.csv")
open(result_path, "w") do io
    println(io, "implementation,n_train,epochs,batchsize,display,setup_seconds,warmup_seconds,training_seconds,iterations,seconds_per_iteration")
    println(
        io,
        join(
            (
                "old_prototype",
                N_TRAIN,
                EPOCHS,
                BATCHSIZE,
                DISPLAY_ITERATIONS,
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

println("implementation=old_prototype")
println("n_train=$(N_TRAIN)")
println("epochs=$(EPOCHS)")
println("batchsize=$(BATCHSIZE)")
println("display=$(DISPLAY_ITERATIONS)")
println("setup_seconds=$(setup_seconds)")
println("warmup_seconds=$(warmup_seconds)")
println("training_seconds=$(training_seconds)")
println("iterations=$(iterations)")
println("seconds_per_iteration=$(seconds_per_iteration)")
