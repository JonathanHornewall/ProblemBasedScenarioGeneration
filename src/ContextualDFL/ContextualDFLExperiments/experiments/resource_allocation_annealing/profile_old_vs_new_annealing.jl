import Dates
import Printf

const REPO_ROOT = normpath(joinpath(@__DIR__, "..", "..", "..", "..", ".."))
const NEW_PROJECT = normpath(joinpath(@__DIR__, "..", ".."))
const PROFILE_EPOCHS = parse(Int, get(ENV, "CDFL_PROFILE_EPOCHS", "3"))
const PROFILE_MODE = get(ENV, "CDFL_PROFILE_MODE", "both")
const PROFILE_REPEATS = parse(Int, get(ENV, "CDFL_PROFILE_REPEATS", "3"))
const PROFILE_WARMUPS = parse(Int, get(ENV, "CDFL_PROFILE_WARMUPS", "2"))
const PROFILE_DELAY = parse(Float64, get(ENV, "CDFL_PROFILE_DELAY", "0.001"))
const PROFILE_MINCOUNT = parse(Int, get(ENV, "CDFL_PROFILE_MINCOUNT", "5"))
const PROFILE_FULL_SAMPLES = parse(Int, get(ENV, "CDFL_PROFILE_FULL_SAMPLES", "100"))
const PROFILE_SMOKE_SAMPLES = parse(Int, get(ENV, "CDFL_PROFILE_SMOKE_SAMPLES", "10"))
const RESULT_ROOT = normpath(
    joinpath(
        @__DIR__,
        "results",
        "profiling_" * replace(string(Dates.now()), ':' => '-'),
    ),
)

function write_temp_script(source)
    path = tempname() * ".jl"
    write(path, source)
    return path
end

function worker_command(script, project, mode, stage_limit, impl, training_samples)
    out_dir = joinpath(RESULT_ROOT, mode, impl)
    mkpath(out_dir)
    return `$(Base.julia_cmd()) --project=$(project) $(script) $(REPO_ROOT) $(out_dir) $(PROFILE_EPOCHS) $(stage_limit) $(mode) $(PROFILE_REPEATS) $(PROFILE_WARMUPS) $(PROFILE_DELAY) $(PROFILE_MINCOUNT) $(training_samples)`
end

function run_worker(label, command)
    println()
    println("Running $(label)...")
    output = IOBuffer()
    io = open(command, "r")
    try
        for line in eachline(io)
            println(line)
            println(output, line)
        end
    finally
        close(io)
    end
    return String(take!(output))
end

function parse_rows(output, prefix, fields)
    rows = NamedTuple[]
    for line in split(output, '\n')
        startswith(line, prefix * "\t") || continue
        parts = split(line, '\t')
        length(parts) == length(fields) + 1 || continue
        values = parts[2:end]
        push!(rows, NamedTuple{Tuple(Symbol.(fields))}(Tuple(values)))
    end
    return rows
end

function print_bucket_table(rows)
    isempty(rows) && return
    println()
    println("Bucket timing summary")
    Printf.@printf("%-6s %-7s %-28s %11s %9s %11s %8s\n", "impl", "mode", "bucket", "seconds", "count", "alloc_MiB", "% total")
    println("-"^90)
    totals = Dict{Tuple{String,String},Float64}()
    for row in rows
        key = (row.impl, row.mode)
        totals[key] = get(totals, key, 0.0) + parse(Float64, row.seconds)
    end
    for row in rows
        seconds = parse(Float64, row.seconds)
        percent = 100 * seconds / max(totals[(row.impl, row.mode)], eps())
        Printf.@printf(
            "%-6s %-7s %-28s %11.3f %9s %11.2f %7.1f%%\n",
            row.impl,
            row.mode,
            row.bucket,
            seconds,
            row.count,
            parse(Float64, row.alloc_mib),
            percent,
        )
    end
end

function print_micro_table(rows)
    isempty(rows) && return
    println()
    println("Micro timing summary")
    Printf.@printf("%-6s %-7s %-28s %10s %10s %10s %11s %s\n", "impl", "mode", "measurement", "min_s", "mean_s", "median_s", "alloc_MiB", "value")
    println("-"^105)
    for row in rows
        Printf.@printf(
            "%-6s %-7s %-28s %10.4f %10.4f %10.4f %11.2f %s\n",
            row.impl,
            row.mode,
            row.measurement,
            parse(Float64, row.min_seconds),
            parse(Float64, row.mean_seconds),
            parse(Float64, row.median_seconds),
            parse(Float64, row.alloc_mib),
            row.value,
        )
    end
end

function print_run_table(rows)
    isempty(rows) && return
    println()
    println("Run summary")
    Printf.@printf("%-6s %-7s %7s %8s %11s %12s %12s\n", "impl", "mode", "stages", "epochs", "iterations", "total_s", "iter_ms")
    println("-"^78)
    for row in rows
        total = parse(Float64, row.total_seconds)
        iterations = parse(Int, row.iterations)
        Printf.@printf(
            "%-6s %-7s %7s %8s %11s %12.3f %12.3f\n",
            row.impl,
            row.mode,
            row.stages,
            row.epochs_per_stage,
            row.iterations,
            total,
            1000 * total / max(iterations, 1),
        )
    end
end

const SHARED_PROFILING_UTILS = raw"""
using LinearAlgebra
using Printf
using Profile
using Serialization
using Statistics

mutable struct BucketStats
    seconds::Float64
    bytes::Int
    count::Int
end

BucketStats() = BucketStats(0.0, 0, 0)

function add_bucket!(buckets, name, sample)
    bucket = get!(buckets, name, BucketStats())
    bucket.seconds += sample.time
    bucket.bytes += sample.bytes
    bucket.count += 1
    return sample.value
end

function time_bucket!(f, buckets, name)
    return add_bucket!(buckets, name, @timed f())
end

function emit_buckets(impl, mode, buckets)
    for name in sort(collect(keys(buckets)))
        bucket = buckets[name]
        Printf.@printf(
            "BUCKET\t%s\t%s\t%s\t%.9f\t%d\t%.6f\n",
            impl,
            mode,
            name,
            bucket.seconds,
            bucket.count,
            bucket.bytes / 1024^2,
        )
    end
end

function summary_value(value)
    if value isa Number
        return Printf.@sprintf("%.6g", Float64(value))
    elseif value isa AbstractArray
        return Printf.@sprintf("array(len=%d,norm=%.6g)", length(value), LinearAlgebra.norm(value))
    else
        return string(typeof(value))
    end
end

function measure(impl, mode, measurement, f, repeats, warmups)
    for _ in 1:warmups
        f()
    end
    samples = NamedTuple[]
    for _ in 1:repeats
        GC.gc()
        push!(samples, @timed f())
    end
    times = [sample.time for sample in samples]
    bytes = [sample.bytes for sample in samples]
    Printf.@printf(
        "MICRO\t%s\t%s\t%s\t%.9f\t%.9f\t%.9f\t%.6f\t%s\n",
        impl,
        mode,
        measurement,
        minimum(times),
        Statistics.mean(times),
        Statistics.median(times),
        Statistics.mean(bytes) / 1024^2,
        summary_value(samples[end].value),
    )
    return samples[end].value
end

function profile_to_file(f, path, delay, mincount)
    mkpath(dirname(path))
    Profile.init(delay=delay)
    Profile.clear()
    result = @profile f()
    open(path, "w") do io
        Profile.print(io; format=:flat, sortedby=:count, mincount=mincount, maxdepth=40)
    end
    return result
end

function emit_profile_file(impl, mode, name, path)
    println("PROFILE_FILE\t$(impl)\t$(mode)\t$(name)\t$(path)")
end
"""

const OLD_WORKER = SHARED_PROFILING_UTILS * raw"""
repo_root = ARGS[1]
out_dir = ARGS[2]
epochs_per_stage = parse(Int, ARGS[3])
stage_limit = parse(Int, ARGS[4])
mode = ARGS[5]
repeats = parse(Int, ARGS[6])
warmups = parse(Int, ARGS[7])
profile_delay = parse(Float64, ARGS[8])
profile_mincount = parse(Int, ARGS[9])
training_samples = parse(Int, ARGS[10])
impl = "old"

import Pkg
env_seconds = @elapsed redirect_stdout(devnull) do
    redirect_stderr(devnull) do
        Pkg.develop(path=joinpath(repo_root, "src", "ProblemBasedScenarioGeneration"))
        Pkg.add(["ChainRulesCore", "Flux", "Plots"])
        Pkg.instantiate()
    end
end
Printf.@printf("ENV_SETUP\t%s\t%s\t%.9f\n", impl, mode, env_seconds)

using ChainRulesCore
using Flux
using Plots
using ProblemBasedScenarioGeneration
using Random

using ProblemBasedScenarioGeneration: ResourceAllocationProblemData,
    ResourceAllocationProblem,
    dataGeneration
import ProblemBasedScenarioGeneration: loss, relative_loss, surrogate_solution

include(joinpath(repo_root, "scripts", "resource_allocation_prototype", "custom_code", "neural_net.jl"))
include(joinpath(repo_root, "scripts", "resource_allocation_prototype", "parameters.jl"))

Random.seed!(1234)

cz = vec(getfield(Main, Symbol("cz")))
qw = vec(getfield(Main, Symbol("qw")))
rho_i = vec(getfield(Main, Symbol("ρᵢ")))
service_rate_parameters = getfield(Main, Symbol("μᵢⱼ"))

problem_data = ResourceAllocationProblemData(service_rate_parameters, cz, qw, rho_i)
problem = ResourceAllocationProblem(problem_data)

Ntraining_samples = training_samples
Ntesting_samples = 30
sigma = 5
p = 2
L = 3
N_xi_per_x = 100
batchsize = 1
step_size = 1e-3
param_list = [1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.08, 0.06, 0.04, 0.02, 0.01]
stage_specs = [(reg, reg) for reg in param_list]
push!(stage_specs, (param_list[end], 0.0))
stage_specs = stage_specs[1:min(stage_limit, length(stage_specs))]

setup_buckets = Dict{String,BucketStats}()
data_set_training, data_set_testing, _, _ = time_bucket!(setup_buckets, "data_generation") do
    dataGeneration(problem, Ntraining_samples, Ntesting_samples, N_xi_per_x, sigma, p, L)
end
model = time_bucket!(setup_buckets, "model_construction") do
    construct_neural_network(problem; nr_of_scenarios=1)
end

xs = collect(keys(data_set_training))
xis = collect(values(data_set_training))
N = length(xs)

function old_loss_mb(loss_fn, model, Xb, Xib)
    return Statistics.mean(loss_fn(model(Xb[:, i:i]), Xib[:, i:i]) for i in 1:size(Xb, 2))
end

function old_relative_loss_mb(relative_fn, model, Xb, Xib)
    return Statistics.mean(relative_fn(model(Xb[:, i:i]), Xib[:, i:i]) for i in 1:size(Xb, 2))
end

function run_epoch!(model, opt, loss_fn, relative_fn, buckets)
    state = time_bucket!(buckets, "optimizer_setup") do
        Flux.setup(opt, model)
    end
    epoch_losses = Float64[]
    for idxs in Iterators.partition(1:N, batchsize)
        Xb = time_bucket!(buckets, "batch_materialization") do
            hcat(xs[idxs]...)
        end
        Xib = time_bucket!(buckets, "batch_materialization") do
            hcat(xis[idxs]...)
        end
        gs = time_bucket!(buckets, "training_gradient") do
            Flux.gradient(model) do m
                old_loss_mb(loss_fn, m, Xb, Xib)
            end
        end
        gmodel = gs isa Tuple ? gs[1] : gs
        time_bucket!(buckets, "optimizer_update") do
            Flux.update!(state, model, gmodel)
        end
        display_value = time_bucket!(buckets, "relative_display_loss") do
            old_relative_loss_mb(relative_fn, model, Xb, Xib)
        end
        push!(epoch_losses, Float64(display_value))
    end
    time_bucket!(buckets, "gc") do
        GC.gc()
    end
    return Statistics.mean(epoch_losses)
end

function run_annealing!(model, buckets)
    stage_seconds = Float64[]
    total_iterations = 0
    for (stage_index, (reg_param_surr, reg_param_prim)) in enumerate(stage_specs)
        println("Starting old stage $(stage_index) with reg_param_surr=$(reg_param_surr), reg_param_prim=$(reg_param_prim), epochs=$(epochs_per_stage)")
        stage_started = time()
        stage_losses = Float64[]
        loss_fn(output, actual) = loss(problem, reg_param_surr, reg_param_prim, output, actual)
        relative_fn(output, actual) = relative_loss(problem, reg_param_surr, reg_param_prim, output, actual)
        for epoch in 1:epochs_per_stage
            average_display = run_epoch!(model, Flux.Adam(step_size), loss_fn, relative_fn, buckets)
            push!(stage_losses, average_display)
            total_iterations += N
            println("Epoch $(epoch) with avg loss $(average_display) ($(N) iterations)")
        end
        time_bucket!(buckets, "save_model") do
            Serialization.serialize(joinpath(out_dir, "old_model_stage_$(stage_index).jls"), model)
        end
        time_bucket!(buckets, "save_state") do
            Serialization.serialize(
                joinpath(out_dir, "old_state_stage_$(stage_index).jls"),
                (; model=model, data_set_training=data_set_training, data_set_testing=data_set_testing, stage=stage_index),
            )
        end
        time_bucket!(buckets, "plot_creation") do
            Plots.plot(1:length(stage_losses), stage_losses; xlabel="Epoch", ylabel="Loss", title="Training Loss")
        end
        stage_elapsed = time() - stage_started
        push!(stage_seconds, stage_elapsed)
        Printf.@printf("STAGE\t%s\t%s\t%d\t%.9f\t%d\n", impl, mode, stage_index, stage_elapsed, epochs_per_stage * N)
    end
    return total_iterations, stage_seconds
end

function run_micro_measurements()
    Xb = hcat(xs[1])
    Xib = hcat(xis[1])
    reg_param_surr = first(stage_specs)[1]
    reg_param_prim = first(stage_specs)[2]
    loss_fn(output, actual) = loss(problem, reg_param_surr, reg_param_prim, output, actual)
    relative_fn(output, actual) = relative_loss(problem, reg_param_surr, reg_param_prim, output, actual)
    predicted_demand = reshape(50.0 .+ 0.1 .* collect(1:size(problem.problem_data.service_rate_parameters, 2)), :, 1)
    actual_demand = reshape(55.0 .+ 0.2 .* collect(1:size(problem.problem_data.service_rate_parameters, 2)), :, 1)
    z_for_cost = surrogate_solution(problem, reg_param_surr, predicted_demand)

    measure(impl, mode, "model_forward", () -> model(Xb), repeats, warmups)
    measure(impl, mode, "loss_forward", () -> loss_fn(model(Xb), Xib), repeats, warmups)
    measure(impl, mode, "training_gradient", () -> Flux.gradient(m -> old_loss_mb(loss_fn, m, Xb, Xib), model)[1], repeats, warmups)
    measure(impl, mode, "relative_display_loss", () -> old_relative_loss_mb(relative_fn, model, Xb, Xib), repeats, warmups)
    measure(impl, mode, "forward_loss_fixed_demand", () -> loss(problem, reg_param_surr, reg_param_prim, predicted_demand, actual_demand), repeats, warmups)
    measure(impl, mode, "gradient_demand", () -> Flux.gradient(d -> loss(problem, reg_param_surr, reg_param_prim, d, actual_demand), predicted_demand)[1], repeats, warmups)
    measure(impl, mode, "surrogate_solve", () -> surrogate_solution(problem, reg_param_surr, predicted_demand), repeats, warmups)
    measure(impl, mode, "recourse_cost", () -> primal_problem_cost(problem, reg_param_prim, actual_demand, z_for_cost), repeats, warmups)
    measure(impl, mode, "recourse_gradient_z", () -> derivative_primal_problem_cost(problem, reg_param_prim, actual_demand, z_for_cost), repeats, warmups)

    gradient_profile_path = joinpath(out_dir, "profile_old_training_gradient.txt")
    profile_to_file(gradient_profile_path, profile_delay, profile_mincount) do
        Flux.gradient(m -> old_loss_mb(loss_fn, m, Xb, Xib), model)[1]
    end
    emit_profile_file(impl, mode, "training_gradient", gradient_profile_path)

    relative_profile_path = joinpath(out_dir, "profile_old_relative_loss.txt")
    profile_to_file(relative_profile_path, profile_delay, profile_mincount) do
        old_relative_loss_mb(relative_fn, model, Xb, Xib)
    end
    emit_profile_file(impl, mode, "relative_loss", relative_profile_path)

    if mode == "smoke"
        iteration_profile_path = joinpath(out_dir, "profile_old_training_iteration.txt")
        profile_to_file(iteration_profile_path, profile_delay, profile_mincount) do
            state = Flux.setup(Flux.Adam(step_size), model)
            gs = Flux.gradient(model) do m
                old_loss_mb(loss_fn, m, Xb, Xib)
            end
            gmodel = gs isa Tuple ? gs[1] : gs
            Flux.update!(state, model, gmodel)
            old_relative_loss_mb(relative_fn, model, Xb, Xib)
        end
        emit_profile_file(impl, mode, "training_iteration", iteration_profile_path)
    end
end

run_micro_measurements()

full_buckets = merge(Dict{String,BucketStats}(), setup_buckets)
run_sample = @timed begin
    run_annealing!(model, full_buckets)
end
run_result = run_sample.value

total_iterations, stage_seconds = run_result
total_seconds = sum(stage_seconds)
Printf.@printf(
    "SUMMARY\t%s\t%s\t%d\t%d\t%d\t%.9f\n",
    impl,
    mode,
    length(stage_specs),
    epochs_per_stage,
    total_iterations,
    total_seconds,
)
emit_buckets(impl, mode, full_buckets)
"""

const NEW_WORKER = SHARED_PROFILING_UTILS * raw"""
repo_root = ARGS[1]
out_dir = ARGS[2]
epochs_per_stage = parse(Int, ARGS[3])
stage_limit = parse(Int, ARGS[4])
mode = ARGS[5]
repeats = parse(Int, ARGS[6])
warmups = parse(Int, ARGS[7])
profile_delay = parse(Float64, ARGS[8])
profile_mincount = parse(Int, ARGS[9])
training_samples = parse(Int, ARGS[10])
impl = "new"

using ContextualDFL
using ContextualDFLExperiments
using Random

const Flux = ContextualDFL.Flux

Random.seed!(1234)

Ntraining_samples = training_samples
Ntesting_samples = 30
sigma = 5
p = 2
L = 3
N_xi_per_x = 100
batchsize = 1
step_size = 1e-3
nr_scenarios = 1
param_list = [1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.08, 0.06, 0.04, 0.02, 0.01]
stage_specs = [(reg, reg) for reg in param_list]
push!(stage_specs, (param_list[end], 0.0))
stage_specs = stage_specs[1:min(stage_limit, length(stage_specs))]

setup_buckets = Dict{String,BucketStats}()
problem = time_bucket!(setup_buckets, "problem_construction") do
    ResourceAllocationProblem(default_resource_allocation_problem_data())
end
program = stochastic_program(problem)
solver = ContextualDFL.Solver(ContextualDFL.IpoptSolver(), ContextualDFL.HiGHSSolver())
vector_decoder = ResourceAllocationDemandVectorDecoder(problem)
parametric_decoder = ResourceAllocationDemandParametricDecoder(problem)
loss_object = ContextualDFL.DflScenLoss(
    vector_decoder,
    parametric_decoder,
    solver,
    program;
    nr_scenarios=nr_scenarios,
)

rng = Random.MersenneTwister(1234)
context_generator = ResourceAllocationContextDataGenerator(rng=rng)
scenario_generator = ResourceAllocationScenarioDataGenerator(
    problem;
    sigma=sigma,
    p=p,
    L=L,
    rng=rng,
)
data_set_training = time_bucket!(setup_buckets, "data_generation") do
    contexts = [Vector{Float64}(context_generator()) for _ in 1:Ntraining_samples]
    scenarios = [scenario_generator(context) for context in contexts]
    generate_contextual_data_set(contexts, scenarios)
end
demand_count = size(problem.problem_data.service_rate_parameters, 2)
model = time_bucket!(setup_buckets, "model_construction") do
    Flux.Chain(
        Flux.Dense(3 => 128, Flux.relu),
        Flux.Dense(128 => 128, Flux.relu),
        Flux.Dense(128 => 128, Flux.relu),
        Flux.Dense(128 => demand_count * nr_scenarios, Flux.relu),
    ) |> Flux.f64
end

N = length(data_set_training)
loss_kwargs = (; nr_scenarios=nr_scenarios)

context_at(index) = data_set_training[index].context
scenario_at(index) = data_set_training[index].scenario_parameters

function new_loss_batch(loss_fn, model, idxs)
    return Statistics.mean(
        loss_fn(model(context_at(index)), scenario_at(index)) for index in idxs
    )
end

function new_relative_batch(relative_fn, model, idxs)
    return Statistics.mean(
        relative_fn(model(context_at(index)), scenario_at(index)) for index in idxs
    )
end

function run_epoch!(model, opt, loss_fn, relative_fn, buckets)
    state = time_bucket!(buckets, "optimizer_setup") do
        Flux.setup(opt, model)
    end
    epoch_losses = Float64[]
    for idxs_iter in Iterators.partition(1:N, batchsize)
        idxs = time_bucket!(buckets, "batch_materialization") do
            collect(idxs_iter)
        end
        loss_value, gradients = time_bucket!(buckets, "training_gradient") do
            Flux.withgradient(model) do trainable_model
                new_loss_batch(loss_fn, trainable_model, idxs)
            end
        end
        time_bucket!(buckets, "optimizer_update") do
            Flux.update!(state, model, gradients[1])
        end
        display_value = time_bucket!(buckets, "relative_display_loss") do
            new_relative_batch(relative_fn, model, idxs)
        end
        push!(epoch_losses, Float64(display_value))
    end
    return Statistics.mean(epoch_losses)
end

function run_annealing!(model, buckets)
    stage_seconds = Float64[]
    total_iterations = 0
    for (stage_index, (mu_in, mu_ref)) in enumerate(stage_specs)
        println("Starting new stage $(stage_index) with mu_in=$(mu_in), mu_ref=$(mu_ref), epochs=$(epochs_per_stage)")
        stage_started = time()
        loss_fn(output, scenario_parameters) =
            loss_object(output, scenario_parameters, mu_in, mu_ref; loss_kwargs...)
        function relative_fn(output, scenario_parameters)
            evaluated_value = loss_object(output, scenario_parameters, mu_in, mu_ref; loss_kwargs...)
            reference_input = reduce(vcat, (scenario.h_eq_xi for scenario in scenario_parameters))
            reference_value = loss_object(reference_input, scenario_parameters, mu_ref, mu_ref; loss_kwargs...)
            return (evaluated_value - reference_value) / abs(reference_value)
        end
        for epoch in 1:epochs_per_stage
            average_display = run_epoch!(model, Flux.Adam(step_size), loss_fn, relative_fn, buckets)
            total_iterations += N
            println("Epoch $(epoch) with avg loss $(average_display) ($(N) iterations)")
        end
        time_bucket!(buckets, "save_model") do
            Serialization.serialize(joinpath(out_dir, "new_model_stage_$(stage_index).jls"), model)
        end
        time_bucket!(buckets, "save_state") do
            Serialization.serialize(
                joinpath(out_dir, "new_state_stage_$(stage_index).jls"),
                (; model=model, data_set_training=data_set_training, problem=problem, stage=stage_index),
            )
        end
        stage_elapsed = time() - stage_started
        push!(stage_seconds, stage_elapsed)
        Printf.@printf("STAGE\t%s\t%s\t%d\t%.9f\t%d\n", impl, mode, stage_index, stage_elapsed, epochs_per_stage * N)
    end
    return total_iterations, stage_seconds
end

function run_micro_measurements()
    idxs = [1]
    mu_in, mu_ref = first(stage_specs)
    loss_fn(output, scenario_parameters) =
        loss_object(output, scenario_parameters, mu_in, mu_ref; loss_kwargs...)
    function relative_fn(output, scenario_parameters)
        evaluated_value = loss_object(output, scenario_parameters, mu_in, mu_ref; loss_kwargs...)
        reference_input = reduce(vcat, (scenario.h_eq_xi for scenario in scenario_parameters))
        reference_value = loss_object(reference_input, scenario_parameters, mu_ref, mu_ref; loss_kwargs...)
        return (evaluated_value - reference_value) / abs(reference_value)
    end

    predicted_demand = 50.0 .+ 0.1 .* collect(1:demand_count)
    actual_demand = 55.0 .+ 0.2 .* collect(1:demand_count)
    actual_collection = [ContextualDFL.ParametricScenario(; h_eq_xi=actual_demand)]
    decoded_predicted = ContextualDFL.decode_scenario_collection(
        vector_decoder,
        predicted_demand;
        nr_scenarios=1,
    )
    decoded_actual = ContextualDFL.decode_scenario_collection(parametric_decoder, actual_collection)
    z_for_cost = ContextualDFL.solve(solver, program, decoded_predicted...; μ=mu_in)[1]

    measure(impl, mode, "model_forward", () -> model(context_at(1)), repeats, warmups)
    measure(impl, mode, "loss_forward", () -> loss_fn(model(context_at(1)), scenario_at(1)), repeats, warmups)
    measure(impl, mode, "training_gradient", () -> Flux.withgradient(m -> new_loss_batch(loss_fn, m, idxs), model)[2][1], repeats, warmups)
    measure(impl, mode, "relative_display_loss", () -> new_relative_batch(relative_fn, model, idxs), repeats, warmups)
    measure(impl, mode, "forward_loss_fixed_demand", () -> loss_object(predicted_demand, actual_collection, mu_in, mu_ref; nr_scenarios=1), repeats, warmups)
    measure(impl, mode, "gradient_demand", () -> Flux.gradient(d -> loss_object(d, actual_collection, mu_in, mu_ref; nr_scenarios=1), predicted_demand)[1], repeats, warmups)
    measure(impl, mode, "surrogate_solve", () -> ContextualDFL.solve(solver, program, decoded_predicted...; μ=mu_in)[1], repeats, warmups)
    measure(impl, mode, "recourse_cost", () -> ContextualDFL.cost_function(program, solver, z_for_cost, decoded_actual...; μ=mu_ref), repeats, warmups)
    measure(impl, mode, "recourse_gradient_z", () -> Flux.gradient(z -> ContextualDFL.cost_function(program, solver, z, decoded_actual...; μ=mu_ref), z_for_cost)[1], repeats, warmups)

    gradient_profile_path = joinpath(out_dir, "profile_new_training_gradient.txt")
    profile_to_file(gradient_profile_path, profile_delay, profile_mincount) do
        Flux.withgradient(m -> new_loss_batch(loss_fn, m, idxs), model)[2][1]
    end
    emit_profile_file(impl, mode, "training_gradient", gradient_profile_path)

    relative_profile_path = joinpath(out_dir, "profile_new_relative_loss.txt")
    profile_to_file(relative_profile_path, profile_delay, profile_mincount) do
        new_relative_batch(relative_fn, model, idxs)
    end
    emit_profile_file(impl, mode, "relative_loss", relative_profile_path)

    if mode == "smoke"
        iteration_profile_path = joinpath(out_dir, "profile_new_training_iteration.txt")
        profile_to_file(iteration_profile_path, profile_delay, profile_mincount) do
            state = Flux.setup(Flux.Adam(step_size), model)
            loss_value, gradients = Flux.withgradient(model) do trainable_model
                new_loss_batch(loss_fn, trainable_model, idxs)
            end
            Flux.update!(state, model, gradients[1])
            new_relative_batch(relative_fn, model, idxs)
        end
        emit_profile_file(impl, mode, "training_iteration", iteration_profile_path)
    end
end

run_micro_measurements()

full_buckets = merge(Dict{String,BucketStats}(), setup_buckets)
run_sample = @timed begin
    run_annealing!(model, full_buckets)
end
run_result = run_sample.value

total_iterations, stage_seconds = run_result
total_seconds = sum(stage_seconds)
Printf.@printf(
    "SUMMARY\t%s\t%s\t%d\t%d\t%d\t%.9f\n",
    impl,
    mode,
    length(stage_specs),
    epochs_per_stage,
    total_iterations,
    total_seconds,
)
emit_buckets(impl, mode, full_buckets)
"""

function modes_to_run()
    if PROFILE_MODE == "both"
        return [("smoke", 1), ("full", 12)]
    elseif PROFILE_MODE == "smoke"
        return [("smoke", 1)]
    elseif PROFILE_MODE == "full"
        return [("full", 12)]
    else
        error("CDFL_PROFILE_MODE must be one of: both, smoke, full")
    end
end

old_script = write_temp_script(OLD_WORKER)
new_script = write_temp_script(NEW_WORKER)
old_project = mktempdir()
mkpath(RESULT_ROOT)

all_outputs = String[]
try
    println("Profile output directory: $(RESULT_ROOT)")
    for (mode, stage_limit) in modes_to_run()
        training_samples = mode == "smoke" ? PROFILE_SMOKE_SAMPLES : PROFILE_FULL_SAMPLES
        push!(
            all_outputs,
            run_worker(
                "old $(mode) profile",
                worker_command(old_script, old_project, mode, stage_limit, "old", training_samples),
            ),
        )
        push!(
            all_outputs,
            run_worker(
                "new $(mode) profile",
                worker_command(new_script, NEW_PROJECT, mode, stage_limit, "new", training_samples),
            ),
        )
    end
finally
    rm(old_script; force=true)
    rm(new_script; force=true)
end

combined_output = join(all_outputs, "\n")
summary_rows = parse_rows(
    combined_output,
    "SUMMARY",
    ["impl", "mode", "stages", "epochs_per_stage", "iterations", "total_seconds"],
)
bucket_rows = parse_rows(
    combined_output,
    "BUCKET",
    ["impl", "mode", "bucket", "seconds", "count", "alloc_mib"],
)
micro_rows = parse_rows(
    combined_output,
    "MICRO",
    [
        "impl",
        "mode",
        "measurement",
        "min_seconds",
        "mean_seconds",
        "median_seconds",
        "alloc_mib",
        "value",
    ],
)

print_run_table(summary_rows)
print_bucket_table(bucket_rows)
print_micro_table(micro_rows)

println()
println("Profile files are under: $(RESULT_ROOT)")
