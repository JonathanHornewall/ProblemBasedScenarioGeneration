import Printf

const REPO_ROOT = normpath(joinpath(@__DIR__, "..", "..", "..", "..", ".."))
const NEW_PROJECT = normpath(joinpath(@__DIR__, "..", ".."))
const REPEATS = parse(Int, get(ENV, "CDFL_BENCH_REPEATS", "3"))
const WARMUPS = parse(Int, get(ENV, "CDFL_BENCH_WARMUPS", "2"))
const RUN_PROFILES = get(ENV, "CDFL_BENCH_PROFILE", "1") == "1"

function write_temp_script(source)
    path = tempname() * ".jl"
    write(path, source)
    return path
end

function benchmark_command(script, project, repo_root)
    return `$(Base.julia_cmd()) --project=$(project) $(script) $(repo_root) $(REPEATS) $(WARMUPS) $(Int(RUN_PROFILES))`
end

function run_benchmark(label, command)
    println("Running $(label) benchmark...")
    output = read(command, String)
    print(output)
    return output
end

function parse_results(output)
    rows = NamedTuple[]
    for line in split(output, '\n')
        startswith(line, "RESULT\t") || continue
        fields = split(line, '\t')
        length(fields) == 8 || continue
        push!(
            rows,
            (;
                implementation=fields[2],
                measurement=fields[3],
                min_seconds=parse(Float64, fields[4]),
                mean_seconds=parse(Float64, fields[5]),
                median_seconds=parse(Float64, fields[6]),
                mean_alloc_mib=parse(Float64, fields[7]),
                value=fields[8],
            ),
        )
    end
    return rows
end

function print_table(rows)
    isempty(rows) && return
    println()
    println("Timing summary")
    Printf.@printf(
        "%-14s  %-24s  %10s  %10s  %10s  %12s  %s\n",
        "impl",
        "measurement",
        "min_s",
        "mean_s",
        "median_s",
        "alloc_MiB",
        "value",
    )
    println("-"^105)
    for row in rows
        Printf.@printf(
            "%-14s  %-24s  %10.4f  %10.4f  %10.4f  %12.2f  %s\n",
            row.implementation,
            row.measurement,
            row.min_seconds,
            row.mean_seconds,
            row.median_seconds,
            row.mean_alloc_mib,
            row.value,
        )
    end
end

const OLD_SCRIPT = raw"""
repo_root = ARGS[1]
repeats = parse(Int, ARGS[2])
warmups = parse(Int, ARGS[3])
run_profiles = parse(Int, ARGS[4]) == 1

import Pkg
try
    redirect_stdout(devnull) do
        redirect_stderr(devnull) do
            Pkg.develop(path=joinpath(repo_root, "src", "ProblemBasedScenarioGeneration"))
            Pkg.add(["Flux", "ChainRulesCore"])
            Pkg.instantiate()
        end
    end
catch error
    @error "temporary old-package environment setup failed" exception=(error, catch_backtrace())
    rethrow()
end

using ChainRulesCore
using Flux
using LinearAlgebra
using Printf
using ProblemBasedScenarioGeneration
using Profile
using Statistics

using ProblemBasedScenarioGeneration: ResourceAllocationProblemData,
    ResourceAllocationProblem,
    LogBarCanLP,
    LogBarCanLP_standard_solver,
    TwoStageSLP,
    diff_opt_b,
    diff_s1_cost,
    s1_cost,
    scenario_collection_realization

import ProblemBasedScenarioGeneration: loss, surrogate_solution

include(joinpath(repo_root, "scripts", "resource_allocation_prototype", "custom_code", "neural_net.jl"))
include(joinpath(repo_root, "src", "ContextualDFL", "ContextualDFLExperiments", "src", "implementations", "resource_allocation_problem", "problem_data", "parameters.jl"))

problem_data = ResourceAllocationProblemData(
    RESOURCE_ALLOCATION_SERVICE_RATE_PARAMETERS,
    vec(Float64.(RESOURCE_ALLOCATION_FIRST_STAGE_COSTS)),
    vec(Float64.(RESOURCE_ALLOCATION_SECOND_STAGE_COSTS)),
    vec(Float64.(RESOURCE_ALLOCATION_YIELD_PARAMETERS)),
)
problem = ResourceAllocationProblem(problem_data)

mu_in = 1.0
mu_ref = 1.0
demand_count = size(problem.problem_data.service_rate_parameters, 2)
predicted_demand = 50.0 .+ 0.1 .* collect(1:demand_count)
actual_demand = 55.0 .+ 0.2 .* collect(1:demand_count)
z_for_cost = surrogate_solution(problem, mu_in, predicted_demand)

function summary_value(value)
    if value isa Number
        return Printf.@sprintf("%.6g", Float64(value))
    elseif value isa AbstractArray
        return Printf.@sprintf("array(len=%d,norm=%.6g)", length(value), LinearAlgebra.norm(value))
    else
        return string(typeof(value))
    end
end

function emit_result(implementation, measurement, samples)
    times = [sample.time for sample in samples]
    bytes = [sample.bytes for sample in samples]
    Printf.@printf(
        "RESULT\t%s\t%s\t%.9f\t%.9f\t%.9f\t%.6f\t%s\n",
        implementation,
        measurement,
        minimum(times),
        Statistics.mean(times),
        Statistics.median(times),
        Statistics.mean(bytes) / 1024^2,
        summary_value(samples[end].value),
    )
end

function measure(implementation, measurement, f)
    for _ in 1:warmups
        f()
    end
    samples = NamedTuple[]
    for _ in 1:repeats
        GC.gc()
        push!(samples, @timed f())
    end
    emit_result(implementation, measurement, samples)
    return samples[end].value
end

function profile_once(implementation, measurement, f)
    run_profiles || return
    for _ in 1:warmups
        f()
    end
    Profile.init(delay=0.001)
    Profile.clear()
    println("PROFILE_START\t$(implementation)\t$(measurement)")
    @profile f()
    Profile.print(format=:flat, sortedby=:count, mincount=5, maxdepth=18)
    println("PROFILE_END\t$(implementation)\t$(measurement)")
end

old_loss(demand) = loss(problem, mu_in, mu_ref, demand, actual_demand)
old_relative_display(demand) = begin
    evaluated = loss(problem, mu_in, mu_ref, demand, actual_demand)
    reference = loss(problem, mu_ref, mu_ref, actual_demand, actual_demand)
    (evaluated - reference) / abs(reference)
end

measure("old", "forward_loss", () -> old_loss(predicted_demand))
measure("old", "gradient_demand", () -> Flux.gradient(d -> old_loss(d), predicted_demand)[1])
measure("old", "surrogate_solve", () -> surrogate_solution(problem, mu_in, predicted_demand))
measure("old", "recourse_cost", () -> primal_problem_cost(problem, mu_ref, actual_demand, z_for_cost))
measure("old", "recourse_gradient_z", () -> derivative_primal_problem_cost(problem, mu_ref, actual_demand, z_for_cost))
measure("old", "relative_display_loss", () -> old_relative_display(predicted_demand))

profile_once("old", "gradient_demand", () -> Flux.gradient(d -> old_loss(d), predicted_demand)[1])
"""

const NEW_SCRIPT = raw"""
repo_root = ARGS[1]
repeats = parse(Int, ARGS[2])
warmups = parse(Int, ARGS[3])
run_profiles = parse(Int, ARGS[4]) == 1

using ContextualDFL
using ContextualDFLExperiments
using LinearAlgebra
using Printf
using Profile
using Statistics

const Flux = ContextualDFL.Flux

mu_kw(value) = NamedTuple{(Symbol(Char(0x03bc)),)}((value,))

problem = ResourceAllocationProblem(default_resource_allocation_problem_data())
program = stochastic_program(problem)
solver = ContextualDFL.Solver(ContextualDFL.IpoptSolver(), ContextualDFL.HiGHSSolver())
vector_decoder = ResourceAllocationDemandVectorDecoder(problem)
parametric_decoder = ResourceAllocationDemandParametricDecoder(problem)
loss_object = ContextualDFL.DflScenLoss(
    vector_decoder,
    parametric_decoder,
    solver,
    program;
    nr_scenarios=1,
)

mu_in = 1.0
mu_ref = 1.0
demand_count = size(problem.problem_data.service_rate_parameters, 2)
predicted_demand = 50.0 .+ 0.1 .* collect(1:demand_count)
actual_demand = 55.0 .+ 0.2 .* collect(1:demand_count)
actual_scenario = ContextualDFL.ParametricScenario(;
    W_eq_xi=Float64[],
    W_ineq_xi=Float64[],
    T_eq_xi=Float64[],
    T_ineq_xi=Float64[],
    h_eq_xi=actual_demand,
    h_ineq_xi=Float64[],
    q_xi=Float64[],
)
actual_collection = [actual_scenario]
decoded_actual = ContextualDFL.decode_scenario_collection(parametric_decoder, actual_collection)
decoded_predicted = ContextualDFL.decode_scenario_collection(
    vector_decoder,
    predicted_demand;
    nr_scenarios=1,
)
z_for_cost = ContextualDFL.solve(solver, program, decoded_predicted...; mu_kw(mu_in)...)[1]

function summary_value(value)
    if value isa Number
        return Printf.@sprintf("%.6g", Float64(value))
    elseif value isa AbstractArray
        return Printf.@sprintf("array(len=%d,norm=%.6g)", length(value), LinearAlgebra.norm(value))
    else
        return string(typeof(value))
    end
end

function emit_result(implementation, measurement, samples)
    times = [sample.time for sample in samples]
    bytes = [sample.bytes for sample in samples]
    Printf.@printf(
        "RESULT\t%s\t%s\t%.9f\t%.9f\t%.9f\t%.6f\t%s\n",
        implementation,
        measurement,
        minimum(times),
        Statistics.mean(times),
        Statistics.median(times),
        Statistics.mean(bytes) / 1024^2,
        summary_value(samples[end].value),
    )
end

function measure(implementation, measurement, f)
    for _ in 1:warmups
        f()
    end
    samples = NamedTuple[]
    for _ in 1:repeats
        GC.gc()
        push!(samples, @timed f())
    end
    emit_result(implementation, measurement, samples)
    return samples[end].value
end

function profile_once(implementation, measurement, f)
    run_profiles || return
    for _ in 1:warmups
        f()
    end
    Profile.init(delay=0.001)
    Profile.clear()
    println("PROFILE_START\t$(implementation)\t$(measurement)")
    @profile f()
    Profile.print(format=:flat, sortedby=:count, mincount=5, maxdepth=18)
    println("PROFILE_END\t$(implementation)\t$(measurement)")
end

function new_loss(demand; kwargs...)
    return loss_object(demand, actual_collection, mu_in, mu_ref; nr_scenarios=1, kwargs...)
end

const reference_input = reduce(vcat, (scenario.h_eq_xi for scenario in actual_collection))
const reference_cache = Dict{Any,Float64}()

function reference_cache_key(; kwargs...)
    return Tuple((key, value) for (key, value) in pairs(kwargs))
end

function cached_reference_value(; kwargs...)
    key = reference_cache_key(; kwargs...)
    return get!(reference_cache, key) do
        Float64(
            loss_object(
                reference_input,
                actual_collection,
                mu_ref,
                mu_ref;
                nr_scenarios=1,
                kwargs...,
            ),
        )
    end
end

function new_relative_display(demand; kwargs...)
    evaluated = loss_object(demand, actual_collection, mu_in, mu_ref; nr_scenarios=1, kwargs...)
    reference = cached_reference_value(; kwargs...)
    return (evaluated - reference) / abs(reference)
end

function surrogate_solve(demand; kwargs...)
    decoded = ContextualDFL.decode_scenario_collection(vector_decoder, demand; nr_scenarios=1)
    return ContextualDFL.solve(solver, program, decoded...; mu_kw(mu_in)..., kwargs...)[1]
end

function recourse_cost(z; kwargs...)
    return ContextualDFL.cost_function(
        program,
        solver,
        z,
        decoded_actual...;
        mu_kw(mu_ref)...,
        kwargs...,
    )
end

function recourse_gradient_z(z; kwargs...)
    return Flux.gradient(z_value -> recourse_cost(z_value; kwargs...), z)[1]
end

function measure_suite(implementation; kwargs...)
    cached_reference_value(; kwargs...)
    measure(implementation, "forward_loss", () -> new_loss(predicted_demand; kwargs...))
    measure(
        implementation,
        "gradient_demand",
        () -> Flux.gradient(d -> new_loss(d; kwargs...), predicted_demand)[1],
    )
    measure(implementation, "surrogate_solve", () -> surrogate_solve(predicted_demand; kwargs...))
    measure(implementation, "recourse_cost", () -> recourse_cost(z_for_cost; kwargs...))
    measure(implementation, "recourse_gradient_z", () -> recourse_gradient_z(z_for_cost; kwargs...))
    measure(
        implementation,
        "relative_display_loss",
        () -> new_relative_display(predicted_demand; kwargs...),
    )
end

measure_suite("new_default")
measure_suite("new_tol_1e-9"; tol=1e-9)

profile_once(
    "new_default",
    "gradient_demand",
    () -> Flux.gradient(d -> new_loss(d), predicted_demand)[1],
)
"""

old_script = write_temp_script(OLD_SCRIPT)
new_script = write_temp_script(NEW_SCRIPT)
old_project = mktempdir()

try
    old_output = run_benchmark(
        "old ProblemBasedScenarioGeneration",
        benchmark_command(old_script, old_project, REPO_ROOT),
    )
    new_output = run_benchmark(
        "new ContextualDFL",
        benchmark_command(new_script, NEW_PROJECT, REPO_ROOT),
    )

    rows = vcat(parse_results(old_output), parse_results(new_output))
    print_table(rows)
finally
    rm(old_script; force=true)
    rm(new_script; force=true)
end
