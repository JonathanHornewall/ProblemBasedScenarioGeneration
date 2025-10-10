import Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

using LinearAlgebra
using Statistics
using Printf
using Random
using ProblemBasedScenarioGeneration
using ProblemBasedScenarioGeneration: ResourceAllocationProblemData, ResourceAllocationProblem,
    dataGeneration, scenario_realization, TwoStageSLP, CanLP, optimal_value

include("parameters.jl")

const DEFAULT_NR_OF_SCENARIOS_PER_SAMPLE = 100
const DEFAULT_NR_OF_SAMPLES = 30
const DEFAULT_SIGMA = 5.0
const DEFAULT_P = 2
const DEFAULT_L = 3
const COLLECTIONS_PER_SAMPLE = 1
const DEFAULT_SEED = 2025

function parse_cli_args(args::Vector{String})
    options = Dict{String, String}()
    for arg in args
        startswith(arg, "--") || continue
        body = arg[3:end]
        parts = split(body, "=", limit = 2)
        if length(parts) == 2
            options[parts[1]] = parts[2]
        else
            options[parts[1]] = "true"
        end
    end
    return options
end

options = parse_cli_args(ARGS)

N = DEFAULT_NR_OF_SCENARIOS_PER_SAMPLE
M = DEFAULT_NR_OF_SAMPLES
σ = DEFAULT_SIGMA
p = DEFAULT_P
L = DEFAULT_L
seed = DEFAULT_SEED

if haskey(options, "scenarios")
    N = parse(Int, options["scenarios"])
end
if haskey(options, "samples")
    M = parse(Int, options["samples"])
end
if haskey(options, "sigma")
    σ = parse(Float64, options["sigma"])
end
if haskey(options, "p")
    p = parse(Int, options["p"])
end
if haskey(options, "L")
    L = parse(Int, options["L"])
end
if haskey(options, "seed")
    seed = parse(Int, options["seed"])
end

Random.seed!(seed)

println("Value-of-prescience experiment")
println("  Samples (contexts): $M")
println("  Scenarios per context: $N")
println("  σ = $σ, p = $p, L = $L, seed = $seed")

cz_vec, qw_vec, ρ_vec = vec(cz), vec(qw), vec(ρᵢ)
problem_data = ResourceAllocationProblemData(μᵢⱼ, cz_vec, qw_vec, ρ_vec)
problem_instance = ResourceAllocationProblem(problem_data)

function build_tslp(problem_instance, scenario_batch)
    A = problem_instance.s1_constraint_matrix
    b = problem_instance.s1_constraint_vector
    c = problem_instance.s1_cost_vector

    Ws, Ts, hs, qs = Matrix{Float64}[], Matrix{Float64}[], Vector{Float64}[], Vector{Float64}[]
    for ξ in scenario_batch
        W, T, h, q = scenario_realization(problem_instance, ξ)
        push!(Ws, W)
        push!(Ts, T)
        push!(hs, h)
        push!(qs, q)
    end

    Ws_array = cat(Ws..., dims = 3)
    Ts_array = cat(Ts..., dims = 3)
    hs_array = hcat(hs...)
    qs_array = hcat(qs...)

    return TwoStageSLP(A, b, c, Ws_array, Ts_array, hs_array, qs_array)
end

function compute_values(problem_instance, ξ_tensor)
    collections = size(ξ_tensor, 1)
    @assert collections == 1 "Expected a single collection, got $(collections)"
    scenario_count = size(ξ_tensor, 2)

    scenario_vectors = [vec(ξ_tensor[1, k, :]) for k in 1:scenario_count]

    saa_slp = build_tslp(problem_instance, scenario_vectors)
    saa_value = optimal_value(CanLP(saa_slp))

    prescient_values = Float64[]
    for ξ in scenario_vectors
        slp_single = build_tslp(problem_instance, (ξ,))
        push!(prescient_values, optimal_value(CanLP(slp_single)))
    end

    avg_prescient = mean(prescient_values)
    rel_diff = (avg_prescient - saa_value) / abs(saa_value)

    return saa_value, avg_prescient, rel_diff
end

datasets = dataGeneration(problem_instance, M, M, N, σ, p, L, COLLECTIONS_PER_SAMPLE)
_, data_set_testing, _, _ = datasets

context_entries = collect(data_set_testing)
if length(context_entries) < M
    @warn "Generated only $(length(context_entries)) contexts; expected $M"
end

for (idx, (x, ξ_tensor)) in enumerate(context_entries)
    saa_value, avg_prescient, rel_diff = compute_values(problem_instance, ξ_tensor)
    println("Context $(idx):")
    println(@sprintf("  SAA optimal value: %.6f", saa_value))
    println(@sprintf("  Avg prescient value: %.6f", avg_prescient))
    println(@sprintf("  Relative difference: %.6e", rel_diff))
end
