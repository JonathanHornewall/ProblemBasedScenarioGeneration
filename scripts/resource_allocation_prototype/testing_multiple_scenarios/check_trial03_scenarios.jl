import Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()

using LinearAlgebra
using Random
using Printf
using ProblemBasedScenarioGeneration
using ProblemBasedScenarioGeneration: ResourceAllocationProblemData, ResourceAllocationProblem,
    dataGeneration, load_trained_model

# Problem setup matches annealing.jl defaults
include(joinpath(@__DIR__, "..", "parameters.jl"))
cz_vec, qw_vec, ρ_vec = vec(cz), vec(qw), vec(ρᵢ)
problem_data = ResourceAllocationProblemData(μᵢⱼ, cz_vec, qw_vec, ρ_vec)
problem_instance = ResourceAllocationProblem(problem_data)

# Load the perturbed model from trial 03 (three-scenario run)
model_path = joinpath(@__DIR__, "results", "run_20251008_134918", "scenarios_3",
    "trial_03", "perturbed", "stage_12_model.jls")
@assert isfile(model_path) "Expected model file not found at $(model_path)"
model = load_trained_model(model_path)

# Generate a fresh context vector x using the same data generator as annealing.jl
Random.seed!(1234)  # deterministic sample for repeatability
Ntrain, Ntest = 1, 1
N_xi_per_x = 100
σ, p, L = 5.0, 2, 3
training_data, _, _, _ = dataGeneration(problem_instance, Ntrain, Ntest,
    N_xi_per_x, σ, p, L)
(x_sample, _) = first(training_data)

# Evaluate the model on the sampled context
scenarios = model(x_sample)
scenario_count = size(scenarios, 2)

@assert scenario_count == 3 "Loaded model does not emit three scenarios (found $(scenario_count))."

function describe_scenarios(matrix::AbstractMatrix; atol::Float64 = 1e-6, rtol::Float64 = 1e-3)
    col_norms = [norm(matrix[:, c]) for c in 1:size(matrix, 2)]
    pair_diffs = Dict{Tuple{Int, Int}, NamedTuple{(:abs_norm, :rel_norm, :max_abs, :cosine, :angle_deg)}}()
    identical = true

    for i in 1:size(matrix, 2) - 1
        for j in i + 1:size(matrix, 2)
            diff_vec = matrix[:, i] - matrix[:, j]
            abs_norm = norm(diff_vec)
            max_abs = maximum(abs.(diff_vec))
            baseline = max(col_norms[i], col_norms[j], atol)
            rel_norm = abs_norm / baseline
            if col_norms[i] > 0 && col_norms[j] > 0
                cosine = clamp(dot(matrix[:, i], matrix[:, j]) / (col_norms[i] * col_norms[j]), -1.0, 1.0)
                angle_deg = acos(cosine) * (180 / π)
            else
                cosine = NaN
                angle_deg = NaN
            end
            pair_diffs[(i, j)] = (abs_norm = abs_norm, rel_norm = rel_norm, max_abs = max_abs,
                cosine = cosine, angle_deg = angle_deg)
            if abs_norm > atol && rel_norm > rtol
                identical = false
            end
        end
    end
    return identical, col_norms, pair_diffs
end

all_identical, col_norms, diff_map = describe_scenarios(scenarios)

println("Context x (first 3 entries): ", x_sample[1:min(end, 3)])
println("Scenario matrix size: ", size(scenarios))

for (idx, col_norm) in enumerate(col_norms)
    println(@sprintf("Scenario %d ℓ₂-norm: %.6e", idx, col_norm))
end

for (cols, diff) in sort(collect(diff_map); by = first)
    stats = diff_map[cols]
    println(@sprintf("‣ Pair (%d, %d): |Δ|₂ = %.6e, rel |Δ|₂ = %.6e, max |Δ|∞ = %.6e, cos θ = %.6f, θ = %.2f°",
        cols[1], cols[2], stats.abs_norm, stats.rel_norm, stats.max_abs, stats.cosine, stats.angle_deg))
end

if all_identical
    println("Result: All three generated scenarios are numerically identical within tolerances (abs ≤ 1e-6 and rel ≤ 1e-3).")
else
    println("Result: Scenarios differ — at least one pair varies above the absolute or relative tolerance.")
end
