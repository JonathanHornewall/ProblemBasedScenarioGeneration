module Artifacts

using Dates
using Logging

export artifact_manifest,
       ensure_step_directories,
       artifacts_present,
       copy_artifacts,
       mark_step_complete,
       write_json_file

const STEP_ARTIFACTS = Dict(
    :generate_testing_data => [
        "artifacts/testing/test_covariates.csv",
        "artifacts/testing/test_scenarios.jls",
        "artifacts/testing/full_context_pool.csv",
        "artifacts/testing/testing_seeds.json",
        "artifacts/testing/problem_parameters.jls"
    ],
    :compute_saa_baselines => [
        "artifacts/testing/saa_runs.csv",
        "artifacts/testing/saa_optima.csv"
    ],
    :generate_training_data => [
        "artifacts/training/training_pairs.jls",
        "artifacts/training/training_covariates.csv",
        "artifacts/training/data_generation_log.json"
    ],
    :train_baselines => [
        "artifacts/models/baselines/ls_model.jls",
        "artifacts/models/baselines/er_saa_model.jls",
        "artifacts/models/baselines/cart_model.jls",
        "artifacts/models/baselines/knn_model.jls",
        "artifacts/models/baselines/nm_model.jls",
        "artifacts/models/baselines/baseline_training_report.json"
    ],
    :train_neural => [
        "artifacts/models/neural/neural_model_final.jls",
        "artifacts/models/neural/neural_training_history.csv",
        "artifacts/models/neural/neural_training_log.json"
    ],
    :run_benchmark => [
        "artifacts/results/benchmark_gaps.csv",
        "artifacts/results/benchmark_summary.json",
        "artifacts/results/gap_boxplot.png"
    ]
)

function artifact_manifest()
    return deepcopy(STEP_ARTIFACTS)
end

function ensure_step_directories(output_dir::AbstractString, step::Symbol)
    for relative_path in STEP_ARTIFACTS[step]
        mkpath(joinpath(output_dir, dirname(relative_path)))
    end
end

function artifacts_present(dir::AbstractString, step::Symbol)
    all(isfile(joinpath(dir, rel)) for rel in STEP_ARTIFACTS[step])
end

function copy_artifacts(src::AbstractString, dest::AbstractString, step::Symbol)
    ensure_step_directories(dest, step)
    for relative_path in STEP_ARTIFACTS[step]
        src_path = joinpath(src, relative_path)
        dest_path = joinpath(dest, relative_path)
        if isfile(src_path)
            mkpath(dirname(dest_path))
            cp(src_path, dest_path; force=true)
        else
            @warn "Missing artifact in input directory" step relative_path
            return false
        end
    end
    mark_step_complete(step, dest)
    return true
end

function mark_step_complete(step::Symbol, dir::AbstractString)
    marker = joinpath(dir, "artifacts", string(step) * ".done")
    mkpath(dirname(marker))
    open(marker, "w") do io
        println(io, "completed_at = \"$(Dates.format(Dates.now(), Dates.RFC3339))\"")
    end
end

function write_json_file(path::AbstractString, data)
    mkpath(dirname(path))
    open(path, "w") do io
        write(io, to_json(data, 0))
        end
end

to_json(x, indent) = string(x)
function to_json(data::Dict{<:Any,<:Any}, indent::Int)
    pieces = String[]
    pad = repeat(" ", indent)
    push!(pieces, "{\n")
    inner_pad = pad * "  "
    keys_list = collect(keys(data))
    for (idx, key) in enumerate(keys_list)
        value = data[key]
        value_str = to_json(value, indent + 2)
        entry = "$(inner_pad)\"$(string(key))\": $value_str"
        idx < length(keys_list) && (entry *= ",")
        push!(pieces, entry * "\n")
    end
    push!(pieces, pad * "}")
    return join(pieces, "")
end

function to_json(vec::AbstractVector, indent::Int)
    items = [to_json(v, indent + 2) for v in vec]
    return "[" * join(items, ", ") * "]"
end

to_json(x::AbstractString, _) = string("\"", x, "\"")
to_json(x::Bool, _) = x ? "true" : "false"
to_json(x::Integer, _) = string(x)
to_json(x::Real, _) = string(x)
to_json(::Nothing, _) = "null"
to_json(x, _) = to_json(string(x), 0)


end # module
