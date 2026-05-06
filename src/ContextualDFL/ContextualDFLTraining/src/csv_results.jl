import CSV
import JSON3

if !isdefined(@__MODULE__, :unix_milliseconds)
    # Explicit Unix epoch milliseconds, independent of the local timezone.
    unix_milliseconds() = round(Int64, time() * 1000)
end

function result_timestamp()
    return string(unix_milliseconds())
end

function write_grid_results(
    results;
    configs=nothing,
    config_results=nothing,
    output_root=joinpath(@__DIR__, "..", "results"),
    timestamp=result_timestamp(),
)
    output_dir = joinpath(output_root, timestamp)
    mkpath(output_dir)

    run_rows = [run_result_row(result) for result in results]
    write_csv(joinpath(output_dir, "runs.csv"), run_rows)

    epoch_rows = epoch_result_rows(results)
    write_csv(joinpath(output_dir, "epochs.csv"), epoch_rows)

    failure_rows = [row for row in run_rows if get(row, :status, "") != "ok"]
    write_csv(joinpath(output_dir, "failures.csv"), failure_rows)

    successful_rows = [row for row in run_rows if get(row, :status, "") == "ok"]
    best_rows = sort(successful_rows; by=row -> get(row, :validation_mse, Inf))
    write_csv(joinpath(output_dir, "best.csv"), best_rows)

    aggregate_rows = config_aggregate_rows(config_results)
    write_csv(joinpath(output_dir, "config_aggregates.csv"), aggregate_rows)

    config_summary_rows = grid_config_summary_rows(configs, results; config_results=config_results)
    write_csv(joinpath(output_dir, "config.csv"), config_summary_rows)

    if configs !== nothing
        config_rows = [flatten_to_dict(config; prefix="config") for config in configs]
        write_csv(joinpath(output_dir, "configs.csv"), config_rows)
    end

    return output_dir
end

function write_incremental_grid_result(output_dir, result)
    mkpath(output_dir)
    append_jsonl(joinpath(output_dir, "runs.jsonl"), run_result_row(result))

    for row in epoch_result_rows((result,))
        append_jsonl(joinpath(output_dir, "epochs.jsonl"), row)
    end

    if getproperty(result, :status) != "ok"
        append_jsonl(joinpath(output_dir, "failures.jsonl"), run_result_row(result))
    end

    return nothing
end

function write_incremental_config_result(output_dir, result)
    mkpath(output_dir)
    append_jsonl(joinpath(output_dir, "config_aggregates.jsonl"), config_aggregate_row(result))
    return nothing
end

function run_result_row(result)
    row = Dict{Symbol,Any}()
    row[:status] = getproperty(result, :status)
    row[:run_id] = getproperty(result, :run_id)
    row[:started_at] = getproperty(result, :started_at)
    row[:finished_at] = getproperty(result, :finished_at)
    row[:elapsed_seconds] = getproperty(result, :elapsed_seconds)
    row[:error] = getproperty(result, :error)

    flatten_to_dict!(row, "config", getproperty(result, :config))
    flatten_to_dict!(row, "", getproperty(result, :worker))
    flatten_to_dict!(row, "", getproperty(result, :final_metrics))
    return row
end

function epoch_result_rows(results)
    rows = Dict{Symbol,Any}[]

    for result in results
        history = getproperty(result, :epoch_history)
        for history_row in history
            row = Dict{Symbol,Any}()
            row[:run_id] = getproperty(result, :run_id)
            row[:status] = getproperty(result, :status)
            flatten_to_dict!(row, "config", getproperty(result, :config))
            flatten_to_dict!(row, "", history_row)
            push!(rows, row)
        end
    end

    return rows
end

function config_aggregate_rows(config_results)
    config_results === nothing && return Dict{Symbol,Any}[]

    return [config_aggregate_row(result) for result in config_results]
end

function config_aggregate_row(result)
    row = Dict{Symbol,Any}()
    row[:status] = getproperty(result, :status)
    row[:run_id] = getproperty(result, :run_id)
    row[:started_at] = getproperty(result, :started_at)
    row[:finished_at] = getproperty(result, :finished_at)
    row[:elapsed_seconds] = getproperty(result, :elapsed_seconds)
    row[:error] = getproperty(result, :error)

    flatten_to_dict!(row, "config", getproperty(result, :config))
    flatten_to_dict!(row, "", getproperty(result, :final_metrics))

    aggregate_metrics = getproperty(result, :aggregate_metrics)
    if aggregate_metrics isa AbstractDict
        for key in sort!(collect(keys(aggregate_metrics)); by=String)
            summary = aggregate_metrics[key]
            for field in (:count, :mean, :median, :min, :max, :std, :stderr)
                row[Symbol(string(key) * "_" * string(field))] = getproperty(summary, field)
            end
        end
    end

    return row
end

function grid_config_summary_rows(configs, results; config_results=nothing)
    rows = Dict{Symbol,Any}[
        Dict(:key => "created_at_unix_ms", :value => unix_milliseconds()),
        Dict(:key => "result_count", :value => length(results)),
        Dict(:key => "successful_count", :value => count(result -> result.status == "ok", results)),
        Dict(:key => "failed_count", :value => count(result -> result.status != "ok", results)),
    ]

    if config_results !== nothing
        push!(rows, Dict(:key => "config_parent_count", :value => length(config_results)))
        push!(
            rows,
            Dict(
                :key => "config_parent_successful_count",
                :value => count(result -> result.status == "ok", config_results),
            ),
        )
        push!(
            rows,
            Dict(
                :key => "config_parent_failed_count",
                :value => count(result -> result.status != "ok", config_results),
            ),
        )
    end

    if configs !== nothing
        push!(rows, Dict(:key => "config_count", :value => length(configs)))
        first_config = isempty(configs) ? nothing : first(configs)
        if first_config !== nothing
            for key in keys(first_config)
                value = getproperty(first_config, key)
                if !(value isa AbstractArray)
                    push!(rows, Dict(:key => "default_" * string(key), :value => value))
                end
            end
        end
    end

    return rows
end

function flatten_to_dict(value; prefix="")
    row = Dict{Symbol,Any}()
    flatten_to_dict!(row, prefix, value)
    return row
end

function flatten_to_dict!(row::Dict{Symbol,Any}, prefix::AbstractString, value::NamedTuple)
    for key in keys(value)
        child_prefix = join_prefix(prefix, key)
        flatten_to_dict!(row, child_prefix, getproperty(value, key))
    end
    return row
end

function flatten_to_dict!(row::Dict{Symbol,Any}, prefix::AbstractString, value::AbstractDict)
    for (key, child_value) in value
        child_prefix = join_prefix(prefix, Symbol(key))
        flatten_to_dict!(row, child_prefix, child_value)
    end
    return row
end

function flatten_to_dict!(row::Dict{Symbol,Any}, prefix::AbstractString, value)
    isempty(prefix) && return row
    row[Symbol(prefix)] = value
    return row
end

function join_prefix(prefix::AbstractString, key)
    key_text = string(key)
    return isempty(prefix) ? key_text : prefix * "_" * key_text
end

function write_csv(path, rows)
    if isempty(rows)
        write(path, "")
        return path
    end

    headers = Symbol[]
    seen = Set{Symbol}()
    for row in rows, header in keys(row)
        header in seen && continue
        push!(seen, header)
        push!(headers, header)
    end
    sort!(headers; by=String)

    columns = map(headers) do header
        header => [something(get(row, header, missing), missing) for row in rows]
    end
    CSV.write(path, (; columns...); missingstring="")
    return path
end

function append_jsonl(path, row)
    mkpath(dirname(path))
    open(path, "a") do io
        JSON3.write(io, jsonl_ready_value(row))
        write(io, "\n")
    end
    return path
end

function jsonl_ready_value(value::AbstractDict)
    output = Dict{String,Any}()
    for (key, child) in value
        output[string(key)] = jsonl_ready_value(child)
    end
    return output
end

function jsonl_ready_value(value::NamedTuple)
    output = Dict{String,Any}()
    for (key, child) in pairs(value)
        output[string(key)] = jsonl_ready_value(child)
    end
    return output
end

function jsonl_ready_value(value::Tuple)
    return Any[jsonl_ready_value(child) for child in value]
end

function jsonl_ready_value(value::AbstractArray)
    return Any[jsonl_ready_value(child) for child in value]
end

function jsonl_ready_value(value)
    value === missing && return nothing
    value === nothing && return nothing
    value isa AbstractString && return value
    value isa Bool && return value
    value isa Integer && return value
    if value isa AbstractFloat
        return isfinite(value) ? value : string(value)
    end
    value isa Number && return value
    value isa Symbol && return string(value)
    return string(value)
end
