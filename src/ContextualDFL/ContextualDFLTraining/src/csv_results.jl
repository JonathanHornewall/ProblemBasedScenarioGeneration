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

    config_summary_rows = grid_config_summary_rows(configs, results)
    write_csv(joinpath(output_dir, "config.csv"), config_summary_rows)

    if configs !== nothing
        config_rows = [flatten_to_dict(config; prefix="config") for config in configs]
        write_csv(joinpath(output_dir, "configs.csv"), config_rows)
    end

    return output_dir
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

function grid_config_summary_rows(configs, results)
    rows = Dict{Symbol,Any}[
        Dict(:key => "created_at_unix_ms", :value => unix_milliseconds()),
        Dict(:key => "result_count", :value => length(results)),
        Dict(:key => "successful_count", :value => count(result -> result.status == "ok", results)),
        Dict(:key => "failed_count", :value => count(result -> result.status != "ok", results)),
    ]

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
    open(path, "w") do io
        if isempty(rows)
            return nothing
        end

        headers = sorted_headers(rows)
        write_csv_row(io, string.(headers))

        for row in rows
            write_csv_row(io, [csv_cell(get(row, header, "")) for header in headers])
        end
    end

    return path
end

function sorted_headers(rows)
    headers = Set{Symbol}()
    for row in rows
        foreach(header -> push!(headers, header), keys(row))
    end
    return sort!(collect(headers); by=String)
end

function csv_cell(value)
    value === nothing && return ""
    value === missing && return ""
    return string(value)
end

function write_csv_row(io, values)
    for (index, value) in enumerate(values)
        index > 1 && print(io, ",")
        print(io, csv_escape(value))
    end
    print(io, "\n")
end

function csv_escape(value::AbstractString)
    if occursin('"', value) || occursin(',', value) || occursin('\n', value) || occursin('\r', value)
        return "\"" * replace(value, "\"" => "\"\"") * "\""
    end
    return value
end
