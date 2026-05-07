#!/usr/bin/env julia

using Printf

const SCRIPT_PATH = abspath(@__FILE__)
const DEFAULT_INPUT_DIR = normpath(joinpath(
    @__DIR__,
    "results",
    "tiny_30ctx_5x100_full_baselines_20260507",
))
const BENCHMARK_NAMES = (
    "resource_allocation",
    "shipment_planning",
    "transshipment_q",
    "transshipment_h",
    "transshipment_h_and_q",
    "random_yield",
    "unreliable_newsvendor",
)
const DETERMINISTIC_POLICY_NAMES = ("saa", "knn", "least_squares", "er_saa")
const REPLICATED_POLICY_NAMES = ("cart", "nn", "ad", "ad_tree", "m5_ad")
const DFL_POLICY_NAMES = ("dfl_mu0_rho0.1", "dfl_mu0_rho0.01", "dfl_mu0_rho0.001")
const EXPECTED_SUCCESS_ROWS = 196

function main(args=ARGS)
    options = parse_options(args)
    files = result_files(options.input)
    isempty(files) && error("No baseline_results_latest.csv files found under $(options.input).")

    rows = NamedTuple[]
    for file in files
        append!(rows, read_result_csv(file))
    end
    ok_rows = [row for row in rows if row.status == "ok"]
    options.validate && validate_tiny_full_rows!(ok_rows; expected_rows=options.expected_rows)

    mkpath(options.output_dir)
    aggregate_path = joinpath(options.output_dir, "tiny_full_baselines_aggregate.csv")
    summary_path = joinpath(options.output_dir, "tiny_full_baselines_summary.csv")
    analysis_path = joinpath(options.output_dir, "tiny_full_baselines_summary.md")

    write_csv(aggregate_path, rows)
    summary_rows = summarize_rows(ok_rows)
    write_csv(summary_path, summary_rows)
    write_summary_markdown(analysis_path, ok_rows, summary_rows, files)

    println("Read $(length(rows)) rows from $(length(files)) file(s).")
    println("Successful rows: $(length(ok_rows))")
    println("Aggregate CSV: $(aggregate_path)")
    println("Summary CSV: $(summary_path)")
    println("Summary analysis: $(analysis_path)")
    return (; rows=rows, summary_rows=summary_rows)
end

function parse_options(args)
    input = DEFAULT_INPUT_DIR
    output_dir = DEFAULT_INPUT_DIR
    expected_rows = EXPECTED_SUCCESS_ROWS
    validate = true

    index = 1
    while index <= length(args)
        arg = args[index]
        if arg == "--input"
            index += 1
            input = abspath(args[index])
        elseif startswith(arg, "--input=")
            input = abspath(split(arg, "=", limit=2)[2])
        elseif arg == "--output-dir"
            index += 1
            output_dir = abspath(args[index])
        elseif startswith(arg, "--output-dir=")
            output_dir = abspath(split(arg, "=", limit=2)[2])
        elseif arg == "--expected-rows"
            index += 1
            expected_rows = parse(Int, args[index])
        elseif startswith(arg, "--expected-rows=")
            expected_rows = parse(Int, split(arg, "=", limit=2)[2])
        elseif arg == "--no-validate"
            validate = false
        else
            input = abspath(arg)
        end
        index += 1
    end

    return (;
        input=input,
        output_dir=output_dir,
        expected_rows=expected_rows,
        validate=validate,
    )
end

function result_files(path)
    if isfile(path)
        return [abspath(path)]
    end
    isdir(path) || error("Input path does not exist: $(path)")

    files = String[]
    for (root, _, names) in walkdir(path)
        for name in names
            name == "baseline_results_latest.csv" || continue
            push!(files, joinpath(root, name))
        end
    end
    return sort!(files)
end

function read_result_csv(path)
    records = csv_records(read(path, String))
    isempty(records) && return NamedTuple[]
    header = Symbol.(parse_csv_record(first(records)))
    rows = NamedTuple[]
    for record in Iterators.drop(records, 1)
        isempty(strip(record)) && continue
        values = parse_csv_record(record)
        length(values) == length(header) ||
            throw(ArgumentError("CSV row in $(path) has $(length(values)) cells; expected $(length(header))."))
        row = Dict{Symbol,Any}(header[index] => values[index] for index in eachindex(header))
        row[:source_file] = path
        push!(rows, (; (key => row[key] for key in sort!(collect(keys(row)); by=String))...))
    end
    return rows
end

function csv_records(text)
    records = String[]
    buffer = IOBuffer()
    in_quotes = false
    index = firstindex(text)
    while index <= lastindex(text)
        char = text[index]
        if char == '"'
            write(buffer, char)
            next_index = nextind(text, index)
            if in_quotes && next_index <= lastindex(text) && text[next_index] == '"'
                index = next_index
                write(buffer, text[index])
            else
                in_quotes = !in_quotes
            end
        elseif (char == '\n' || char == '\r') && !in_quotes
            record = String(take!(buffer))
            isempty(record) || push!(records, record)
            if char == '\r'
                next_index = nextind(text, index)
                next_index <= lastindex(text) && text[next_index] == '\n' && (index = next_index)
            end
        else
            write(buffer, char)
        end
        index = nextind(text, index)
    end
    tail = String(take!(buffer))
    isempty(tail) || push!(records, tail)
    return records
end

function parse_csv_record(record)
    cells = String[]
    buffer = IOBuffer()
    in_quotes = false
    index = firstindex(record)
    while index <= lastindex(record)
        char = record[index]
        if char == '"'
            next_index = nextind(record, index)
            if in_quotes && next_index <= lastindex(record) && record[next_index] == '"'
                write(buffer, '"')
                index = next_index
            else
                in_quotes = !in_quotes
            end
        elseif char == ',' && !in_quotes
            push!(cells, String(take!(buffer)))
        else
            write(buffer, char)
        end
        index = nextind(record, index)
    end
    push!(cells, String(take!(buffer)))
    return cells
end

function validate_tiny_full_rows!(rows; expected_rows)
    length(rows) == expected_rows ||
        throw(ArgumentError("Expected $(expected_rows) successful rows, found $(length(rows))."))

    for row in rows
        parse(Int, row.test_contexts) == 30 ||
            throw(ArgumentError("Row $(row.benchmark)/$(row.policy) has test_contexts=$(row.test_contexts)."))
        parse(Int, row.test_scenarios_per_context) == 500 ||
            throw(ArgumentError("Row $(row.benchmark)/$(row.policy) has test_scenarios_per_context=$(row.test_scenarios_per_context)."))
        parse(Int, row.evaluation_batches) == 5 ||
            throw(ArgumentError("Row $(row.benchmark)/$(row.policy) has evaluation_batches=$(row.evaluation_batches)."))
    end

    for benchmark in BENCHMARK_NAMES
        for policy in DETERMINISTIC_POLICY_NAMES
            count = row_count(rows, benchmark, policy)
            count == 1 ||
                throw(ArgumentError("Expected one $(benchmark)/$(policy) row, found $(count)."))
        end
        for policy in (REPLICATED_POLICY_NAMES..., DFL_POLICY_NAMES...)
            count = row_count(rows, benchmark, policy)
            count == 3 ||
                throw(ArgumentError("Expected three $(benchmark)/$(policy) rows, found $(count)."))
        end
    end

    rejected = [
        row for row in rows
        if row.test_scenarios_per_context == "1000" || row.evaluation_batches == "50"
    ]
    isempty(rejected) ||
        throw(ArgumentError("Found rows from a rejected 30x1000 or 50-batch protocol."))
    return nothing
end

row_count(rows, benchmark, policy) =
    count(row -> row.benchmark == benchmark && row.policy == policy, rows)

function summarize_rows(rows)
    summaries = NamedTuple[]
    for benchmark in BENCHMARK_NAMES
        policies = sort!(unique(row.policy for row in rows if row.benchmark == benchmark))
        for policy in policies
            group = [row for row in rows if row.benchmark == benchmark && row.policy == policy]
            push!(summaries, summary_row(benchmark, policy, group))
        end
    end
    return summaries
end

function summary_row(benchmark, policy, rows)
    regrets = parse_float_column(rows, :regret_mean)
    relative_regrets = parse_float_column(rows, :relative_regret_mean)
    fit_seconds = parse_float_column(rows, :fit_seconds)
    eval_seconds = parse_float_column(rows, :eval_seconds)
    return (;
        benchmark=benchmark,
        policy=policy,
        status_ok_count=length(rows),
        replica_count=length(unique(row.replica_index for row in rows)),
        regret_mean=mean_value(regrets),
        regret_std=std_value(regrets),
        relative_regret_mean=mean_value(relative_regrets),
        relative_regret_std=std_value(relative_regrets),
        fit_seconds_mean=mean_value(fit_seconds),
        eval_seconds_mean=mean_value(eval_seconds),
    )
end

function parse_float_column(rows, column)
    values = Float64[]
    for row in rows
        value = getproperty(row, column)
        isempty(value) || push!(values, parse(Float64, value))
    end
    return values
end

mean_value(values) = isempty(values) ? NaN : sum(values) / length(values)

function std_value(values)
    length(values) <= 1 && return 0.0
    mean = mean_value(values)
    return sqrt(sum((value - mean)^2 for value in values) / (length(values) - 1))
end

function write_summary_markdown(path, rows, summary_rows, files)
    open(path, "w") do io
        println(io, "# Tiny 30ctx 5x100 Full Baseline Summary")
        println(io)
        println(io, "- Source files: $(length(files))")
        println(io, "- Successful rows: $(length(rows))")
        println(io, "- Test contexts: 30")
        println(io, "- Test scenarios per context: 500")
        println(io, "- Evaluation batches: 5")
        println(io)
        println(io, "## Best Deterministic Policy By Problem")
        println(io)
        println(io, "| Benchmark | Policy | Regret Mean | Relative Regret Mean |")
        println(io, "| --- | --- | ---: | ---: |")
        for benchmark in BENCHMARK_NAMES
            candidates = [
                row for row in summary_rows
                if row.benchmark == benchmark && row.policy in DETERMINISTIC_POLICY_NAMES
            ]
            isempty(candidates) && continue
            best = candidates[argmin([row.regret_mean for row in candidates])]
            println(
                io,
                "| $(best.benchmark) | $(best.policy) | $(fmt(best.regret_mean)) | $(fmt(best.relative_regret_mean)) |",
            )
        end
        println(io)
        println(io, "## Replicated Policy Summary")
        println(io)
        println(io, "| Benchmark | Policy | Rows | Regret Mean | Regret Std | Relative Regret Mean | Relative Regret Std |")
        println(io, "| --- | --- | ---: | ---: | ---: | ---: | ---: |")
        for row in summary_rows
            row.policy in DETERMINISTIC_POLICY_NAMES && continue
            println(
                io,
                "| $(row.benchmark) | $(row.policy) | $(row.status_ok_count) | " *
                "$(fmt(row.regret_mean)) | $(fmt(row.regret_std)) | " *
                "$(fmt(row.relative_regret_mean)) | $(fmt(row.relative_regret_std)) |",
            )
        end
    end
    return path
end

fmt(value) = isfinite(value) ? @sprintf("%.6g", value) : string(value)

function write_csv(path, rows)
    columns = csv_columns(rows)
    open(path, "w") do io
        println(io, join(String.(columns), ","))
        for row in rows
            println(io, join((csv_cell(getproperty(row, column)) for column in columns), ","))
        end
    end
    return path
end

function csv_columns(rows)
    isempty(rows) && return Symbol[]
    columns = Symbol[]
    for row in rows
        for key in keys(row)
            key in columns || push!(columns, key)
        end
    end
    return columns
end

function csv_cell(value)
    text = string(value)
    if any(contains(text, needle) for needle in (",", "\"", "\n", "\r"))
        return "\"" * replace(text, "\"" => "\"\"") * "\""
    end
    return text
end

if abspath(PROGRAM_FILE) == SCRIPT_PATH
    main()
end
