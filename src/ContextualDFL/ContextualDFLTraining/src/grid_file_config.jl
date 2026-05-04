using Configurations: @option, from_dict
using JSON3
using SHA
using YAML

@option struct GridScheduleSegmentConfig
    epochs::Int
    value::Float64
end

@option struct GridScheduleConfig
    kind::String = ""
    type::String = ""
    start::Union{Nothing,Float64} = nothing
    stop::Union{Nothing,Float64} = nothing
    value::Union{Nothing,Float64} = nothing
    values::Union{Nothing,Vector{Float64}} = nothing
    segments::Vector{GridScheduleSegmentConfig} = GridScheduleSegmentConfig[]
end

@option struct GridFileConfig
    version::Int
    name::Union{Nothing,String} = nothing
    description::Union{Nothing,String} = nothing
    base::Dict{String,Any} = Dict{String,Any}()
    fixed::Dict{String,Any} = Dict{String,Any}()
    grid::Dict{String,Any} = Dict{String,Any}()
    schedules::Dict{String,GridScheduleConfig} = Dict{String,GridScheduleConfig}()
    run_id_template::Union{Nothing,String} = nothing
end

struct GridSearchSpec
    path::String
    format::Symbol
    version::Int
    name::String
    base::Dict{Symbol,Any}
    fixed::Dict{Symbol,Any}
    grid::Dict{Symbol,Vector{Any}}
    schedules::Dict{Symbol,Any}
    schedule_grid::Vector{Dict{Symbol,Any}}
    run_id_template::Union{Nothing,String}
    raw::Dict{String,Any}
end

const GRID_SYMBOL_KEYS = Set(
    [
        :activation,
        :loss,
        :method,
        :mu_ref_schedule,
        :mu_schedule,
        :optimality_evaluate_mode,
        :rho_ref_schedule,
        :rho_schedule,
    ],
)

const GRID_INT_KEYS = Set(
    [
        :batch_size,
        :candidate_index,
        :collection_duplicates_per_context,
        :depth,
        :display_real,
        :epochs,
        :hidden_size,
        :Nr_contexts,
        :optimality_evaluation_batches,
        :optimality_test_sample_count,
        :optimality_train_sample_count,
        :optimality_validation_sample_count,
        :replicate_index,
        :repeat_count,
        :scenarios_per_context,
        :seed,
        :test_scenarios_per_context,
        :training_context_count,
        :training_scenarios_per_context,
        :warmup_epochs,
    ],
)

const GRID_FLOAT_KEYS = Set(
    [
        :dropout,
        :generated_split_test_fraction,
        :learning_rate,
        :mu,
        :mu_end,
        :mu_ref,
        :mu_ref_end,
        :mu_ref_start,
        :mu_start,
        :optimality_mu,
        :optimality_rho,
        :policy_inference_mu,
        :policy_inference_rho,
        :rho,
        :rho_end,
        :rho_ref,
        :rho_ref_end,
        :rho_ref_start,
        :rho_start,
        :test_fraction,
        :tolerance_absolute_floor,
        :tolerance_relative,
        :validation_fraction,
    ],
)

const GRID_BOOL_KEYS = Set(
    [
        :annealing,
        :display_smooth,
        :fine_tuning,
        :knn_homogenization,
        :log_barrier_inference,
        :mlflow_enabled,
        :mlflow_upload_model_artifact,
        :optimality_evaluation,
        :reset_optimizer_each_epoch,
        :shuffle,
    ],
)

function load_grid_config(path::AbstractString)
    absolute_path = abspath(path)
    isfile(absolute_path) ||
        throw(ArgumentError("grid config file does not exist: $absolute_path"))

    format = grid_config_format(absolute_path)
    raw = grid_config_data(read_grid_config_file(absolute_path, format))
    parsed = parse_grid_file_config(raw, absolute_path)

    version = parsed.version
    version == 1 ||
        throw(ArgumentError("unsupported grid config version $version in $absolute_path"))

    name = parsed.name === nothing ? splitext(basename(absolute_path))[1] : string(parsed.name)
    isempty(strip(name)) &&
        throw(ArgumentError("grid config name must not be empty in $absolute_path"))

    base = settings_section(parsed.base)
    fixed = settings_section(parsed.fixed)
    raw_grid = copy(parsed.grid)
    schedule_grid = schedule_grid_section(pop_grid_schedules!(raw_grid))
    grid = grid_section(raw_grid)
    schedules = schedule_settings(parsed.schedules)
    run_id_template = parsed.run_id_template === nothing ?
        nothing :
        string(parsed.run_id_template)

    return GridSearchSpec(
        absolute_path,
        format,
        version,
        name,
        base,
        fixed,
        grid,
        schedules,
        schedule_grid,
        run_id_template,
        raw,
    )
end

function grid_config_format(path::AbstractString)
    extension = lowercase(splitext(path)[2])
    extension in (".yaml", ".yml") && return :yaml
    extension == ".json" && return :json
    throw(ArgumentError("grid config must be .yaml, .yml, or .json, got: $path"))
end

function read_grid_config_file(path::AbstractString, format::Symbol)
    format == :yaml && return YAML.load_file(path)
    format == :json && return JSON3.read(read(path, String))
    throw(ArgumentError("unsupported grid config format $format"))
end

function grid_config_data(value)
    if value isa AbstractDict || value isa JSON3.Object
        output = Dict{String,Any}()
        for (key, item) in pairs(value)
            output[string(key)] = grid_config_data(item)
        end
        return output
    elseif value isa AbstractVector || value isa JSON3.Array
        return Any[grid_config_data(item) for item in value]
    elseif value isa AbstractString || value isa Number || value isa Bool ||
           value === nothing || value === missing
        return value
    end
    return string(value)
end

function parse_grid_file_config(raw, path::AbstractString)
    raw isa AbstractDict ||
        throw(ArgumentError("grid config $path must contain a mapping/object at the top level."))

    normalize_grid_schedule_aliases!(raw)
    try
        return from_dict(GridFileConfig, raw)
    catch error
        throw(ArgumentError("invalid grid config $path: $(sprint(showerror, error))"))
    end
end

function normalize_grid_schedule_aliases!(raw)
    normalize_schedule_aliases_in_section!(get(raw, "schedules", nothing))

    grid = get(raw, "grid", nothing)
    if grid isa AbstractDict
        grid_schedules = get(grid, "schedules", nothing)
        if grid_schedules isa AbstractDict
            for (_, schedules) in grid_schedules
                if schedules isa AbstractVector
                    for schedule in schedules
                        normalize_single_schedule_aliases!(schedule)
                    end
                else
                    normalize_single_schedule_aliases!(schedules)
                end
            end
        end
    end

    return raw
end

function normalize_schedule_aliases_in_section!(schedules)
    schedules isa AbstractDict || return schedules
    for (_, schedule) in schedules
        normalize_single_schedule_aliases!(schedule)
    end
    return schedules
end

function normalize_single_schedule_aliases!(schedule)
    schedule isa AbstractDict || return schedule
    if haskey(schedule, "end")
        haskey(schedule, "stop") || (schedule["stop"] = schedule["end"])
        delete!(schedule, "end")
    end
    return schedule
end

function pop_grid_schedules!(grid_values::Dict{String,Any})
    haskey(grid_values, "schedules") || return nothing
    schedules = grid_values["schedules"]
    delete!(grid_values, "schedules")
    return schedules
end

function settings_section(section)
    output = Dict{Symbol,Any}()
    for (setting_key, value) in section
        symbol_key = Symbol(setting_key)
        output[symbol_key] = normalize_grid_setting_value(symbol_key, value)
    end
    return output
end

function grid_section(grid_values)
    output = Dict{Symbol,Vector{Any}}()
    for (setting_key, values) in grid_values
        values isa AbstractVector ||
            throw(ArgumentError("grid entry '$setting_key' must be a non-empty array."))
        isempty(values) &&
            throw(ArgumentError("grid entry '$setting_key' must not be empty."))
        symbol_key = Symbol(setting_key)
        output[symbol_key] = [
            normalize_grid_setting_value(symbol_key, value) for value in values
        ]
    end
    return output
end

function normalize_grid_setting_value(key::Symbol, value)
    value === nothing && return nothing
    value === missing && return missing

    if key in GRID_SYMBOL_KEYS
        return Symbol(value)
    elseif key in GRID_INT_KEYS
        return Int(value)
    elseif key in GRID_FLOAT_KEYS
        return Float64(value)
    elseif key in GRID_BOOL_KEYS
        return Bool(value)
    elseif value isa AbstractDict
        return Dict(Symbol(k) => normalize_grid_setting_value(Symbol(k), v) for (k, v) in value)
    elseif value isa AbstractVector
        return Any[normalize_grid_setting_value(key, item) for item in value]
    end

    return value
end

function schedule_settings(schedules)
    output = Dict{Symbol,Any}()

    for (name, spec) in schedules
        schedule_name = Symbol(name)
        merge!(output, normalize_schedule(schedule_name, spec))
    end

    return output
end

function schedule_grid_section(schedules)
    schedules === nothing && return [Dict{Symbol,Any}()]
    schedules isa AbstractDict ||
        throw(ArgumentError("grid.schedules must be a mapping/object."))

    options = Dict{Symbol,Vector{Dict{Symbol,Any}}}()
    for (name, values) in schedules
        values isa AbstractVector ||
            throw(ArgumentError("grid.schedules entry '$name' must be a non-empty array."))
        isempty(values) &&
            throw(ArgumentError("grid.schedules entry '$name' must not be empty."))

        schedule_name = Symbol(name)
        options[schedule_name] = [
            normalize_schedule(schedule_name, grid_schedule_config(schedule_name, value)) for
            value in values
        ]
    end

    return schedule_grid_candidates(options)
end

function grid_schedule_config(schedule_name::Symbol, value)
    value isa GridScheduleConfig && return value
    value isa AbstractDict || throw(
        ArgumentError(
            "grid.schedules entry '$schedule_name' candidates must be schedule mappings/objects.",
        ),
    )

    string_keyed = Dict{String,Any}(string(key) => item for (key, item) in value)
    normalize_single_schedule_aliases!(string_keyed)
    try
        return from_dict(GridScheduleConfig, string_keyed)
    catch error
        throw(
            ArgumentError(
                "invalid grid.schedules entry '$schedule_name': $(sprint(showerror, error))",
            ),
        )
    end
end

function schedule_grid_candidates(options::Dict{Symbol,Vector{Dict{Symbol,Any}}})
    keys_sorted = sort!(collect(keys(options)); by=string)
    isempty(keys_sorted) && return [Dict{Symbol,Any}()]

    reversed_keys = reverse(keys_sorted)
    return vec([
        merge_schedule_candidate_values(keys_sorted, reverse(values)) for
        values in Iterators.product((options[key] for key in reversed_keys)...)
    ])
end

function merge_schedule_candidate_values(keys_sorted, values)
    output = Dict{Symbol,Any}()
    for (_, settings) in zip(keys_sorted, values)
        merge!(output, settings)
    end
    return output
end

function normalize_schedule(name::Symbol, spec::GridScheduleConfig)
    raw_kind = isempty(strip(spec.kind)) ? spec.type : spec.kind
    kind = Symbol(strip(raw_kind))
    kind === Symbol("") &&
        throw(ArgumentError("schedule '$name' must define a kind."))

    if name == :mu
        return normalize_mu_schedule(kind, spec)
    elseif name == :mu_ref
        return normalize_mu_ref_schedule(kind, spec)
    elseif name == :rho
        return normalize_rho_schedule(kind, spec)
    elseif name == :rho_ref
        return normalize_rho_ref_schedule(kind, spec)
    end

    throw(ArgumentError("unsupported schedule '$name'; supported schedules are mu, mu_ref, rho, and rho_ref."))
end

function normalize_mu_schedule(kind::Symbol, spec)
    manual_values = manual_schedule_values(kind, spec, "mu")
    manual_values === nothing || return Dict{Symbol,Any}(:mu_schedule => manual_values)

    kind in (:constant, :linear, :geometric, :exponential) ||
        throw(ArgumentError("unsupported mu schedule kind '$kind'."))

    output = Dict{Symbol,Any}(:mu_schedule => kind)
    if kind == :constant
        spec.value !== nothing && (output[:mu] = Float64(spec.value))
        return output
    end

    output[:mu_start] = Float64(required_schedule_value(spec, :start, "mu"))
    output[:mu_end] = Float64(schedule_stop_value(spec, "mu"))
    return output
end

function normalize_mu_ref_schedule(kind::Symbol, spec)
    manual_values = manual_schedule_values(kind, spec, "mu_ref")
    manual_values === nothing || return Dict{Symbol,Any}(:mu_ref_schedule => manual_values)

    if kind in (:match_input, :same, :input, :zero, :zeros, :none)
        return Dict{Symbol,Any}(:mu_ref_schedule => kind)
    end
    kind in (:constant, :linear, :geometric, :exponential) ||
        throw(ArgumentError("unsupported mu_ref schedule kind '$kind'."))

    output = Dict{Symbol,Any}(:mu_ref_schedule => kind)
    if kind == :constant
        spec.value !== nothing && (output[:mu_ref] = Float64(spec.value))
        return output
    end

    output[:mu_ref_start] = Float64(required_schedule_value(spec, :start, "mu_ref"))
    output[:mu_ref_end] = Float64(schedule_stop_value(spec, "mu_ref"))
    return output
end

function normalize_rho_schedule(kind::Symbol, spec)
    manual_values = manual_schedule_values(kind, spec, "rho")
    manual_values === nothing || return Dict{Symbol,Any}(:rho_schedule => manual_values)

    kind in (:constant, :linear, :geometric, :exponential) ||
        throw(ArgumentError("unsupported rho schedule kind '$kind'."))

    output = Dict{Symbol,Any}(:rho_schedule => kind)
    if kind == :constant
        spec.value !== nothing && (output[:rho] = Float64(spec.value))
        return output
    end

    output[:rho_start] = Float64(required_schedule_value(spec, :start, "rho"))
    output[:rho_end] = Float64(schedule_stop_value(spec, "rho"))
    return output
end

function normalize_rho_ref_schedule(kind::Symbol, spec)
    manual_values = manual_schedule_values(kind, spec, "rho_ref")
    manual_values === nothing || return Dict{Symbol,Any}(:rho_ref_schedule => manual_values)

    if kind in (:match_input, :same, :input, :zero, :zeros, :none)
        return Dict{Symbol,Any}(:rho_ref_schedule => kind)
    end
    kind in (:constant, :linear, :geometric, :exponential) ||
        throw(ArgumentError("unsupported rho_ref schedule kind '$kind'."))

    output = Dict{Symbol,Any}(:rho_ref_schedule => kind)
    if kind == :constant
        spec.value !== nothing && (output[:rho_ref] = Float64(spec.value))
        return output
    end

    output[:rho_ref_start] = Float64(required_schedule_value(spec, :start, "rho_ref"))
    output[:rho_ref_end] = Float64(schedule_stop_value(spec, "rho_ref"))
    return output
end

function manual_schedule_values(
    kind::Symbol,
    spec::GridScheduleConfig,
    schedule_name::AbstractString,
)
    if kind in (:values, :manual)
        spec.values !== nothing ||
            throw(ArgumentError("schedule '$schedule_name' must define non-empty values."))
        isempty(spec.values) &&
            throw(ArgumentError("schedule '$schedule_name' values must not be empty."))
        return Float64.(spec.values)
    elseif kind in (:piecewise, :step)
        isempty(spec.segments) &&
            throw(ArgumentError("schedule '$schedule_name' segments must not be empty."))

        values = Float64[]
        for (index, segment) in enumerate(spec.segments)
            segment.epochs > 0 || throw(
                ArgumentError(
                    "schedule '$schedule_name' segment $index epochs must be positive.",
                ),
            )
            append!(values, fill(Float64(segment.value), segment.epochs))
        end
        return values
    end

    return nothing
end

function required_schedule_value(
    spec::GridScheduleConfig,
    key::Symbol,
    schedule_name::AbstractString,
)
    value = getproperty(spec, key)
    value === nothing &&
        throw(ArgumentError("schedule '$schedule_name' must define '$key'."))
    return value
end

function schedule_stop_value(spec::GridScheduleConfig, schedule_name::AbstractString)
    spec.stop !== nothing && return spec.stop
    throw(ArgumentError("schedule '$schedule_name' must define 'stop' or 'end'."))
end

function resolve_grid_configs(
    spec::GridSearchSpec;
    base_config::NamedTuple=NamedTuple(),
    defaults::NamedTuple=DEFAULT_RUN_SETTINGS,
)
    static_config = Dict{Symbol,Any}()
    merge!(static_config, Dict{Symbol,Any}(pairs(defaults)))
    merge!(static_config, Dict{Symbol,Any}(pairs(base_config)))
    merge!(static_config, spec.base)
    merge!(static_config, spec.fixed)

    configs_without_digest = NamedTuple[]
    index = 0
    for candidate_values in grid_candidates(spec.grid),
        schedule_candidate_values in spec.schedule_grid

        index += 1
        config = copy(static_config)
        merge!(config, spec.schedules)
        merge!(config, candidate_values)
        merge!(config, schedule_candidate_values)

        config[:grid_config_name] = spec.name
        config[:grid_config_path] = spec.path
        config[:grid_config_version] = spec.version
        config[:grid_candidate_index] = index

        validate_explicit_schedule_lengths!(config)

        haskey(config, :run_id) || (config[:run_id] = grid_run_id(spec, config, index))
        push!(configs_without_digest, namedtuple_from_dict(config))
    end

    digest = grid_config_digest(configs_without_digest)
    return [
        merge(config, (; grid_config_digest=digest)) for
        config in configs_without_digest
    ]
end

function validate_explicit_schedule_lengths!(config::Dict{Symbol,Any})
    epochs = Int(get(config, :epochs, 0))
    for key in (:mu_schedule, :mu_ref_schedule, :rho_schedule, :rho_ref_schedule)
        haskey(config, key) || continue
        values = config[key]
        values isa AbstractVector || continue
        length(values) == epochs || throw(
            ArgumentError(
                "$(key) vector length $(length(values)) must equal epochs $(epochs).",
            ),
        )
    end
    return nothing
end

function grid_candidates(grid::Dict{Symbol,Vector{Any}})
    keys_sorted = sort!(collect(keys(grid)); by=string)
    isempty(keys_sorted) && return [Dict{Symbol,Any}()]

    reversed_keys = reverse(keys_sorted)
    return [
        Dict{Symbol,Any}(key => value for (key, value) in zip(keys_sorted, reverse(values))) for
        values in Iterators.product((grid[key] for key in reversed_keys)...)
    ]
end

function grid_run_id(spec::GridSearchSpec, config::Dict{Symbol,Any}, index::Integer)
    hash = bytes2hex(sha1(JSON3.write(json_ready(config))))[1:8]
    index_text = lpad(string(index), 4, "0")
    template = spec.run_id_template
    template === nothing &&
        return "grid_" * safe_identifier(spec.name) * "_" * index_text * "_" * hash

    output = replace(template, "{index}" => index_text)
    output = replace(output, "{name}" => safe_identifier(spec.name))
    output = replace(output, "{hash}" => hash)
    return output
end

function safe_identifier(value)
    text = lowercase(string(value))
    text = replace(text, r"[^A-Za-z0-9_.=-]+" => "_")
    return strip(text, ['_'])
end

function namedtuple_from_dict(values::Dict{Symbol,Any})
    keys_sorted = sort!(collect(keys(values)); by=string)
    return NamedTuple{Tuple(keys_sorted)}(Tuple(values[key] for key in keys_sorted))
end

function grid_config_digest(configs)
    data = Vector{UInt8}(codeunits(resolved_grid_json(configs)))
    return "sha256:" * bytes2hex(sha256(data))
end

const RESOLVED_GRID_JSON_OMITTED_KEYS = Set(["grid_config_digest"])

function resolved_grid_json(configs)
    return sprint(io -> JSON3.pretty(io, json_ready(configs, RESOLVED_GRID_JSON_OMITTED_KEYS)))
end

function json_ready(value, omitted_keys::Set{String}=Set{String}())
    value === missing && return nothing
    value === nothing && return nothing
    value isa Symbol && return string(value)
    value isa Number && return value
    value isa Bool && return value
    value isa AbstractString && return value

    if value isa NamedTuple
        ready_keys = Symbol[
            key for key in sort!(collect(keys(value)); by=string) if
            !(string(key) in omitted_keys)
        ]
        return NamedTuple{Tuple(ready_keys)}(
            Tuple(json_ready(getproperty(value, key), omitted_keys) for key in ready_keys),
        )
    elseif value isa AbstractDict
        source_keys = [
            key for key in sort!(collect(keys(value)); by=string) if
            !(string(key) in omitted_keys)
        ]
        ready_keys = Symbol.(string.(source_keys))
        return NamedTuple{Tuple(ready_keys)}(
            Tuple(json_ready(value[key], omitted_keys) for key in source_keys),
        )
    elseif value isa AbstractVector || value isa Tuple
        return Any[json_ready(item, omitted_keys) for item in value]
    end

    return string(value)
end
