using JSON3
using SHA
using YAML

struct GridSearchSpec
    path::String
    format::Symbol
    version::Int
    name::String
    base::Dict{Symbol,Any}
    fixed::Dict{Symbol,Any}
    grid::Dict{Symbol,Vector{Any}}
    schedules::Dict{Symbol,Any}
    run_id_template::Union{Nothing,String}
    digest::String
    raw::Dict{String,Any}
end

const GRID_CONFIG_TOP_LEVEL_KEYS = Set(
    [
        "version",
        "name",
        "description",
        "base",
        "fixed",
        "grid",
        "schedules",
        "run_id_template",
    ],
)

const GRID_SYMBOL_KEYS = Set(
    [
        :loss,
        :method,
        :mu_ref_schedule,
        :mu_schedule,
        :problem,
        :solver,
    ],
)

const GRID_INT_KEYS = Set(
    [
        :batch_size,
        :candidate_index,
        :context_terms,
        :depth,
        :epochs,
        :hidden_size,
        :n_samples,
        :nr_scenarios,
        :optimality_test_sample_count,
        :optimality_train_sample_count,
        :optimality_validation_sample_count,
        :replicate_index,
        :seed,
        :warmup_epochs,
    ],
)

const GRID_FLOAT_KEYS = Set(
    [
        :demand_power,
        :dropout,
        :learning_rate,
        :mu,
        :mu_end,
        :mu_ref,
        :mu_ref_end,
        :mu_ref_start,
        :mu_start,
        :optimality_mu,
        :policy_inference_mu,
        :rho,
        :sigma,
        :test_fraction,
        :tolerance_absolute_floor,
        :tolerance_relative,
        :validation_fraction,
    ],
)

const GRID_BOOL_KEYS = Set(
    [
        :annealing,
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
    raw = normalize_grid_config_data(read_grid_config_file(absolute_path, format))
    validate_grid_config_top_level!(raw, absolute_path)

    version = required_int(raw, "version", absolute_path)
    version == 1 ||
        throw(ArgumentError("unsupported grid config version $version in $absolute_path"))

    name = string(get(raw, "name", splitext(basename(absolute_path))[1]))
    isempty(strip(name)) &&
        throw(ArgumentError("grid config name must not be empty in $absolute_path"))

    base = settings_section(raw, "base")
    fixed = settings_section(raw, "fixed")
    grid = grid_section(raw)
    schedules = schedule_settings(raw)
    run_id_template = haskey(raw, "run_id_template") ?
        string(raw["run_id_template"]) :
        nothing

    return GridSearchSpec(
        absolute_path,
        format,
        version,
        name,
        base,
        fixed,
        grid,
        schedules,
        run_id_template,
        grid_config_digest(absolute_path),
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

function normalize_grid_config_data(value)
    if value isa AbstractDict || value isa JSON3.Object
        output = Dict{String,Any}()
        for (key, item) in pairs(value)
            output[string(key)] = normalize_grid_config_data(item)
        end
        return output
    elseif value isa AbstractVector || value isa JSON3.Array
        return Any[normalize_grid_config_data(item) for item in value]
    elseif value isa AbstractString || value isa Number || value isa Bool ||
           value === nothing || value === missing
        return value
    end
    return string(value)
end

function validate_grid_config_top_level!(raw, path)
    raw isa AbstractDict ||
        throw(ArgumentError("grid config $path must contain a mapping/object at the top level."))

    unknown = sort!(setdiff(collect(keys(raw)), collect(GRID_CONFIG_TOP_LEVEL_KEYS)))
    isempty(unknown) ||
        throw(ArgumentError("unknown top-level grid config key(s) in $path: $(join(unknown, ", "))"))
    return nothing
end

function required_int(raw, key::AbstractString, path::AbstractString)
    haskey(raw, key) || throw(ArgumentError("grid config $path is missing required key '$key'."))
    return Int(raw[key])
end

function section_dict(raw, key::AbstractString)
    value = get(raw, key, Dict{String,Any}())
    value isa AbstractDict ||
        throw(ArgumentError("grid config section '$key' must be a mapping/object."))
    return value
end

function settings_section(raw, key::AbstractString)
    output = Dict{Symbol,Any}()
    for (setting_key, value) in section_dict(raw, key)
        symbol_key = Symbol(setting_key)
        output[symbol_key] = normalize_grid_setting_value(symbol_key, value)
    end
    return output
end

function grid_section(raw)
    output = Dict{Symbol,Vector{Any}}()
    for (setting_key, values) in section_dict(raw, "grid")
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

function schedule_settings(raw)
    output = Dict{Symbol,Any}()
    schedules = section_dict(raw, "schedules")

    for (name, spec) in schedules
        schedule_name = Symbol(name)
        spec isa AbstractDict ||
            throw(ArgumentError("schedule '$name' must be a mapping/object."))
        merge!(output, normalize_schedule(schedule_name, spec))
    end

    return output
end

function normalize_schedule(name::Symbol, spec)
    kind = Symbol(get(spec, "kind", get(spec, "type", "")))
    kind === Symbol("") &&
        throw(ArgumentError("schedule '$name' must define a kind."))

    if name == :mu
        return normalize_mu_schedule(kind, spec)
    elseif name == :mu_ref
        return normalize_mu_ref_schedule(kind, spec)
    end

    throw(ArgumentError("unsupported schedule '$name'; supported schedules are mu and mu_ref."))
end

function normalize_mu_schedule(kind::Symbol, spec)
    kind in (:constant, :linear, :geometric, :exponential) ||
        throw(ArgumentError("unsupported mu schedule kind '$kind'."))

    output = Dict{Symbol,Any}(:mu_schedule => kind)
    if kind == :constant
        haskey(spec, "value") && (output[:mu] = Float64(spec["value"]))
        return output
    end

    output[:mu_start] = Float64(required_schedule_value(spec, "start", "mu"))
    output[:mu_end] = Float64(schedule_stop_value(spec, "mu"))
    return output
end

function normalize_mu_ref_schedule(kind::Symbol, spec)
    if kind in (:match_input, :same, :input, :zero, :zeros, :none)
        return Dict{Symbol,Any}(:mu_ref_schedule => kind)
    end
    kind in (:constant, :linear, :geometric, :exponential) ||
        throw(ArgumentError("unsupported mu_ref schedule kind '$kind'."))

    output = Dict{Symbol,Any}(:mu_ref_schedule => kind)
    if kind == :constant
        haskey(spec, "value") && (output[:mu_ref] = Float64(spec["value"]))
        return output
    end

    output[:mu_ref_start] = Float64(required_schedule_value(spec, "start", "mu_ref"))
    output[:mu_ref_end] = Float64(schedule_stop_value(spec, "mu_ref"))
    return output
end

function required_schedule_value(spec, key::AbstractString, schedule_name::AbstractString)
    haskey(spec, key) ||
        throw(ArgumentError("schedule '$schedule_name' must define '$key'."))
    return spec[key]
end

function schedule_stop_value(spec, schedule_name::AbstractString)
    if haskey(spec, "stop")
        return spec["stop"]
    elseif haskey(spec, "end")
        return spec["end"]
    end
    throw(ArgumentError("schedule '$schedule_name' must define 'stop' or 'end'."))
end

function resolve_grid_configs(experiment, spec::GridSearchSpec)
    static_config = Dict{Symbol,Any}()
    merge!(static_config, Dict{Symbol,Any}(pairs(DEFAULT_RUN_SETTINGS)))
    merge!(static_config, Dict{Symbol,Any}(pairs(experiment_base_config(experiment))))
    merge!(static_config, spec.base)
    merge!(static_config, spec.fixed)

    configs = NamedTuple[]
    for (index, candidate_values) in enumerate(grid_candidates(spec.grid))
        config = copy(static_config)
        merge!(config, spec.schedules)
        merge!(config, candidate_values)

        config[:grid_config_name] = spec.name
        config[:grid_config_path] = spec.path
        config[:grid_config_digest] = spec.digest
        config[:grid_config_version] = spec.version
        config[:grid_candidate_index] = index

        haskey(config, :run_id) || (config[:run_id] = grid_run_id(spec, config, index))
        push!(configs, namedtuple_from_dict(config))
    end

    return configs
end

function grid_candidates(grid::Dict{Symbol,Vector{Any}})
    keys_sorted = sort!(collect(keys(grid)); by=string)
    isempty(keys_sorted) && return [Dict{Symbol,Any}()]

    candidates = Dict{Symbol,Any}[]
    function visit(index, values)
        if index > length(keys_sorted)
            push!(candidates, copy(values))
            return nothing
        end
        key = keys_sorted[index]
        for value in grid[key]
            values[key] = value
            visit(index + 1, values)
        end
        delete!(values, key)
        return nothing
    end
    visit(1, Dict{Symbol,Any}())
    return candidates
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

function grid_config_digest(path::AbstractString)
    return "sha256:" * bytes2hex(sha256(read(path)))
end

function resolved_grid_json(configs)
    return sprint(io -> JSON3.pretty(io, json_ready(configs)))
end

function json_ready(value)
    value === missing && return nothing
    value === nothing && return nothing
    value isa Symbol && return string(value)
    value isa Number && return value
    value isa Bool && return value
    value isa AbstractString && return value

    if value isa NamedTuple
        return Dict(string(key) => json_ready(getproperty(value, key)) for key in keys(value))
    elseif value isa AbstractDict
        return Dict(string(key) => json_ready(item) for (key, item) in value)
    elseif value isa AbstractVector || value isa Tuple
        return Any[json_ready(item) for item in value]
    end

    return string(value)
end
