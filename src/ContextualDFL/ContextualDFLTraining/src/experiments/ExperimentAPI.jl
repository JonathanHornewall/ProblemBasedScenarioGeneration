using Dates
using Serialization
using SHA

const EXPERIMENT_CONFIG_FILENAME = "Config.jl"
const EXPERIMENTS_ROOT = @__DIR__
const OPTIMAL_RESULTS_FORMAT_VERSION = 1

struct ExperimentSpec
    id::String
    name::String
    module_name::Symbol
    module_ref::Module
    root_dir::String
    config_path::String
end

const REQUIRED_EXPERIMENT_FUNCTIONS = (
    :experiment_id,
    :experiment_name,
    :artifact_dir,
    :base_config,
    :grid_configs,
    :smoke_configs,
    :training_objects,
    :optimality_splits,
    :optimal_results_path,
)

function load_experiment()
    throw(
        ArgumentError(
            "load_experiment requires an explicit selector. Scripts should pass --experiment <experiment>.",
        ),
    )
end

function load_experiment(selector)
    selector isa ExperimentSpec && return selector
    selector_string = string(selector)
    path = resolve_experiment_config_path(selector_string)
    spec = load_experiment_config(path)

    if !experiment_selector_matches(spec, selector_string)
        throw(
            ArgumentError(
                "experiment selector '$selector_string' resolved to $(spec.config_path), but that config declares experiment_id=$(spec.id), experiment_name=$(spec.name), module_name=$(spec.module_name)",
            ),
        )
    end

    return spec
end

function resolve_experiment_config_path(selector::AbstractString)
    if isfile(selector)
        return abspath(selector)
    end

    direct = direct_experiment_config_path(selector)
    direct === nothing || return direct

    matches = ExperimentSpec[]
    for path in experiment_config_paths()
        spec = load_experiment_config(path)
        experiment_selector_matches(spec, selector) && push!(matches, spec)
    end

    isempty(matches) && throw(
        ArgumentError(
            "no experiment config found for selector '$selector'. Expected a file path, an experiment id such as resource_allocation/experiment_1, or a declared module/name under $EXPERIMENTS_ROOT.",
        ),
    )
    length(matches) == 1 || throw(
        ArgumentError(
            "experiment selector '$selector' matched multiple configs: $(join((spec.config_path for spec in matches), ", "))",
        ),
    )

    return only(matches).config_path
end

function direct_experiment_config_path(selector::AbstractString)
    normalized = replace(strip(selector), "\\" => "/")
    isempty(normalized) && return nothing

    candidates = String[]
    push!(candidates, joinpath(EXPERIMENTS_ROOT, split(normalized, "/")..., EXPERIMENT_CONFIG_FILENAME))
    push!(candidates, joinpath(EXPERIMENTS_ROOT, normalized))

    for candidate in candidates
        isfile(candidate) && return abspath(candidate)
    end

    return nothing
end

function experiment_config_paths()
    paths = String[]
    for (root, _, files) in walkdir(EXPERIMENTS_ROOT)
        EXPERIMENT_CONFIG_FILENAME in files &&
            push!(paths, abspath(joinpath(root, EXPERIMENT_CONFIG_FILENAME)))
    end
    sort!(paths)
    return paths
end

function load_experiment_config(path::AbstractString)
    absolute_path = abspath(path)
    isfile(absolute_path) ||
        throw(ArgumentError("experiment config file does not exist: $absolute_path"))

    module_name = dynamic_experiment_module_name(absolute_path)
    module_ref = Module(module_name)
    Core.eval(module_ref, :(import ContextualDFLTraining))
    Core.eval(module_ref, :(import ContextualDFLExperiments))
    Core.eval(module_ref, :(import ContextualDFL))
    Base.include(module_ref, absolute_path)

    validate_experiment_module!(module_ref, absolute_path)

    id = String(experiment_call(module_ref, :experiment_id))
    name = String(experiment_call(module_ref, :experiment_name))
    declared_module_name = if isdefined(module_ref, :experiment_module_name)
        Symbol(experiment_call(module_ref, :experiment_module_name))
    else
        module_name
    end

    spec = ExperimentSpec(
        id,
        name,
        declared_module_name,
        module_ref,
        dirname(absolute_path),
        absolute_path,
    )
    validate_experiment!(spec)
    return spec
end

function dynamic_experiment_module_name(path::AbstractString)
    relative = try
        relpath(path, EXPERIMENTS_ROOT)
    catch
        basename(path)
    end
    safe = replace(relative, r"[^A-Za-z0-9_]" => "_")
    digest = bytes2hex(sha1(path))[1:8]
    return Symbol(:ContextualDFLTrainingExperiment_, safe, :_, digest)
end

function validate_experiment_module!(module_ref::Module, path::AbstractString)
    missing = Symbol[]
    for name in REQUIRED_EXPERIMENT_FUNCTIONS
        isdefined(module_ref, name) || push!(missing, name)
    end
    isempty(missing) || throw(
        ArgumentError(
            "experiment config $path is missing required function(s): $(join(string.(missing), ", "))",
        ),
    )
    return nothing
end

function validate_experiment!(spec::ExperimentSpec)
    isempty(spec.id) && throw(ArgumentError("experiment_id() must not be empty."))
    isempty(spec.name) && throw(ArgumentError("experiment_name() must not be empty."))
    isabspath(experiment_artifact_dir(spec)) ||
        throw(ArgumentError("artifact_dir() must return an absolute path."))
    return spec
end

function experiment_selector_matches(spec::ExperimentSpec, selector::AbstractString)
    normalized = normalize_experiment_selector(selector)
    return normalized in (
        normalize_experiment_selector(spec.id),
        normalize_experiment_selector(spec.name),
        normalize_experiment_selector(string(spec.module_name)),
        normalize_experiment_selector(spec.config_path),
        normalize_experiment_selector(relpath(spec.root_dir, EXPERIMENTS_ROOT)),
    )
end

function normalize_experiment_selector(selector::AbstractString)
    value = lowercase(strip(selector))
    value = replace(value, "\\" => "/")
    value = replace(value, r"[^a-z0-9/]+" => "_")
    value = replace(value, r"_+" => "_")
    return strip(value, ['_', '/'])
end

function experiment_call(spec::ExperimentSpec, name::Symbol, args...; kwargs...)
    return experiment_call(spec.module_ref, name, args...; kwargs...)
end

function experiment_call(module_ref::Module, name::Symbol, args...; kwargs...)
    isdefined(module_ref, name) ||
        throw(ArgumentError("experiment module $(module_ref) does not define $name"))
    fn = Base.invokelatest(getfield, module_ref, name)
    fn isa Function ||
        throw(ArgumentError("experiment binding $name must be a function, got $(typeof(fn))"))
    return Base.invokelatest(fn, args...; kwargs...)
end

function experiment_has_function(spec::ExperimentSpec, name::Symbol)
    return isdefined(spec.module_ref, name) &&
           Base.invokelatest(getfield, spec.module_ref, name) isa Function
end

experiment_artifact_dir(spec::ExperimentSpec) = abspath(String(experiment_call(spec, :artifact_dir)))

function experiment_base_config(spec::ExperimentSpec)
    return with_experiment_metadata(spec, experiment_call(spec, :base_config))
end

function experiment_grid_configs(spec::ExperimentSpec; kwargs...)
    return [
        with_experiment_metadata(spec, config) for
        config in experiment_call(spec, :grid_configs; kwargs...)
    ]
end

function experiment_smoke_configs(spec::ExperimentSpec; kwargs...)
    return [
        with_experiment_metadata(spec, config) for
        config in experiment_call(spec, :smoke_configs; kwargs...)
    ]
end

function with_experiment_metadata(spec::ExperimentSpec, config::NamedTuple)
    return merge(
        config,
        (;
            experiment_id=spec.id,
            experiment_name=spec.name,
            experiment_module_name=spec.module_name,
            experiment_config_path=spec.config_path,
            experiment_artifact_dir=experiment_artifact_dir(spec),
        ),
    )
end

function experiment_from_config(config)
    if config isa ExperimentSpec
        return config
    end
    config isa NamedTuple || return nothing

    for key in (:experiment_id, :experiment_module_name, :experiment_name)
        if hasproperty(config, key)
            return load_experiment(getproperty(config, key))
        end
    end

    return nothing
end

function training_objects_for_config(config::NamedTuple)
    spec = experiment_from_config(config)
    spec === nothing && return resource_allocation_training_objects(config)
    return experiment_call(spec, :training_objects, with_experiment_metadata(spec, config))
end

function optimality_splits_for_config(objects, config::NamedTuple)
    spec = experiment_from_config(config)
    spec === nothing && return optimality_evaluation_datasets(objects, config)
    return experiment_call(spec, :optimality_splits, objects, with_experiment_metadata(spec, config))
end

function optimal_results_path(spec::ExperimentSpec, split_name::Symbol)
    return abspath(String(experiment_call(spec, :optimal_results_path, split_name)))
end

function optimal_results_path(config::NamedTuple, split_name::Symbol)
    spec = experiment_from_config(config)
    spec === nothing && throw(
        ArgumentError(
            "config must declare experiment_id, experiment_module_name, or experiment_name to locate optimal results.",
        ),
    )
    return optimal_results_path(spec, split_name)
end

function load_optimal_results(
    spec::ExperimentSpec,
    split_name::Symbol;
    dataset=nothing,
)
    path = optimal_results_path(spec, split_name)
    isfile(path) || throw(
        ArgumentError(
            "missing precomputed optimal results for experiment $(spec.id), split $(split_name). Expected $path. Run ContextualDFLTraining/generate_optimal_solutions.jl for this experiment first.",
        ),
    )

    payload = open(Serialization.deserialize, path)
    results = optimal_results_from_payload(payload)
    validate_optimal_results_payload(spec, split_name, dataset, payload, path)
    return results
end

function load_optimal_results(config::NamedTuple, split_name::Symbol; dataset=nothing)
    spec = experiment_from_config(config)
    spec === nothing && throw(
        ArgumentError(
            "optimality_evaluation=true requires config.experiment_id so precomputed optimal results can be loaded.",
        ),
    )
    return load_optimal_results(spec, split_name; dataset=dataset)
end

function save_optimal_results!(
    spec::ExperimentSpec,
    split_name::Symbol,
    results;
    dataset=nothing,
    metadata=NamedTuple(),
)
    path = optimal_results_path(spec, split_name)
    mkpath(dirname(path))
    payload = merge(
        metadata,
        (;
            format_version=OPTIMAL_RESULTS_FORMAT_VERSION,
            experiment_id=spec.id,
            experiment_name=spec.name,
            split_name=Symbol(split_name),
            generated_at=Dates.format(now(UTC), dateformat"yyyy-mm-ddTHH:MM:SS.sZ"),
            dataset_digest=dataset === nothing ? missing : experiment_dataset_digest(dataset),
            optimal_results=results,
        ),
    )

    temp_path = tempname(dirname(path))
    open(temp_path, "w") do io
        Serialization.serialize(io, payload)
    end
    mv(temp_path, path; force=true)
    return path
end

function optimal_results_from_payload(payload)
    if payload isa NamedTuple && hasproperty(payload, :optimal_results)
        return payload.optimal_results
    end
    return payload
end

function validate_optimal_results_payload(
    spec::ExperimentSpec,
    split_name::Symbol,
    dataset,
    payload,
    path::AbstractString,
)
    payload isa NamedTuple || return nothing

    if hasproperty(payload, :format_version)
        payload.format_version == OPTIMAL_RESULTS_FORMAT_VERSION ||
            throw(ArgumentError("unsupported optimal-results format in $path"))
    end
    if hasproperty(payload, :experiment_id)
        String(payload.experiment_id) == spec.id ||
            throw(ArgumentError("optimal-results artifact $path belongs to experiment $(payload.experiment_id), expected $(spec.id)."))
    end
    if hasproperty(payload, :split_name)
        Symbol(payload.split_name) == split_name ||
            throw(ArgumentError("optimal-results artifact $path belongs to split $(payload.split_name), expected $(split_name)."))
    end
    if dataset !== nothing && hasproperty(payload, :dataset_digest) &&
       payload.dataset_digest !== missing
        expected = experiment_dataset_digest(dataset)
        String(payload.dataset_digest) == expected ||
            throw(ArgumentError("optimal-results artifact $path does not match the current dataset for split $(split_name). Regenerate it with generate_optimal_solutions.jl."))
    end

    return nothing
end

function experiment_dataset_digest(dataset)
    io = IOBuffer()
    Serialization.serialize(io, dataset)
    return "sha1:" * bytes2hex(sha1(take!(io)))
end
