using Dates
using Serialization
using SHA

const EXPERIMENT_CONFIG_FILENAME = "Config.jl"
const EXPERIMENTS_ROOT = @__DIR__
const OPTIMAL_RESULTS_FORMAT_VERSION = 1
const TEST_DATA_FORMAT_VERSION = 1
const TRAINING_DATA_FORMAT_VERSION = 1
const DEFAULT_TEST_DATA_SEED = 1
const DEFAULT_TEST_DATA_SET_SIZE = 30

struct ExperimentSpec
    id::String
    name::String
    module_name::Symbol
    module_ref::Module
    root_dir::String
    config_path::String
end

struct LatestExperimentFunction <: Function
    module_ref::Module
    name::Symbol
end

experiment_binding_isdefined(module_ref::Module, name::Symbol) =
    Base.invokelatest(isdefined, module_ref, name)

function experiment_binding(module_ref::Module, name::Symbol)
    experiment_binding_isdefined(module_ref, name) ||
        throw(ArgumentError("experiment module $(module_ref) does not define $name"))
    return Base.invokelatest(getfield, module_ref, name)
end

function (fn::LatestExperimentFunction)(args...; kwargs...)
    target = experiment_binding(fn.module_ref, fn.name)
    target isa Function ||
        throw(ArgumentError("experiment binding $(fn.name) must be a function, got $(typeof(target))"))
    return Base.invokelatest(target, args...; kwargs...)
end

const REQUIRED_EXPERIMENT_FUNCTIONS = (
    :experiment_id,
    :experiment_name,
    :artifact_dir,
    :base_config,
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
    declared_module_name = if experiment_binding_isdefined(module_ref, :experiment_module_name)
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
        experiment_binding_isdefined(module_ref, name) || push!(missing, name)
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
    fn = experiment_binding(module_ref, name)
    fn isa Function ||
        throw(ArgumentError("experiment binding $name must be a function, got $(typeof(fn))"))
    return Base.invokelatest(fn, args...; kwargs...)
end

function experiment_has_function(spec::ExperimentSpec, name::Symbol)
    return experiment_binding_isdefined(spec.module_ref, name) &&
           experiment_binding(spec.module_ref, name) isa Function
end

experiment_artifact_dir(spec::ExperimentSpec) = abspath(String(experiment_call(spec, :artifact_dir)))

function experiment_base_config(spec::ExperimentSpec)
    return with_experiment_metadata(spec, experiment_call(spec, :base_config))
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

    if hasproperty(config, :experiment_config_path)
        return load_experiment(getproperty(config, :experiment_config_path))
    end

    for key in (:experiment_id, :experiment_module_name, :experiment_name)
        if hasproperty(config, key)
            return load_experiment(getproperty(config, key))
        end
    end

    return nothing
end

function training_objects_for_config(config::NamedTuple)
    spec = experiment_from_config(config)
    spec === nothing && throw(
        ArgumentError(
            "config must declare experiment_id, experiment_module_name, experiment_name, or experiment_config_path to construct training objects.",
        ),
    )
    config = with_experiment_metadata(spec, config)

    if experiment_has_function(spec, :training_data)
        artifact = load_or_generate_training_data(spec, config)
        objects = experiment_call(spec, :training_objects, config, artifact.dataset)
        return with_training_data_metadata(objects, artifact)
    end

    return experiment_call(spec, :training_objects, config)
end

function experiment_config_value(config, key::Symbol, default)
    config isa NamedTuple || return default
    return key in keys(config) ? getproperty(config, key) : default
end

function training_data_cache_enabled(config)
    return Bool(experiment_config_value(config, :training_data_cache, true)) &&
           Bool(experiment_config_value(config, :write_training_data_artifact, true))
end

function training_data_dir(spec::ExperimentSpec, config::NamedTuple)
    return abspath(
        string(
            experiment_config_value(
                config,
                :training_data_dir,
                joinpath(experiment_artifact_dir(spec), "training_data"),
            ),
        ),
    )
end

function training_dataset_name(spec::ExperimentSpec, config::NamedTuple)
    if experiment_has_function(spec, :training_dataset_name)
        return string(experiment_call(spec, :training_dataset_name, config))
    end

    training_seed = experiment_config_value(config, :training_data_seed, nothing)
    return isnothing(training_seed) ? spec.name : string(spec.name, "-", Int(training_seed))
end

function training_data_identity(spec::ExperimentSpec, config::NamedTuple)
    if experiment_has_function(spec, :training_data_identity)
        return experiment_call(spec, :training_data_identity, config)
    end

    return (;
        experiment_id=spec.id,
        experiment_name=spec.name,
        training_data_seed=experiment_config_value(config, :training_data_seed, missing),
    )
end

function training_data_identity_digest(spec::ExperimentSpec, config::NamedTuple)
    return experiment_dataset_digest(training_data_identity(spec, config))
end

function training_data_digest_slug(digest)
    text = replace(string(digest), r"[^A-Za-z0-9]+" => "-")
    return text[1:min(lastindex(text), 24)]
end

function safe_training_dataset_name(name)
    return replace(string(name), r"[^A-Za-z0-9_.=-]+" => "_")
end

function training_data_path(spec::ExperimentSpec, config::NamedTuple, name, identity_digest)
    filename = string(
        safe_training_dataset_name(name),
        "_",
        training_data_digest_slug(identity_digest),
        ".jls",
    )
    return joinpath(training_data_dir(spec, config), filename)
end

function load_or_generate_training_data(spec::ExperimentSpec, config::NamedTuple)
    name = training_dataset_name(spec, config)
    identity = training_data_identity(spec, config)
    identity_digest = experiment_dataset_digest(identity)
    path = training_data_path(spec, config, name, identity_digest)

    if !training_data_cache_enabled(config)
        dataset = experiment_call(spec, :training_data, config)
        return training_data_artifact(
            dataset;
            path="",
            name=name,
            identity_digest=identity_digest,
            cache_hit=false,
        )
    end

    isfile(path) && return load_training_data_artifact(spec, path, identity_digest; cache_hit=true)

    mkpath(dirname(path))
    return with_training_data_lock(path, config) do
        if isfile(path)
            load_training_data_artifact(spec, path, identity_digest; cache_hit=true)
        else
            dataset = experiment_call(spec, :training_data, config)
            save_training_data_artifact!(
                spec,
                path,
                dataset;
                name=name,
                identity=identity,
                identity_digest=identity_digest,
            )
            load_training_data_artifact(spec, path, identity_digest; cache_hit=false)
        end
    end
end

function with_training_data_lock(callback, path::AbstractString, config::NamedTuple)
    lock_dir = path * ".lock"
    timeout_seconds = Float64(
        experiment_config_value(config, :training_data_lock_timeout_seconds, 600.0),
    )
    started = time()
    acquired = false

    while !acquired
        try
            mkdir(lock_dir)
            acquired = true
        catch error
            if isdir(lock_dir)
                time() - started <= timeout_seconds ||
                    throw(ErrorException("timed out waiting for training data cache lock: $lock_dir"))
                sleep(0.25)
            else
                rethrow()
            end
        end
    end

    try
        return callback()
    finally
        rm(lock_dir; force=true, recursive=true)
    end
end

function save_training_data_artifact!(
    spec::ExperimentSpec,
    path::AbstractString,
    dataset;
    name,
    identity,
    identity_digest,
)
    mkpath(dirname(path))
    payload = (;
        format_version=TRAINING_DATA_FORMAT_VERSION,
        experiment_id=spec.id,
        experiment_name=spec.name,
        dataset_name=string(name),
        generated_at=Dates.format(now(UTC), dateformat"yyyy-mm-ddTHH:MM:SS.sZ"),
        training_data_identity=identity,
        training_data_identity_digest=string(identity_digest),
        dataset_digest=experiment_dataset_digest(dataset),
        dataset=dataset,
    )
    atomic_serialize(path, payload)
    return path
end

function load_training_data_artifact(
    spec::ExperimentSpec,
    path::AbstractString,
    identity_digest;
    cache_hit::Bool,
)
    payload = open(Serialization.deserialize, path)
    validate_training_data_payload(spec, payload, path, identity_digest)
    return training_data_artifact(
        training_data_from_payload(payload);
        path=path,
        name=payload.dataset_name,
        identity_digest=payload.training_data_identity_digest,
        dataset_digest=payload.dataset_digest,
        cache_hit=cache_hit,
    )
end

function training_data_from_payload(payload)
    payload isa NamedTuple && hasproperty(payload, :dataset) && return payload.dataset
    throw(ArgumentError("training data artifact payload must contain a dataset field."))
end

function validate_training_data_payload(
    spec::ExperimentSpec,
    payload,
    path::AbstractString,
    identity_digest,
)
    payload isa NamedTuple ||
        throw(ArgumentError("training data artifact $path must contain a metadata payload."))
    hasproperty(payload, :format_version) && payload.format_version == TRAINING_DATA_FORMAT_VERSION ||
        throw(ArgumentError("unsupported training-data format in $path"))
    String(payload.experiment_id) == spec.id ||
        throw(ArgumentError("training data artifact $path belongs to experiment $(payload.experiment_id), expected $(spec.id)."))
    String(payload.training_data_identity_digest) == string(identity_digest) ||
        throw(ArgumentError("training data artifact $path does not match the current training data identity."))

    dataset = training_data_from_payload(payload)
    expected = experiment_dataset_digest(dataset)
    String(payload.dataset_digest) == expected ||
        throw(ArgumentError("training data artifact $path has an invalid dataset digest."))
    return nothing
end

function training_data_artifact(
    dataset;
    path,
    name,
    identity_digest,
    dataset_digest=experiment_dataset_digest(dataset),
    cache_hit,
)
    metadata = (;
        path=string(path),
        dataset_name=string(name),
        dataset_digest=string(dataset_digest),
        training_data_identity_digest=string(identity_digest),
        cache_hit=Bool(cache_hit),
    )
    return (; dataset=dataset, metadata=metadata)
end

function with_training_data_metadata(objects, artifact)
    objects isa NamedTuple ||
        throw(ArgumentError("training objects must be a NamedTuple when using central training_data memoization."))

    existing_metadata = if hasproperty(objects, :data_metadata) &&
                           getproperty(objects, :data_metadata) isa NamedTuple
        getproperty(objects, :data_metadata)
    else
        NamedTuple()
    end

    metadata = artifact.metadata
    path = metadata.path
    central_metadata = (;
        dataset_name=metadata.dataset_name,
        dataset_digest=metadata.dataset_digest,
        dataset_path=path,
        dataset_source=isempty(path) ? metadata.training_data_identity_digest : path,
        dataset_identity_digest=metadata.training_data_identity_digest,
        training_data_artifact=path,
        training_data_cache_hit=metadata.cache_hit,
    )

    return merge(
        objects,
        (;
            data_metadata=merge(existing_metadata, central_metadata),
            training_data_artifact=metadata,
        ),
    )
end

function optimality_splits_for_config(objects, config::NamedTuple)
    spec = experiment_from_config(config)
    spec === nothing && throw(
        ArgumentError(
            "config must declare experiment_id, experiment_module_name, experiment_name, or experiment_config_path to select optimality splits.",
        ),
    )
    return experiment_call(spec, :optimality_splits, objects, with_experiment_metadata(spec, config))
end

function experiment_test_data_config(
    spec::ExperimentSpec;
    seed::Integer=DEFAULT_TEST_DATA_SEED,
    data_set_size::Integer=DEFAULT_TEST_DATA_SET_SIZE,
    overrides...,
)
    if experiment_has_function(spec, :test_data_config)
        return with_experiment_metadata(
            spec,
            experiment_call(
                spec,
                :test_data_config;
                seed=Int(seed),
                data_set_size=Int(data_set_size),
                overrides...,
            ),
        )
    end

    return merge(
        experiment_base_config(spec),
        (;
            seed=Int(seed),
            test_data_seed=Int(seed),
            data_set_size=Int(data_set_size),
        ),
        NamedTuple(overrides),
    )
end

function experiment_test_data_bundle(
    spec::ExperimentSpec;
    seed::Integer=DEFAULT_TEST_DATA_SEED,
    data_set_size::Integer=DEFAULT_TEST_DATA_SET_SIZE,
    overrides...,
)
    config = experiment_test_data_config(
        spec;
        seed=seed,
        data_set_size=data_set_size,
        overrides...,
    )
    if experiment_has_function(spec, :test_data_bundle)
        return experiment_call(
            spec,
            :test_data_bundle,
            config;
            seed=Int(seed),
            data_set_size=Int(data_set_size),
        )
    end

    throw(
        ArgumentError(
            "experiment $(spec.id) does not define test_data_bundle; generated test data must be owned by the experiment config.",
        ),
    )
end

function test_data_dir(spec::ExperimentSpec)
    if experiment_has_function(spec, :test_data_dir)
        return abspath(String(experiment_call(spec, :test_data_dir)))
    end
    return joinpath(experiment_artifact_dir(spec), "test_data")
end

function test_data_path(spec::ExperimentSpec, seed::Integer=DEFAULT_TEST_DATA_SEED)
    if experiment_has_function(spec, :test_data_path)
        return abspath(String(experiment_call(spec, :test_data_path, Int(seed))))
    end
    return joinpath(test_data_dir(spec), "test_data_seed$(Int(seed)).jls")
end

function test_optimal_results_path(
    spec::ExperimentSpec,
    seed::Integer=DEFAULT_TEST_DATA_SEED,
)
    if experiment_has_function(spec, :test_optimal_results_path)
        return abspath(String(experiment_call(spec, :test_optimal_results_path, Int(seed))))
    end
    return joinpath(test_data_dir(spec), "optimal_solutions_seed$(Int(seed)).jls")
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
    if split_name == :test && uses_generated_test_data(spec)
        test_artifact = load_test_data_artifact(spec)
        test_seeds = test_artifact.metadata.seeds
        dataset_digests_by_seed = Dict(
            seed => digest for
            (seed, digest) in zip(test_seeds, test_artifact.metadata.dataset_digests)
        )
        paths = test_artifact_paths(spec, "optimal_solutions")
        isempty(paths) && throw(
            ArgumentError(
                "missing generated optimal-results artifacts for experiment $(spec.id) in $(test_data_dir(spec)). Run ContextualDFLTraining/generate_test_data.jl --experiment $(spec.id) first.",
            ),
        )

        results_by_seed = Dict{Int,Any}()
        for path in paths
            payload = open(Serialization.deserialize, path)
            hasproperty(payload, :test_data_seed) ||
                throw(ArgumentError("optimal-results artifact $path is missing test_data_seed."))
            hasproperty(payload, :data_set_size) ||
                throw(ArgumentError("optimal-results artifact $path is missing data_set_size."))
            hasproperty(payload, :dataset_digest) ||
                throw(ArgumentError("optimal-results artifact $path is missing dataset_digest."))

            seed = Int(payload.test_data_seed)
            haskey(results_by_seed, seed) &&
                throw(ArgumentError("duplicate optimal-results artifact for seed $seed."))
            haskey(dataset_digests_by_seed, seed) || throw(
                ArgumentError(
                    "optimal-results artifact $path has seed $seed, but no matching test-data artifact was loaded.",
                ),
            )

            results = optimal_results_from_payload(payload)
            length(results) == Int(payload.data_set_size) ||
                throw(ArgumentError("optimal-results artifact $path has $(length(results)) rows, expected $(payload.data_set_size)."))
            String(payload.dataset_digest) == dataset_digests_by_seed[seed] || throw(
                ArgumentError(
                    "optimal-results artifact $path does not match test-data artifact seed $seed. Regenerate it with generate_test_data.jl.",
                ),
            )
            validate_optimal_results_payload(spec, split_name, nothing, payload, path)
            results_by_seed[seed] = results
        end

        missing_seeds = setdiff(Set(test_seeds), Set(keys(results_by_seed)))
        isempty(missing_seeds) || throw(
            ArgumentError(
                "missing optimal-results artifact(s) for test data seed(s): $(join(sort!(collect(missing_seeds)), ", ")).",
            ),
        )

        results = vcat([results_by_seed[seed] for seed in test_seeds]...)
        if dataset !== nothing
            full_dataset = test_artifact.dataset
            if length(dataset) == length(full_dataset)
                experiment_dataset_digest(dataset) == experiment_dataset_digest(full_dataset) ||
                    throw(ArgumentError("optimal-results artifacts do not match the current dataset for split test. Regenerate them with generate_test_data.jl."))
            elseif length(dataset) < length(full_dataset)
                prefix_dataset = full_dataset[1:length(dataset)]
                experiment_dataset_digest(dataset) == experiment_dataset_digest(prefix_dataset) ||
                    throw(ArgumentError("limited test dataset must be a prefix of the generated test-data artifacts."))
                return results[1:length(dataset)]
            else
                throw(DimensionMismatch("test dataset has $(length(dataset)) rows, but generated optimal-results artifacts cover $(length(full_dataset)) rows."))
            end
        end
        return results
    end

    path = optimal_results_path(spec, split_name)
    isfile(path) || throw(
        ArgumentError(
            "missing precomputed optimal results for experiment $(spec.id), split $(split_name). Expected $path. Run ContextualDFLTraining/generate_test_data.jl for this experiment first.",
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

function uses_generated_test_data(spec::ExperimentSpec)
    return experiment_has_function(spec, :test_data_config) ||
           experiment_has_function(spec, :test_data_bundle) ||
           experiment_has_function(spec, :test_data_dir) ||
           experiment_has_function(spec, :test_data_path) ||
           experiment_has_function(spec, :test_optimal_results_path)
end

function test_artifact_paths(spec::ExperimentSpec, prefix::AbstractString)
    dir = test_data_dir(spec)
    isdir(dir) || return String[]
    pattern = Regex("^" * prefix * "_seed([0-9]+)\\.jls\$")
    paths = [
        joinpath(dir, name) for name in readdir(dir) if occursin(pattern, name)
    ]
    return sort!(
        paths;
        by=path -> parse(Int, only(match(pattern, basename(path)).captures)),
    )
end

function save_test_data!(
    spec::ExperimentSpec,
    seed::Integer,
    dataset;
    data_set_size::Integer=length(dataset),
    metadata=NamedTuple(),
)
    seed = Int(seed)
    data_set_size = Int(data_set_size)
    data_set_size > 0 || throw(ArgumentError("data_set_size must be positive."))
    length(dataset) == data_set_size ||
        throw(ArgumentError("test dataset length $(length(dataset)) != data_set_size $data_set_size."))

    path = test_data_path(spec, seed)
    mkpath(dirname(path))
    context_dimension = dataset_context_dimension(dataset)
    scenarios_per_context = dataset_scenarios_per_context(dataset)
    payload = merge(
        metadata,
        (;
            format_version=TEST_DATA_FORMAT_VERSION,
            experiment_id=spec.id,
            experiment_name=spec.name,
            split_name=:test,
            generated_at=Dates.format(now(UTC), dateformat"yyyy-mm-ddTHH:MM:SS.sZ"),
            test_data_seed=seed,
            data_set_size=data_set_size,
            context_dimension=context_dimension,
            scenarios_per_context=scenarios_per_context,
            dataset_digest=experiment_dataset_digest(dataset),
            dataset=dataset,
        ),
    )
    atomic_serialize(path, payload)
    return path
end

function load_test_data_artifact(spec::ExperimentSpec)
    paths = test_artifact_paths(spec, "test_data")
    isempty(paths) && throw(
        ArgumentError(
            "missing generated test data artifacts for experiment $(spec.id) in $(test_data_dir(spec)). Run ContextualDFLTraining/generate_test_data.jl --experiment $(spec.id) first.",
        ),
    )

    datasets = Any[]
    seeds = Int[]
    data_set_sizes = Int[]
    dataset_digests = String[]
    expected_data_set_size = nothing
    expected_context_dimension = nothing
    expected_scenarios_per_context = nothing

    for path in paths
        payload = open(Serialization.deserialize, path)
        dataset = test_data_from_payload(payload)
        validate_test_data_payload(spec, dataset, payload, path)
        isempty(dataset) &&
            throw(ArgumentError("test data artifact $path must not contain an empty dataset."))

        seed = Int(payload.test_data_seed)
        seed in seeds &&
            throw(ArgumentError("duplicate test data artifact for seed $seed."))
        data_set_size = Int(payload.data_set_size)
        context_dimension = dataset_context_dimension(dataset)
        scenarios_per_context = dataset_scenarios_per_context(dataset)

        if expected_data_set_size === nothing
            expected_data_set_size = data_set_size
            expected_context_dimension = context_dimension
            expected_scenarios_per_context = scenarios_per_context
        elseif data_set_size != expected_data_set_size ||
               context_dimension != expected_context_dimension ||
               scenarios_per_context != expected_scenarios_per_context
            throw(
                ArgumentError(
                    "test data artifact $path has shape (rows=$data_set_size, context_dimension=$context_dimension, scenarios_per_context=$scenarios_per_context), expected (rows=$expected_data_set_size, context_dimension=$expected_context_dimension, scenarios_per_context=$expected_scenarios_per_context).",
                ),
            )
        end

        push!(datasets, dataset)
        push!(seeds, seed)
        push!(data_set_sizes, data_set_size)
        push!(dataset_digests, String(payload.dataset_digest))
    end

    dataset = vcat(datasets...)
    return (;
        dataset=dataset,
        metadata=(;
            path=join(paths, ","),
            paths=paths,
            seed=first(seeds),
            seeds=seeds,
            data_set_size=length(dataset),
            data_set_sizes=data_set_sizes,
            dataset_digest=experiment_dataset_digest(dataset),
            dataset_digests=dataset_digests,
            context_dimension=expected_context_dimension,
            scenarios_per_context=expected_scenarios_per_context,
        ),
    )
end

function load_test_data_artifact(config::NamedTuple)
    spec = experiment_from_config(config)
    spec === nothing && throw(
        ArgumentError(
            "loading generated test data requires config.experiment_id, experiment_module_name, or experiment_name.",
        ),
    )
    return load_test_data_artifact(spec)
end

load_test_data(spec::ExperimentSpec) = load_test_data_artifact(spec).dataset
load_test_data(config::NamedTuple) = load_test_data_artifact(config).dataset

function save_test_optimal_results!(
    spec::ExperimentSpec,
    seed::Integer,
    results;
    dataset,
    data_set_size::Integer=length(dataset),
    metadata=NamedTuple(),
)
    seed = Int(seed)
    data_set_size = Int(data_set_size)
    data_set_size > 0 || throw(ArgumentError("data_set_size must be positive."))
    length(dataset) == data_set_size ||
        throw(ArgumentError("test dataset length $(length(dataset)) != data_set_size $data_set_size."))
    length(results) == data_set_size ||
        throw(ArgumentError("optimal results length $(length(results)) != data_set_size $data_set_size."))
    optimal_metadata = optimal_results_metadata(results)
    path = test_optimal_results_path(spec, seed)
    mkpath(dirname(path))
    payload = merge(
        metadata,
        optimal_metadata,
        (;
            format_version=OPTIMAL_RESULTS_FORMAT_VERSION,
            experiment_id=spec.id,
            experiment_name=spec.name,
            split_name=:test,
            generated_at=Dates.format(now(UTC), dateformat"yyyy-mm-ddTHH:MM:SS.sZ"),
            test_data_seed=seed,
            data_set_size=data_set_size,
            dataset_digest=experiment_dataset_digest(dataset),
            optimal_results=results,
        ),
    )
    atomic_serialize(path, payload)
    return path
end

function save_optimal_results!(
    spec::ExperimentSpec,
    split_name::Symbol,
    results;
    dataset=nothing,
    metadata=NamedTuple(),
)
    if dataset !== nothing && length(results) != length(dataset)
        throw(ArgumentError("optimal results length $(length(results)) != dataset length $(length(dataset))."))
    end
    optimal_metadata = optimal_results_metadata(results)
    path = optimal_results_path(spec, split_name)
    mkpath(dirname(path))
    payload = merge(
        metadata,
        optimal_metadata,
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

    atomic_serialize(path, payload)
    return path
end

function atomic_serialize(path::AbstractString, payload)
    temp_path = tempname(dirname(path))
    open(temp_path, "w") do io
        Serialization.serialize(io, payload)
    end
    mv(temp_path, path; force=true)
    return path
end

function test_data_from_payload(payload)
    payload isa NamedTuple && hasproperty(payload, :dataset) && return payload.dataset
    throw(ArgumentError("test data artifact payload must contain a dataset field."))
end

function validate_test_data_payload(
    spec::ExperimentSpec,
    dataset,
    payload,
    path::AbstractString,
)
    payload isa NamedTuple ||
        throw(ArgumentError("test data artifact $path must contain a metadata payload."))

    if hasproperty(payload, :format_version)
        payload.format_version == TEST_DATA_FORMAT_VERSION ||
            throw(ArgumentError("unsupported test-data format in $path"))
    end
    if hasproperty(payload, :experiment_id)
        String(payload.experiment_id) == spec.id ||
            throw(ArgumentError("test data artifact $path belongs to experiment $(payload.experiment_id), expected $(spec.id)."))
    end
    if hasproperty(payload, :split_name)
        Symbol(payload.split_name) == :test ||
            throw(ArgumentError("test data artifact $path belongs to split $(payload.split_name), expected test."))
    end
    hasproperty(payload, :test_data_seed) ||
        throw(ArgumentError("test data artifact $path is missing test_data_seed."))
    hasproperty(payload, :data_set_size) ||
        throw(ArgumentError("test data artifact $path is missing data_set_size."))
    hasproperty(payload, :dataset_digest) ||
        throw(ArgumentError("test data artifact $path is missing dataset_digest."))
    length(dataset) == Int(payload.data_set_size) ||
        throw(ArgumentError("test data artifact $path has $(length(dataset)) rows, expected $(payload.data_set_size)."))
    expected = experiment_dataset_digest(dataset)
    String(payload.dataset_digest) == expected ||
        throw(ArgumentError("test data artifact $path has an invalid dataset digest."))
    if hasproperty(payload, :context_dimension)
        Int(payload.context_dimension) == dataset_context_dimension(dataset) ||
            throw(ArgumentError("test data artifact $path has the wrong context_dimension."))
    end
    if hasproperty(payload, :scenarios_per_context)
        Int(payload.scenarios_per_context) == dataset_scenarios_per_context(dataset) ||
            throw(ArgumentError("test data artifact $path has the wrong scenarios_per_context."))
    end

    return nothing
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
    if !(payload isa NamedTuple)
        validate_optimal_results_collection(optimal_results_from_payload(payload), path)
        return nothing
    end

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
    results = optimal_results_from_payload(payload)
    validate_optimal_results_collection(results, path)
    if hasproperty(payload, :evaluation_batches)
        payload_batches = Int(payload.evaluation_batches)
        result_batches = optimal_results_evaluation_batches(results)
        payload_batches == result_batches ||
            throw(ArgumentError("optimal-results artifact $path declares evaluation_batches=$payload_batches, but results contain $result_batches batches."))
    end

    return nothing
end

function experiment_dataset_digest(dataset)
    io = IOBuffer()
    Serialization.serialize(io, dataset)
    return "sha1:" * bytes2hex(sha1(take!(io)))
end

function dataset_context_dimension(dataset)
    isempty(dataset) && return 0
    dimension = length(first(dataset).context)
    for (index, data_point) in enumerate(dataset)
        length(data_point.context) == dimension ||
            throw(ArgumentError(
                "dataset row $index has context dimension $(length(data_point.context)), expected $dimension.",
            ))
    end
    return dimension
end

function dataset_scenarios_per_context(dataset)
    isempty(dataset) && return 0
    scenario_count = length(first(dataset).scenario_parameters)
    for (index, data_point) in enumerate(dataset)
        length(data_point.scenario_parameters) == scenario_count ||
            throw(ArgumentError(
                "dataset row $index has $(length(data_point.scenario_parameters)) scenarios, expected $scenario_count.",
            ))
    end
    return scenario_count
end

function optimal_results_metadata(results)
    validate_optimal_results_collection(results, "optimal results")
    return (; evaluation_batches=optimal_results_evaluation_batches(results))
end

function validate_optimal_results_collection(results, source)
    results isa AbstractVector ||
        throw(ArgumentError("$source must contain an AbstractVector of optimal results."))
    isempty(results) && return nothing

    expected_batches = nothing
    for (index, result) in enumerate(results)
        values = optimal_result_objective_values(result, "$source row $index")
        batch_count = length(values)
        if expected_batches === nothing
            expected_batches = batch_count
        elseif batch_count != expected_batches
            throw(ArgumentError("$source contains mixed evaluation batch counts."))
        end
    end
    return nothing
end

function optimal_results_evaluation_batches(results)
    isempty(results) && return 0
    return length(optimal_result_objective_values(first(results), "optimal results row 1"))
end

function optimal_result_objective_values(result, source)
    values = if hasproperty(result, :objective_values)
        Float64.(collect(result.objective_values))
    elseif hasproperty(result, :objective_value)
        [Float64(result.objective_value)]
    else
        throw(ArgumentError("$source must contain objective_values or objective_value."))
    end

    isempty(values) &&
        throw(ArgumentError("$source must contain at least one objective value."))
    all(isfinite, values) ||
        throw(DomainError(values, "$source contains non-finite objective values."))

    if hasproperty(result, :objective_value)
        objective_value = Float64(result.objective_value)
        isfinite(objective_value) ||
            throw(DomainError(result.objective_value, "$source has a non-finite objective_value."))
        mean_value = sum(values) / length(values)
        isapprox(objective_value, mean_value; rtol=1e-10, atol=1e-10) ||
            throw(ArgumentError("$source objective_value=$objective_value does not equal mean(objective_values)=$mean_value."))
    end
    if hasproperty(result, :evaluation_batches)
        Int(result.evaluation_batches) == length(values) ||
            throw(ArgumentError("$source declares evaluation_batches=$(result.evaluation_batches), but has $(length(values)) objective_values."))
    end

    return values
end
