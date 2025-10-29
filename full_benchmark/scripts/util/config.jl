module Config

using Dates
using Random
using Base.Filesystem: expanduser

export ExperimentConfig,
       load_config,
       seed_rng!,
       default_output_dir,
       StepOrder,
       resolve_flag_priority

const StepOrder = [
    :generate_testing_data,
    :compute_saa_baselines,
    :generate_training_data,
    :train_baselines,
    :train_neural,
    :run_benchmark
]

struct ExperimentConfig
    output_dir::String
    input_dir::Union{Nothing,String}
    full_training::Bool
    method_training::Bool
    neural_training::Bool
    full_testing::Bool
    training_size::Int
    training_scenarios_per_context::Int
    testing_covariates::Int
    testing_collections_per_covariate::Int
    testing_scenarios_per_collection::Int
    training_param_p::Float64
    testing_param_p::Float64
    annealing_schedule::Vector{Float64}
    epoch_schedule::Vector{Int}
    step_size_schedule::Vector{Float64}
    batch_size_schedule::Vector{Int}
    surrogate_parameter::Float64
    testing_covariate_seed::Int
    testing_scenario_seed::Int
    training_covariate_seed::Int
    training_scenario_seed::Int
    neural_seed::Int
    num_products::Int
    feature_dim::Int
    sigma::Float64
    omega::Float64
end

default_output_dir() = joinpath(pwd(), "full_benchmark", "outputs", Dates.format(Dates.now(), "yyyymmdd-HHMMSS"))

function load_config(args::Vector{String})
    output_dir = default_output_dir()
    input_dir = nothing
    full_training = false
    method_training = false
    neural_training = false
    full_testing = false
    training_size = 100
    training_scenarios_per_context = 1
    testing_covariates = 30
    testing_collections_per_covariate = 1
    testing_scenarios_per_collection = 100
    training_param_p = 2.0
    testing_param_p = 2.0
    annealing_schedule = [1.0, 0.1, 0.01]
    epoch_schedule = [100, 30, 30, 30]
    step_size_schedule = [1e-3, 1e-4, 1e-4, 1e-4]
    batch_size_schedule = [10, 25, 25, 25]
    surrogate_parameter = annealing_schedule[end]
    testing_covariate_seed = 11_042
    testing_scenario_seed = 11_043
    training_covariate_seed = 42_001
    training_scenario_seed = 42_002
    neural_seed = 77_777
    num_products = 30
    feature_dim = 3
    sigma = 5.0
    omega = 1.0

    i = 1
    while i <= length(args)
        arg = args[i]
        next_arg = i < length(args) ? args[i+1] : nothing
        function take_value(default)
            if next_arg === nothing || startswith(next_arg, "--")
                return default, 0
            else
                return next_arg, 1
            end
        end

        if startswith(arg, "--output-dir")
            value, consumed = parse_key_value(arg, next_arg)
            output_dir = expanduser(abspath(value))
            i += consumed
        elseif startswith(arg, "--input-dir")
            value, consumed = parse_key_value(arg, next_arg)
            input_dir = expanduser(abspath(value))
            i += consumed
        elseif arg == "--full_training"
            full_training = true
        elseif arg == "--method_training"
            method_training = true
        elseif arg == "--neural_training"
            neural_training = true
        elseif arg == "--full_testing"
            full_testing = true
        elseif startswith(arg, "--training-size")
            value, consumed = parse_key_value(arg, next_arg)
            training_size = parse(Int, value)
            i += consumed
        elseif startswith(arg, "--training-scenarios-per-context")
            value, consumed = parse_key_value(arg, next_arg)
            training_scenarios_per_context = parse(Int, value)
            i += consumed
        elseif startswith(arg, "--testing-covariates")
            value, consumed = parse_key_value(arg, next_arg)
            testing_covariates = parse(Int, value)
            i += consumed
        elseif startswith(arg, "--testing-collections")
            value, consumed = parse_key_value(arg, next_arg)
            testing_collections_per_covariate = parse(Int, value)
            i += consumed
        elseif startswith(arg, "--testing-scenarios")
            value, consumed = parse_key_value(arg, next_arg)
            testing_scenarios_per_collection = parse(Int, value)
            i += consumed
        elseif startswith(arg, "--training-p")
            value, consumed = parse_key_value(arg, next_arg)
            training_param_p = parse(Float64, value)
            i += consumed
        elseif startswith(arg, "--testing-p")
            value, consumed = parse_key_value(arg, next_arg)
            testing_param_p = parse(Float64, value)
            i += consumed
        elseif startswith(arg, "--annealing")
            value, consumed = parse_key_value(arg, next_arg)
            annealing_schedule = parse_float_vector(value)
            i += consumed
        elseif startswith(arg, "--epochs")
            value, consumed = parse_key_value(arg, next_arg)
            epoch_schedule = parse_int_vector(value)
            i += consumed
        elseif startswith(arg, "--step-sizes")
            value, consumed = parse_key_value(arg, next_arg)
            step_size_schedule = parse_float_vector(value)
            i += consumed
        elseif startswith(arg, "--batch-sizes")
            value, consumed = parse_key_value(arg, next_arg)
            batch_size_schedule = parse_int_vector(value)
            i += consumed
        elseif startswith(arg, "--surrogate-param")
            value, consumed = parse_key_value(arg, next_arg)
            surrogate_parameter = parse(Float64, value)
            i += consumed
        elseif startswith(arg, "--testing-covariate-seed")
            value, consumed = parse_key_value(arg, next_arg)
            testing_covariate_seed = parse(Int, value)
            i += consumed
        elseif startswith(arg, "--testing-scenario-seed")
            value, consumed = parse_key_value(arg, next_arg)
            testing_scenario_seed = parse(Int, value)
            i += consumed
        elseif startswith(arg, "--training-covariate-seed")
            value, consumed = parse_key_value(arg, next_arg)
            training_covariate_seed = parse(Int, value)
            i += consumed
        elseif startswith(arg, "--training-scenario-seed")
            value, consumed = parse_key_value(arg, next_arg)
            training_scenario_seed = parse(Int, value)
            i += consumed
        elseif startswith(arg, "--neural-seed")
            value, consumed = parse_key_value(arg, next_arg)
            neural_seed = parse(Int, value)
            i += consumed
        elseif startswith(arg, "--sigma")
            value, consumed = parse_key_value(arg, next_arg)
            sigma = parse(Float64, value)
            i += consumed
        elseif startswith(arg, "--omega")
            value, consumed = parse_key_value(arg, next_arg)
            omega = parse(Float64, value)
            i += consumed
        else
            error("Unrecognised argument: $arg")
        end
        i += 1
    end

    return ExperimentConfig(output_dir,
                            input_dir,
                            full_training,
                            method_training,
                            neural_training,
                            full_testing,
                            training_size,
                            training_scenarios_per_context,
                            testing_covariates,
                            testing_collections_per_covariate,
                            testing_scenarios_per_collection,
                            training_param_p,
                            testing_param_p,
                            annealing_schedule,
                            epoch_schedule,
                            step_size_schedule,
                            batch_size_schedule,
                            surrogate_parameter,
                            testing_covariate_seed,
                            testing_scenario_seed,
                            training_covariate_seed,
                            training_scenario_seed,
                            neural_seed,
                            num_products,
                            feature_dim,
                            sigma,
                            omega)
end

function parse_key_value(arg::String, next_arg::Union{Nothing,String})
    if occursin("=", arg)
        key, value = split(arg, "=", limit=2)
        return value, 0
    elseif next_arg === nothing
        error("Argument $arg expects a value")
    else
        return next_arg, 1
    end
end

parse_float_vector(str::String) = [parse(Float64, strip(s)) for s in split(str, ",")]
parse_int_vector(str::String) = [parse(Int, strip(s)) for s in split(str, ",")]

seed_rng!(_scope::Symbol, seed::Integer) = MersenneTwister(seed)

function resolve_flag_priority(config::ExperimentConfig)
    if config.full_training
        return :full_training
    elseif config.full_testing
        return :full_testing
    elseif config.method_training
        return :method_training
    elseif config.neural_training
        return :neural_training
    else
        return :none
    end
end

end # module
