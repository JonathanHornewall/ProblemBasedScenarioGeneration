"""
Model checkpoint save/load for ProblemBasedScenarioGeneration.

Checkpoints store model weights (via `Flux.state`), architecture config,
problem metadata, and training history in a serialized `Dict{String,Any}`.
"""

using Serialization, Dates, Flux

# -----------------------------------------------------------------------
# Activation name mapping
# -----------------------------------------------------------------------

const ACTIVATION_MAP = Dict{String,Function}(
    "relu"     => relu,
    "tanh"     => tanh,
    "sigmoid"  => sigmoid,
    "softplus" => softplus,
    "identity" => identity,
)

const ACTIVATION_NAME = Dict{Function,String}(v => k for (k, v) in ACTIVATION_MAP)

function _activation_name(f::Function)
    return get(ACTIVATION_NAME, f, string(f))
end

function _activation_func(name::String)
    haskey(ACTIVATION_MAP, name) || error("Unknown activation: $name. Choose from: $(join(keys(ACTIVATION_MAP), ", "))")
    return ACTIVATION_MAP[name]
end

# -----------------------------------------------------------------------
# Problem type name mapping
# -----------------------------------------------------------------------

const PROBLEM_TYPE_MAP = Dict{String,Any}(
    "resource_allocation" => ResourceAllocationProblem,
    "shipment_planning"   => ShipmentPlanningProblem,
    "newsvendor"          => UnreliableNewsvendorProblem,
)

function _problem_type_name(prob::ProblemInstance)
    if prob isa ResourceAllocationProblem
        return "resource_allocation"
    elseif prob isa ShipmentPlanningProblem
        return "shipment_planning"
    elseif prob isa UnreliableNewsvendorProblem
        return "newsvendor"
    else
        return string(typeof(prob))
    end
end

function resolve_problem(name::String)
    haskey(PROBLEM_TYPE_MAP, name) || error("Unknown problem: $name. Choose from: $(join(keys(PROBLEM_TYPE_MAP), ", "))")
    return PROBLEM_TYPE_MAP[name]()
end

# -----------------------------------------------------------------------
# Save checkpoint
# -----------------------------------------------------------------------

"""
    save_checkpoint(path, model, prob, config::Dict, loss_history;
                    nr_of_scenarios=1)

Serialize a training checkpoint to `path`.

The checkpoint is a `Dict{String,Any}` containing:
- Model weights (via `Flux.state`)
- Architecture config (dims, layers, activation)
- Problem type and noise pattern
- Training hyperparameters
- Loss history
"""
function save_checkpoint(
    path::String,
    model,
    prob::ProblemInstance,
    config::Dict,
    loss_history::Vector{Float64};
    nr_of_scenarios::Int = 1
)
    # Infer model architecture from the Chain
    layers = model.layers
    input_dim  = size(layers[1].weight, 2)
    output_dim = size(layers[end].weight, 1)
    hidden_dim = size(layers[1].weight, 1)
    # n_layers = number of hidden layers (build_generator adds +1 output layer)
    n_layers   = length(layers) - 1
    # Activation of hidden layers (first layer)
    activation_name = _activation_name(layers[1].σ)

    checkpoint = Dict{String,Any}(
        "model_state"    => Flux.state(model),
        "model_config"   => Dict{String,Any}(
            "input_dim"  => input_dim,
            "output_dim" => output_dim,
            "hidden_dim" => hidden_dim,
            "n_layers"   => n_layers,
            "activation" => activation_name,
        ),
        "problem_type"     => _problem_type_name(prob),
        "noise_pattern"    => string(noise_pattern(prob)),
        "nr_of_scenarios"  => nr_of_scenarios,
        "training_config"  => config,
        "loss_history"     => loss_history,
        "total_epochs"     => get(config, "total_epochs", length(loss_history)),
        "timestamp"        => Dates.format(now(), dateformat"yyyy-mm-ddTHH:MM:SS"),
    )

    Serialization.serialize(path, checkpoint)
    return path
end

# -----------------------------------------------------------------------
# Load checkpoint
# -----------------------------------------------------------------------

"""
    load_checkpoint(path) -> Dict{String,Any}

Deserialize a training checkpoint from `path`.
"""
function load_checkpoint(path::String)
    isfile(path) || error("Checkpoint file not found: $path")
    return Serialization.deserialize(path)
end

# -----------------------------------------------------------------------
# Restore model from checkpoint
# -----------------------------------------------------------------------

"""
    restore_model(checkpoint::Dict) -> Flux.Chain

Rebuild the neural network architecture from checkpoint metadata
and load saved weights into it.
"""
function restore_model(checkpoint::Dict)
    mc = checkpoint["model_config"]
    activation = _activation_func(mc["activation"])

    model = build_generator(
        mc["input_dim"], mc["output_dim"];
        hidden_dim = mc["hidden_dim"],
        n_layers   = mc["n_layers"],
        activation = activation,
    )

    Flux.loadmodel!(model, checkpoint["model_state"])
    return model
end

# -----------------------------------------------------------------------
# Print checkpoint info
# -----------------------------------------------------------------------

"""
    print_checkpoint_info(io, checkpoint)

Pretty-print checkpoint metadata.
"""
function print_checkpoint_info(io::IO, checkpoint::Dict)
    mc = checkpoint["model_config"]
    tc = get(checkpoint, "training_config", Dict())

    println(io, "=== Checkpoint Info ===")
    println(io, "Timestamp:        ", get(checkpoint, "timestamp", "unknown"))
    println(io, "Problem:          ", get(checkpoint, "problem_type", "unknown"))
    println(io, "Noise pattern:    ", get(checkpoint, "noise_pattern", "unknown"))
    println(io, "Nr of scenarios:  ", get(checkpoint, "nr_of_scenarios", "unknown"))
    println(io)
    println(io, "--- Model Architecture ---")
    println(io, "Input dim:        ", mc["input_dim"])
    println(io, "Output dim:       ", mc["output_dim"])
    println(io, "Hidden dim:       ", mc["hidden_dim"])
    println(io, "Layers:           ", mc["n_layers"])
    println(io, "Activation:       ", mc["activation"])
    n_params = _count_params(mc)
    println(io, "Total parameters: ", n_params)
    println(io)
    println(io, "--- Training Config ---")
    for (k, v) in sort(collect(tc), by=first)
        println(io, "  ", rpad(k, 20), v)
    end
    println(io)
    losses = get(checkpoint, "loss_history", Float64[])
    println(io, "--- Training History ---")
    println(io, "Total epochs:     ", get(checkpoint, "total_epochs", length(losses)))
    if !isempty(losses)
        println(io, "Initial loss:     ", losses[1])
        println(io, "Final loss:       ", losses[end])
        println(io, "Best loss:        ", minimum(losses))
    end
end

function print_checkpoint_info(checkpoint::Dict)
    print_checkpoint_info(stdout, checkpoint)
end

function _count_params(mc::Dict)
    input_dim  = mc["input_dim"]
    output_dim = mc["output_dim"]
    hidden_dim = mc["hidden_dim"]
    n_layers   = mc["n_layers"]
    # First layer: input -> hidden
    total = input_dim * hidden_dim + hidden_dim
    # Middle layers: hidden -> hidden
    for _ in 2:n_layers
        total += hidden_dim * hidden_dim + hidden_dim
    end
    # Output layer: hidden -> output
    total += hidden_dim * output_dim + output_dim
    return total
end
