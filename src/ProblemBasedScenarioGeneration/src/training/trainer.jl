"""
Training loop for scenario generation models.

Uses Flux.jl's Adam optimizer with stochastic gradient updates.
The loss function is the decision regret (expected cost of surrogate
decision on the actual scenario).
"""

using Flux, Statistics

"""
    train!(model, prob, dataset;
           mu_surr=1.0, mu_prim=0.0,
           opt=Adam(1e-3), epochs=30, batchsize=1,
           verbose=false, opt_state=nothing) -> Vector{Float64}

Train a scenario generator `model` using decision regret as the loss.

Arguments:
- `model`: Flux model mapping context x ∈ ℝ^d to scenario parameters
- `prob`: `ProblemInstance` defining the optimization problem
- `dataset`: Vector of `(context, Scenario)` tuples

Keyword arguments:
- `mu_surr`: barrier regularization for surrogate solve
- `mu_prim`: barrier regularization for cost evaluation
- `opt`: Flux optimizer
- `epochs`: number of training epochs
- `batchsize`: mini-batch size (default 1 = SGD)
- `verbose`: print epoch losses

Returns a vector of per-epoch average losses.
"""
function train!(
    model,
    prob::ProblemInstance,
    dataset::Vector{<:Tuple};
    mu_surr::Float64  = 1.0,
    mu_prim::Float64  = 0.0,
    opt               = Adam(1e-3),
    epochs::Int       = 30,
    batchsize::Int    = 1,
    verbose::Bool     = false,
    opt_state         = nothing
)
    opt_state = isnothing(opt_state) ? Flux.setup(opt, model) : opt_state
    loss_history = Float64[]
    N = length(dataset)

    for epoch in 1:epochs
        epoch_losses = Float64[]
        indices = randperm(N)

        for batch_idx in Iterators.partition(indices, batchsize)
            batch = dataset[batch_idx]

            # Compute batch loss and gradients
            gs = Flux.gradient(model) do m
                batch_loss = mean(batch) do (x, actual_sc)
                    # Model predicts scenario parameters from context
                    raw_params = m(x)
                    # Convert to scenarios
                    predicted_sc = _params_to_scenarios(prob, raw_params)
                    # Compute decision regret
                    decision_regret(prob, mu_surr, mu_prim, predicted_sc, actual_sc)
                end
                batch_loss
            end

            gmodel = gs isa Tuple ? gs[1] : gs
            Flux.update!(opt_state, model, gmodel)

            # Track loss (without gradient)
            bl = mean(batch) do (x, actual_sc)
                raw_params = model(x)
                predicted_sc = _params_to_scenarios(prob, raw_params)
                decision_regret(prob, mu_surr, mu_prim, predicted_sc, actual_sc)
            end
            push!(epoch_losses, bl)
        end

        avg_loss = mean(epoch_losses)
        push!(loss_history, avg_loss)
        verbose && @info "Epoch $epoch: loss = $avg_loss"

        GC.gc()
    end

    return loss_history
end

"""
    _params_to_scenarios(prob, raw_params) -> Vector{Scenario}

Convert raw model output (vector) to a `Vector{Scenario}`.
For problems with H_ONLY noise, treats each column of a reshaped
output as a scenario parameter vector.
"""
function _params_to_scenarios(prob::ProblemInstance, raw_params::AbstractVector)
    sc_dim = _scenario_param_dim(prob)
    n_sc   = length(raw_params) ÷ sc_dim
    params_mat = reshape(raw_params, sc_dim, n_sc)
    return [scenario_realization(prob, params_mat[:, s]) for s in 1:n_sc]
end

function _params_to_scenarios(prob::ProblemInstance, raw_params::AbstractMatrix)
    # If batch dimension, use first column (single sample)
    return _params_to_scenarios(prob, raw_params[:, 1])
end
