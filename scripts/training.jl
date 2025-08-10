"""
    train!(model, data, loss; opt = Descent(1e-3), epochs = 1)

Stochastic (batch-size = 1) training loop.

* `model` – any callable Flux model
* `data`  – an iterable of `(x, y)` tuples
* `loss`  – your custom loss function `loss(ŷ, y)`
"""
function train!(problem_instance::ResourceAllocationProblem,
                regularization_parameter,
                model, data; opt = Adam(1e-3), epochs = 1,
                display_iterations = false)

    state = Flux.setup(opt, model)         # Optimisers-style state
    for epoch_number in 1:epochs
        display_iterations && println("Epoch: ", epoch_number)
        for (x, ξ) in data                 # batch size = 1
            gs = Flux.gradient(model) do m # grads w.r.t. MODEL, not params
                loss(problem_instance, regularization_parameter, m(x), ξ)
            end
            Flux.update!(state, model, gs[1]) # update with state + model-shaped grads
        end
    end
    return model
end