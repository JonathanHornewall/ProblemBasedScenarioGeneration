"""
    train!(model, data, loss; opt = Descent(1e-3), epochs = 1)

Stochastic (batch-size = 1) training loop.

* `model` – any callable Flux model
* `data`  – an iterable of `(x, y)` tuples
* `loss`  – your custom loss function `loss(ŷ, y)`
"""
function train!(problem_instance::ResourceAllocationProblem, regularization_parameter, model, data; opt = Adam(1e-3), epochs = 1)
    ps = params(model)                           # model parameters
    for _ in 1:epochs
        for (x, ξ) in data                       # batch size = 1
            gs = gradient(ps) do
                loss(problem_instance, regularization_parameter, model(x), ξ)
            end
            Flux.Optimise.update!(opt, ps, gs)   # backward + step
        end
    end
    return model
end