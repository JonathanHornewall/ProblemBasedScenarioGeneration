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
    cross_epoch_losses::Vector{Float64} = []

    for epoch_number in 1:epochs
        display_iterations && print("Epoch ", epoch_number)
        epoch_losses::Vector{Float64} = []
        # Flux.train!(model, data, state) do m, x, ξ
        #     loss(problem_instance, regularization_parameter, m(x), ξ)
        # end
        # The previous Flux.train! call is the same as this :
        state = Flux.setup(opt, model)

        for (x, ξ) in data
            gs = Flux.gradient(model) do m
                loss(problem_instance, regularization_parameter, m(x), ξ)
            end
            # Some versions may return a 1-tuple; unwrap defensively.
            gmodel = gs isa Tuple ? gs[1] : gs
            @show gs  
            Flux.update!(state, model, gmodel)
            
            
            if display_iterations
                δ = loss(problem_instance, regularization_parameter, model(x), ξ)
                # println("Loss is ", δ)
                push!(epoch_losses, δ)
            end
        end

        if display_iterations
            avg_epoch_loss = mean(epoch_losses)
            println(" with avg loss ", avg_epoch_loss, " (", length(epoch_losses), " iterations)")
            push!(cross_epoch_losses, avg_epoch_loss)
        end
    end

    if display_iterations
        plt = plot(
            1:epochs,
            cross_epoch_losses,
            xlabel="Epoch",
            ylabel="Loss",
            title="Training Loss"
        )
        display(plt)  # forces rendering for VS Code
    end
end