# Training functions for neural network models
# This file focuses on training logic and calls load_experiment.jl for data persistence

using Flux, Statistics, Plots, Serialization

# Include data persistence functionality
include("load_experiment.jl")

"""
    train!(model, data, loss; opt = Descent(1e-3), epochs = 1)

Stochastic (batch-size = 1) training loop.

* `model` – any callable Flux model
* `data  – an iterable of `(x, y)` tuples
* `loss`  – your custom loss function `loss(ŷ, y)`
"""

function train!(problem_instance,
                reg_param_surr, reg_param_prim,
                model, data; opt = Adam(1e-3), epochs = 1, batchsize = 1,
                display_iterations = false, save_model = false, 
                model_save_path = "trained_model.jld2")

    state = Flux.setup(opt, model)         # Optimisers-style state
    cross_epoch_losses::Vector{Float64} = []

    # Set up batch data
    xs  = collect(keys(data))
    xis = collect(values(data))
    X  = hcat(xs...)
    Ξ  = hcat(xis...)
    loader = DataLoader((X, Ξ), batchsize)
    # batched loss functions
    loss_mb(problem_instance, reg_param_surr, reg_param_prim, model, Xb, Ξb) = 
    mean(loss(problem_instance, reg_param_surr, reg_param_prim, vec(model(Xb[:, i])), vec(Ξb[:, i])) for i in 1:size(Xb, 2))
    # Relative loss 
    relative_loss_mb(problem_instance, reg_param_surr, reg_param_prim, model, Xb, Ξb) = 
    mean(relative_loss(problem_instance, reg_param_surr, reg_param_prim, vec(model(Xb[:, i])), vec(Ξb[:, i])) for i in 1:size(Xb, 2))
    
    for epoch_number in 1:epochs
        display_iterations && print("Epoch ", epoch_number)
        epoch_losses::Vector{Float64} = []
        # Flux.train!(model, data, state) do m, x, ξ
        #     loss(problem_instance, regularization_parameter, m(x), ξ)
        # end
        # The previous Flux.train! call is the same as this :
        state = Flux.setup(opt, model)

       # for (x, ξ) in data
           #= gs = Flux.gradient(model) do m
                loss(problem_instance, reg_param_surr, reg_param_prim, m(x), ξ)
            end
            # Some versions may return a 1-tuple; unwrap defensively.
            gmodel = gs isa Tuple ? gs[1] : gs
            Flux.update!(state, model, gmodel)
            =#
        for (x, ξ) in loader
            gs = Flux.gradient(model) do m
                loss_mb(problem_instance, reg_param_surr, reg_param_prim, m, x, ξ)
            end
            gmodel = gs isa Tuple ? gs[1] : gs
            Flux.update!(state, model, gmodel)
        
            
            if display_iterations
                δ = relative_loss_mb(problem_instance, reg_param_surr, reg_param_prim, model, x, ξ)
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

    # Save the trained model if requested
    if save_model
        save_trained_model(model, model_save_path)
        println("Model saved to: $model_save_path")
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


function train!(loss, relative_loss, model, data; opt = Adam(1e-3), epochs = 1, batchsize = 1,
    display_iterations = false, save_model = false, 
    model_save_path = "trained_model.jld2")

state = Flux.setup(opt, model)         # Optimisers-style state
cross_epoch_losses::Vector{Float64} = []

# Set up batch data
xs  = collect(keys(data))
xis = collect(values(data))
X  = hcat(xs...)
Ξ  = hcat(xis...)
loader = DataLoader((X, Ξ), batchsize)
# batched loss functions
loss_mb(model, Xb, Ξb) = 
    mean( loss( vec(model(Xb[:, i])), vec(Ξb[:, i]) ) for i in 1:size(Xb, 2) )

# Relative loss 
relative_loss_mb(model, Xb, Ξb) = 
    mean(relative_loss(vec(model(Xb[:, i])), vec(Ξb[:, i])) for i in 1:size(Xb, 2))

for epoch_number in 1:epochs
display_iterations && print("Epoch ", epoch_number)
epoch_losses::Vector{Float64} = []
# Flux.train!(model, data, state) do m, x, ξ
#     loss(problem_instance, regularization_parameter, m(x), ξ)
# end
# The previous Flux.train! call is the same as this :
state = Flux.setup(opt, model)

# for (x, ξ) in data
#= gs = Flux.gradient(model) do m
    loss(problem_instance, reg_param_surr, reg_param_prim, m(x), ξ)
end
# Some versions may return a 1-tuple; unwrap defensively.
gmodel = gs isa Tuple ? gs[1] : gs
Flux.update!(state, model, gmodel)
=#
for (x, ξ) in loader
gs = Flux.gradient(model) do m
    loss_mb(m, x, ξ)
end
gmodel = gs isa Tuple ? gs[1] : gs
Flux.update!(state, model, gmodel)


if display_iterations
    δ = relative_loss_mb(model, x, ξ)
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

# Save the trained model if requested
if save_model
save_trained_model(model, model_save_path)
println("Model saved to: $model_save_path")
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