using Flux, Statistics, Plots, Serialization

"""
    train!(model, data, loss; opt = Descent(1e-3), epochs = 1)

Stochastic (batch-size = 1) training loop.

* `model` – any callable Flux model
* `data  – an iterable of `(x, y)` tuples
* `loss`  – your custom loss function `loss(ŷ, y)`
"""
function train!(problem_instance,
                reg_param_surr, reg_param_prim,
                model, data; opt = Adam(1e-3), epochs = 1,
                display_iterations = false, save_model = false, 
                model_save_path = "trained_model.jld2")

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
                loss(problem_instance, reg_param_surr, reg_param_prim, m(x), ξ)
            end
            # Some versions may return a 1-tuple; unwrap defensively.
            gmodel = gs isa Tuple ? gs[1] : gs
            Flux.update!(state, model, gmodel)
            
            
            if display_iterations
                δ = loss(problem_instance, reg_param_surr, reg_param_prim, model(x), ξ)
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

"""
    save_trained_model(model, filepath)

Save a trained Flux model to a file using Julia's built-in Serialization.
"""
function save_trained_model(model, filepath)
    serialize(filepath, model)
end

"""
    load_trained_model(filepath)

Load a trained Flux model to a file using Julia's built-in Serialization.
"""
function load_trained_model(filepath)
    return deserialize(filepath)
end

"""
    save_training_data(training_data, testing_data, filepath)

Save training and testing datasets to a file using Julia's built-in Serialization.
"""
function save_training_data(training_data, testing_data, filepath)
    serialize(filepath, (training_data, testing_data))
end

"""
    load_training_data(filepath)

Load training and testing datasets from a file using Julia's built-in Serialization.
"""
function load_training_data(filepath)
    training_data, testing_data = deserialize(filepath)
    return training_data, testing_data
end

"""
    save_experiment_state(model, training_data, testing_data, problem_instance, 
                         reg_params, filepath)

Save the complete experiment state including model, data, and parameters.
"""
function save_experiment_state(model, training_data, testing_data, problem_instance, 
                             reg_params; filepath = "experiment_state.jls")
    
    # Extract key parameters from problem instance
    problem_data = Dict(
        "s1_constraint_matrix" => problem_instance.s1_constraint_matrix,
        "s1_constraint_vector" => problem_instance.s1_constraint_vector,
        "s1_cost_vector" => problem_instance.s1_cost_vector
    )
    
    # Save everything
    serialize(filepath, (model, training_data, testing_data, problem_data, reg_params))
    
    println("Complete experiment state saved to: $filepath")
end

"""
    load_experiment_state(filepath)

Load the complete experiment state from a file.
"""
function load_experiment_state(filepath)
    
    model, training_data, testing_data, problem_data, reg_params = deserialize(filepath)
    
    # Reconstruct problem instance
    problem_instance = ResourceAllocationProblem(ResourceAllocationProblemData(
        problem_data["s1_constraint_matrix"], 
        problem_data["s1_constraint_vector"], 
        problem_data["s1_cost_vector"]
    ))
    
    return model, training_data, testing_data, problem_instance, reg_params
end