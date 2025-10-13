function NNmodel(Nsamples,x,ξ,μᵢⱼ, cz, qw, ρᵢ)   

    in_sample = []
    for i in 1:Nsamples
        push!(in_sample, (x[i,:], ξ[:,i]))
    end

    cz, qw, ρᵢ, = vec(cz), vec(qw), vec(ρᵢ)
    
    data_set_training = Dict(in_sample)

    problem_data = ResourceAllocationProblemData(μᵢⱼ, cz, qw, ρᵢ)
    problem_instance = ResourceAllocationProblem(problem_data)



    model = construct_neural_network(problem_instance; nr_of_scenarios = 3)
    # Train the neural network model
    reg_param_surr = 1.0
    reg_param_prim = 0.0
    reg_param_ref = 0.0
    batchsize = 1
    epochs = 1
    step_size = 1e-3
    save_model_training = true

    state_dir = joinpath("experiment_states", "main")
    mkpath(state_dir)

    # Defining closure for loss function to run generic neural network training with custom functions
    #input_loss(ξ_output, ξ_actual) = loss(problem_instance, reg_param_surr, reg_param_prim, reshape(ξ_output, :, 1), reshape(ξ_actual, :, 1))
    #input_relative_loss(ξ_output, ξ_actual) = relative_loss(problem_instance, reg_param_surr, reg_param_prim, reshape(ξ_output, :, 1), reshape(ξ_actual, :, 1))

    # Defining closure for loss function to run generic neural network training with loss function from ProblemBasedScenarioGeneration.jl
    input_loss(ξ_output, ξ_actual) = loss(problem_instance, reg_param_surr, reg_param_prim, ξ_output, ξ_actual)
    input_relative_loss(ξ_output, ξ_actual) = relative_loss(problem_instance, reg_param_surr, reg_param_prim, ξ_output, ξ_actual)


    # Train with original loss functions
    model_save_path = joinpath(state_dir, "trained_model.jls")
    train!(input_loss, input_relative_loss, model, data_set_training; 
            opt = Flux.Adam(step_size), epochs = epochs, batchsize = batchsize, display_iterations = true, 
            save_model = save_model_training, model_save_path = model_save_path)

    return model



end