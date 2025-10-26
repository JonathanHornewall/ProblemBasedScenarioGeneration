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
    reg_param_ref = 0.0
    batchsize = 1
    epochs = 5
    step_size = 1e-3
    save_model_training = true

    state_dir = joinpath("experiment_states", "main")
    mkpath(state_dir)

    # we keep only the model from the last annealing stage because of overwriting
    model_save_path = joinpath(state_dir, "trained_model_$(Nsamples).jls")

    #param_list = [1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.08, 0.06, 0.04, 0.02, 0.01]
    param_list = [0.01]
    #param_list = [0.01]
    epoch_list = fill(epochs, length(param_list)) # configurable epochs per stage
    #epoch_list[11] = 20

    function run_training_stage(reg_param_surr_stage, reg_param_prim_stage, stage_epochs)
            input_loss(ξ_output, ξ_actual) = loss(problem_instance, reg_param_surr_stage, reg_param_prim_stage, ξ_output, ξ_actual)
            input_relative_loss(ξ_output, ξ_actual) = relative_loss(problem_instance, reg_param_surr_stage, reg_param_prim_stage, ξ_output, ξ_actual)

            train!(input_loss, input_relative_loss, model, data_set_training;
                    opt = Flux.Adam(step_size), epochs = stage_epochs, batchsize = batchsize, display_iterations = true,
                    save_model = save_model_training, model_save_path = model_save_path)

    end

    for (idx, reg_param_surr) in enumerate(param_list)
            stage_epochs = epoch_list[idx]
            if idx == length(param_list)
                    reg_param_prim_stage = 0.0
            else
                    reg_param_prim_stage = reg_param_surr
            end
            println("Starting annealing stage $(idx) with reg_param_surr = $(reg_param_surr), reg_param_prim = $(reg_param_prim_stage), epochs = $(stage_epochs)")
            run_training_stage(reg_param_surr, reg_param_prim_stage, stage_epochs)
    end


    return model



end
