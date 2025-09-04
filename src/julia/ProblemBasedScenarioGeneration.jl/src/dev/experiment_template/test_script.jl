"""
    benchmark_model(problem_instance, traning_data, step_size, batch_size, epoch_nr, reg_param_surr, reg_param_prim=0, reg_param_test=0)
Trains a neural network surrogate model on the provided training data and evaluates its performance. Stores the result in a table.
Records everything of interest about the set up.
"""
function benchmark_model(problem_instance::ResourceAllocationProblem, training_data, step_size, batch_size, epoch_nr, reg_param_surr, save_file_path; reg_param_prim=0, reg_param_test=0)
    error("not yet implemented")
end