module Models

export BaselineModels, NeuralModelArtifacts

"""
    BaselineModels

Bundle for the fitted non-neural baselines.
"""
struct BaselineModels
    ls
    er_saa
    cart
    knn
    nelder_mead
end

"""
    NeuralModelArtifacts

Bookkeeping struct for the neural network training outputs.
"""
struct NeuralModelArtifacts
    model_path::String
    history_path::String
    metadata::Dict{String,Any}
end

end # module
