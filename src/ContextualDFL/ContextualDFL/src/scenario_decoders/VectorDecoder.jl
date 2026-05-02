"""
    VectorDecoder

Abstract decoder for vector-valued model outputs, especially neural net outputs.
Concrete subtypes decode one scenario from an `AbstractVector`; the collection
method below splits a longer vector into `nr_scenarios` equally sized scenario
vectors before assembling the decoded scenario collection.
"""
abstract type VectorDecoder <: ScenarioDecoder end

(decoder::VectorDecoder)(vector::AbstractVector) =
    error("Vector decoding is not defined for $(typeof(decoder)).")

function decode_scenario_collection(
    decoder::VectorDecoder,
    vector_or_collection::AbstractVector;
    nr_scenarios=nothing,
)
    if isnothing(nr_scenarios)
        return _decode_scenario_collection(decoder, vector_or_collection)
    end

    nr_scenarios isa Integer && nr_scenarios > 0 ||
        throw(ArgumentError("nr_scenarios must be a positive integer."))

    L = length(vector_or_collection)
    L % nr_scenarios == 0 ||
        throw(ArgumentError("vector length $L is not divisible by nr_scenarios=$nr_scenarios."))

    scenario_width = L ÷ nr_scenarios
    scenario_matrix = reshape(vector_or_collection, scenario_width, nr_scenarios)
    scenario_parameter_collection = [
        view(scenario_matrix, :, scenario_index)
        for scenario_index in 1:nr_scenarios
    ]

    return decode_scenario_collection(decoder, scenario_parameter_collection)
end
