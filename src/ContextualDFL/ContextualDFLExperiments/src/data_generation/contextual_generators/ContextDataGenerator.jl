abstract type ContextDataGenerator end

(generator::ContextDataGenerator)() =
    error("Context data generation is not defined for $(typeof(generator)).")
