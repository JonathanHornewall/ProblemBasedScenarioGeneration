using ContextualDFL
using Test

@testset "ContextualDFL" begin
    include("linear_programming/runtests.jl")
    include("learning/runtests.jl")
    include("loss_functions/runtests.jl")
    include("scenario_decoders/runtests.jl")
    include("stochastic_programming/runtests.jl")
    include("resource_allocation_training/runtests.jl")
end
