using ContextualDFLExperiments
using Test

@testset "ContextualDFLExperiments" begin
    script = joinpath(@__DIR__, "..", "scripts", "train_resource_allocation_refactored.jl")
    include(script)
    value = main(n_samples=2, epochs=1, seed=7)
    @test isfinite(value)
end
