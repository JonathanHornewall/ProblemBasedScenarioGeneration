using Test
using ProblemBasedScenarioGeneration
using LinearAlgebra

@testset "ProblemBasedScenarioGeneration.jl" begin
    include("test_barrier_solver.jl")
    include("test_implicit_diff.jl")
    include("test_problems.jl")
    include("test_decision_regret.jl")
end
