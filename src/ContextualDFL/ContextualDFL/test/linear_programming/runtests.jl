@testset "linear_programming" begin
    include("test_helpers.jl")
    include("base_suite/runtests.jl")
    include("extended_suite/runtests.jl")
end
