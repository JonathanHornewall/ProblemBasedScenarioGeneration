"""
    TwoStageSLP{T<:Real}

Two-stage stochastic linear program. Stores first-stage data plus a vector
of second-stage scenarios with associated probabilities.

Fields:
- `A`: (m₁ × n₁) first-stage constraint matrix
- `b`: (m₁,) first-stage RHS
- `c`: (n₁,) first-stage cost vector
- `scenarios`: Vector of `Scenario{T}`, length S
- `p`: (S,) probability vector, sums to 1
"""
struct TwoStageSLP{T<:Real}
    A::Matrix{T}
    b::Vector{T}
    c::Vector{T}
    scenarios::Vector{Scenario{T}}
    p::Vector{T}

    function TwoStageSLP(
        A::Matrix{T}, b::Vector{T}, c::Vector{T},
        scenarios::Vector{Scenario{T}}, p::Vector{T}
    ) where {T<:Real}
        S = length(scenarios)
        S == length(p) || error("Number of scenarios must equal length of probability vector")
        isempty(scenarios) || begin
            n1 = size(A, 2)
            for (s, sc) in enumerate(scenarios)
                size(sc.T, 2) == n1 ||
                    error("Scenario $s coupling matrix T has wrong number of columns (expected n₁=$n1)")
            end
        end
        abs(sum(p) - 1) < 1e-10 || error("Probabilities must sum to 1, got $(sum(p))")
        all(p .> 0) || error("All probabilities must be positive")
        new{T}(A, b, c, scenarios, p)
    end
end

"""
    TwoStageSLP(A, b, c, scenarios, p)

Outer constructor that promotes all inputs to a common element type.
"""
function TwoStageSLP(A, b, c, scenarios::Vector{<:Scenario}, p)
    T = promote_type(eltype(A), eltype(b), eltype(c), eltype(p))
    for sc in scenarios
        T = promote_type(T, eltype(sc.W), eltype(sc.T), eltype(sc.h), eltype(sc.q))
    end
    scenarios_T = [Scenario(T.(sc.W), T.(sc.T), T.(sc.h), T.(sc.q)) for sc in scenarios]
    return TwoStageSLP(T.(A), T.(b), T.(c), scenarios_T, T.(p))
end

"""
    TwoStageSLP(A, b, c, scenarios)

Constructs a TwoStageSLP with equiprobable scenarios.
"""
function TwoStageSLP(A, b, c, scenarios::Vector{<:Scenario})
    S = length(scenarios)
    p = fill(1.0 / S, S)
    return TwoStageSLP(A, b, c, scenarios, p)
end

Base.show(io::IO, slp::TwoStageSLP{T}) where {T} =
    print(io, "TwoStageSLP{$T}(n₁=$(length(slp.c)), S=$(length(slp.scenarios)))")
