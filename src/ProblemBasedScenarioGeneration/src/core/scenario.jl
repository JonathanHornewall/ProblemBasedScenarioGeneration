"""
    Scenario{T<:Real}

Data for a single second-stage scenario in a two-stage stochastic LP.

Fields:
- `W`: (m₂ × n₂) second-stage constraint matrix
- `T`: (m₂ × n₁) coupling matrix (linking first-stage to second-stage)
- `h`: (m₂,) second-stage right-hand side
- `q`: (n₂,) second-stage cost vector
"""
struct Scenario{T<:Real}
    W::Matrix{T}   # (m₂, n₂) second-stage constraint matrix
    T::Matrix{T}   # (m₂, n₁) coupling matrix
    h::Vector{T}   # (m₂,) second-stage RHS
    q::Vector{T}   # (n₂,) second-stage cost

    function Scenario(W::Matrix{T}, T_mat::Matrix{T}, h::Vector{T}, q::Vector{T}) where {T<:Real}
        m2, n2 = size(W)
        m2 == size(T_mat, 1) || error("W and T must have the same number of rows (m₂)")
        m2 == length(h) || error("W rows must match h length (m₂)")
        n2 == length(q) || error("W columns must match q length (n₂)")
        new{T}(W, T_mat, h, q)
    end
end

"""
    scenario_from_tuple(W, T, h, q) -> Scenario

Convenience constructor for `Scenario` from individual arrays.
Promotes all arrays to a common element type.
"""
function scenario_from_tuple(W, T_mat, h, q)
    ET = promote_type(eltype(W), eltype(T_mat), eltype(h), eltype(q))
    return Scenario(ET.(W), ET.(T_mat), ET.(h), ET.(q))
end

Base.show(io::IO, s::Scenario{T}) where {T} =
    print(io, "Scenario{$T}(m₂=$(size(s.W,1)), n₂=$(size(s.W,2)), n₁=$(size(s.T,2)))")
