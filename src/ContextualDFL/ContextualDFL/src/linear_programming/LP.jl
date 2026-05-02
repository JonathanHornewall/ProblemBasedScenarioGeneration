const _MaybeMatrix = Union{Nothing,AbstractMatrix}
const _MaybeVector = Union{Nothing,AbstractVector}

struct LP{
    TAeq<:AbstractMatrix,
    TAineq<:AbstractMatrix,
    Tbeq<:AbstractVector,
    Tbineq<:AbstractVector,
    Tc<:AbstractVector,
}
    A_eq::TAeq
    A_ineq::TAineq
    b_eq::Tbeq
    b_ineq::Tbineq
    c::Tc

    function LP(
        A_eq::TAeq,
        A_ineq::TAineq,
        b_eq::Tbeq,
        b_ineq::Tbineq,
        c::Tc,
    ) where {
        TAeq<:AbstractMatrix,
        TAineq<:AbstractMatrix,
        Tbeq<:AbstractVector,
        Tbineq<:AbstractVector,
        Tc<:AbstractVector,
    }
        _validate_lp_dimensions(A_eq, A_ineq, b_eq, b_ineq, c)
        return new{TAeq,TAineq,Tbeq,Tbineq,Tc}(A_eq, A_ineq, b_eq, b_ineq, c)
    end
end

function LP(
    A_eq::_MaybeMatrix,
    A_ineq::_MaybeMatrix,
    b_eq::_MaybeVector,
    b_ineq::_MaybeVector,
    c::_MaybeVector,
)
    return LP(_canonical_lp_data(A_eq, A_ineq, b_eq, b_ineq, c)...)
end

LP(; A_eq=nothing, A_ineq=nothing, b_eq=nothing, b_ineq=nothing, c=nothing) =
    LP(A_eq, A_ineq, b_eq, b_ineq, c)

function _canonical_lp_data(A_eq, A_ineq, b_eq, b_ineq, c)
    n_variables = _infer_variable_count(A_eq, A_ineq, c)
    T = _infer_lp_eltype(A_eq, A_ineq, b_eq, b_ineq, c)

    A_eq, b_eq = _canonical_constraint_pair(:A_eq, :b_eq, A_eq, b_eq, n_variables, T)
    A_ineq, b_ineq =
        _canonical_constraint_pair(:A_ineq, :b_ineq, A_ineq, b_ineq, n_variables, T)
    c = isnothing(c) ? zeros(T, n_variables) : c

    return A_eq, A_ineq, b_eq, b_ineq, c
end

function _infer_variable_count(A_eq, A_ineq, c)
    counts = Int[]
    isnothing(A_eq) || push!(counts, size(A_eq, 2))
    isnothing(A_ineq) || push!(counts, size(A_ineq, 2))
    isnothing(c) || push!(counts, length(c))

    isempty(counts) && return 0

    n_variables = first(counts)
    all(==(n_variables), counts) ||
        throw(DimensionMismatch("LP inputs disagree on the number of variables."))

    return n_variables
end

function _infer_lp_eltype(values...)
    types = Type[]

    for value in values
        if !isnothing(value) && (!isempty(value) || eltype(value) !== Any)
            push!(types, eltype(value))
        end
    end

    return isempty(types) ? Float64 : promote_type(types...)
end

function _canonical_constraint_pair(A_name, b_name, A, b, n_variables, T)
    if isnothing(A)
        if isnothing(b) || isempty(b)
            return Matrix{T}(undef, 0, n_variables), Vector{T}(undef, 0)
        end

        throw(ArgumentError("$(A_name) must be provided when $(b_name) has entries."))
    end

    if isnothing(b)
        return A, zeros(T, size(A, 1))
    end

    return A, b
end

function _validate_lp_dimensions(A_eq, A_ineq, b_eq, b_ineq, c)
    n_variables = length(c)

    size(A_eq, 2) == n_variables ||
        throw(DimensionMismatch("A_eq must have length(c) columns."))
    size(A_ineq, 2) == n_variables ||
        throw(DimensionMismatch("A_ineq must have length(c) columns."))
    size(A_eq, 1) == length(b_eq) ||
        throw(DimensionMismatch("A_eq and b_eq must have matching row counts."))
    size(A_ineq, 1) == length(b_ineq) ||
        throw(DimensionMismatch("A_ineq and b_ineq must have matching row counts."))

    return nothing
end
