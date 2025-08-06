"""
------------------------------------------------------------------------------------------------
Linear Programming functions
------------------------------------------------------------------------------------------------
"""

"""
Abstract type representing a linear program
"""
abstract type LP end

"""
Concrete type representing a linear program in canonical form
This type is used to represent linear programs in the form:
    min c'x
    subject to Ax = b
    x >= 0
where A is the constraint matrix, b is the right-hand side vector, and c is the cost vector.
"""
mutable struct CanLP{R<:Real} <: LP
    constraint_matrix::AbstractMatrix{R}  # Matrix of constraints (A)
    constraint_vector::AbstractVector{R}  # Right-hand side vector (b)
    cost_vector::AbstractVector{R}       # Cost vector (c)
    """
        CanLP(constraint_matrix::A, constraint_vector::B, cost_vector::C) where {A<:AbstractMatrix, B<:AbstractVector, C<:AbstractVector}
    Constructs a canonical linear program with the given constraint matrix, constraint vector, and cost vector.
    """
    function CanLP(constraint_matrix::AbstractMatrix{R}, constraint_vector::AbstractVector{R}, cost_vector::AbstractVector{R}) where {R<:Real}
        size(constraint_matrix, 2) == length(cost_vector) || error("Mismatched cost vector size")
        size(constraint_matrix, 1) == length(constraint_vector) || error("Mismatched constraint vector size")
        new{R}(constraint_matrix, constraint_vector, cost_vector)
    end
end

function CanLP(constraint_matrix::AbstractMatrix{<:Real}, constraint_vector::AbstractVector{<:Real}, cost_vector::AbstractVector{<:Real})
    T = eltype(constraint_matrix)
    if !(eltype(constraint_matrix)== eltype(constraint_vector) == eltype(cost_vector))
        T = promote_type(eltype(constraint_matrix), eltype(constraint_vector), eltype(cost_vector))
    end
    return CanLP(T.(constraint_matrix), T.(constraint_vector), T.(cost_vector))
end

"""
Concrete type representing a log barrier regularized linear program in canonical form
"""
struct LogBarCanLP{T<:Real} <: LP
    linear_program :: CanLP{T}   # the underlying canonical LP
    regularization_parameters :: AbstractVector{T}   # the μ vector (or scalar wrapped in a 1‑vector)
    function LogBarCanLP(linear_program::CanLP{T}, regularization_parameters::AbstractVector{T})  where {T <:Real}
        length(regularization_parameters) == size(linear_program.constraint_matrix, 2) || error("Regularization parameters must match the number of decision variables")
        new{T}(linear_program, regularization_parameters)
    end
end

"""
Constructor for log barrier regularized linear program in canonical form in case the regularization parameter is a scalar
"""
function LogBarCanLP(linear_program::CanLP{R}, regularization_parameter::R) where {R<:Real}
    regularization_parameters = regularization_parameter * ones(size(linear_program.constraint_matrix, 2))
    LogBarCanLP(linear_program, regularization_parameters)
end


"""
    cost(instance::LogBarCanLP, decision)
cost function for log barrier regularized canonical form problem evaluated at a given decision.
"""
function cost(instance::LogBarCanLP, decision)
    LP = instance.linear_program
    A = LP.constraint_matrix
    b = LP.constraint_vector
    c = LP.cost_vector
    mu = instance.regularization_parameters
    x = decision
    @assert Ax == b
    return dot(c, x) -dot(mu, log.(x))
end