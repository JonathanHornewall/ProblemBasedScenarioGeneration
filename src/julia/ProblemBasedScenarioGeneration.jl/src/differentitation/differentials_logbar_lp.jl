"""
------------------------------------------------------------------------------------------------
Linear Programming Structs and Basic Functions
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

function isfeasible(instance::CanLP, decision; feasibility_margin = 1e-8)
    !all(decision .>= -feasibility_margin) && println("negative decision: ", decision)
    !all(isapprox.(instance.constraint_matrix * decision, instance.constraint_vector; atol=feasibility_margin)) && println("inequality constraint violation: ", maximum(abs.(instance.constraint_matrix * decision - instance.constraint_vector)))
    return all(isapprox.(instance.constraint_matrix * decision, instance.constraint_vector; atol=feasibility_margin)) && all(decision .>= -feasibility_margin)
end

function isfeasible(instance::LogBarCanLP, decision; feasibility_margin = 1e-8)  
    if iszero(instance.regularization_parameters)
        return isfeasible(instance.linear_program, decision; feasibility_margin = feasibility_margin)
    else
        return all(isapprox.(instance.linear_program.constraint_matrix * decision, instance.linear_program.constraint_vector; atol=feasibility_margin)) && all(decision .> 0)
    end
end

"""
    cost(instance::LogBarCanLP, decision)
cost function for log barrier regularized canonical form problem evaluated at a given decision.
"""
function cost(instance::LogBarCanLP, decision)
    LP = instance.linear_program
    c = LP.cost_vector
    mu = instance.regularization_parameters
    x = decision
    iszero(mu) && return cost(LP, x)
    if !isfeasible(instance, decision) 
        if !all(decision .> 0)
            println("Positivity error")
        elseif !all(isapprox.(instance.linear_program.constraint_matrix * decision, instance.linear_program.constraint_vector; atol=feasibility_margin))
            println("Equality constraint violation")
        end
        error("Decision is not feasible")
    end
    return dot(c, x) - dot(mu, log.(x))
end

function cost(instance::CanLP, decision) 
    !isfeasible(instance, decision) && error("Decision is not feasible")
    return dot(instance.cost_vector, decision) 
end

"""
------------------------------------------------------------------------------------------------
Linear Program Differentiation Functions
------------------------------------------------------------------------------------------------
"""


"""
    diff_KKT_Y(instance::LogBarCanLP, state, dual_state)
Differentiate the l.h.s. of the KKT condition for optimality for a log barrier regularized linear program in canonincal form
with respect to the primal dual variable pair Y = (x, lambda).
"""
function diff_KKT_Y(instance::LogBarCanLP, state)
    A = instance.linear_program.constraint_matrix
    # Rename variables for notational convenience
    x = state
    A = instance.linear_program.constraint_matrix
    mu = instance.regularization_parameters

    # D is the diagonal of log-barrier Hessian
    D = Diagonal(mu ./ (x .^ 2) )
    # KKT matrix: [D  A'; A  0]
    K = Symmetric([D  A'; A  zeros(eltype(D), size(A,1), size(A,1))])
    return K
end


"""
    diff_KKT_A(instance::LogBarCanLP, state, dual_state)
Differentiate the l.h.s. of the KKT condition for optimality for a log barrier regularized linear program in canonincal form
with respect to the constraint matrix.
"""
function diff_KKT_A(instance::LogBarCanLP, state, dual_state)
    n = length(instance.linear_program.cost_vector)
    m = length(instance.linear_program.constraint_vector)
    x = state
    lambda = dual_state
    D_A = zeros(Float64, n+m, m, n)
    for j in 1:m
        for k in 1:n
            D_A[k,j,k] = lambda[j]      # Derivative wrt A in primal block
            D_A[n+j,j,k] = x[k]         # Derivative wrt A in dual block
        end
    end
    return D_A
end

"""
    diff_KKT_b(instance::LogBarCanLP, state, dual_state)
Differentiate the l.h.s. of the KKT condition for optimality for a log barrier regularized linear program in canonincal form
with respect to the constraint vector.
"""
function diff_KKT_b(instance::LogBarCanLP, state, dual_state)
    n = length(instance.linear_program.cost_vector)
    m = length(instance.linear_program.constraint_vector)

    D_b = vcat(zeros(n, m), -I(m))
    D_b = float(D_b)  # Ensure the type is Float64

    return D_b
end
"""
    function diff_KKT_c(instance::LogBarCanLP, state, dual_state)
Differentiate the l.h.s. of the KKT condition for optimality for a log barrier regularized linear program in canonincal form
with respect to the cost vector.
"""
function diff_KKT_c(instance::LogBarCanLP, state, dual_state)
    n = length(instance.linear_program.cost_vector)
    m = length(instance.linear_program.constraint_vector)
    D_c = zeros(Float64, n+m, n)
    for j in 1:n
        D_c[j, j] = 1.0  # Only set the diagonal element, not the entire column
        # D_c[n+1:end, j] remains 0.0 (already initialized)
    end
    return D_c
end

"""
    diff_cache_computation(instance, optimal_state=[], optimal_dual=[], KKT_matrix=[], solver=LogBarCanLP_standard_solver)
Computes the optimal state, optimal dual solution, and a factorization of the KKT matrix. This makes it quick to retrieve the other derivatives.
"""
function diff_cache_computation(instance, optimal_state=[], optimal_dual=[], KKT_matrix=[], solver=LogBarCanLP_standard_solver)
    if optimal_state == []
        optimal_state, optimal_dual = solver(instance)
    end
    if KKT_matrix == []
        KKT_matrix = diff_KKT_Y(instance, optimal_state)
        #KKT_matrix =  bunchkaufman(KKT_matrix)  # Perform factorization
    end
    return optimal_state, optimal_dual, KKT_matrix
end


"""
    diff_opt_A(instance::LogBarCanLP, optimal_state=[], optimal_dual=[], KKT_matrix=[], solver=LogBarCanLP_standard_solver)
Derivative of optimal solution with respect to constraint matrix
"""
function diff_opt_A(instance::LogBarCanLP, optimal_state=[], optimal_dual=[], KKT_matrix=[], solver=LogBarCanLP_standard_solver)
    optimal_state, optimal_dual, KKT_matrix = diff_cache_computation(instance, optimal_state, optimal_dual, KKT_matrix, solver)
    m, n = size(instance.linear_program.constraint_matrix)
    D_A_KKT = diff_KKT_A(instance, optimal_state, optimal_dual)
    D_A_KKT = reshape(D_A_KKT, m + n, :)
    D_A = - KKT_matrix \ D_A_KKT
    D_A = reshape(D_A, m + n, m, n)
    D_A = D_A[1:n, :, :]  # To get the derivative for the optimal solution specifically, ignoring the dual
    return D_A
end

"""
    diff_opt_b(instance::LogBarCanLP, optimal_state=[], optimal_dual=[], KKT_matrix=[], solver=LogBarCanLP_standard_solver)
Derivative of optimal solution with respect to constraint vector
"""
function diff_opt_b(instance::LogBarCanLP, optimal_state=[], optimal_dual=[], KKT_matrix=[]; solver=LogBarCanLP_standard_solver)
    optimal_state, optimal_dual, KKT_matrix = diff_cache_computation(instance, optimal_state, optimal_dual, KKT_matrix, solver)
    n = length(instance.linear_program.cost_vector)
    D_b_KKT = diff_KKT_b(instance, optimal_state, optimal_dual)
    D_b = - (KKT_matrix \ D_b_KKT)
    D_b = D_b[1:n, :]  # To get the derivative for the optimal solution specifically, ignoring the dual
    return D_b
end

"""
    diff_opt_c(instance::LogBarCanLP, optimal_state=[], optimal_dual=[], KKT_matrix=[], solver=LogBarCanLP_standard_solver)
Derivative of optimal solution with respect to cost vector
"""
function diff_opt_c(instance::LogBarCanLP, optimal_state=[], optimal_dual=[], KKT_matrix=[], solver=LogBarCanLP_standard_solver)
    optimal_state, optimal_dual, KKT_matrix = diff_cache_computation(instance, optimal_state, optimal_dual, KKT_matrix, solver)
    n = length(instance.linear_program.cost_vector)
    D_c_KKT = diff_KKT_c(instance, optimal_state, optimal_dual)
    D_c = - KKT_matrix \ D_c_KKT
    D_c = D_c[1:n, :]  # To get the derivative for the optimal solution specifically, ignoring the dual
    return D_c
end

"""
    diff_opt(instance::LogBarCanLP, optimal_state=[], optimal_dual=[], KKT_matrix=[], solver=LogBarCanLP_standard_solver, params=["A", "b", "c"])
Returns a collection consisting of all derivatives.
"""
function diff_opt(instance::LogBarCanLP, optimal_state=[], optimal_dual=[], KKT_matrix=[], solver=LogBarCanLP_standard_solver, params=["A", "b", "c"])
    if !all(x -> x in ["A", "b", "c"], params)
        error("Can not differentiate with respect to parameter ", params)
    end

    n = length(instance.linear_program.cost_vector)
    m = length(instance.linear_program.constraint_vector)
    D_A = zeros(Float64, n, m, n)
    D_b = zeros(Float64, n, m) 
    D_c = zeros(Float64, n, n)

    optimal_state, optimal_dual, KKT_matrix = diff_cache_computation(instance, optimal_state, optimal_dual, KKT_matrix, solver)

    if "A" in params
        D_A = diff_opt_A(instance, optimal_state, optimal_dual, KKT_matrix, solver)
    end
    if "b" in params
        D_b = diff_opt_b(instance, optimal_state, optimal_dual, KKT_matrix, solver)
    end
    if "c" in params
        D_c = diff_opt_c(instance, optimal_state, optimal_dual, KKT_matrix, solver)
    end
    return D_A, D_b, D_c
end