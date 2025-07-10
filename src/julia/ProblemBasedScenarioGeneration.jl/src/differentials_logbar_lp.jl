"""
Abstract type representing a linear program
"""
abstract type LP end

"""
Concrete type representing a linear program in canonical form
"""
mutable struct CanLP{A<:AbstractMatrix, B<:AbstractVector, C<:AbstractVector} <: LP
    constraint_matrix::A  # Matrix of constraints (A)
    constraint_vector::B  # Right-hand side vector (b)
    cost_vector::C        # Cost vector (c)
end

"""
Constructor for linear program in canonical form
"""
function CanLP(constraint_matrix::A, constraint_vector::B, cost_vector::C) where {A<:AbstractMatrix, B<:AbstractVector, C<:AbstractVector}
    size(constraint_matrix, 2) == length(cost_vector) || error("Mismatched cost vector size")
    size(constraint_matrix, 1) == length(constraint_vector) || error("Mismatched constraint vector size")
    new{A, B, C}(constraint_matrix, constraint_vector, cost_vector)
end

"""
Concrete type representing a log barrier regularized linear program in canonical form
"""
struct LogBarCanLP{U <: Real}
    linear_program::LP              # Underlying linear program
    regularization_parameter::U     # Log-barrier regularization parameter (mu)
end

"""
    diff_KKT_Y(instance::LogBarCanLP, state, dual_state)
Differentiate the l.h.s. of the KKT condition for optimality for a log barrier regularized linear program in canonincal form
with respect to the primal dual variable pair Y = (x, lambda).
"""
function diff_KKT_Y(instance::LogBarCanLP, state)
    # Rename variables for notational convenience
    x = state
    A = instance.linear_program.constraint_matrix
    mu = instance.regularization_parameter

    # D is the diagonal of log-barrier Hessian
    D = [mu/y^2 for y in x]
    # KKT matrix: [D  A'; A  0]
    K = Symmetric([D  A'; A  zeros(eltype(D), size(A,1), size(A,1))])
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
    D_b = zeros(Float64, n+m, m)
    for j in 1:m
        # Fill in the derivative structure for b
        D_b[:,j][zeros(n), -ones(m)]
    end
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
    D_b = zeros(Float64, n+m, n)
    for j in 1:n
        # Fill in the derivative structure for c
        D_b[:,j][ones(n), zeros(m)]
    return D_c
    end
end

"""
    diff_cache_computation(instance, optimal_state=[], optimal_dual=[], KKT_matrix=[], solver=standard_solver)
Computes the optimal state, optimal dual solution, and a factorization of the KKT matrix. This makes it quick to retrieve the other derivatives.
"""
function diff_cache_computation(instance, optimal_state=[], optimal_dual=[], KKT_matrix=[], solver=standard_solver)
    if optimal_state == [] 
            optimal_state, optimal_dual = solve(solver, instance)
    end
    if KKT_matrix == []
        KKT_matrix = diff_KKT_Y(instance, optimal_solution)
        KKT_matrix = ldlt(KKT_matrix)  # Perform factorization   
    end
    return optimal_state, optimal_dual, KKT_matrix
end


"""
    diff_opt_A(instance::LogBarCanLP, optimal_state=[], optimal_dual=[], KKT_matrix::AbstractMatrix=[], solver=standard_solver)
Derivative of optimal solution with respect to constraint matrix
"""
function diff_opt_A(instance::LogBarCanLP, optimal_state=[], optimal_dual=[], KKT_matrix::AbstractMatrix=[], solver=standard_solver)
    optimal_state, optimal_dual, KKT_matrix = diff_cache_computation(instance, optimal_solution, KKT_matrix, solver)
    n = size(instance.linear_program.constraint_matrix, 2)
    n = size(instance.linear_program.constraint_matrix, 1)
    D_A_KKT = diff_KKT_A(instance, optimal_state, optimal_dual)
    D_A_KKT=reshape(A, n+m, :)
    D_A = - KKT_matrix \ D_A_KKT
    D_A=reshape(D_A, n+m, m, n)
    D_A = D_A[1:n, :]  # To get the derivative for the optimal solution specifically, ignoring the dual
    return D_A
end

"""
    diff_opt_b(instance::LogBarCanLP, optimal_state=[], optimal_dual=[], KKT_matrix=[], solver=standard_solver)
Derivative of optimal solution with respect to constraint vector
"""
function diff_opt_b(instance::LogBarCanLP, optimal_state=[], optimal_dual=[], KKT_matrix=[], solver=standard_solver)
    optimal_state, optimal_dual, KKT_matrix = diff_cache_computation(instance, optimal_state, optimal_dual, KKT_matrix, solver)
    D_b_KKT = diff_KKT_b(instance, optimal_state, optimal_dual)
    D_b = - KKT_matrix \ D_b_KKT
    D_b = D_b[1:n, :]  # To get the derivative for the optimal solution specifically, ignoring the dual
    return D_b
end

"""
    diff_opt_c(instance::LogBarCanLP, optimal_state=[], optimal_dual=[], KKT_matrix=[], solver=standard_solver)
Derivative of optimal solution with respect to cost vector
"""
function diff_opt_c(instance::LogBarCanLP, optimal_state=[], optimal_dual=[], KKT_matrix=[], solver=standard_solver)
    optimal_state, optimal_dual, KKT_matrix = diff_cache_computation(instance, optimal_state, optimal_dual, KKT_matrix, solver)
    D_c_KKT = diff_KKT_c(instance, optimal_state, optimal_dual)
    D_c = - KKT_matrix \ D_c_KKT
    D_c = D_c[1:n, :]  # To get the derivative for the optimal solution specifically, ignoring the dual 
    return D_c
end

"""
    diff_opt(instance::LogBarCanLP, optimal_state=[], optimal_dual=[], KKT_matrix=[], solver=standard_solver, params=["A", "b", "c"])
Returns a collection consisting of all derivatives.
"""
function diff_opt(instance::LogBarCanLP, optimal_state=[], optimal_dual=[], KKT_matrix=[], solver=standard_solver, params=["A", "b", "c"])
    if !all(x -> x in ["A", "b", "c"], params)
        error("Can not differentiate with respect to parameter ", params)
    end

    optimal_solution, KKT_matrix = diff_cache_computation(instance, optimal_state, optimal_dual, KKT_matrix, solver)
    D_A = zeros(Float64, n, n, m)
    D_b = zeros(Float64, m, n)
    D_c = zeros(Float64, n, n)
        
    if "A" in params
        D_A = diff_opt_A(instance::LogBarCanLP, optimal_solution, KKT_matrix)
    end

    if " b" in params
        D_b = diff_opt_b(instance::LogBarCanLP, optimal_state, optimal_dual, KKT_matrix)
    end
    if "c" in params
        D_c = diff_opt_c(instance::LogBarCanLP, optimal_state, optimal_dual, KKT_matrix)
        end
    return D_A, D_b, D_c
    end