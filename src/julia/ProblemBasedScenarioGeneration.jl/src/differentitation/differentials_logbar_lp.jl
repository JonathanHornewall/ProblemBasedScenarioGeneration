"""
    diff_KKT_Y(instance::LogBarCanLP, state, dual_state)
Differentiate the l.h.s. of the KKT condition for optimality for a log barrier regularized linear program in canonincal form
with respect to the primal dual variable pair Y = (x, lambda).
"""
function diff_KKT_Y(instance::LogBarCanLP, state)
    # Rename variables for notational convenience
    x = state
    A = instance.linear_program.constraint_matrix
    mu = instance.regularization_parameters

    # D is the diagonal of log-barrier Hessian
    D = [mu[i]/x[i]^2 for i in eachindex(x)]
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
    D_b = zeros(Float64, n+m, m)
    for j in 1:m
        D_b[1:n, j] .= 0.0
        D_b[n+1:end, j] .= 1.0
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
    D_c = zeros(Float64, n+m, n)
    for j in 1:n
        D_c[1:n, j] .= 1.0
        D_c[n+1:end, j] .= 0.0
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
        KKT_matrix = ldlt(KKT_matrix)  # Perform factorization
    end
    return optimal_state, optimal_dual, KKT_matrix
end


"""
    diff_opt_A(instance::LogBarCanLP, optimal_state=[], optimal_dual=[], KKT_matrix::AbstractMatrix=[], solver=LogBarCanLP_standard_solver)
Derivative of optimal solution with respect to constraint matrix
"""
function diff_opt_A(instance::LogBarCanLP; optimal_state=[], optimal_dual=[], KKT_matrix::AbstractMatrix=[], solver=LogBarCanLP_standard_solver)
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
function diff_opt_b(instance::LogBarCanLP, optimal_state=[], optimal_dual=[], KKT_matrix=[], solver=LogBarCanLP_standard_solver)
    optimal_state, optimal_dual, KKT_matrix = diff_cache_computation(instance, optimal_state, optimal_dual, KKT_matrix, solver)
    n = length(instance.linear_program.cost_vector)
    D_b_KKT = diff_KKT_b(instance, optimal_state, optimal_dual)
    D_b = - KKT_matrix \ D_b_KKT
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