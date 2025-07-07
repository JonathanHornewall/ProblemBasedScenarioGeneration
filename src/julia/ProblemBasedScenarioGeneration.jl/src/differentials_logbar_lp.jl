abstract type LP end

mutable struct CanLP{A<:AbstractMatrix, B<:AbstractVector, C<:AbstractVector} <: LP
    constraint_matrix::A
    constraint_vector::B
    cost_vector::C
end

function CanLP(constraint_matrix::A, constraint_vector::B, cost_vector::C) where {A<:AbstractMatrix, B<:AbstractVector, C<:AbstractVector}
    size(constraint_matrix, 2) == length(cost_vector) || error("Mismatched cost vector size")
    size(constraint_matrix, 1) == length(constraint_vector) || error("Mismatched constraint vector size")
    new{A, B, C}(constraint_matrix, constraint_vector, cost_vector)
end

struct LogBarCanLP{U <: Real}
    linear_program::LP
    regularization_parameter::U
end

function diff_KKT_Y(instance::LogBarCanLP, state)
    # Rename variables for notational convenience
    x = state
    A = instance.linear_program.constraint_matrix
    mu = instance.regularization_parameter

    D = [mu/y^2 for y in x]
    K = Symmetric([D  A'; A  zeros(eltype(D), size(A,1), size(A,1))])
end

function diff_KKT_A(instance::LogBarCanLP, state, dual_state)
    n = length(instance.linear_program.cost_vector)
    m = length(instance.linear_program.constraint_vector)
    x = state
    lambda = dual_state
    D_A = zeros(Float64, n+m, m, n)
    for j in 1:m
        for k in 1:n
            D_A[k,j,k] = lambda[j]
            D_A[n+j,j,k] = x[k]
        end
    return D_A
end

function diff_KKT_b(instance::LogBarCanLP, state, dual_state)
    n = length(instance.linear_program.cost_vector)
    m = length(instance.linear_program.constraint_vector)
    D_b = zeros(Float64, n+m, m)
    for j in 1:m
        D_b[:,j][zeros(n), -ones(m)]
    end
    return D_b
end

function diff_KKT_c(instance::LogBarCanLP, state, dual_state)
    n = length(instance.linear_program.cost_vector)
    m = length(instance.linear_program.constraint_vector)
    D_b = zeros(Float64, n+m, n)
    for j in 1:n
        D_b[:,j][ones(n), zeros(m)]
    return D_c
    end
end

function diff_cache_computation(instance, optimal_state=[], optimal_dual=[], KKT_matrix=[], solver=standard_solver)
    if optimal_state == [] 
            optimal_state = solve(solver, instance)
            # Here you add the logic for computing the optimal dual
    if KKT_matrix == []
        KKT_matrix = diff_KKT_Y(instance, optimal_solution)
        KKT_matrix = ldlt(KKT_matrix)  # Perform factorization   
    end
    return optimal_state, optimal_dual, KKT_matrix
end

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
end

function diff_opt_b(instance::LogBarCanLP, optimal_state=[], optimal_dual=[], KKT_matrix=[], solver=standard_solver)
    optimal_state, optimal_dual, KKT_matrix = diff_cache_computation(instance, optimal_state, optimal_dual, KKT_matrix, solver)
    D_b_KKT = diff_KKT_b(instance, optimal_state, optimal_dual)
    D_b = - KKT_matrix \ D_b_KKT
    D_b = D_b[1:n, :]  # To get the derivative for the optimal solution specifically, ignoring the dual
    return D_b
end

function diff_opt_c(instance::LogBarCanLP, optimal_state=[], optimal_dual=[], KKT_matrix=[], solver=standard_solver)
    optimal_state, optimal_dual, KKT_matrix = diff_cache_computation(instance, optimal_state, optimal_dual, KKT_matrix, solver)
    D_c_KKT = diff_KKT_c(instance, optimal_state, optimal_dual)
    D_c = - KKT_matrix \ D_c_KKT
    D_c = D_c[1:n, :]  # To get the derivative for the optimal solution specifically, ignoring the dual 
    return D_c
end

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
    end
    return D_A, D_b, D_c
end