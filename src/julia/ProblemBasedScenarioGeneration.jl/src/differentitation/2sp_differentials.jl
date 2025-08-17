"""
---------------------------------------------------------------------------------------------
Basic functionalities for two-stage stochastic linear programs
---------------------------------------------------------------------------------------------
"""

"""
Struct encoding the data of a two-stage stochastic linear program in extensive form and canonical formulation
"""
struct TwoStageSLP{T<:Real,                # the numeric scalar type
                M<:AbstractMatrix{T},   # matrix whose entries are T
                V<:AbstractVector{T}}   # vector whose entries are T
    A1 :: M           # first–stage constraint matrix
    b1 :: V           # first–stage RHS
    c1 :: V           # first–stage cost
    Ws :: Vector{M}   # second–stage constraint matrices
    Ts :: Vector{M}   # coupling matrices
    hs :: Vector{V}   # second–stage RHS vectors
    qs :: Vector{V}   # second–stage cost vectors
    ps :: Vector{T}   # scenario probabilities
end


"""
    TwoStageSLP(A_1, b1, c1, Ws, Ts, hs, qs, ps)
Constructor for TwoStageSLP
"""
function TwoStageSLP(A_1, b1, c1, Ws, Ts, hs, qs, ps = nothing)
    if isnothing(ps)
        ps = ones(length(Ws)) ./ length(Ws)  # Default to equiprobable scenarios
    end
    @assert length(Ws) == length(Ts) == length(hs) == length(qs) == length(ps)
    n_1 = size(A_1, 2)  # Number of first stage decision variables
    m_1 = size(A_1, 1)  # Number of first stage constraints
    n_2 = size(Ws[1], 2)  # Number of second stage decision variables
    m_2 = size(Ws[1], 1)  # Number of second stage constraints
    @assert length(Ws) == length(Ts) == length(hs) == length(qs) == length(ps)  # Number of scenarios must be consistent across all parameters
    @assert all(size(Ws[s]) == (m_2, n_2) for s in eachindex(Ws))  # Each second stage constraint matrix must have the same size
    @assert all(size(Ts[s]) == (m_2, n_1) for s in eachindex(Ts))  # Each coupling matrix must have the same size
    @assert all(length(hs[s]) == m_2 for s in eachindex(hs))  # Each second stage constraint vector must have the same size
    @assert all(length(qs[s]) == n_2 for s in eachindex(qs))  # Each second stage cost vector must have the same size
    @assert all(p -> isa(p, Real) && p > 0, ps)  # Each scenario probability must be a positive real number
    @assert sum(ps) ≈ 1.0  # Scenario probabilities must sum to 1.0

    return TwoStageSLP{eltype(A_1), typeof(A_1), typeof(b1)}(A_1, b1, c1, Ws, Ts, hs, qs, ps)
end

"""
    extensive_form_canonical(two_slp::TwoStageSLP)
Generates an extensive form of two stage stochastic linear program in canonical form, with log barrier regularization.
"""
function extensive_form_canonical(two_slp::TwoStageSLP)
    s1_constraint_matrix = two_slp.A1
    s1_constraint_vector = two_slp.b1
    s1_cost_vector = two_slp.c1
    coupling_matrices = two_slp.Ts
    s2_constraint_matrices = two_slp.Ws 
    s2_constraint_vectors = two_slp.hs
    s2_cost_vectors = two_slp.qs
    s2_probability_vector = two_slp.ps
    if !(length(coupling_matrices) == length(s2_constraint_matrices) == length(s2_constraint_vectors) == length(s2_cost_vectors))
        error("Number of scenarios inconsistent across problem data")
    end

    S = length(coupling_matrices)  # number of scenarios
    n_1 = length(s1_cost_vector)  # dimension of first stage decision
    m_1  = size(s1_constraint_matrix, 1)  # number of first stage constraints
    n_2 = length(s2_cost_vectors[1])  # dimension of second stage decision
    m_2 = size(s2_constraint_matrices[1], 1)  # number of second stage constraints

    probability_adjusted_s2_cost_vectors = [s2_probability_vector[s] * s2_cost_vectors[s] for s in 1:S]
    c_e = vcat(s1_cost_vector,vcat(probability_adjusted_s2_cost_vectors...))  # cost vector of extensive form program
    b_e = vcat(s1_constraint_vector, vcat(s2_constraint_vectors...))

    # We build the extensive form constraint matrix

    list_of_rows = []
    # We implement the first stage constraint matrix

    matrix_row = zeros(m_1, n_1 + S * n_2)  # First row
    matrix_row[:, 1:n_1] = s1_constraint_matrix
    push!(list_of_rows, matrix_row)

    # We implement the second stage stage constraint matrix
    for s in 1:S
        # Coupling matrix
        matrix_row = zeros(m_2, n_1+S*n_2)
        matrix_row[:, 1:n_1] = coupling_matrices[s]
        # Constraint matrix
        constraint_matrix_start_col = n_1 + (s-1) * n_2 + 1
        constraint_matrix_end_col = n_1 + s * n_2
        matrix_row[:, constraint_matrix_start_col:constraint_matrix_end_col] = s2_constraint_matrices[s]
        push!(list_of_rows, matrix_row)
    end

    # Build extensive form constraint matrix by combining all the rows
    A_e = vcat(list_of_rows...)

    # Account for possibility of lack of constraints in first-stage decision
    if iszero(A_e[1,:])
        A_e = A_e[2:end, :]  # Remove the first row if it is all zeros
        b_e = b_e[2:end]  # Remove the first element of b_e if it is all zeros
    end


    extensive_form_LP = CanLP(A_e, b_e, c_e)
    return extensive_form_LP
end

function LogBarCanLP(two_slp::TwoStageSLP, regularization_parameter::Real)
    """
    Constructor for log barrier regularized version of a two-stage stochastic linear program in canonical form.
    """
    lp = extensive_form_canonical(two_slp)
    n_1 = length(two_slp.c1)  # Number of first stage decision variables
    n_2 = length(two_slp.qs[1])  # Number of second stage decision variables
    S = length(two_slp.Ts)  # Number of scenarios
    p = two_slp.ps  # Scenario probabilities
    regularization_parameters = regularization_parameter * ones(n_1)
    for s in 1:S
        regularization_parameters = vcat(regularization_parameters, regularization_parameter * p[s] * ones(n_2))
    end
    return LogBarCanLP(lp, regularization_parameters)
end

"""
---------------------------------------------------------------------------------------------
Differentiation functionalities for cost function
---------------------------------------------------------------------------------------------
"""


"""
    cost_2s_LogBarCanLP(two_slp::TwoStageSLP, s1_decision, regularization_parameters, solver=LogBarCanLP_standard_solver, project_derivative=false)
Gives the cost function of a two-stage stochastic linear program with respect to the first-stage decision.
"""
function cost_2s_LogBarCanLP(two_slp::TwoStageSLP, s1_decision, regularization_parameter,
    solver=LogBarCanLP_standard_solver, project_derivative=false)
    s1_constraint_matrix = two_slp.A1
    s1_constraint_vector = two_slp.b1
    s1_cost_vector = two_slp.c1
    coupling_matrices = two_slp.Ts
    s2_constraint_matrices = two_slp.Ws 
    s2_constraint_vectors = two_slp.hs
    s2_cost_vectors = two_slp.qs
    s2_probability_vector = two_slp.ps
    S = length(coupling_matrices)  # Represents the number of scenarios

    s1_lp = CanLP(s1_constraint_matrix, s1_constraint_vector, s1_cost_vector)
    s1_reg_lp = LogBarCanLP(s1_lp, regularization_parameter)
    final_cost = cost(s1_reg_lp, s1_decision)
    for s in 1:S
        constraint_matrix = s2_constraint_matrices[s]
        constraint_vector = s2_constraint_vectors[s] - coupling_matrices[s] * s1_decision
        cost_vector = s2_cost_vectors[s] * s2_probability_vector[s]
        s2_lp = CanLP(constraint_matrix, constraint_vector, cost_vector)
        s2_reg_lp = LogBarCanLP(s2_lp, regularization_parameter * s2_probability_vector[s])
        
        # Solve the second-stage problem to find optimal second-stage decision given fixed first-stage decision
        optimal_s2_decision, _ = solver(s2_reg_lp)
        
        # Evaluate the cost at the optimal second-stage decision
        final_cost += cost(s2_reg_lp, optimal_s2_decision)
    end
    return final_cost
end

"""
    recourse_derivative_canLP(coupling_matrix, s2_logbar_lp::LogBarCanLP, solver=LogBarCanLP_standard_solver)
Computes the derivative of one scenario component of the recourse function for a two-stage stochastic log barrier regularized linear program, with respect to the 
first stage decision variable.
"""
function recourse_derivative_canLP(coupling_matrix, s2_logbar_lp::LogBarCanLP, s2_probability, solver=LogBarCanLP_standard_solver)
    optimal_solution, optimal_dual = solver(s2_logbar_lp)
    return - s2_probability * coupling_matrix' * optimal_dual  # The dual variable is the derivative of the recourse function with respect to the first stage decision
end

"""
    recourse_derivative_canLP(s1_decision, coupling_matrix, s2_constraint_matrix, s2_constraint_vector, s2_cost_vector, regularization_parameter,
    solver=LogBarCanLP_standard_solver)
Computes the derivative of one scenario component of the recourse function for a two-stage stochastic log barrier regularized linear program, with respect to the 
first stage decision variable.
"""
function recourse_derivative_canLP(s1_decision, coupling_matrix, s2_constraint_matrix, s2_constraint_vector, s2_cost_vector, s2_probability, regularization_parameter,
    solver=LogBarCanLP_standard_solver)
    # Rename variables for notational convenience
    A = s2_constraint_matrix
    b = s2_constraint_vector - coupling_matrix * s1_decision
    c = s2_cost_vector
    mu = regularization_parameter
    s2_logbar_lp = LogBarCanLP(CanLP(A, b, c), mu)
    return recourse_derivative_canLP(coupling_matrix, s2_logbar_lp, s2_probability, solver)
end

"""
    project_to_affine_space(point, matrix, rhs_vector)
Performs a projection on to an affine space defined by a matrix and a right-hand-side(rhs) vector. A helper function for diff_cost_2s_LogBarCanLP.
"""
function project_to_affine_space(point::AbstractVector, matrix::AbstractMatrix, rhs_vector::AbstractVector)
    y = point
    A = matrix
    b = rhs_vector
    # If A has no rows or rank 0 → no constraints → projection is y
    if isempty(A) || rank(A) == 0
        return y
    end
    r = A*y - b
    λ = pinv(A*A') * r   # works for any rank
    return y - A' * λ
end


"""
    diff_cost_2s_LogBarCanLP(s1_decision, two_slp::TwoStageSLP, regularization_parameter,
    solver=LogBarCanLP_standard_solver, project_derivative=false)
Returns the derivative of the cost function with respect to the first stage decision, for a two-stage linear program in canonical form with log-barrier regularization
"""
function diff_cost_2s_LogBarCanLP(two_slp::TwoStageSLP, regularization_parameter, s1_decision,
    solver=LogBarCanLP_standard_solver, project_derivative=true)
    s1_constraint_matrix = two_slp.A1
    s1_constraint_vector = two_slp.b1
    s1_cost_vector = two_slp.c1
    coupling_matrices = two_slp.Ts
    s2_constraint_matrices = two_slp.Ws 
    s2_constraint_vectors = two_slp.hs
    s2_cost_vectors = two_slp.qs
    s2_probability_vector = two_slp.ps
    @assert (length(coupling_matrices) == length(s2_constraint_matrices) == length(s2_constraint_vectors) == length(s2_cost_vectors))

    S = length(coupling_matrices)  # number of scenarios
    
    Dx = s1_cost_vector .- regularization_parameter ./ s1_decision  # Initialize the derivative with respect to the first stage decision
    for s in 1:S
        Dx += recourse_derivative_canLP(s1_decision, coupling_matrices[s], s2_constraint_matrices[s], s2_constraint_vectors[s], s2_cost_vectors[s], 
        s2_probability_vector[s], regularization_parameter, solver)
    end
    if project_derivative==true
        Dx = project_to_affine_space(Dx, s1_constraint_matrix, s1_constraint_vector)
    end
    return Dx
end

"""
---------------------------------------------------------------------------------------------
Differentiation functionalities for scenario parameters
---------------------------------------------------------------------------------------------
"""

"""
    ScenarioType{W<:Bool, T<:Bool, H<:Bool,Q<:Bool} end   # each parameter is boolean

Encodes, at *compile time*, which of the four
parameters T, W, H, Q vary between different scenarios.

Example
-------
julia> Flags(:T, :H)
Flags{true, false, true, false}()
"""
struct ScenarioType{W<:Bool, T<:Bool, H<:Bool,Q<:Bool} end   # each parameter is boolean

"""
Constructor for ScenarioType    
"""
ScenarioType(params::Symbol...) = ScenarioType{(:W in params), (:T in params), (:H in params), (:Q in params)}()

"""
    D_xiY(two_slp::TwoStageSLP, regularization_parameter, solver=LogBarCanLP_standard_solver)
Derivative of optimal first-stage decision with respect to the scenario parameters. Leverages an extensive form formulation of the optimization problem.
NOTE: This should be rewritten in a way so that we can tune which scenarios are variable and which ones aren't.
"""
function D_xiY(two_slp::TwoStageSLP, regularization_parameter, scenariotype=ScenarioType(:T, :W, :H, :Q), solver=LogBarCanLP_standard_solver)

    # Renaming for notational convenience
    s1_constraint_matrix = two_slp.A1
    s1_constraint_vector = two_slp.b1
    s1_cost_vector = two_slp.c1
    coupling_matrices = two_slp.Ts
    s2_constraint_matrices = two_slp.Ws 
    s2_constraint_vectors = two_slp.hs
    s2_cost_vectors = two_slp.qs
    s2_probability_vector = two_slp.ps
    S = length(coupling_matrices)  # Represents the number of scenarios

    # Compute derivatives of extensive form lp
    extensive_prob = extensive_form_canonical(two_slp)
    regularization_parameters = regularization_parameter * ones(length(extensive_prob.c))
    for s in S
        regularization_parameters = vcat(regularization_parameters, regularization_parameter * s2_probability_vector[s] * ones(length(s2_cost_vectors[s])))
    end
    extensive_prob_regularized = LogBarCanLP(extensive_prob, regularization_parameters)
    D_A, D_b, D_c = diff_opt(extensive_prob_regularized)

    S = length(coupling_matrices)  # number of scenarios
    n_1 = length(s1_cost_vector)  # dimension of first stage decision
    m_1  = length(s1_constraint_vector)  # number of first stage constraints
    n_2 = length(s2_cost_vectors[1])  # dimension of second stage decision
    m_2 = length(s2_constraint_vectors[1])  # number of second stage constraints

    has_W, has_T, has_h, has_q = typeof(scenariotype).parameters

    D_Ws = []; D_Ts = []; D_hs = []; D_qs=[]

    # Derivative with respect to second stage constraint matrices W
    if has_W
        for s in 1:S
            start_index_row = 1 + m_1 + (s-1)*m_2
            end_index_row = m_1 + s*m_2
            start_index_column = 1
            end_index_column = n_1
            D_W = D_A[:, start_index_row:end_index_row, start_index_column:end_index_column]  # Extract D_W from full extensive form derivative
            push!(D_Ws, D_W)
        end
    end

    # Derivative with respect to coupling matrices T
    if has_T==true
    for s in 1:S
        start_index_row = 1 + m_1 + (s-1)*m_2
        end_index_row = m_1 + (s)* m_2
        start_index_column = 1 + n_1 + (s-1) * n_2
        end_index_column = n_1 + s * n_2
        D_T = D_A[:, start_index_row: end_index_row, start_index_column:end_index_column]  # Extract correct derivatives
        push!(D_Ts, D_T)
    end

    if has_h==true
    # Derivative with respect to second stage constraint vector h
        for s in 1:S
            start_index = 1 + m_1 + (s-1)*m_2
            end_index = m_1 + s * m_2
            D_h = D_b[:, start_index:end_index]
            push!(D_hs, D_h)
        end
    end
    end

    # Derivative with respect to second stage cost vectors q
    if has_q==true
    for s in 1:S
        start_index = 1 + n_1 + (s-1)*n_2
        end_index = n_1 + s * n_2
        D_q = D_c[:, start_index:end_index]
        push!(D_qs, D_q)
    end
    end
    return D_Ws, D_Ts, D_hs, D_qs
    end