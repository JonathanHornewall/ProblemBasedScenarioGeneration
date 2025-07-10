"""
Struct encoding the data of a two-stage stochastic linear program in extensive form and canonical formulation
"""
struct TwoStageSLP{T<:Real, M<:AbstractMatrix{T}, V<:AbstractVector{T}}
    A1 :: M
    b1 :: V
    c1 :: V
    Ts :: Vector{M}
    Ws :: Vector{M}
    hs :: Vector{V}
    qs :: Vector{V}
end



"""
    recourse_derivative_canLP(s1_decision, coupling_matrix, s2_logbar_lp::LogBarCanLP, solver=standard_solver)
Computes the derivative of one scenario component of the recourse function for a two-stage stochastic log barrier regularized linear program, with respect to the 
first stage decision variable.
"""
function recourse_derivative_canLP(s1_decision, coupling_matrix, s2_logbar_lp::LogBarCanLP, solver=standard_solver)
    return diff_opt_b(s2_logbar_lp; solver=solver) * coupling_matrix * s1_decision
end

function recourse_derivative_canLP(s1_decision, coupling_matrix, s2_constraint_matrix, s2_constraint_vector, s2_cost_vector, regularization_parameter,
    solver=standard_solver)
    # Rename variables for notational convenience
    A = s2_constraint_matrix
    b = s2_constraint_vector - coupling_matrix * s1_decision
    c = s2_cost_vector
    mu = regularization_parameter
    s2_logbar_lp = LogBarCanLP(CanLP(A, b, c), mu)
    return diff_opt_b(s2_logbar_lp; solver=solver) * coupling_matrix
end

"""
    project_to_affine_space(point, matrix, rhs_vector)
Performs a projection on to an affine space defined by a matrix and a right-hand-side(rhs) vector. A simple helper function.
"""
function project_to_affine_space(point, matrix, rhs_vector)
    y = point
    A = matrix
    b = rhs_vector
    return y + A \ (b - A*y)
end


"""
    diff_value_2ScanLP_can(s1_decision, s1_constraint_matrix, s1_constraint_vector, s1_cost_vector, coupling_matrices::Vector{Matrix{P}}, 
    s2_constraint_matrices::Vector{Matrix{P}}, s2_constraint_vectors::Vector{Matrix{P}}, s2_cost_vectors::Vector{Vector{P}}, regularization_parameter,
    solver=standard_solver, project_derivative=false)
Returns the derivative of the value function with respect to the first stage decision, for a two-stage linear program in canonical form with log-barrier regularization
"""
function diff_value_2ScanLP_can(s1_decision, two_slp::TwoStageSLP, regularization_parameter,
    solver=standard_solver, project_derivative=false)
    s1_constraint_matrix = two_slp.A1
    s1_constraint_vector = two_slp.b1
    s1_cost_vector = two_slp.c1
    coupling_matrices = two_slp.Ts
    s2_constraint_matrices = two_slp.Ws 
    s2_constraint_vectors = two_slp.hs
    s2_cost_vectors = two_slp.qs
    if !(length(coupling_matrices) == length(s2_constraint_matrices) == s2_constraint_vectors == s2_cost_vectors)
        error("Number of scenarios inconsistent across problem data")
    end

    S = length(coupling_matrices)  # number of scenarios
    Dx = s1_cost_vector
    for s in 1:S
        Dx += recourse_derivative_canLP(s1_decision, coupling_matrices, s2_constraint_matrices[s], s2_constraint_vectors[s], s2_cost_vectors[s], regularization_parameter,
        solver)
    end
    if project_derivative==true
        Dx = project_to_affine_space(Dx, s1_constraint_matrix, s1_constraint_vector)
    end
    return Dx
end

"""
    extensive_form_2s_can_lp(s1_constraint_matrix, s1_constraint_vector, s1_cost_vector, coupling_matrices::Vector{Matrix{P}}, 
    s2_constraint_matrices::Vector{Matrix{P}}, s2_constraint_vectors::Vector{Matrix{P}}, s2_cost_vectors::Vector{Vector{P}}, regularization_parameter,
    solver=standard_solver, projection=false) where P
Generates a large, extensive form of a two stage stochastic linear program in canonical form, with log barrier regularization.
"""
function extensive_form_2s_can_log_lp(two_slp::TwoStageSLP, regularization_parameter)
    s1_constraint_matrix = two_slp.A1
    s1_constraint_vector = two_slp.b1
    s1_cost_vector = two_slp.c1
    coupling_matrices = two_slp.Ts
    s2_constraint_matrices = two_slp.Ws 
    s2_constraint_vectors = two_slp.hs
    s2_cost_vectors = two_slp.qs
    if !(length(coupling_matrices) == length(s2_constraint_matrices) == s2_constraint_vectors == s2_cost_vectors)
        error("Number of scenarios inconsistent across problem data")
    end

    S = length(coupling_matrices)  # number of scenarios
    n_1 = length(s1_cost_vector)  # dimension of first stage decision
    m_1  = size(s1_constraint_matrix, 1)  # number of first stage constraints
    n_2 = size(s2_cost_vectors[1])  # dimension of second stage decision
    m_2 = size(s2_constraint_matrices[1], 1)  # number of second stage constraints

    c_e = vcat(s1_cost_vector,vcat(s2_cost_vectors...))  # cost vector of extensive form program
    b_e = vcat(s1_constraint_vector, vcat(s2_constraint_vectors))

    # We build the extensive form constraint matrix
    A_e = zeros(m_1, n_1 + S * n_2)  # First row
    A_e[:, 1:n_1] = s1_constraint_matrix
    for s in 1:S
        matrix_row = zeros(m_2, n_1+S*n_2)
        matrix_row[:, 1:n_1] = coupling_matrices[s]
        matrix_row[:, n_1+(s-1)*n_2:n_1+s*n_2] = s2_constraint_matrices[s]
        vcat(A_e, matrix_row)
    end

    extensive_form_LP = LogBarCanLP(CanLP(A_e, b_e, c_e), regularization_parameter)
    return extensive_form_LP
end

"""
    D_xiY(optimal_state, optimal_dual, )
Derivative of optimal first-stage decision with respect to the scenario parameters. Leverages an extensive form formulation of the optimization problem.
"""
function D_xiY(optimal_state, optimal_dual, )
end