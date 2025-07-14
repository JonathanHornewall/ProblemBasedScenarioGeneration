"""
Struct encoding the data of a two-stage stochastic linear program in extensive form and canonical formulation
"""
struct TwoStageSLP{T<:Real, M<:AbstractMatrix{T}, V<:AbstractVector{T}}
    A1 :: M  # First stage constraint matrix
    b1 :: V  # First stage constraint vector
    c1 :: V  # First stage cost vector
    Ts :: Vector{M}  # Coupling matrix
    Ws :: Vector{M}  # Second stage constraint matrix
    hs :: Vector{V}  # Second stage constraint vector
    qs :: Vector{V}  # Second stage cost vector
end

"""
    recourse_derivative_canLP(coupling_matrix, s2_logbar_lp::LogBarCanLP, solver=standard_solver)
Computes the derivative of one scenario component of the recourse function for a two-stage stochastic log barrier regularized linear program, with respect to the 
first stage decision variable.
"""
function recourse_derivative_canLP(coupling_matrix, s2_logbar_lp::LogBarCanLP, solver=standard_solver)
    return - diff_opt_b(s2_logbar_lp; solver=solver) * coupling_matrix
end

"""
    recourse_derivative_canLP(s1_decision, coupling_matrix, s2_constraint_matrix, s2_constraint_vector, s2_cost_vector, regularization_parameter,
    solver=standard_solver)
Provides the derivative of the second stage recourse function for a two-stage stochastic log-barrier regularized linear program
"""
function recourse_derivative_canLP(s1_decision, coupling_matrix, s2_constraint_matrix, s2_constraint_vector, s2_cost_vector, regularization_parameter,
    solver=standard_solver)
    # Rename variables for notational convenience
    A = s2_constraint_matrix
    b = s2_constraint_vector - coupling_matrix * s1_decision
    c = s2_cost_vector
    mu = regularization_parameter
    s2_logbar_lp = LogBarCanLP(CanLP(A, b, c), mu)
    return recourse_derivative_canLP(coupling_matrix, s2_logbar_lp, solver)
end

"""
    project_to_affine_space(point, matrix, rhs_vector)
Performs a projection on to an affine space defined by a matrix and a right-hand-side(rhs) vector. A simple helper function.
"""
function project_to_affine_space(point, matrix, rhs_vector)
    y = point
    A = matrix
    b = rhs_vector

    At = transpose(A)
    correction = A * ((At * A) \ (At * (A * y - b)))
    return y - correction
end

"""
    diff_value_2ScanLP_can(s1_decision, two_slp::TwoStageSLP, regularization_parameter,
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
    if !(length(coupling_matrices) == length(s2_constraint_matrices) == length(s2_constraint_vectors) == length(s2_cost_vectors))
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
    extensive_form_canonical(two_slp::TwoStageSLP)
Generates a large, extensive form of a two stage stochastic linear program in canonical form, with log barrier regularization.
"""
function extensive_form_canonical(two_slp::TwoStageSLP)
    s1_constraint_matrix = two_slp.A1
    s1_constraint_vector = two_slp.b1
    s1_cost_vector = two_slp.c1
    coupling_matrices = two_slp.Ts
    s2_constraint_matrices = two_slp.Ws 
    s2_constraint_vectors = two_slp.hs
    s2_cost_vectors = two_slp.qs
    if !(length(coupling_matrices) == length(s2_constraint_matrices) == length(s2_constraint_vectors) == length(s2_cost_vectors))
        error("Number of scenarios inconsistent across problem data")
    end

    S = length(coupling_matrices)  # number of scenarios
    n_1 = length(s1_cost_vector)  # dimension of first stage decision
    m_1  = size(s1_constraint_matrix, 1)  # number of first stage constraints
    n_2 = length(s2_cost_vectors[1])  # dimension of second stage decision
    m_2 = size(s2_constraint_matrices[1], 1)  # number of second stage constraints

    c_e = vcat(s1_cost_vector,vcat(s2_cost_vectors...))  # cost vector of extensive form program
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
        constraint_matrix_start_col = n_1 + (s-1) * n_2
        constraint_matrix_end_col = n_1 + s * n_2
        matrix_row[:, constraint_matrix_start_col:constraint_matrix_end_col] = s2_constraint_matrices[s]
        push!(list_of_rows, matrix_row)
    end

    # Build extensive form constraint matrix by combining all the rows
    A_e = vcat(list_of_rows...)

    extensive_form_LP = CanLP(A_e, b_e, c_e)
    return extensive_form_LP
end


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

ScenarioType(params::Symbol...) = ScenarioType{(:W in params), (:T in params), (:H in params), (:Q in params)}()



"""
    D_xiY(two_slp::TwoStageSLP, regularization_parameter, solver=standard_solver)
Derivative of optimal first-stage decision with respect to the scenario parameters. Leverages an extensive form formulation of the optimization problem.
NOTE: This should be rewritten in a way so that we can tune which scenarios are variable and which ones aren't.
"""
function D_xiY(two_slp::TwoStageSLP, regularization_parameter, scenariotype=ScenarioType(:T, :W, :H, :Q), solver=standard_solver)
    # Compute derivatives for extensive form problem
    extensive_prob = extensive_form_canonical(two_slp)
    extensive_prob_regularized = LogBarCanLP(extensive_prob, regularization_parameter)
    D_A, D_b, D_c = diff_opt(extensive_prob_regularized)

    # Recover derivatives with respect to relevant scenarios
    s1_constraint_matrix = two_slp.A1
    s1_constraint_vector = two_slp.b1
    s1_cost_vector = two_slp.c1
    coupling_matrices = two_slp.Ts
    s2_constraint_matrices = two_slp.Ws 
    s2_constraint_vectors = two_slp.hs
    s2_cost_vectors = two_slp.qs

    S = length(coupling_matrices)  # number of scenarios
    n_1 = length(s1_cost_vector)  # dimension of first stage decision
    m_1  = length(s1_constraint_vector)  # number of first stage constraints
    n_2 = length(s2_cost_vectors[1])  # dimension of second stage decision
    m_2 = length(s2_constraint_vectors[1])  # number of second stage constraints
    return_value = ()

    has_W, has_T, has_h, has_q = typeof(scenariotype).parameters

    # Derivative with respect to second stage constraint matrices W
    if has_W==true
    D_Ws = []
    for s in 1:S
        start_index_row = 1 + m_1 + (s-1)*m_2
        end_index_row = m_1 + s*m_2
        start_index_column = 1
        end_index_column = n_1
        D_W = D_A[:, start_index_row:end_index_row, start_index_column:end_index_column]  # Extract D_W from full extensive form derivative
        push!(D_Ws, D_W)
    end
    return_value = (return_value..., D_Ws)
    end

    # Derivative with respect to coupling matrices T
    if has_T==true
    D_Ts = []
    for s in 1:S
        start_index_row = 1 + m_1 + (s-1)*m_2
        end_index_row = m_1 + (s)* m_2
        start_index_column = 1 + n_1 + (s-1) * n_2
        end_index_column = n_1 + s * n_2
        D_T = D_A[:, start_index_row: end_index_row, start_index_column:end_index_column]  # Extract correct derivatives
        push!(D_Ts, D_T)
    return_value = (return_value..., D_Ts)
    end

    if has_h==true
    # Derivative with respect to second stage constraint vector h
    D_hs = []  
    for s in 1:S
        start_index = 1 + m_1 + (s-1)*m_2
        end_index = m_1 + s * m_2
        D_h = D_b[:, start_index:end_index]
        push!(D_hs, D_h)
    end
    return_value = (return_value..., D_hs)
    end

    # Derivative with respect to second stage cost vectors q
    if has_q==true
    D_qs = [] 
    for s in 1:S
        start_index = 1 + n_1 + (s-1)*n_2
        end_index = n_1 + s * n_2
        D_q = D_c[:, start_index:end_index]
        push!(D_qs, D_q)
    end
    return_value = (return_value..., D_qs)
    end
end
    return return_value
end