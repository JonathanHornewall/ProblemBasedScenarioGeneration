function construct_lp(
    sp::StochasticProgram,
    W_eq_array,
    W_ineq_array,
    T_eq_array,
    T_ineq_array,
    h_eq_array,
    h_ineq_array,
    q_array;
    probabilities=nothing,
)
    K = _sp_n_scenarios(W_eq_array, W_ineq_array, T_eq_array, T_ineq_array, h_eq_array, h_ineq_array, q_array)
    first_stage_lp = sp.first_stage_lp
    T = _sp_eltype(
        first_stage_lp.A_eq,
        first_stage_lp.A_ineq,
        first_stage_lp.b_eq,
        first_stage_lp.b_ineq,
        first_stage_lp.c,
        W_eq_array,
        W_ineq_array,
        T_eq_array,
        T_ineq_array,
        h_eq_array,
        h_ineq_array,
        q_array,
    )

    p_vector = if isnothing(probabilities)
        fill(one(T) / K, K)
    else
        length(probabilities) == K ||
            throw(DimensionMismatch("probabilities must have one entry per scenario."))
        probabilities
    end
    T = promote_type(T, eltype(p_vector))

    # The extensive-form variable is v = [z; y_1; ...; y_K].
    nz = size(first_stage_lp.A_eq, 2)
    ny = size(q_array, 1)
    nvars = nz + K * ny

    m1_eq = length(first_stage_lp.b_eq)
    m1_ineq = length(first_stage_lp.b_ineq)
    m2_eq = size(W_eq_array, 1)
    m2_ineq = size(W_ineq_array, 1)

    A_eq = spzeros(T, m1_eq + K * m2_eq, nvars)
    A_ineq = spzeros(T, m1_ineq + K * m2_ineq, nvars)
    b_eq = zeros(T, m1_eq + K * m2_eq)
    b_ineq = zeros(T, m1_ineq + K * m2_ineq)
    c = zeros(T, nvars)

    # First-stage rows: A_eq z = b_eq and A_ineq z <= b_ineq.
    z_cols = 1:nz
    A_eq[1:m1_eq, z_cols] = first_stage_lp.A_eq
    A_ineq[1:m1_ineq, z_cols] = first_stage_lp.A_ineq
    b_eq[1:m1_eq] = first_stage_lp.b_eq
    b_ineq[1:m1_ineq] = first_stage_lp.b_ineq
    c[z_cols] = first_stage_lp.c

    for k in 1:K
        y_cols = (nz + (k - 1) * ny + 1):(nz + k * ny)

        # Scenario-k equality rows: T_eq_array[k] z + W_eq_array[k] y_k = h_eq_array[k].
        eq_rows = (m1_eq + (k - 1) * m2_eq + 1):(m1_eq + k * m2_eq)
        A_eq[eq_rows, z_cols] = view(T_eq_array, :, :, k)
        A_eq[eq_rows, y_cols] = view(W_eq_array, :, :, k)
        b_eq[eq_rows] = view(h_eq_array, :, k)

        # Scenario-k inequality rows: T_ineq_array[k] z + W_ineq_array[k] y_k <= h_ineq_array[k].
        ineq_rows = (m1_ineq + (k - 1) * m2_ineq + 1):(m1_ineq + k * m2_ineq)
        A_ineq[ineq_rows, z_cols] = view(T_ineq_array, :, :, k)
        A_ineq[ineq_rows, y_cols] = view(W_ineq_array, :, :, k)
        b_ineq[ineq_rows] = view(h_ineq_array, :, k)

        # Expected second-stage objective: sum_k p_k q_k' y_k.
        c[y_cols] = p_vector[k] .* view(q_array, :, k)
    end

    return LP(A_eq, A_ineq, b_eq, b_ineq, c)
end

function _sp_n_scenarios(
    W_eq_array,
    W_ineq_array,
    T_eq_array,
    T_ineq_array,
    h_eq_array,
    h_ineq_array,
    q_array,
)
    # Scenario data always uses the last array index for scenarios.
    # Matrices are 3D arrays; vectors are stored as columns of 2D matrices.
    matrix_arrays = (W_eq_array, W_ineq_array, T_eq_array, T_ineq_array)
    vector_arrays = (h_eq_array, h_ineq_array, q_array)
    all(component -> ndims(component) == 3, matrix_arrays) ||
        throw(ArgumentError("W_eq_array, W_ineq_array, T_eq_array, and T_ineq_array must be 3D arrays."))
    all(component -> ndims(component) == 2, vector_arrays) ||
        throw(ArgumentError("h_eq_array, h_ineq_array, and q_array must be matrices with one scenario per column."))

    K = size(W_eq_array, 3)
    all(component -> size(component, 3) == K, matrix_arrays) &&
        all(component -> size(component, 2) == K, vector_arrays) ||
        throw(DimensionMismatch("Scenario components disagree on the number of scenarios."))
    return K
end

function _sp_eltype(values...)
    types = [eltype(value) for value in values]
    return promote_type(types...)
end
