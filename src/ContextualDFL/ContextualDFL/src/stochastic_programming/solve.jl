function solve(
    solver::Solver,
    sp::StochasticProgram,
    W_eq_array,
    W_ineq_array,
    T_eq_array,
    T_ineq_array,
    h_eq_array,
    h_ineq_array,
    q_array;
    probabilities=nothing,
    μ=0,
    kwargs...,
)
    lp = construct_lp(
        sp,
        W_eq_array,
        W_ineq_array,
        T_eq_array,
        T_ineq_array,
        h_eq_array,
        h_ineq_array,
        q_array;
        probabilities=probabilities,
    )
    result = solve(solver, lp; μ=μ, kwargs...)
    return _split_stochastic_solution(sp, result, W_eq_array, W_ineq_array, q_array)
end

function _split_stochastic_solution(sp::StochasticProgram, result, W_eq_array, W_ineq_array, q_array)
    first_stage_lp = sp.first_stage_lp
    K = size(W_eq_array, 3)
    nz = length(first_stage_lp.c)
    ny = size(q_array, 1)

    m1_eq = length(first_stage_lp.b_eq)
    m1_ineq = length(first_stage_lp.b_ineq)
    m2_eq = size(W_eq_array, 1)
    m2_ineq = size(W_ineq_array, 1)

    z = result.z[1:nz]
    y = reshape(result.z[(nz + 1):(nz + K * ny)], ny, K)

    λ_b_eq = result.dual_eq[1:m1_eq]
    λ_b_ineq = result.dual_ineq[1:m1_ineq]
    λ_h_eq_array = reshape(
        result.dual_eq[(m1_eq + 1):(m1_eq + K * m2_eq)],
        m2_eq,
        K,
    )
    λ_h_ineq_array = reshape(
        result.dual_ineq[(m1_ineq + 1):(m1_ineq + K * m2_ineq)],
        m2_ineq,
        K,
    )

    return (
        z,
        y,
        λ_b_eq,
        λ_b_ineq,
        λ_h_eq_array,
        λ_h_ineq_array,
    )
end
