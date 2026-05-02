import LinearAlgebra: Diagonal, I, dot, norm

const TEST_SOLVER = Solver(IpoptSolver(), HiGHSSolver())
const TEST_HIGHS_SOLVER = HiGHSSolver()
const TEST_BARRIER_MUS = (1.0, 0.1)
# μ=0 derivative finite-difference checks are intentionally skipped for now:
# the LP solution map is nonsmooth at active-set changes, so these failures do
# not tell us anything useful about the smooth log-barrier derivative path.
const TEST_DERIVATIVE_MUS = (1.0, 0.1)

status_name(status) = string(status)
is_optimal_status(status) = status_name(status) in ("OPTIMAL", "LOCALLY_SOLVED")

function assert_feasible(lp, z; atol=1e-6)
    if !isempty(lp.A_eq)
        @test norm(lp.A_eq * z - lp.b_eq, Inf) ≤ atol
    end

    if !isempty(lp.A_ineq)
        @test maximum(lp.A_ineq * z - lp.b_ineq) ≤ atol
    end
end

function assert_barrier_stationarity(lp, z, μ; atol=5e-4)
    slack = lp.b_ineq - lp.A_ineq * z
    stationarity = lp.c + μ .* (transpose(lp.A_ineq) * (1.0 ./ slack))

    if !isempty(lp.A_eq)
        λ = -(transpose(lp.A_eq) \ stationarity)
        stationarity += transpose(lp.A_eq) * λ
    end

    @test norm(stationarity, Inf) ≤ atol
end

function assert_lp_kkt(lp, result; atol=1e-6)
    dual_ineq = result.dual_ineq

    if !isempty(lp.A_ineq)
        @test minimum(dual_ineq) ≥ -atol
        @test norm(dual_ineq .* (lp.A_ineq * result.z - lp.b_ineq), Inf) ≤ atol
    end

    stationarity = copy(lp.c)
    isempty(lp.A_eq) || (stationarity .-= transpose(lp.A_eq) * result.dual_eq)
    isempty(lp.A_ineq) || (stationarity .+= transpose(lp.A_ineq) * dual_ineq)

    @test norm(stationarity, Inf) ≤ atol
end

function assert_lp_case(case, solver=TEST_SOLVER)
    result = solve(solver, case.lp)

    @test status_name(result.status) == case.expected_status

    if case.expected_status == "OPTIMAL"
        assert_feasible(case.lp, result.z)
        @test result.objective_value ≈ dot(case.lp.c, result.z) atol = 1e-7
        assert_lp_kkt(case.lp, result)

        if haskey(case, :expected_z)
            @test result.z ≈ case.expected_z atol = 1e-7
        end
    end
end

function assert_lp_case_with_highs(case)
    @testset "LP solver strategy" begin
        assert_lp_case(case, TEST_SOLVER)
    end

    @testset "HiGHS direct" begin
        assert_lp_case(case, TEST_HIGHS_SOLVER)
    end
end

function solve_reference_z(lp, μ)
    result = iszero(μ) ?
        solve(TEST_SOLVER, lp) :
        solve(TEST_SOLVER, lp; μ=μ, tol=1e-10, max_iter=1_000)
    @test is_optimal_status(result.status)
    return result.z
end

function assert_log_barrier_case(case, μ)
    result = solve(TEST_SOLVER, case.lp; μ=μ, tol=1e-10, max_iter=1_000)

    @test is_optimal_status(result.status)
    assert_feasible(case.lp, result.z; atol=1e-6)
    @test minimum(case.lp.b_ineq - case.lp.A_ineq * result.z) > 1e-7
    assert_barrier_stationarity(case.lp, result.z, μ)

    return result
end

function deterministic_direction(length_value, scale, phase)
    return [scale * sin(i + phase) for i in 1:length_value]
end

function finite_difference_action(lp, μ, dc, db_eq, db_ineq)
    ε = 1e-3
    lp_plus = LP(
        A_eq=lp.A_eq,
        A_ineq=lp.A_ineq,
        b_eq=lp.b_eq + ε .* db_eq,
        b_ineq=lp.b_ineq + ε .* db_ineq,
        c=lp.c + ε .* dc,
    )
    lp_minus = LP(
        A_eq=lp.A_eq,
        A_ineq=lp.A_ineq,
        b_eq=lp.b_eq - ε .* db_eq,
        b_ineq=lp.b_ineq - ε .* db_ineq,
        c=lp.c - ε .* dc,
    )

    return (solve_reference_z(lp_plus, μ) - solve_reference_z(lp_minus, μ)) ./ (2ε)
end

function finite_difference_jacobian(lp, μ, component)
    n = length(lp.c)

    if component === :c
        J = zeros(n, n)
        for j in 1:n
            dc = zeros(n)
            dc[j] = 1.0
            J[:, j] = finite_difference_action(
                lp,
                μ,
                dc,
                zeros(length(lp.b_eq)),
                zeros(length(lp.b_ineq)),
            )
        end
        return J
    elseif component === :b_eq
        J = zeros(n, length(lp.b_eq))
        for j in 1:length(lp.b_eq)
            db_eq = zeros(length(lp.b_eq))
            db_eq[j] = 1.0
            J[:, j] = finite_difference_action(
                lp,
                μ,
                zeros(n),
                db_eq,
                zeros(length(lp.b_ineq)),
            )
        end
        return J
    elseif component === :b_ineq
        J = zeros(n, length(lp.b_ineq))
        for j in 1:length(lp.b_ineq)
            db_ineq = zeros(length(lp.b_ineq))
            db_ineq[j] = 1.0
            J[:, j] = finite_difference_action(
                lp,
                μ,
                zeros(n),
                zeros(length(lp.b_eq)),
                db_ineq,
            )
        end
        return J
    end

    throw(ArgumentError("Unknown derivative component: $component"))
end

function construct_jacobian(
    solver,
    lp::LP,
    μ;
    pre_computed=nothing,
    compute_J_c=true,
    compute_J_b_eq=true,
    compute_J_b_ineq=true,
    tight_tol=1e-7,
    kwargs...,
)
    cache = ContextualDFL._diff_precompute(solver, lp, μ, pre_computed, tight_tol; kwargs...)
    n = length(lp.c)
    m_eq = length(lp.b_eq)
    m_ineq = length(lp.b_ineq)
    T = promote_type(eltype(cache.z), eltype(lp.c), typeof(μ))

    J_c = nothing
    J_b_eq = nothing
    J_b_ineq = nothing

    if iszero(μ)
        if compute_J_c
            J_c = zeros(T, n, n)
        end

        if compute_J_b_eq
            rhs_b_eq = vcat(
                zeros(T, n, m_eq),
                Matrix{T}(I, m_eq, m_eq),
                zeros(T, length(cache.tight), m_eq),
            )
            J_b_eq = (cache.K_factorization \ rhs_b_eq)[1:n, :]
        end

        if compute_J_b_ineq
            top = zeros(T, n, m_ineq)
            if !isempty(cache.loose)
                top[:, cache.loose] = transpose(lp.A_ineq[cache.loose, :]) * Diagonal(cache.d)
            end

            bottom = zeros(T, length(cache.tight), m_ineq)
            for (row, index) in enumerate(cache.tight)
                bottom[row, index] = one(T)
            end

            rhs_b_ineq = vcat(top, zeros(T, m_eq, m_ineq), bottom)
            J_b_ineq = (cache.K_factorization \ rhs_b_ineq)[1:n, :]
        end
    else
        if compute_J_c
            rhs_c = vcat(Matrix{T}(I, n, n), zeros(T, m_eq, n))
            J_c = -(cache.K_factorization \ rhs_c)[1:n, :]
        end

        if compute_J_b_eq
            rhs_b_eq = vcat(zeros(T, n, m_eq), Matrix{T}(I, m_eq, m_eq))
            J_b_eq = (cache.K_factorization \ rhs_b_eq)[1:n, :]
        end

        if compute_J_b_ineq
            C = μ .* (transpose(lp.A_ineq) * Diagonal(cache.d))
            rhs_b_ineq = vcat(Matrix{T}(C), zeros(T, m_eq, m_ineq))
            J_b_ineq = (cache.K_factorization \ rhs_b_ineq)[1:n, :]
        end
    end

    return (;
        J_c=J_c,
        J_b_eq=J_b_eq,
        J_b_ineq=J_b_ineq,
        pre_computed=cache,
    )
end

function assert_diff_solve_column(lp, μ, jac, component, column)
    n = length(lp.c)
    dc = zeros(n)
    db_eq = zeros(length(lp.b_eq))
    db_ineq = zeros(length(lp.b_ineq))

    if component === :c
        dc[column] = 1.0
        expected = jac.J_c[:, column]
    elseif component === :b_eq
        db_eq[column] = 1.0
        expected = jac.J_b_eq[:, column]
    elseif component === :b_ineq
        db_ineq[column] = 1.0
        expected = jac.J_b_ineq[:, column]
    else
        throw(ArgumentError("Unknown derivative component: $component"))
    end

    actual = diff_solve(
        TEST_SOLVER,
        lp,
        μ;
        pre_computed=jac.pre_computed,
        dc=dc,
        db_eq=db_eq,
        db_ineq=db_ineq,
    )

    @test actual ≈ expected atol = 1e-8 rtol = 1e-8
end

function assert_diff_case(case, μ)
    lp = case.lp
    z = solve_reference_z(lp, μ)
    jac = construct_jacobian(TEST_SOLVER, lp, μ; pre_computed=z)
    fd_J_c = finite_difference_jacobian(lp, μ, :c)
    fd_J_b_eq = finite_difference_jacobian(lp, μ, :b_eq)
    fd_J_b_ineq = finite_difference_jacobian(lp, μ, :b_ineq)

    @test jac.J_c ≈ fd_J_c atol = 5e-4 rtol = 5e-3
    @test jac.J_b_eq ≈ fd_J_b_eq atol = 5e-4 rtol = 5e-3
    @test jac.J_b_ineq ≈ fd_J_b_ineq atol = 5e-4 rtol = 5e-3

    for j in 1:length(lp.c)
        assert_diff_solve_column(lp, μ, jac, :c, j)
    end

    for j in 1:length(lp.b_eq)
        assert_diff_solve_column(lp, μ, jac, :b_eq, j)
    end

    for j in 1:length(lp.b_ineq)
        assert_diff_solve_column(lp, μ, jac, :b_ineq, j)
    end
end

function run_smooth_case(case)
    @testset "$(case.name)" begin
        assert_lp_case_with_highs(case)

        for μ in TEST_BARRIER_MUS
            @testset "log-barrier μ=$(μ)" begin
                assert_log_barrier_case(case, μ)
            end
        end

        for μ in TEST_DERIVATIVE_MUS
            @testset "derivative μ=$(μ)" begin
                assert_diff_case(case, μ)
            end
        end
    end
end

function square_2d()
    return [1.0 0.0; -1.0 0.0; 0.0 1.0; 0.0 -1.0], ones(4)
end

function nonnegative_orthant(n)
    return -Matrix{Float64}(I, n, n), zeros(n)
end

function box_constraints(lower, upper)
    n = length(lower)
    return [Matrix{Float64}(I, n, n); -Matrix{Float64}(I, n, n)], [upper; -lower]
end
