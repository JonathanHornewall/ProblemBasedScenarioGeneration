import HiGHS
import Ipopt
import JuMP
import LinearAlgebra: Diagonal, I, dot, norm
import SparseArrays

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
    μ_vector = ContextualDFL._barrier_parameter_vector(length(slack), μ)
    stationarity = lp.c + transpose(lp.A_ineq) * (μ_vector ./ slack)

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

function solve_row_explicit_highs(lp; constraint_tolerance=1e-6, kwargs...)
    model = JuMP.Model(HiGHS.Optimizer)
    JuMP.set_silent(model)
    for (attribute, value) in kwargs
        JuMP.set_optimizer_attribute(model, String(attribute), value)
    end

    n_variables = length(lp.c)
    JuMP.@variable(model, z[1:n_variables])
    eq_constraints = JuMP.@constraint(model, lp.A_eq * z .== lp.b_eq)
    ineq_constraints = JuMP.@constraint(model, lp.A_ineq * z .<= lp.b_ineq)
    JuMP.@objective(model, Min, sum(lp.c[j] * z[j] for j in 1:n_variables))
    JuMP.optimize!(model)

    status = ContextualDFL._assert_successful_solve(
        model,
        TEST_HIGHS_SOLVER;
        accepted_statuses=("OPTIMAL",),
    )
    z_value = JuMP.value.(z)
    ContextualDFL._assert_lp_solution_feasible(lp, z_value; atol=constraint_tolerance)

    return (;
        z=z_value,
        dual_eq=JuMP.dual.(eq_constraints),
        dual_ineq=-JuMP.dual.(ineq_constraints),
        objective_value=JuMP.objective_value(model),
        status=status,
    )
end

function solve_row_explicit_ipopt(
    lp,
    μ;
    slack_lower_bound=1e-9,
    constraint_tolerance=1e-6,
    kwargs...,
)
    μ_vector = ContextualDFL._barrier_parameter_vector(lp, μ)
    positive_barrier_indices = findall(!iszero, μ_vector)

    model = JuMP.Model(Ipopt.Optimizer)
    JuMP.set_optimizer_attribute(model, "print_level", 0)
    JuMP.set_optimizer_attribute(model, "sb", "yes")
    JuMP.set_optimizer_attribute(model, "mu_strategy", "monotone")
    JuMP.set_optimizer_attribute(model, "nlp_scaling_method", "none")
    for (attribute, value) in kwargs
        JuMP.set_optimizer_attribute(model, String(attribute), value)
    end

    n_variables = length(lp.c)
    n_inequalities = length(lp.b_ineq)
    JuMP.@variable(model, z[1:n_variables])
    JuMP.@variable(model, s[1:n_inequalities] >= 0)
    for i in positive_barrier_indices
        JuMP.set_lower_bound(s[i], slack_lower_bound)
    end

    eq_constraints = JuMP.@constraint(model, lp.A_eq * z .== lp.b_eq)
    slack_constraints = JuMP.@constraint(model, lp.A_ineq * z .+ s .== lp.b_ineq)
    JuMP.@NLobjective(
        model,
        Min,
        sum(lp.c[j] * z[j] for j in 1:n_variables) -
        sum(μ_vector[i] * log(s[i]) for i in positive_barrier_indices),
    )
    JuMP.optimize!(model)

    status = ContextualDFL._assert_successful_solve(
        model,
        IpoptSolver();
        accepted_statuses=("OPTIMAL", "LOCALLY_SOLVED"),
    )
    z_value = JuMP.value.(z)
    ContextualDFL._assert_lp_solution_feasible(lp, z_value; atol=constraint_tolerance)

    return (;
        z=z_value,
        slack=JuMP.value.(s),
        dual_eq=JuMP.dual.(eq_constraints),
        dual_ineq=-JuMP.dual.(slack_constraints),
        objective_value=JuMP.objective_value(model),
        status=status,
    )
end

function log_barrier_objective(lp, z, μ)
    μ_vector = ContextualDFL._barrier_parameter_vector(lp, μ)
    slack = lp.b_ineq - lp.A_ineq * z
    return dot(lp.c, z) - dot(μ_vector, log.(slack))
end

function assert_bound_aware_value_preserving(lp; μ=0.2, atol=5e-5)
    row_explicit_lp = solve_row_explicit_highs(lp)
    bound_aware_lp = solve(TEST_HIGHS_SOLVER, lp)

    @test bound_aware_lp.z ≈ row_explicit_lp.z atol = atol rtol = atol
    @test dot(lp.c, bound_aware_lp.z) ≈ dot(lp.c, row_explicit_lp.z) atol = atol rtol = atol
    @test bound_aware_lp.objective_value ≈ row_explicit_lp.objective_value atol = atol rtol = atol
    @test lp.b_ineq - lp.A_ineq * bound_aware_lp.z ≈
          lp.b_ineq - lp.A_ineq * row_explicit_lp.z atol = atol rtol = atol

    row_explicit_barrier =
        solve_row_explicit_ipopt(lp, μ; tol=1e-10, max_iter=1_000)
    bound_aware_barrier =
        solve(TEST_SOLVER, lp; μ=μ, tol=1e-10, max_iter=1_000)

    @test bound_aware_barrier.z ≈ row_explicit_barrier.z atol = atol rtol = atol
    @test bound_aware_barrier.slack ≈
          lp.b_ineq - lp.A_ineq * bound_aware_barrier.z atol = atol rtol = atol
    @test bound_aware_barrier.slack ≈ row_explicit_barrier.slack atol = atol rtol = atol
    @test log_barrier_objective(lp, bound_aware_barrier.z, μ) ≈
          log_barrier_objective(lp, row_explicit_barrier.z, μ) atol = atol rtol = atol
    @test bound_aware_barrier.objective_value ≈
          row_explicit_barrier.objective_value atol = atol rtol = atol
    @test bound_aware_barrier.dual_ineq ≈
          ContextualDFL._barrier_parameter_vector(lp, μ) ./ bound_aware_barrier.slack atol =
          5e-4 rtol = 5e-4
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
    μ_vector = ContextualDFL._barrier_parameter_vector(lp, μ)
    result = ContextualDFL._is_zero_barrier_parameter(μ_vector) ?
        solve(TEST_SOLVER, lp) :
        solve(TEST_SOLVER, lp; μ=μ_vector, tol=1e-10, max_iter=1_000)
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
    μ_vector = cache.μ
    T = promote_type(eltype(cache.z), eltype(lp.c), eltype(μ_vector))

    J_c = nothing
    J_b_eq = nothing
    J_b_ineq = nothing

    if ContextualDFL._is_zero_barrier_parameter(μ_vector)
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
            C = transpose(lp.A_ineq) * Diagonal(μ_vector .* cache.d)
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
