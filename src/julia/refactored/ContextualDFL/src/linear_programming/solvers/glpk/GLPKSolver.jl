using JuMP
using GLPK

struct GLPKSolver{C} <: Solver
    config::C
end

GLPKSolver(mu::Real) = GLPKSolver((mu=mu, tol=1e-8, max_iter=100))
GLPKSolver(; mu=0.0, tol=1e-8, max_iter=100) = GLPKSolver((mu=mu, tol=tol, max_iter=max_iter))

#=
struct GLPKImplementation{T} <: LPImplementation
    implementation::T
end
=#

function implement(solver::GLPKSolver, lp::LP, config=nothing)
    return canonical_form(lp)
end

function solve(solver::GLPKSolver, lp::LP, config=nothing; kwargs...)
    A, b, c, n_original = canonical_form(
        lp;
        A_eq=get(kwargs, :A_eq, nothing),
        b=get(kwargs, :b, nothing),
        c=get(kwargs, :c, nothing),
    )
    mu = _solver_value(solver, config, :mu, 0.0)
    tol = _solver_value(solver, config, :tol, 1e-8)
    max_iter = Int(_solver_value(solver, config, :max_iter, 100))

    if _positive_mu(mu)
        cache = _solve_barrier(A, b, c, mu; tol=tol, max_iter=max_iter)
        x_can = cache.x
        lambda = cache.lambda
        cache_out = cache
    else
        x_can, lambda = _solve_glpk_canonical(A, b, c; tol=tol)
        cache_out = nothing
    end

    x_original = x_can[1:n_original]
    return (
        primal=x_original,
        canonical_primal=x_can,
        dual=lambda,
        cache=cache_out,
        objective_value=dot(c[1:n_original], x_original),
        canonical_objective_value=dot(c, x_can),
        canonical=(A=A, b=b, c=c, n_original=n_original),
        lp=lp,
    )
end

function _positive_mu(mu)
    mu isa Number && return mu > 0
    return any(mu .> 0)
end

function _mu_vector(mu, n::Integer, ::Type{T}) where {T}
    if mu isa Number
        return fill(T(mu), n)
    end
    length(mu) == n || error("Barrier parameter has length $(length(mu)) but LP has $n variables")
    return T.(vec(mu))
end

function _solve_glpk_canonical(A, b, c; tol::Real=1e-8)
    m, n = size(A)
    length(b) == m || error("Canonical RHS length $(length(b)) does not match $m rows")
    length(c) == n || error("Canonical objective length $(length(c)) does not match $n columns")

    model = Model(GLPK.Optimizer)
    set_silent(model)
    @variable(model, x[1:n] >= 0)
    con = if m == 0
        nothing
    else
        @constraint(model, A * x .== b)
    end
    @objective(model, Min, dot(c, x))
    optimize!(model)

    status = termination_status(model)
    if status != OPTIMAL
        raw = try
            raw_status(model)
        catch
            string(status)
        end
        error("GLPK failed to solve canonical LP: $status ($raw)")
    end

    x_opt = value.(x)
    lambda = con === nothing ? zeros(eltype(x_opt), 0) : dual.(con)
    if m > 0
        violation = maximum(abs.(A * x_opt .- b))
        violation <= max(tol, 1e-7) || error("GLPK solution violates Ax=b by $violation")
    end
    return x_opt, lambda
end

function _barrier_initial_point(A::Matrix{T}, b::Vector{T}) where {T}
    m, n = size(A)
    m == 0 && return ones(T, n)

    eps = T(1e-7)
    model = Model(GLPK.Optimizer)
    set_silent(model)
    @variable(model, x[1:n] >= eps)
    @constraint(model, A * x .== b)
    @objective(model, Min, sum(x))
    optimize!(model)

    if termination_status(model) == OPTIMAL
        return max.(T.(value.(x)), eps)
    end

    x0 = pinv(A) * b
    min_val = minimum(x0)
    if min_val <= eps
        x0 = x0 .+ abs(min_val) .+ one(T)
    end
    correction = pinv(A) * (A * x0 - b)
    return max.(x0 - correction, eps)
end

function _solve_barrier(A, b, c, mu; tol::Real=1e-8, max_iter::Int=100)
    T = promote_type(eltype(A), eltype(b), eltype(c), Float64)
    A = Matrix{T}(A)
    b = T.(vec(b))
    c = T.(vec(c))
    m, n = size(A)
    mus = _mu_vector(mu, n, T)

    if m == 0
        positive_c = max.(c, T(1e-8))
        x = max.(mus ./ positive_c, T(1e-8))
        return BarrierCache(x, zeros(T, 0), mus, A, b, c)
    end

    x = _barrier_initial_point(A, b)
    lambda = zeros(T, m)
    for _ in 1:max_iter
        r_dual = c - mus ./ x + A' * lambda
        r_primal = A * x - b
        residual = sqrt(sum(abs2, r_dual) + sum(abs2, r_primal))
        residual <= tol && break

        D = mus ./ (x .^ 2)
        K = vcat(
            hcat(Diagonal(D), A'),
            hcat(A, zeros(T, m, m)),
        )
        rhs = -vcat(r_dual, r_primal)
        step = try
            K \ rhs
        catch
            (K + T(1e-10) * I) \ rhs
        end
        dx = step[1:n]
        dlambda = step[(n + 1):end]

        alpha = one(T)
        neg = dx .< 0
        if any(neg)
            alpha = min(alpha, T(0.99) * minimum(-x[neg] ./ dx[neg]))
        end
        while any(x .+ alpha .* dx .<= 0)
            alpha *= T(0.5)
            alpha < T(1e-12) && break
        end
        x = x + alpha .* dx
        lambda = lambda + alpha .* dlambda
    end

    return BarrierCache(x, lambda, mus, A, b, c)
end
