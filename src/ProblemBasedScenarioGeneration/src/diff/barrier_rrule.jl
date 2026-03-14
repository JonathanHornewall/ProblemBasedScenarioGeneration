"""
ChainRules rrule for `solve_barrier`, enabling AD through the barrier solver.

The pullback uses implicit differentiation (see implicit_diff.jl):
    d(loss)/db = (∂x*/∂b)' * d(loss)/dx
    d(loss)/dc = (∂x*/∂c)' * d(loss)/dx
"""

using ChainRulesCore

"""
    ChainRulesCore.rrule(::typeof(solve_barrier), A, b, c, mu; ...)

Differentiable wrapper around `solve_barrier`.

The pullback propagates upstream gradients w.r.t. the cache's `x` field
back to the inputs `b` and `c` using implicit differentiation. Tangents
for `A` and `mu` are `NoTangent`.
"""
function ChainRulesCore.rrule(
    ::typeof(solve_barrier), A, b, c, mu;
    tol = 1e-10, max_iter = 200, x0 = nothing, lambda0 = nothing
)
    cache = solve_barrier(A, b, c, mu; tol=tol, max_iter=max_iter, x0=x0, lambda0=lambda0)

    function solve_barrier_pullback(cache_bar)
        # Upstream gradient w.r.t. x
        Δx = cache_bar isa AbstractZero ? zeros(eltype(cache.x), length(cache.x)) :
             (hasproperty(cache_bar, :x) ? cache_bar.x : zeros(eltype(cache.x), length(cache.x)))

        # Lazy computation of derivatives via implicit differentiation
        db_thunk = @thunk begin
            Dh = implicit_diff_h(cache)   # (n, m)
            Dh' * Δx                       # (m,)
        end

        dc_thunk = @thunk begin
            Dq = implicit_diff_q(cache)   # (n, n)
            Dq' * Δx                       # (n,)
        end

        return (
            NoTangent(),   # typeof(solve_barrier)
            NoTangent(),   # A
            db_thunk,      # b
            dc_thunk,      # c
            NoTangent()    # mu
        )
    end

    return cache, solve_barrier_pullback
end
