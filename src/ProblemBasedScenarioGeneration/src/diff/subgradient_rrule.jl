"""
ChainRules rrule for `solve_lp` (μ = 0 case).

When the barrier parameter is zero, we use a subgradient approximation:
    d(loss)/db ≈ -λ*   (dual variable as subgradient of optimal value)
    d(loss)/dc ≈ 0     (cost subgradient is approximately the optimal basis indicator)

This is the standard approach for differentiating through an LP at μ = 0,
used during the fine-tuning phase of continuation training.
"""

using ChainRulesCore

"""
    ChainRulesCore.rrule(::typeof(solve_lp), A, b, c; tol=1e-9)

Differentiable wrapper for `solve_lp` using the LP dual as subgradient.

The pullback for `b` uses: d(loss)/db = -(∂V/∂b) ≈ λ* (negated dual)
where V(b) = min_x c'x s.t. Ax=b, x≥0.

By duality, ∂V/∂b = -λ* (the optimal dual variable), so:
    d(loss)/db = d(loss)/dV * (-λ*) = -Δx_effective * λ*

In practice, the upstream gradient acts on x, not V, so we use the
dual variable as a proxy subgradient for the primal-dual relationship.
"""
function ChainRulesCore.rrule(::typeof(solve_lp), A, b, c; tol=1e-9)
    x_opt, lambda_opt = solve_lp(A, b, c; tol=tol)

    function solve_lp_pullback(result_bar)
        # result_bar is a tuple tangent for (x, lambda)
        # We extract the x component
        if result_bar isa Tangent
            Δx = result_bar[1]
        elseif result_bar isa AbstractVector
            Δx = result_bar
        else
            Δx = zeros(eltype(x_opt), length(x_opt))
        end

        # Subgradient approximation: db ≈ -lambda * (Δx · x / ||x||²)
        # or simply use the dual as the derivative of the value function
        db_thunk = @thunk(-lambda_opt .* sum(Δx))

        # Cost derivative: approximately zero at a vertex (subgradient is 0)
        dc_thunk = @thunk(x_opt .* sum(Δx))

        return (
            NoTangent(),   # typeof(solve_lp)
            NoTangent(),   # A
            db_thunk,      # b
            dc_thunk       # c
        )
    end

    return (x_opt, lambda_opt), solve_lp_pullback
end
