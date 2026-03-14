"""
Canonical form utilities for two-stage stochastic LPs.

Canonical form: min c'x  s.t. Ax = b, x ≥ 0

Converting standard form (with equality/inequality constraints and free variables)
to canonical form by variable splitting and adding slack variables.
"""

"""
    to_canonical(A, b, c) -> (A_can, b_can, c_can)

Convert a standard-form LP `min c'x s.t. Ax = b, x ≥ 0` to canonical form
by splitting each variable x_i = x_i⁺ - x_i⁻ and adding slack variables.

The canonical form has `2n + m` variables: [x⁺; x⁻; slack].

Note: When A has a zero row (unconstrained first-stage), that row is removed.
"""
function to_canonical(A, b, c)
    m, n = size(A)
    T = promote_type(eltype(A), eltype(b), eltype(c), Float64)
    A, b, c = T.(A), T.(b), T.(c)

    # A_can = [A | -A | I_m]
    A_can = hcat(A, -A, Matrix{T}(I, m, m))
    # c_can = [c; -c; 0_m]  (slack variables have zero cost)
    c_can = vcat(c, -c, zeros(T, m))
    b_can = b

    # Remove zero rows (unconstrained first stage)
    nonzero_rows = findall(i -> !iszero(A_can[i, :]), 1:m)
    if length(nonzero_rows) < m
        A_can = A_can[nonzero_rows, :]
        b_can = b_can[nonzero_rows]
    end

    return A_can, b_can, c_can
end

"""
    to_canonical_decision(x, A, b) -> x_can

Convert a decision vector from standard form to canonical form.
Returns `[x⁺; x⁻; slack]` where x⁺ = max.(x, 0), x⁻ = max.(-x, 0),
and slack = b - A*x (projected onto feasible set).
"""
function to_canonical_decision(x, A, b)
    T = promote_type(eltype(x), eltype(A), eltype(b), Float64)
    x = T.(x)
    x_plus  = max.(x, zero(T))
    x_minus = max.(-x, zero(T))
    slack   = T.(b) - T.(A) * x
    return vcat(x_plus, x_minus, max.(slack, zero(T)))
end

"""
    extensive_form(slp::TwoStageSLP) -> (A_ext, b_ext, c_ext)

Build the extensive-form LP for a two-stage stochastic program.

Decision variables: [x₁; y₁; y₂; ...; y_S]
where x₁ is n₁-dimensional and each yₛ is n₂-dimensional.

The constraint matrix has the block structure:
  [A₁    0    0  ...  0  ]
  [T₁    W₁   0  ...  0  ]
  [T₂    0    W₂ ...  0  ]
  [⋮     ⋮    ⋮  ⋱   ⋮  ]
  [T_S   0    0  ...  W_S]

The cost vector is [c₁; p₁q₁; p₂q₂; ...; p_Sq_S].

If the first-stage has a zero-row constraint (unconstrained), that row is dropped.
"""
function extensive_form(slp::TwoStageSLP{T}) where {T}
    S = length(slp.scenarios)
    n1 = length(slp.c)
    m1 = size(slp.A, 1)

    if S == 0
        error("TwoStageSLP must have at least one scenario")
    end

    n2 = size(slp.scenarios[1].W, 2)
    m2 = size(slp.scenarios[1].W, 1)

    # Cost vector
    c_ext = vcat(slp.c, [slp.p[s] .* slp.scenarios[s].q for s in 1:S]...)

    # RHS vector
    b_ext = vcat(slp.b, [slp.scenarios[s].h for s in 1:S]...)

    # First-stage constraint row: [A₁ | 0 ... 0]
    first_stage_row = hcat(slp.A, zeros(T, m1, S * n2))

    # Second-stage rows: one block per scenario
    second_stage_rows = Matrix{T}[]
    for s in 1:S
        sc = slp.scenarios[s]
        zeros_before = zeros(T, m2, (s - 1) * n2)
        zeros_after  = zeros(T, m2, (S - s) * n2)
        row = hcat(sc.T, zeros_before, sc.W, zeros_after)
        push!(second_stage_rows, row)
    end

    A_ext = vcat(first_stage_row, second_stage_rows...)

    # Drop zero rows (unconstrained first stage)
    nonzero = findall(i -> !iszero(A_ext[i, :]), 1:size(A_ext, 1))
    if length(nonzero) < size(A_ext, 1)
        A_ext = A_ext[nonzero, :]
        b_ext = b_ext[nonzero]
    end

    return A_ext, b_ext, c_ext
end
