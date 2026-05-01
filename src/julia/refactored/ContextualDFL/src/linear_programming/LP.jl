struct LP{AEQ,AIN,BEQ,BIN,CEQ,CIN}
    A_eq::AEQ
    A_in::AIN
    b_eq::BEQ
    b_in::BIN
    c_eq::CEQ
    c_in::CIN
end

function LP(A_eq::AbstractMatrix, b_eq::AbstractVector, c::AbstractVector)
    n = size(A_eq, 2)
    return LP(A_eq, zeros(eltype(A_eq), 0, n), b_eq, zeros(eltype(b_eq), 0), c, nothing)
end

function LP(;
    A_eq,
    A_in=zeros(eltype(A_eq), 0, size(A_eq, 2)),
    b_eq,
    b_in=zeros(eltype(b_eq), 0),
    c,
    metadata=nothing,
)
    return LP(A_eq, A_in, b_eq, b_in, c, metadata)
end

_objective(lp::LP) = lp.c_eq

function _matrix_or_empty(A, rows::Integer, cols::Integer, ::Type{T}) where {T}
    A === nothing && return zeros(T, rows, cols)
    return T.(A)
end

function _vector_or_empty(v, len::Integer, ::Type{T}) where {T}
    v === nothing && return zeros(T, len)
    return T.(v)
end

function _nonzero_rows(A::AbstractMatrix, b::AbstractVector; atol::Real=1e-10, inequality::Bool=false)
    keep = Int[]
    for i in axes(A, 1)
        if any(abs.(A[i, :]) .> atol)
            push!(keep, i)
        elseif inequality
            b[i] >= -atol || error("Infeasible zero inequality row with RHS $(b[i])")
        else
            abs(b[i]) <= atol || error("Infeasible zero equality row with RHS $(b[i])")
        end
    end
    return keep
end

"""
    canonical_form(lp::LP; A_eq=nothing, b=nothing, c=nothing)

Return `(A, b, c, n_original)` for the canonical LP
`min c'x` subject to `A*x = b, x >= 0`.

`LP` stores equality rows and `<=` inequality rows. Inequalities are converted
with nonnegative slacks. The original decision variables always occupy the
first `n_original` entries of the canonical decision vector.
"""
function canonical_form(lp::LP; A_eq=nothing, b=nothing, c=nothing)
    c_raw = c === nothing ? _objective(lp) : c
    T = promote_type(
        eltype(lp.A_eq),
        eltype(lp.b_eq),
        eltype(c_raw),
        lp.A_in === nothing ? Float64 : eltype(lp.A_in),
        lp.b_in === nothing ? Float64 : eltype(lp.b_in),
        Float64,
    )

    c_vec = T.(vec(c_raw))
    n = length(c_vec)
    Aeq = A_eq === nothing ? _matrix_or_empty(lp.A_eq, size(lp.A_eq, 1), n, T) : T.(A_eq)
    Ain = _matrix_or_empty(lp.A_in, 0, n, T)

    beq = b === nothing ? _vector_or_empty(lp.b_eq, size(Aeq, 1), T) : T.(vec(b))
    bin = _vector_or_empty(lp.b_in, size(Ain, 1), T)

    size(Aeq, 2) == n || error("Equality matrix has $(size(Aeq, 2)) columns but objective has length $n")
    size(Ain, 2) == n || error("Inequality matrix has $(size(Ain, 2)) columns but objective has length $n")
    length(beq) == size(Aeq, 1) || error("Equality RHS length $(length(beq)) does not match $(size(Aeq, 1)) rows")
    length(bin) == size(Ain, 1) || error("Inequality RHS length $(length(bin)) does not match $(size(Ain, 1)) rows")

    eq_rows = _nonzero_rows(Aeq, beq)
    in_rows = _nonzero_rows(Ain, bin; inequality=true)
    Aeq = Aeq[eq_rows, :]
    beq = beq[eq_rows]
    Ain = Ain[in_rows, :]
    bin = bin[in_rows]

    m_eq = size(Aeq, 1)
    m_in = size(Ain, 1)
    if m_in == 0
        A_can = Aeq
        b_can = beq
        c_can = c_vec
    else
        A_can = vcat(
            hcat(Aeq, zeros(T, m_eq, m_in)),
            hcat(Ain, Matrix{T}(I, m_in, m_in)),
        )
        b_can = vcat(beq, bin)
        c_can = vcat(c_vec, zeros(T, m_in))
    end

    return A_can, b_can, c_can, n
end

canonical_form(A_eq, b_eq, c) = canonical_form(LP(A_eq, b_eq, c))
