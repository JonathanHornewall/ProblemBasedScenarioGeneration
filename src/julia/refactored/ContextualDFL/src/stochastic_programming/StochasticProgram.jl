struct StochasticProgram{AEQ,AIN,B,C}
    A_eq::AEQ
    A_in::AIN
    b::B
    c::C
end

function StochasticProgram(A_eq::AbstractMatrix, b::AbstractVector, c::AbstractVector)
    return StochasticProgram(A_eq, zeros(eltype(A_eq), 0, size(A_eq, 2)), b, c)
end

function _first_stage_rhs(program::StochasticProgram)
    m_eq = size(program.A_eq, 1)
    m_in = size(program.A_in, 1)
    b = vec(program.b)
    if length(b) == m_eq + m_in
        return b[1:m_eq], b[(m_eq + 1):end]
    elseif length(b) == m_eq && m_in == 0
        return b, similar(b, 0)
    elseif length(b) == m_eq
        return b, zeros(eltype(b), m_in)
    else
        error("First-stage RHS length $(length(b)) does not match equality/inequality rows $(m_eq + m_in)")
    end
end
