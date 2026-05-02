struct StochasticProgram{TLP<:LP}
    first_stage_lp::TLP
end

StochasticProgram(args...; kwargs...) = StochasticProgram(LP(args...; kwargs...))

function Base.getproperty(sp::StochasticProgram, name::Symbol)
    name === :A_eq && return getfield(sp, :first_stage_lp).A_eq
    name === :A_ineq && return getfield(sp, :first_stage_lp).A_ineq
    name === :b_eq && return getfield(sp, :first_stage_lp).b_eq
    name === :b_ineq && return getfield(sp, :first_stage_lp).b_ineq
    name === :c && return getfield(sp, :first_stage_lp).c
    return getfield(sp, name)
end

function Base.propertynames(sp::StochasticProgram, private::Bool=false)
    names = (:first_stage_lp, :A_eq, :A_ineq, :b_eq, :b_ineq, :c)
    return private ? (names..., fieldnames(typeof(sp))...) : names
end
