abstract type Instance end

function first_stage_data(inst::Instance)
    error("Not implemented for $(typeof(inst))")
end

function default_second_stage_data(inst::Instance)
    error("Not implemented for $(typeof(inst))")
end

function scenario_realization(inst::Instance, ξ::AbstractVector)
    error("Not implemented for $(typeof(inst)) and $(typeof(ξ))")
end

function n1(inst)
    error("Not implemented for $(typeof(inst))")
end

function n2(inst)
    error("Not implemented for $(typeof(inst))")
end

function m1(inst)
    error("Not implemented for $(typeof(inst))")
end

function m2(inst)
    error("Not implemented for $(typeof(inst))")
end

struct Scenario{R<:Real}
    W::Matrix{R}
    T::Matrix{R}
    h::Vector{R}
    q::Vector{R}

    function Scenario(W::Matrix{R}, T::Matrix{R}, h::Vector{R}, q::Vector{R}) where {R<:Real}
        @assert size(W, 1) == size(T, 1) == length(h)
        @assert size(W, 2) == length(q)
        new{R}(W, T, h, q)
    end
end



