struct BaseScenario{WEQ,WIN,TEQ,TIN,H,Q}
    W_eq::WEQ
    W_ineq::WIN
    T_eq::TEQ
    T_ineq::TIN
    h::H
    q::Q
end

function BaseScenario(; W_eq, W_ineq, T_eq, T_ineq, h, q)
    return BaseScenario(W_eq, W_ineq, T_eq, T_ineq, h, q)
end

function Base.getproperty(scenario::BaseScenario, name::Symbol)
    if name === :W
        return getfield(scenario, :W_eq)
    elseif name === :T
        return getfield(scenario, :T_eq)
    else
        return getfield(scenario, name)
    end
end
