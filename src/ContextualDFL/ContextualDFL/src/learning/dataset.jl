struct ParametricScenario{W_EQ,W_INEQ,T_EQ,T_INEQ,H_EQ,H_INEQ,Q}
    W_eq_xi::W_EQ
    W_ineq_xi::W_INEQ
    T_eq_xi::T_EQ
    T_ineq_xi::T_INEQ
    h_eq_xi::H_EQ
    h_ineq_xi::H_INEQ
    q_xi::Q
end

function ParametricScenario(;
    W_eq_xi=0,
    W_ineq_xi=0,
    T_eq_xi=0,
    T_ineq_xi=0,
    h_eq_xi=0,
    h_ineq_xi=0,
    q_xi=0,
)
    return ParametricScenario(
        W_eq_xi,
        W_ineq_xi,
        T_eq_xi,
        T_ineq_xi,
        h_eq_xi,
        h_ineq_xi,
        q_xi,
    )
end

struct ContextualDataPoint{TContext<:AbstractVector,TScenarioParameter<:ParametricScenario}
    context::TContext
    scenario_parameters::Vector{TScenarioParameter}
end

const ContextualDataSet{TDataPoint<:ContextualDataPoint} = Vector{TDataPoint}
