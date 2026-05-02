struct DataSet{TX,TW,TT,TH,TQ}
    x_data::TX
    xi_W_data::TW
    xi_T_data::TT
    xi_h_data::TH
    xi_q_data::TQ
end

"""
    ContextualDataSet{TContext,TScenarioParameter}

A typed list of `(context, scenario_parameters)` training examples. Each context
is an `AbstractVector`, and each scenario collection is a `Vector` of stored
scenario parameter objects.
"""
const ContextualDataSet{TContext<:AbstractVector,TScenarioParameter} =
    Vector{Tuple{TContext,Vector{TScenarioParameter}}}
