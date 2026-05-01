function stochasticity_diagnostics(problem_instance, data_set; kwargs...)
    return (
        n=length(data_set),
        context_dimension=length(first(data_set).x),
        h_dimension=first(data_set).xi_h === nothing ? 0 : length(first(data_set).xi_h),
    )
end

function value_of_stochasticity(problem_instance, data_set; kwargs...)
    return (value=0.0, diagnostics=stochasticity_diagnostics(problem_instance, data_set; kwargs...))
end
