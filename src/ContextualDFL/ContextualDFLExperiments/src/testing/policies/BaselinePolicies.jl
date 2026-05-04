import LinearAlgebra

const PARAMETRIC_SCENARIO_COMPONENTS = (
    :W_eq_xi,
    :W_ineq_xi,
    :T_eq_xi,
    :T_ineq_xi,
    :h_eq_xi,
    :h_ineq_xi,
    :q_xi,
)

"""
    SampleAverageApproximationPolicy(data_set, solver, program, parametric_decoder; kwargs...)

Context-free SAA baseline. The policy solves one stochastic program using all
scenario parameters in `data_set` and returns that same first-stage decision for
every context.
"""
struct SampleAverageApproximationPolicy{
    TDecision,
    TScenarios,
    TSolver,
    TProgram,
    TDecoder,
    TMu,
    TRho,
    TKwargs,
} <: Policy
    decision::TDecision
    scenario_parameters::TScenarios
    solver::TSolver
    program::TProgram
    parametric_decoder::TDecoder
    mu::TMu
    rho::TRho
    solve_kwargs::TKwargs
end

function SampleAverageApproximationPolicy(
    scenario_parameters::AbstractVector{<:ContextualDFL.ParametricScenario},
    solver,
    program,
    parametric_decoder;
    mu=0,
    rho=0,
    kwargs...,
)
    checked_scenarios = _checked_scenario_collection(scenario_parameters)
    solve_kwargs = (; kwargs...)
    decision = _solve_scenario_collection(
        solver,
        program,
        parametric_decoder,
        checked_scenarios;
        mu=mu,
        rho=rho,
        solve_kwargs...,
    )

    return SampleAverageApproximationPolicy(
        decision,
        checked_scenarios,
        solver,
        program,
        parametric_decoder,
        mu,
        rho,
        solve_kwargs,
    )
end

function SampleAverageApproximationPolicy(
    contextual_data_set::AbstractVector,
    solver,
    program,
    parametric_decoder;
    mu=0,
    rho=0,
    kwargs...,
)
    scenario_parameters = _flatten_scenario_collections(
        _scenario_collections(contextual_data_set),
    )
    return SampleAverageApproximationPolicy(
        scenario_parameters,
        solver,
        program,
        parametric_decoder;
        mu=mu,
        rho=rho,
        kwargs...,
    )
end

infer(policy::SampleAverageApproximationPolicy, context) = copy(policy.decision)

"""
    LeastSquaresPolicy(data_set, solver, program, parametric_decoder; kwargs...)

Ordinary least-squares certainty-equivalent baseline. The policy fits a linear
map from contexts to one vector-valued `ParametricScenario` component, predicts
that component for a new context, solves the one-scenario stochastic program,
and returns the first-stage decision.

The default `target_component` is `:h_eq_xi`, matching the resource-allocation
demand scenarios. Other simple fixed-structure problems can pass another
component, such as `:h_ineq_xi` or `:q_xi`.
"""
struct LeastSquaresPolicy{
    TCoefficients,
    TTemplate,
    TSolver,
    TProgram,
    TDecoder,
    TPostprocess,
    TMu,
    TRho,
    TKwargs,
} <: Policy
    coefficients::TCoefficients
    scenario_template::TTemplate
    target_component::Symbol
    target_length::Int
    solver::TSolver
    program::TProgram
    parametric_decoder::TDecoder
    postprocess_prediction::TPostprocess
    mu::TMu
    rho::TRho
    solve_kwargs::TKwargs
end

function LeastSquaresPolicy(
    contextual_data_set::AbstractVector,
    solver,
    program,
    parametric_decoder;
    target_component=:h_eq_xi,
    postprocess_prediction=identity,
    validate_fixed_components=true,
    mu=0,
    rho=0,
    kwargs...,
)
    regression = _fit_scenario_target_regression(
        contextual_data_set,
        target_component;
        validate_fixed_components=validate_fixed_components,
    )

    return LeastSquaresPolicy(
        regression.coefficients,
        first(regression.scenario_templates),
        regression.target_component,
        regression.target_length,
        solver,
        program,
        parametric_decoder,
        postprocess_prediction,
        mu,
        rho,
        (; kwargs...),
    )
end

function infer(policy::LeastSquaresPolicy, context)
    target_vector = _processed_prediction(
        policy.postprocess_prediction,
        _predict_target(policy.coefficients, context),
        policy.target_length,
    )
    scenario = _scenario_from_target_vector(
        policy.scenario_template,
        policy.target_component,
        target_vector,
    )

    return _solve_scenario_collection(
        policy.solver,
        policy.program,
        policy.parametric_decoder,
        [scenario];
        mu=policy.mu,
        rho=policy.rho,
        policy.solve_kwargs...,
    )
end

"""
    ResidualSampleAverageApproximationPolicy(data_set, solver, program, parametric_decoder; kwargs...)

Empirical-residual SAA baseline (ER-SAA). It fits the same OLS model as
`LeastSquaresPolicy`, stores training residuals, and at inference time solves
SAA over `prediction(context) + residual` scenarios.
"""
struct ResidualSampleAverageApproximationPolicy{
    TCoefficients,
    TResiduals,
    TTemplates,
    TSolver,
    TProgram,
    TDecoder,
    TPostprocess,
    TMu,
    TRho,
    TKwargs,
} <: Policy
    coefficients::TCoefficients
    residuals::TResiduals
    scenario_templates::TTemplates
    target_component::Symbol
    target_length::Int
    solver::TSolver
    program::TProgram
    parametric_decoder::TDecoder
    postprocess_prediction::TPostprocess
    mu::TMu
    rho::TRho
    solve_kwargs::TKwargs
end

function ResidualSampleAverageApproximationPolicy(
    contextual_data_set::AbstractVector,
    solver,
    program,
    parametric_decoder;
    target_component=:h_eq_xi,
    postprocess_prediction=identity,
    validate_fixed_components=true,
    mu=0,
    rho=0,
    kwargs...,
)
    regression = _fit_scenario_target_regression(
        contextual_data_set,
        target_component;
        validate_fixed_components=validate_fixed_components,
    )

    return ResidualSampleAverageApproximationPolicy(
        regression.coefficients,
        regression.residuals,
        regression.scenario_templates,
        regression.target_component,
        regression.target_length,
        solver,
        program,
        parametric_decoder,
        postprocess_prediction,
        mu,
        rho,
        (; kwargs...),
    )
end

function infer(policy::ResidualSampleAverageApproximationPolicy, context)
    base_target = _predict_target(policy.coefficients, context)
    scenario_parameters = [
        _scenario_from_target_vector(
            policy.scenario_templates[index],
            policy.target_component,
            _processed_prediction(
                policy.postprocess_prediction,
                base_target .+ view(policy.residuals, index, :),
                policy.target_length,
            ),
        ) for index in axes(policy.residuals, 1)
    ]

    return _solve_scenario_collection(
        policy.solver,
        policy.program,
        policy.parametric_decoder,
        scenario_parameters;
        mu=policy.mu,
        rho=policy.rho,
        policy.solve_kwargs...,
    )
end

"""
    KNearestNeighborsPolicy(data_set, solver, program, parametric_decoder; k, kwargs...)

kNN-SAA baseline. At inference time, the policy finds the `k` nearest training
contexts, pools their scenario parameters, solves the resulting SAA problem, and
returns the first-stage decision.
"""
struct KNearestNeighborsPolicy{
    TContexts,
    TScenarioCollections,
    TSolver,
    TProgram,
    TDecoder,
    TMu,
    TRho,
    TKwargs,
} <: Policy
    training_contexts::TContexts
    training_scenario_collections::TScenarioCollections
    k::Int
    solver::TSolver
    program::TProgram
    parametric_decoder::TDecoder
    mu::TMu
    rho::TRho
    solve_kwargs::TKwargs
end

function KNearestNeighborsPolicy(
    contextual_data_set::AbstractVector,
    solver,
    program,
    parametric_decoder;
    k=default_knn_k(length(contextual_data_set)),
    mu=0,
    rho=0,
    kwargs...,
)
    _check_nonempty_data_set(contextual_data_set)
    training_contexts = _contexts(contextual_data_set)
    training_scenario_collections = _scenario_collections(contextual_data_set)
    checked_k = _checked_neighbor_count(k, length(training_contexts))

    return KNearestNeighborsPolicy(
        training_contexts,
        training_scenario_collections,
        checked_k,
        solver,
        program,
        parametric_decoder,
        mu,
        rho,
        (; kwargs...),
    )
end

function infer(policy::KNearestNeighborsPolicy, context)
    neighbor_indices = _nearest_neighbor_indices(
        policy.training_contexts,
        context,
        policy.k,
    )
    scenario_parameters = _flatten_scenario_collections(
        policy.training_scenario_collections[index] for index in neighbor_indices
    )

    return _solve_scenario_collection(
        policy.solver,
        policy.program,
        policy.parametric_decoder,
        scenario_parameters;
        mu=policy.mu,
        rho=policy.rho,
        policy.solve_kwargs...,
    )
end

function default_knn_k(sample_count::Integer; scale=5.0, exponent=0.4)
    n = Int(sample_count)
    n > 0 || throw(ArgumentError("sample_count must be positive, got $n."))

    return min(n, max(1, round(Int, scale * n^exponent)))
end

function _fit_scenario_target_regression(
    contextual_data_set,
    target_component;
    validate_fixed_components,
)
    checked_target_component = _checked_target_component(target_component)
    observations = _scenario_target_observations(
        contextual_data_set,
        checked_target_component,
    )
    validate_fixed_components &&
        _check_fixed_scenario_components(
            observations.scenario_templates,
            checked_target_component,
        )

    design = _design_matrix(observations.contexts)
    coefficients = design \ observations.targets
    fitted_values = design * coefficients

    return (;
        coefficients=coefficients,
        residuals=observations.targets - fitted_values,
        scenario_templates=observations.scenario_templates,
        target_component=checked_target_component,
        target_length=size(observations.targets, 2),
    )
end

function _scenario_target_observations(contextual_data_set, target_component::Symbol)
    _check_nonempty_data_set(contextual_data_set)

    context_vectors = Vector{Vector{Float64}}()
    target_vectors = Vector{Vector{Float64}}()
    scenario_templates = ContextualDFL.ParametricScenario[]
    context_dimension = 0
    target_length = 0

    for data_point in contextual_data_set
        context_vector = Float64.(collect(data_point.context))
        isempty(context_vector) &&
            throw(ArgumentError("contexts must contain at least one feature."))
        if context_dimension == 0
            context_dimension = length(context_vector)
        elseif length(context_vector) != context_dimension
            throw(DimensionMismatch("all contexts must have the same dimension."))
        end

        for scenario in _checked_scenario_collection(data_point.scenario_parameters)
            target_vector = _target_feature_vector(scenario, target_component)
            if target_length == 0
                target_length = length(target_vector)
            elseif length(target_vector) != target_length
                throw(DimensionMismatch(
                    "all target components must have the same flattened length.",
                ))
            end

            push!(context_vectors, context_vector)
            push!(target_vectors, target_vector)
            push!(scenario_templates, scenario)
        end
    end

    isempty(target_vectors) &&
        throw(ArgumentError("regression baselines require at least one scenario."))

    context_matrix = zeros(Float64, length(context_vectors), context_dimension)
    target_matrix = zeros(Float64, length(target_vectors), target_length)
    for index in eachindex(context_vectors)
        context_matrix[index, :] = context_vectors[index]
        target_matrix[index, :] = target_vectors[index]
    end

    return (;
        contexts=context_matrix,
        targets=target_matrix,
        scenario_templates=scenario_templates,
    )
end

function _design_matrix(context_matrix::AbstractMatrix)
    return hcat(context_matrix, ones(Float64, size(context_matrix, 1)))
end

function _predict_target(coefficients::AbstractMatrix, context)
    context_vector = Float64.(collect(context))
    length(context_vector) + 1 == size(coefficients, 1) ||
        throw(DimensionMismatch(
            "context has length $(length(context_vector)); expected $(size(coefficients, 1) - 1).",
        ))

    return vec(transpose(vcat(context_vector, 1.0)) * coefficients)
end

function _checked_target_component(target_component)
    component = Symbol(target_component)
    component in PARAMETRIC_SCENARIO_COMPONENTS ||
        throw(ArgumentError(
            "target_component must be one of $(PARAMETRIC_SCENARIO_COMPONENTS); got $(repr(component)).",
        ))

    return component
end

function _target_feature_vector(scenario, target_component::Symbol)
    target = getproperty(scenario, target_component)
    vector = _numeric_feature_vector(target; name=target_component)
    isempty(vector) &&
        throw(ArgumentError("target component $(target_component) must not be empty."))
    return vector
end

function _numeric_feature_vector(value; name)
    if value isa Number
        return [Float64(value)]
    elseif value isa AbstractArray
        return Float64.(vec(value))
    end

    throw(ArgumentError("$(name) must be numeric or an array of numeric values."))
end

function _processed_prediction(postprocess_prediction, target_vector, target_length::Integer)
    processed = postprocess_prediction(collect(target_vector))
    processed_vector = _numeric_feature_vector(processed; name=:postprocess_prediction)
    length(processed_vector) == target_length ||
        throw(DimensionMismatch(
            "postprocess_prediction returned length $(length(processed_vector)); expected $target_length.",
        ))

    return processed_vector
end

function _scenario_from_target_vector(
    scenario_template,
    target_component::Symbol,
    target_vector::AbstractVector,
)
    replacement = _reshape_target_vector(
        target_vector,
        getproperty(scenario_template, target_component),
        target_component,
    )

    return ContextualDFL.ParametricScenario(;
        W_eq_xi=target_component == :W_eq_xi ? replacement :
                _copy_scenario_component(scenario_template.W_eq_xi),
        W_ineq_xi=target_component == :W_ineq_xi ? replacement :
                  _copy_scenario_component(scenario_template.W_ineq_xi),
        T_eq_xi=target_component == :T_eq_xi ? replacement :
                _copy_scenario_component(scenario_template.T_eq_xi),
        T_ineq_xi=target_component == :T_ineq_xi ? replacement :
                  _copy_scenario_component(scenario_template.T_ineq_xi),
        h_eq_xi=target_component == :h_eq_xi ? replacement :
                _copy_scenario_component(scenario_template.h_eq_xi),
        h_ineq_xi=target_component == :h_ineq_xi ? replacement :
                  _copy_scenario_component(scenario_template.h_ineq_xi),
        q_xi=target_component == :q_xi ? replacement :
             _copy_scenario_component(scenario_template.q_xi),
    )
end

function _reshape_target_vector(target_vector, template_value, target_component)
    if template_value isa Number
        length(target_vector) == 1 ||
            throw(DimensionMismatch(
                "target vector for scalar $(target_component) must have length 1.",
            ))
        return only(target_vector)
    elseif template_value isa AbstractVector
        length(target_vector) == length(template_value) ||
            throw(DimensionMismatch(
                "target vector for $(target_component) has length $(length(target_vector)); expected $(length(template_value)).",
            ))
        return collect(target_vector)
    elseif template_value isa AbstractArray
        length(target_vector) == length(template_value) ||
            throw(DimensionMismatch(
                "target vector for $(target_component) has length $(length(target_vector)); expected $(length(template_value)).",
            ))
        return reshape(collect(target_vector), size(template_value))
    end

    throw(ArgumentError("template component $(target_component) must be numeric or an array."))
end

_copy_scenario_component(value::AbstractArray) = copy(value)
_copy_scenario_component(value) = value

function _check_fixed_scenario_components(scenario_templates, target_component::Symbol)
    base_scenario = first(scenario_templates)
    for scenario in Iterators.drop(scenario_templates, 1)
        for component in PARAMETRIC_SCENARIO_COMPONENTS
            component == target_component && continue
            isequal(getproperty(base_scenario, component), getproperty(scenario, component)) ||
                throw(ArgumentError(
                    "Least-squares baselines require fixed non-target scenario components; $(component) varies.",
                ))
        end
    end

    return nothing
end

function _check_nonempty_data_set(contextual_data_set)
    isempty(contextual_data_set) &&
        throw(ArgumentError("contextual_data_set must not be empty."))
    return nothing
end

function _contexts(contextual_data_set)
    _check_nonempty_data_set(contextual_data_set)
    return [collect(data_point.context) for data_point in contextual_data_set]
end

function _scenario_collections(contextual_data_set)
    _check_nonempty_data_set(contextual_data_set)
    return map(contextual_data_set) do data_point
        _checked_scenario_collection(data_point.scenario_parameters)
    end
end

function _checked_scenario_collection(
    scenario_parameters::AbstractVector{<:ContextualDFL.ParametricScenario},
)
    isempty(scenario_parameters) &&
        throw(ArgumentError("scenario collections must not be empty."))
    return collect(scenario_parameters)
end

function _flatten_scenario_collections(scenario_collections)
    scenario_parameters = ContextualDFL.ParametricScenario[]
    for collection in scenario_collections
        checked_collection = _checked_scenario_collection(collection)
        append!(scenario_parameters, checked_collection)
    end

    isempty(scenario_parameters) &&
        throw(ArgumentError("scenario collections must not be empty."))
    return scenario_parameters
end

function _checked_neighbor_count(k, sample_count::Integer)
    k isa Integer ||
        throw(ArgumentError("k must be an integer, got $(typeof(k))."))

    neighbor_count = Int(k)
    1 <= neighbor_count <= sample_count ||
        throw(ArgumentError("k must be between 1 and $sample_count, got $neighbor_count."))

    return neighbor_count
end

function _nearest_neighbor_indices(training_contexts, context, k::Integer)
    query_context = collect(context)
    distances = [
        _squared_euclidean_distance(training_context, query_context)
        for training_context in training_contexts
    ]
    return partialsortperm(distances, 1:k)
end

function _squared_euclidean_distance(a, b)
    length(a) == length(b) ||
        throw(DimensionMismatch("context dimensions must match."))

    total = 0.0
    @inbounds for index in eachindex(a, b)
        difference = Float64(a[index]) - Float64(b[index])
        total += difference * difference
    end
    return total
end

function _solve_scenario_collection(
    solver,
    program,
    parametric_decoder,
    scenario_parameters::AbstractVector{<:ContextualDFL.ParametricScenario};
    mu=0,
    rho=0,
    kwargs...,
)
    checked_scenarios = _checked_scenario_collection(scenario_parameters)
    W_eq, W_ineq, T_eq, T_ineq, h_eq, h_ineq, q =
        ContextualDFL.decode_scenario_collection(parametric_decoder, checked_scenarios)

    z, _, _, _, _, _ = ContextualDFL.solve(
        solver,
        program,
        W_eq,
        W_ineq,
        T_eq,
        T_ineq,
        h_eq,
        h_ineq,
        q;
        μ=mu,
        ρ=rho,
        kwargs...,
    )

    return collect(z)
end
