import LinearAlgebra
import HiGHS
import JuMP
import MathOptInterface as MOI
import Optim
import Random

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
map from contexts to one or more vector-valued `ParametricScenario` components,
predicts those components for a new context, solves the one-scenario stochastic
program, and returns the first-stage decision.

The default `target_component` is `:h_eq_xi`, matching the resource-allocation
demand scenarios. Other simple fixed-structure problems can pass another
component, such as `:h_ineq_xi` or `:q_xi`, or a tuple such as
`(:h_eq_xi, :q_xi)`.
"""
struct LeastSquaresPolicy{
    TCoefficients,
    TTemplate,
    TTargetComponent,
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
    target_component::TTargetComponent
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
    TTargetComponent,
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
    target_component::TTargetComponent
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
    DecisionFocusedLinearPolicy(data_set, solver, program, parametric_decoder; kwargs...)

Decision-focused linear certainty-equivalent baseline. The policy starts from
the same ordinary least-squares map as `LeastSquaresPolicy`, then optionally
refines those coefficients with Nelder-Mead on the realized training decision
cost induced by each predicted scenario.
"""
struct DecisionFocusedLinearPolicy{
    TCoefficients,
    TTemplate,
    TTargetComponent,
    TSolver,
    TProgram,
    TDecoder,
    TPostprocess,
    TTrainingTransform,
    TOptimizationResult,
    TMu,
    TRho,
    TKwargs,
} <: Policy
    coefficients::TCoefficients
    initial_coefficients::TCoefficients
    scenario_template::TTemplate
    target_component::TTargetComponent
    target_length::Int
    solver::TSolver
    program::TProgram
    parametric_decoder::TDecoder
    postprocess_prediction::TPostprocess
    training_prediction_transform::TTrainingTransform
    optimization_result::TOptimizationResult
    mu::TMu
    rho::TRho
    solve_kwargs::TKwargs
end

function DecisionFocusedLinearPolicy(
    contextual_data_set::AbstractVector,
    solver,
    program,
    parametric_decoder;
    target_component=:h_eq_xi,
    postprocess_prediction=identity,
    training_prediction_transform=nothing,
    validate_fixed_components=true,
    optimize=true,
    optimizer=Optim.NelderMead(),
    optimizer_options=Optim.Options(f_reltol=1e-4),
    mu=0,
    rho=0,
    kwargs...,
)
    regression = _fit_scenario_target_regression(
        contextual_data_set,
        target_component;
        validate_fixed_components=validate_fixed_components,
    )
    active_transform = training_prediction_transform === nothing ?
        _zero_penalty_training_transform :
        training_prediction_transform

    coefficients = copy(regression.coefficients)
    optimization_result = nothing
    if optimize
        coefficients, optimization_result = _fit_decision_focused_coefficients(
            regression.coefficients,
            regression.contexts,
            regression.scenario_templates,
            regression.target_component,
            regression.target_length,
            solver,
            program,
            parametric_decoder,
            active_transform;
            optimizer=optimizer,
            optimizer_options=optimizer_options,
            mu=mu,
            rho=rho,
            kwargs...,
        )
    end

    return DecisionFocusedLinearPolicy(
        coefficients,
        regression.coefficients,
        first(regression.scenario_templates),
        regression.target_component,
        regression.target_length,
        solver,
        program,
        parametric_decoder,
        postprocess_prediction,
        active_transform,
        optimization_result,
        mu,
        rho,
        (; kwargs...),
    )
end

function infer(policy::DecisionFocusedLinearPolicy, context)
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
    LexSPOLinearPolicy(data_set, solver, program, parametric_decoder; kwargs...)

Lex-SPO-style direct empirical linear RHS baseline. The policy starts from the
same ordinary least-squares RHS predictor as `LeastSquaresPolicy`, then
optionally refines those coefficients by minimizing realized downstream
training cost. Predicted single-scenario LPs are solved with deterministic
lexicographic tie-breaking on first-stage decisions.

This is a rushed direct empirical baseline, not a full Estes-Richard convex
surrogate or cutting-plane implementation. It supports RHS uncertainty only:
`:h_eq_xi`, `:h_ineq_xi`, or a tuple containing only those components.
"""
struct LexSPOLinearPolicy{
    TCoefficients,
    TTemplate,
    TTargetComponent,
    TSolver,
    TProgram,
    TDecoder,
    TPostprocess,
    TTrainingTransform,
    TOptimizationResult,
    TKwargs,
    TMetadata,
} <: Policy
    coefficients::TCoefficients
    initial_coefficients::TCoefficients
    scenario_template::TTemplate
    target_component::TTargetComponent
    target_length::Int
    solver::TSolver
    program::TProgram
    parametric_decoder::TDecoder
    postprocess_prediction::TPostprocess
    training_prediction_transform::TTrainingTransform
    optimization_result::TOptimizationResult
    lex_objective_atol::Float64
    lex_objective_rtol::Float64
    lex_variable_atol::Float64
    solve_kwargs::TKwargs
    metadata::TMetadata
end

function LexSPOLinearPolicy(
    contextual_data_set::AbstractVector,
    solver,
    program,
    parametric_decoder;
    target_component=:h_eq_xi,
    postprocess_prediction=identity,
    training_prediction_transform=nothing,
    validate_fixed_components=true,
    optimize=true,
    optimizer=Optim.NelderMead(),
    optimizer_options=Optim.Options(f_reltol=1e-4),
    lex_objective_atol=1e-7,
    lex_objective_rtol=1e-7,
    lex_variable_atol=1e-7,
    mu=0,
    rho=0,
    kwargs...,
)
    iszero(mu) && iszero(rho) ||
        throw(ArgumentError("LexSPOLinearPolicy currently supports only unsmoothed LP solves: mu=0, rho=0."))
    _check_lex_spo_tolerances(lex_objective_atol, lex_objective_rtol, lex_variable_atol)
    solve_kwargs = (; kwargs...)
    _check_lex_spo_solve_kwargs(solve_kwargs)
    checked_target_component = _checked_lex_spo_target_component(target_component)

    regression = _fit_scenario_target_regression(
        contextual_data_set,
        checked_target_component;
        validate_fixed_components=validate_fixed_components,
    )
    active_transform = training_prediction_transform === nothing ?
        _zero_penalty_training_transform :
        training_prediction_transform

    coefficients = copy(regression.coefficients)
    optimization_result = nothing
    if optimize
        coefficients, optimization_result = _fit_lex_spo_coefficients(
            regression.coefficients,
            regression.contexts,
            regression.scenario_templates,
            regression.target_component,
            regression.target_length,
            solver,
            program,
            parametric_decoder,
            active_transform;
            optimizer=optimizer,
            optimizer_options=optimizer_options,
            lex_objective_atol=lex_objective_atol,
            lex_objective_rtol=lex_objective_rtol,
            lex_variable_atol=lex_variable_atol,
            solve_kwargs...,
        )
    end

    return LexSPOLinearPolicy(
        coefficients,
        regression.coefficients,
        first(regression.scenario_templates),
        regression.target_component,
        regression.target_length,
        solver,
        program,
        parametric_decoder,
        postprocess_prediction,
        active_transform,
        optimization_result,
        Float64(lex_objective_atol),
        Float64(lex_objective_rtol),
        Float64(lex_variable_atol),
        solve_kwargs,
        (;
            target_component=regression.target_component,
            optimize=optimize,
            rhs_only=true,
        ),
    )
end

function infer(policy::LexSPOLinearPolicy, context)
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

    return _lex_solve_scenario_collection(
        policy.solver,
        policy.program,
        policy.parametric_decoder,
        [scenario];
        lex_objective_atol=policy.lex_objective_atol,
        lex_objective_rtol=policy.lex_objective_rtol,
        lex_variable_atol=policy.lex_variable_atol,
        policy.solve_kwargs...,
    )
end

function nonnegative_prediction_penalty_transform(; lower_bound=0.0, penalty_weight=1000.0)
    checked_lower_bound = Float64(lower_bound)
    checked_penalty_weight = Float64(penalty_weight)
    checked_penalty_weight >= 0.0 ||
        throw(ArgumentError("penalty_weight must be non-negative."))

    return target -> begin
        values = Float64.(target)
        deficits = max.(checked_lower_bound .- values, 0.0)
        (;
            target=max.(values, checked_lower_bound),
            penalty=checked_penalty_weight * sum(deficits),
        )
    end
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

"""
    AdaptiveDecisionTreePolicy(data_set, solver, program, parametric_decoder; kwargs...)

Decision-focused fixed-depth tree baseline. The policy partitions contextual
observations with a MIP-fitted binary tree and assigns each leaf a representative
training-scenario decision. At inference time, it routes the context through the
tree and returns the representative first-stage decision stored at the reached
leaf.
"""
struct AdaptiveDecisionTreePolicy{
    TContexts,
    TFeatureMin,
    TFeatureScale,
    TBranchFeatures,
    TBranchThresholds,
    TLeafRepresentatives,
    TLeafDecisions,
    TMetadata,
} <: Policy
    normalized_contexts::TContexts
    feature_min::TFeatureMin
    feature_scale::TFeatureScale
    depth::Int
    branch_features::TBranchFeatures
    branch_thresholds::TBranchThresholds
    leaf_representative_indices::TLeafRepresentatives
    leaf_decisions::TLeafDecisions
    metadata::TMetadata
end

function AdaptiveDecisionTreePolicy(
    contextual_data_set::AbstractVector,
    solver,
    program,
    parametric_decoder;
    depth=2,
    min_leaf=2,
    mip_optimizer=HiGHS.Optimizer,
    mip_optimizer_attributes=(; threads=1),
    mu=0,
    rho=0,
    target_component=:h_eq_xi,
    postprocess_prediction=identity,
    validate_fixed_components=false,
    kwargs...,
)
    checked_depth = _checked_ad_tree_depth(depth)
    observations = _ad_tree_observations(contextual_data_set)
    checked_min_leaf = _checked_ad_tree_min_leaf(min_leaf)
    branch_nodes, leaf_nodes = _ad_tree_nodes(checked_depth)
    length(observations.scenarios) >= checked_min_leaf * length(leaf_nodes) ||
        throw(ArgumentError(
            "AD-tree requires at least min_leaf * leaf_count observations; got " *
            "$(length(observations.scenarios)) observations, min_leaf=$(checked_min_leaf), " *
            "leaf_count=$(length(leaf_nodes)).",
        ))

    if validate_fixed_components
        checked_target_component = _checked_target_component(target_component)
        _check_fixed_scenario_components(observations.scenarios, checked_target_component)
    end

    normalized = _normalize_ad_tree_contexts(observations.contexts)
    representative_decisions = _ad_tree_representative_decisions(
        observations.scenarios,
        solver,
        program,
        parametric_decoder;
        mu=mu,
        rho=rho,
        kwargs...,
    )
    cost_matrix = _ad_tree_cost_matrix(
        representative_decisions,
        observations.scenarios,
        solver,
        program,
        parametric_decoder;
        mu=mu,
        rho=rho,
        kwargs...,
    )
    fit = _fit_ad_tree_mip(
        normalized.contexts,
        cost_matrix;
        depth=checked_depth,
        min_leaf=checked_min_leaf,
        optimizer=mip_optimizer,
        optimizer_attributes=mip_optimizer_attributes,
    )

    leaf_decisions = zeros(Float64, size(representative_decisions, 1), length(leaf_nodes))
    for leaf_position in eachindex(leaf_nodes)
        representative_index = fit.leaf_representative_indices[leaf_position]
        leaf_decisions[:, leaf_position] = representative_decisions[:, representative_index]
    end

    metadata = (;
        depth=checked_depth,
        min_leaf=checked_min_leaf,
        branch_nodes=branch_nodes,
        leaf_nodes=leaf_nodes,
        objective_value=fit.objective_value,
        termination_status=fit.termination_status,
        primal_status=fit.primal_status,
        representative_count=size(representative_decisions, 2),
        observation_count=size(observations.contexts, 1),
        target_component=target_component,
        postprocess_prediction=postprocess_prediction,
    )

    return AdaptiveDecisionTreePolicy(
        normalized.contexts,
        normalized.feature_min,
        normalized.feature_scale,
        checked_depth,
        fit.branch_features,
        fit.branch_thresholds,
        fit.leaf_representative_indices,
        leaf_decisions,
        metadata,
    )
end

function infer(policy::AdaptiveDecisionTreePolicy, context)
    normalized_context = _normalize_ad_tree_context(
        context,
        policy.feature_min,
        policy.feature_scale,
    )
    leaf_position = _ad_tree_leaf_position(
        normalized_context,
        policy.depth,
        policy.branch_features,
        policy.branch_thresholds,
    )
    return copy(view(policy.leaf_decisions, :, leaf_position))
end

function _ad_tree_observations(contextual_data_set)
    _check_nonempty_data_set(contextual_data_set)

    context_vectors = Vector{Vector{Float64}}()
    scenario_parameters = ContextualDFL.ParametricScenario[]
    context_dimension = 0

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
            push!(context_vectors, context_vector)
            push!(scenario_parameters, scenario)
        end
    end

    contexts = zeros(Float64, length(context_vectors), context_dimension)
    for index in eachindex(context_vectors)
        contexts[index, :] = context_vectors[index]
    end

    return (; contexts=contexts, scenarios=scenario_parameters)
end

function _normalize_ad_tree_contexts(contexts::AbstractMatrix)
    feature_min = vec(minimum(contexts; dims=1))
    feature_max = vec(maximum(contexts; dims=1))
    feature_scale = feature_max .- feature_min
    normalized = zeros(Float64, size(contexts))

    for feature in axes(contexts, 2)
        if iszero(feature_scale[feature])
            normalized[:, feature] .= 0.0
            feature_scale[feature] = 1.0
        else
            normalized[:, feature] =
                (contexts[:, feature] .- feature_min[feature]) ./ feature_scale[feature]
        end
    end

    return (; contexts=normalized, feature_min=feature_min, feature_scale=feature_scale)
end

function _normalize_ad_tree_context(context, feature_min, feature_scale)
    context_vector = Float64.(collect(context))
    length(context_vector) == length(feature_min) ||
        throw(DimensionMismatch(
            "context has length $(length(context_vector)); expected $(length(feature_min)).",
        ))
    return clamp.((context_vector .- feature_min) ./ feature_scale, 0.0, 1.0)
end

function _ad_tree_representative_decisions(
    scenarios,
    solver,
    program,
    parametric_decoder;
    mu,
    rho,
    kwargs...,
)
    decisions = Vector{Vector{Float64}}(undef, length(scenarios))
    for index in eachindex(scenarios)
        decisions[index] = _solve_scenario_collection(
            solver,
            program,
            parametric_decoder,
            [scenarios[index]];
            mu=mu,
            rho=rho,
            kwargs...,
        )
    end
    decision_length = length(first(decisions))
    decision_matrix = zeros(Float64, decision_length, length(decisions))
    for index in eachindex(decisions)
        length(decisions[index]) == decision_length ||
            throw(DimensionMismatch("all representative decisions must have the same length."))
        decision_matrix[:, index] = decisions[index]
    end
    return decision_matrix
end

function _ad_tree_cost_matrix(
    representative_decisions::AbstractMatrix,
    scenarios,
    solver,
    program,
    parametric_decoder;
    mu,
    rho,
    kwargs...,
)
    observation_count = length(scenarios)
    representative_count = size(representative_decisions, 2)
    costs = zeros(Float64, observation_count, representative_count)

    decoded = map(scenarios) do scenario
        ContextualDFL.decode_scenario_collection(parametric_decoder, [scenario])
    end

    for observation in 1:observation_count
        W_eq, W_ineq, T_eq, T_ineq, h_eq, h_ineq, q = decoded[observation]
        for representative in 1:representative_count
            costs[observation, representative] = Float64(ContextualDFL.cost_function(
                program,
                solver,
                representative_decisions[:, representative],
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
            ))
        end
    end

    return costs
end

function _fit_ad_tree_mip(
    normalized_contexts::AbstractMatrix,
    cost_matrix::AbstractMatrix;
    depth,
    min_leaf,
    optimizer,
    optimizer_attributes,
)
    observation_count, feature_count = size(normalized_contexts)
    representative_count = size(cost_matrix, 2)
    branch_nodes, leaf_nodes = _ad_tree_nodes(depth)
    epsilon, epsilon_max = _ad_tree_feature_epsilons(normalized_contexts)
    big_m = 1.0 + epsilon_max

    model = JuMP.Model(optimizer)
    JuMP.set_silent(model)
    _set_ad_tree_optimizer_attributes(model, optimizer_attributes)

    JuMP.@variable(model, assign[1:observation_count, leaf_nodes], Bin)
    JuMP.@variable(model, representative[leaf_nodes, 1:representative_count], Bin)
    JuMP.@variable(model, link[1:observation_count, leaf_nodes, 1:representative_count], Bin)
    JuMP.@variable(model, feature_select[branch_nodes, 1:feature_count], Bin)
    JuMP.@variable(model, 0 <= threshold[branch_nodes] <= 1)

    JuMP.@constraint(model, [i in 1:observation_count], sum(assign[i, leaf] for leaf in leaf_nodes) == 1)
    JuMP.@constraint(model, [leaf in leaf_nodes], sum(assign[i, leaf] for i in 1:observation_count) >= min_leaf)
    JuMP.@constraint(model, [leaf in leaf_nodes], sum(representative[leaf, r] for r in 1:representative_count) == 1)
    JuMP.@constraint(model, [node in branch_nodes], sum(feature_select[node, p] for p in 1:feature_count) == 1)

    JuMP.@constraint(model, [i in 1:observation_count, leaf in leaf_nodes, r in 1:representative_count], link[i, leaf, r] <= assign[i, leaf])
    JuMP.@constraint(model, [i in 1:observation_count, leaf in leaf_nodes, r in 1:representative_count], link[i, leaf, r] <= representative[leaf, r])
    JuMP.@constraint(model, [i in 1:observation_count, leaf in leaf_nodes, r in 1:representative_count], link[i, leaf, r] >= assign[i, leaf] + representative[leaf, r] - 1)

    for leaf in leaf_nodes
        for node in _ad_tree_left_ancestors(leaf)
            JuMP.@constraint(
                model,
                [i in 1:observation_count],
                sum(feature_select[node, p] * normalized_contexts[i, p] for p in 1:feature_count) <=
                    threshold[node] + big_m * (1 - assign[i, leaf]),
            )
        end
        for node in _ad_tree_right_ancestors(leaf)
            JuMP.@constraint(
                model,
                [i in 1:observation_count],
                sum(feature_select[node, p] * (normalized_contexts[i, p] - epsilon[p]) for p in 1:feature_count) >=
                    threshold[node] - big_m * (1 - assign[i, leaf]),
            )
        end
    end

    JuMP.@objective(
        model,
        Min,
        sum(
            cost_matrix[i, r] * link[i, leaf, r]
            for i in 1:observation_count, leaf in leaf_nodes, r in 1:representative_count
        ) / observation_count,
    )

    JuMP.optimize!(model)
    termination_status = JuMP.termination_status(model)
    termination_status in (MOI.OPTIMAL, MOI.LOCALLY_SOLVED, MOI.TIME_LIMIT) ||
        throw(ArgumentError("AD-tree MIP failed with status $(termination_status)."))
    primal_status = JuMP.primal_status(model)
    primal_status in (MOI.FEASIBLE_POINT, MOI.NEARLY_FEASIBLE_POINT) ||
        throw(ArgumentError("AD-tree MIP did not return a feasible tree: primal_status=$(primal_status)."))

    feature_values = JuMP.value.(feature_select)
    representative_values = JuMP.value.(representative)
    branch_features = [
        _argmax_index([feature_values[node, p] for p in 1:feature_count])
        for node in branch_nodes
    ]
    branch_thresholds = [Float64(JuMP.value(threshold[node])) for node in branch_nodes]
    leaf_representative_indices = [
        _argmax_index([representative_values[leaf, r] for r in 1:representative_count])
        for leaf in leaf_nodes
    ]

    return (;
        branch_features=branch_features,
        branch_thresholds=branch_thresholds,
        leaf_representative_indices=leaf_representative_indices,
        objective_value=Float64(JuMP.objective_value(model)),
        termination_status=termination_status,
        primal_status=primal_status,
    )
end

function _set_ad_tree_optimizer_attributes(model, attributes)
    for (attribute, value) in pairs(attributes)
        JuMP.set_optimizer_attribute(model, String(attribute), value)
    end
    return nothing
end

function _ad_tree_nodes(depth::Integer)
    checked_depth = _checked_ad_tree_depth(depth)
    first_leaf = 2^checked_depth
    last_leaf = 2^(checked_depth + 1) - 1
    return collect(1:(first_leaf - 1)), collect(first_leaf:last_leaf)
end

function _checked_ad_tree_depth(depth)
    depth isa Integer || throw(ArgumentError("depth must be an integer."))
    checked = Int(depth)
    checked > 0 || throw(ArgumentError("depth must be positive."))
    checked <= 6 || throw(ArgumentError("depth above 6 is not supported by the AD-tree baseline."))
    return checked
end

function _checked_ad_tree_min_leaf(min_leaf)
    min_leaf isa Integer || throw(ArgumentError("min_leaf must be an integer."))
    checked = Int(min_leaf)
    checked > 0 || throw(ArgumentError("min_leaf must be positive."))
    return checked
end

function _ad_tree_left_ancestors(leaf::Integer)
    return first(_ad_tree_path_ancestors(leaf))
end

function _ad_tree_right_ancestors(leaf::Integer)
    return last(_ad_tree_path_ancestors(leaf))
end

function _ad_tree_path_ancestors(leaf::Integer)
    left = Int[]
    right = Int[]
    node = Int(leaf)
    while node > 1
        parent = fld(node, 2)
        if iseven(node)
            push!(left, parent)
        else
            push!(right, parent)
        end
        node = parent
    end
    return reverse(left), reverse(right)
end

function _ad_tree_feature_epsilons(normalized_contexts::AbstractMatrix)
    feature_count = size(normalized_contexts, 2)
    epsilons = zeros(Float64, feature_count)
    for feature in 1:feature_count
        values = sort(unique(normalized_contexts[:, feature]))
        positive_differences = Float64[]
        for index in 2:length(values)
            difference = values[index] - values[index - 1]
            difference > 0.0 && push!(positive_differences, difference)
        end
        epsilons[feature] = isempty(positive_differences) ? 1e-6 : minimum(positive_differences)
    end
    return epsilons, maximum(epsilons)
end

function _ad_tree_leaf_position(context, depth, branch_features, branch_thresholds)
    node = 1
    first_leaf = 2^depth
    while node < first_leaf
        feature = branch_features[node]
        node = context[feature] <= branch_thresholds[node] ? 2 * node : 2 * node + 1
    end
    return node - first_leaf + 1
end

function _argmax_index(values)
    best_index = firstindex(values)
    best_value = values[best_index]
    for index in Iterators.drop(eachindex(values), 1)
        if values[index] > best_value
            best_index = index
            best_value = values[index]
        end
    end
    return Int(best_index)
end

"""
    CARTPolicy(data_set, solver, program, parametric_decoder; kwargs...)

Regression-tree certainty-equivalent baseline. The policy fits scikit-learn's
`DecisionTreeRegressor` from contexts to one or more vector-valued
`ParametricScenario` components, predicts those components for a new context,
solves the one-scenario stochastic program, and returns the first-stage
decision.

The defaults mirror the old resource-allocation CART baseline:
`criterion="squared_error"`, `min_samples_leaf=25`, `test_size=0.2`, and
`random_state=42`.
"""
struct CARTPolicy{
    TRegressor,
    TTemplate,
    TTargetComponent,
    TSolver,
    TProgram,
    TDecoder,
    TPostprocess,
    TMu,
    TRho,
    TKwargs,
    TMetadata,
} <: Policy
    regressor::TRegressor
    scenario_template::TTemplate
    target_component::TTargetComponent
    target_length::Int
    solver::TSolver
    program::TProgram
    parametric_decoder::TDecoder
    postprocess_prediction::TPostprocess
    mu::TMu
    rho::TRho
    solve_kwargs::TKwargs
    metadata::TMetadata
end

function CARTPolicy(
    contextual_data_set::AbstractVector,
    solver,
    program,
    parametric_decoder;
    target_component=:h_eq_xi,
    postprocess_prediction=identity,
    validate_fixed_components=true,
    min_samples_leaf=25,
    test_size=0.2,
    random_state=42,
    mu=0,
    rho=0,
    kwargs...,
)
    fit = _fit_cart_target_regressor(
        contextual_data_set,
        target_component;
        validate_fixed_components=validate_fixed_components,
        min_samples_leaf=min_samples_leaf,
        test_size=test_size,
        random_state=random_state,
    )

    return CARTPolicy(
        fit.regressor,
        first(fit.scenario_templates),
        fit.target_component,
        fit.target_length,
        solver,
        program,
        parametric_decoder,
        postprocess_prediction,
        mu,
        rho,
        (; kwargs...),
        fit.metadata,
    )
end

function infer(policy::CARTPolicy, context)
    target_vector = _processed_prediction(
        policy.postprocess_prediction,
        _cart_predict(policy.regressor, context, policy.target_length),
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
    M5ADPolicy(data_set, solver, program, parametric_decoder; kwargs...)

Leaf-local decision-focused linear baseline. This mirrors the old resource-
allocation "M5 + AD" script: fit a multi-output CART model, route each query
context to a tree leaf, optimize the AD linear model on the training samples in
that leaf, and solve the certainty-equivalent stochastic program from the local
linear prediction.
"""
struct M5ADPolicy{
    TRegressor,
    TCoefficients,
    TInitialCoefficients,
    TTemplate,
    TTargetComponent,
    TSolver,
    TProgram,
    TDecoder,
    TPostprocess,
    TTrainingTransform,
    TOptimizationResults,
    TMu,
    TRho,
    TKwargs,
    TMetadata,
} <: Policy
    regressor::TRegressor
    leaf_coefficients::TCoefficients
    initial_coefficients::TInitialCoefficients
    scenario_template::TTemplate
    target_component::TTargetComponent
    target_length::Int
    feature_count::Int
    solver::TSolver
    program::TProgram
    parametric_decoder::TDecoder
    postprocess_prediction::TPostprocess
    training_prediction_transform::TTrainingTransform
    leaf_optimization_results::TOptimizationResults
    mu::TMu
    rho::TRho
    solve_kwargs::TKwargs
    metadata::TMetadata
end

function M5ADPolicy(
    contextual_data_set::AbstractVector,
    solver,
    program,
    parametric_decoder;
    target_component=:h_eq_xi,
    postprocess_prediction=identity,
    training_prediction_transform=nothing,
    validate_fixed_components=true,
    optimize=true,
    optimizer=Optim.NelderMead(),
    optimizer_options=Optim.Options(f_reltol=1e-4),
    min_samples_leaf=25,
    test_size=0.2,
    random_state=42,
    mu=0,
    rho=0,
    kwargs...,
)
    regression = _fit_scenario_target_regression(
        contextual_data_set,
        target_component;
        validate_fixed_components=validate_fixed_components,
    )
    split = _cart_train_test_split(
        regression.contexts,
        regression.targets,
        test_size,
        random_state,
    )
    requested_min_samples_leaf = _checked_cart_min_samples_leaf(min_samples_leaf)
    effective_min_samples_leaf = min(requested_min_samples_leaf, size(split.X_train, 1))

    regressor = _cart_fit_regressor(
        split.X_train,
        split.y_train;
        min_samples_leaf=effective_min_samples_leaf,
    )
    metrics = _cart_holdout_metrics(regressor, split)

    active_transform = training_prediction_transform === nothing ?
        _zero_penalty_training_transform :
        training_prediction_transform
    leaf_ids = _cart_leaf_ids(regressor, split.X_train)
    leaf_coefficients = Dict{Int,Matrix{Float64}}()
    leaf_optimization_results = Dict{Int,Any}()
    leaf_counts = Dict{Int,Int}()

    for leaf_id in sort(unique(leaf_ids))
        local_positions = findall(==(leaf_id), leaf_ids)
        local_indices = split.train_indices[local_positions]
        local_contexts = regression.contexts[local_indices, :]
        local_templates = regression.scenario_templates[local_indices]
        coefficients = copy(regression.coefficients)
        optimization_result = nothing

        if optimize
            coefficients, optimization_result = _fit_decision_focused_coefficients(
                regression.coefficients,
                local_contexts,
                local_templates,
                regression.target_component,
                regression.target_length,
                solver,
                program,
                parametric_decoder,
                active_transform;
                optimizer=optimizer,
                optimizer_options=optimizer_options,
                mu=mu,
                rho=rho,
                kwargs...,
            )
        end

        leaf_coefficients[leaf_id] = coefficients
        leaf_optimization_results[leaf_id] = optimization_result
        leaf_counts[leaf_id] = length(local_positions)
    end

    return M5ADPolicy(
        regressor,
        leaf_coefficients,
        regression.coefficients,
        first(regression.scenario_templates),
        regression.target_component,
        regression.target_length,
        size(regression.contexts, 2),
        solver,
        program,
        parametric_decoder,
        postprocess_prediction,
        active_transform,
        leaf_optimization_results,
        mu,
        rho,
        (; kwargs...),
        (;
            min_samples_leaf=requested_min_samples_leaf,
            effective_min_samples_leaf=effective_min_samples_leaf,
            test_size=split.test_size,
            random_state=random_state,
            score=metrics.score,
            mean_squared_error=metrics.mean_squared_error,
            leaf_count=length(leaf_coefficients),
            leaf_sample_counts=leaf_counts,
            train_indices=split.train_indices,
            test_indices=split.test_indices,
            target_component=regression.target_component,
            optimize=optimize,
        ),
    )
end

function infer(policy::M5ADPolicy, context)
    leaf_id = _cart_leaf_id(policy.regressor, context, policy.feature_count)
    haskey(policy.leaf_coefficients, leaf_id) ||
        throw(ArgumentError("CART routed context to unknown leaf $(leaf_id)."))
    target_vector = _processed_prediction(
        policy.postprocess_prediction,
        _predict_target(policy.leaf_coefficients[leaf_id], context),
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
        contexts=observations.contexts,
        targets=observations.targets,
        scenario_templates=observations.scenario_templates,
        target_component=checked_target_component,
        target_length=size(observations.targets, 2),
    )
end

function _decision_focused_linear_objective(
    flat_coefficients,
    coefficient_size,
    contexts,
    scenario_templates,
    target_component,
    target_length,
    solver,
    program,
    parametric_decoder,
    training_prediction_transform;
    mu,
    rho,
    kwargs...,
)
    coefficients = reshape(collect(flat_coefficients), coefficient_size)
    total = 0.0
    for index in axes(contexts, 1)
        raw_prediction = _predict_target(coefficients, view(contexts, index, :))
        processed_prediction = _processed_training_prediction(
            training_prediction_transform,
            raw_prediction,
            target_length,
        )
        predicted_scenario = _scenario_from_target_vector(
            scenario_templates[index],
            target_component,
            processed_prediction.target,
        )
        decision = _solve_scenario_collection(
            solver,
            program,
            parametric_decoder,
            [predicted_scenario];
            mu=mu,
            rho=rho,
            kwargs...,
        )
        total += _scenario_collection_cost(
            solver,
            program,
            parametric_decoder,
            decision,
            [scenario_templates[index]];
            mu=mu,
            rho=rho,
            kwargs...,
        )
        total += processed_prediction.penalty
    end
    return total
end

function _fit_decision_focused_coefficients(
    initial_coefficients,
    contexts,
    scenario_templates,
    target_component,
    target_length,
    solver,
    program,
    parametric_decoder,
    training_prediction_transform;
    optimizer,
    optimizer_options,
    mu,
    rho,
    kwargs...,
)
    objective = θ -> _decision_focused_linear_objective(
        θ,
        size(initial_coefficients),
        contexts,
        scenario_templates,
        target_component,
        target_length,
        solver,
        program,
        parametric_decoder,
        training_prediction_transform;
        mu=mu,
        rho=rho,
        kwargs...,
    )
    optimization_result = Optim.optimize(
        objective,
        vec(copy(initial_coefficients)),
        optimizer,
        optimizer_options,
    )
    coefficients = reshape(
        collect(Optim.minimizer(optimization_result)),
        size(initial_coefficients),
    )
    return coefficients, optimization_result
end

function _lex_spo_linear_objective(
    flat_coefficients,
    coefficient_size,
    contexts,
    scenario_templates,
    target_component,
    target_length,
    solver,
    program,
    parametric_decoder,
    training_prediction_transform;
    lex_objective_atol,
    lex_objective_rtol,
    lex_variable_atol,
    kwargs...,
)
    coefficients = reshape(collect(flat_coefficients), coefficient_size)
    total = 0.0
    for index in axes(contexts, 1)
        raw_prediction = _predict_target(coefficients, view(contexts, index, :))
        processed_prediction = _processed_training_prediction(
            training_prediction_transform,
            raw_prediction,
            target_length,
        )
        predicted_scenario = _scenario_from_target_vector(
            scenario_templates[index],
            target_component,
            processed_prediction.target,
        )
        decision = _lex_solve_scenario_collection(
            solver,
            program,
            parametric_decoder,
            [predicted_scenario];
            lex_objective_atol=lex_objective_atol,
            lex_objective_rtol=lex_objective_rtol,
            lex_variable_atol=lex_variable_atol,
            kwargs...,
        )
        true_cost = _scenario_collection_cost(
            solver,
            program,
            parametric_decoder,
            decision,
            [scenario_templates[index]];
            mu=0,
            rho=0,
            kwargs...,
        )
        total += true_cost + processed_prediction.penalty
    end
    return total / size(contexts, 1)
end

function _fit_lex_spo_coefficients(
    initial_coefficients,
    contexts,
    scenario_templates,
    target_component,
    target_length,
    solver,
    program,
    parametric_decoder,
    training_prediction_transform;
    optimizer,
    optimizer_options,
    lex_objective_atol,
    lex_objective_rtol,
    lex_variable_atol,
    kwargs...,
)
    objective = θ -> _lex_spo_linear_objective(
        θ,
        size(initial_coefficients),
        contexts,
        scenario_templates,
        target_component,
        target_length,
        solver,
        program,
        parametric_decoder,
        training_prediction_transform;
        lex_objective_atol=lex_objective_atol,
        lex_objective_rtol=lex_objective_rtol,
        lex_variable_atol=lex_variable_atol,
        kwargs...,
    )
    optimization_result = Optim.optimize(
        objective,
        vec(copy(initial_coefficients)),
        optimizer,
        optimizer_options,
    )
    coefficients = reshape(
        collect(Optim.minimizer(optimization_result)),
        size(initial_coefficients),
    )
    return coefficients, optimization_result
end

_zero_penalty_training_transform(target) = (; target=target, penalty=0.0)

function _processed_training_prediction(
    training_prediction_transform,
    target_vector,
    target_length::Integer,
)
    raw_result = training_prediction_transform(collect(target_vector))
    target = raw_result
    penalty = 0.0

    if raw_result isa NamedTuple && hasproperty(raw_result, :target)
        target = raw_result.target
        penalty = hasproperty(raw_result, :penalty) ? Float64(raw_result.penalty) : 0.0
    elseif raw_result isa Tuple && length(raw_result) == 2
        target, penalty = raw_result
        penalty = Float64(penalty)
    end

    processed_vector = _numeric_feature_vector(target; name=:training_prediction_transform)
    length(processed_vector) == target_length ||
        throw(DimensionMismatch(
            "training_prediction_transform returned length $(length(processed_vector)); expected $target_length.",
        ))
    isfinite(penalty) ||
        throw(DomainError(penalty, "training_prediction_transform returned a non-finite penalty."))

    return (; target=processed_vector, penalty=penalty)
end

function _scenario_collection_cost(
    solver,
    program,
    parametric_decoder,
    decision,
    scenario_parameters::AbstractVector{<:ContextualDFL.ParametricScenario};
    mu=0,
    rho=0,
    kwargs...,
)
    checked_scenarios = _checked_scenario_collection(scenario_parameters)
    W_eq, W_ineq, T_eq, T_ineq, h_eq, h_ineq, q =
        ContextualDFL.decode_scenario_collection(parametric_decoder, checked_scenarios)

    return Float64(ContextualDFL.cost_function(
        program,
        solver,
        decision,
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
    ))
end

function _fit_cart_target_regressor(
    contextual_data_set,
    target_component;
    validate_fixed_components,
    min_samples_leaf,
    test_size,
    random_state,
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

    split = _cart_train_test_split(
        observations.contexts,
        observations.targets,
        test_size,
        random_state,
    )
    requested_min_samples_leaf = _checked_cart_min_samples_leaf(min_samples_leaf)
    effective_min_samples_leaf = min(requested_min_samples_leaf, size(split.X_train, 1))

    regressor = _cart_fit_regressor(
        split.X_train,
        split.y_train;
        min_samples_leaf=effective_min_samples_leaf,
    )

    metrics = _cart_holdout_metrics(regressor, split)

    return (;
        regressor=regressor,
        scenario_templates=observations.scenario_templates,
        target_component=checked_target_component,
        target_length=size(observations.targets, 2),
        metadata=(;
            min_samples_leaf=requested_min_samples_leaf,
            effective_min_samples_leaf=effective_min_samples_leaf,
            test_size=split.test_size,
            random_state=random_state,
            score=metrics.score,
            mean_squared_error=metrics.mean_squared_error,
        ),
    )
end

struct _JuliaCARTNode
    prediction::Vector{Float64}
    feature::Int
    threshold::Float64
    left::Any
    right::Any
    leaf_id::Int
end

struct _JuliaCARTRegressor
    root::_JuliaCARTNode
    feature_count::Int
    target_count::Int
    min_samples_leaf::Int
end

function _cart_fit_regressor(
    contexts::AbstractMatrix,
    targets::AbstractMatrix;
    min_samples_leaf::Integer,
)
    size(contexts, 1) == size(targets, 1) ||
        throw(DimensionMismatch("CART contexts and targets must have the same row count."))
    size(contexts, 1) > 0 || throw(ArgumentError("CART requires at least one observation."))
    checked_min_leaf = _checked_cart_min_samples_leaf(min_samples_leaf)
    leaf_counter = Ref(0)
    root = _cart_build_node(
        Matrix{Float64}(contexts),
        Matrix{Float64}(targets),
        collect(1:size(contexts, 1)),
        checked_min_leaf,
        leaf_counter,
    )
    return _JuliaCARTRegressor(
        root,
        size(contexts, 2),
        size(targets, 2),
        checked_min_leaf,
    )
end

function _cart_build_node(
    contexts::AbstractMatrix,
    targets::AbstractMatrix,
    indices::Vector{Int},
    min_samples_leaf::Integer,
    leaf_counter::Base.RefValue{Int},
)
    prediction = _cart_target_mean(targets, indices)
    parent_sse = _cart_sse(targets, indices, prediction)
    split = length(indices) >= 2 * min_samples_leaf ?
            _cart_best_split(contexts, targets, indices, min_samples_leaf, parent_sse) :
            nothing

    if split === nothing
        leaf_counter[] += 1
        return _JuliaCARTNode(prediction, 0, 0.0, nothing, nothing, leaf_counter[])
    end

    left = _cart_build_node(
        contexts,
        targets,
        split.left_indices,
        min_samples_leaf,
        leaf_counter,
    )
    right = _cart_build_node(
        contexts,
        targets,
        split.right_indices,
        min_samples_leaf,
        leaf_counter,
    )
    return _JuliaCARTNode(prediction, split.feature, split.threshold, left, right, 0)
end

function _cart_best_split(
    contexts::AbstractMatrix,
    targets::AbstractMatrix,
    indices::Vector{Int},
    min_samples_leaf::Integer,
    parent_sse::Real,
)
    best_feature = 0
    best_threshold = 0.0
    best_score = Inf
    best_left = Int[]
    best_right = Int[]

    for feature in axes(contexts, 2)
        values = sort(unique(Float64(contexts[index, feature]) for index in indices))
        length(values) > 1 || continue

        for value_index in 1:(length(values) - 1)
            threshold = 0.5 * (values[value_index] + values[value_index + 1])
            left_indices = [index for index in indices if contexts[index, feature] <= threshold]
            right_indices = [index for index in indices if contexts[index, feature] > threshold]
            length(left_indices) >= min_samples_leaf || continue
            length(right_indices) >= min_samples_leaf || continue

            score = _cart_sse(targets, left_indices) + _cart_sse(targets, right_indices)
            if score < best_score
                best_feature = Int(feature)
                best_threshold = threshold
                best_score = score
                best_left = left_indices
                best_right = right_indices
            end
        end
    end

    best_feature == 0 && return nothing
    best_score < Float64(parent_sse) - 1e-12 || return nothing
    return (;
        feature=best_feature,
        threshold=best_threshold,
        left_indices=best_left,
        right_indices=best_right,
    )
end

function _cart_target_mean(targets::AbstractMatrix, indices::AbstractVector{<:Integer})
    prediction = zeros(Float64, size(targets, 2))
    for index in indices
        prediction .+= view(targets, index, :)
    end
    prediction ./= length(indices)
    return prediction
end

function _cart_sse(
    targets::AbstractMatrix,
    indices::AbstractVector{<:Integer},
    prediction=_cart_target_mean(targets, indices),
)
    total = 0.0
    for index in indices
        for target_index in axes(targets, 2)
            total += abs2(targets[index, target_index] - prediction[target_index])
        end
    end
    return total
end

function _cart_centered_sum_of_squares(targets::AbstractMatrix)
    indices = collect(axes(targets, 1))
    isempty(indices) && return 0.0
    return _cart_sse(targets, indices)
end

function _cart_predict_matrix(regressor::_JuliaCARTRegressor, contexts::AbstractMatrix)
    size(contexts, 2) == regressor.feature_count ||
        throw(DimensionMismatch(
            "CART contexts have $(size(contexts, 2)) features; expected $(regressor.feature_count).",
        ))
    predictions = zeros(Float64, size(contexts, 1), regressor.target_count)
    for row in axes(contexts, 1)
        leaf = _cart_leaf_node(regressor.root, view(contexts, row, :))
        predictions[row, :] = leaf.prediction
    end
    return predictions
end

function _cart_apply_matrix(regressor::_JuliaCARTRegressor, contexts::AbstractMatrix)
    size(contexts, 2) == regressor.feature_count ||
        throw(DimensionMismatch(
            "CART contexts have $(size(contexts, 2)) features; expected $(regressor.feature_count).",
        ))
    leaf_ids = Vector{Int}(undef, size(contexts, 1))
    for row in axes(contexts, 1)
        leaf_ids[row] = _cart_leaf_node(regressor.root, view(contexts, row, :)).leaf_id
    end
    return leaf_ids
end

function _cart_leaf_node(node::_JuliaCARTNode, context)
    current = node
    while current.feature != 0
        current = context[current.feature] <= current.threshold ? current.left : current.right
    end
    return current
end

function _cart_train_test_split(contexts, targets, test_size, random_state)
    checked_test_size = _checked_cart_test_size(test_size)
    all_indices = collect(1:size(contexts, 1))
    if checked_test_size === nothing
        return (;
            X_train=Matrix{Float64}(contexts),
            X_test=zeros(Float64, 0, size(contexts, 2)),
            y_train=Matrix{Float64}(targets),
            y_test=zeros(Float64, 0, size(targets, 2)),
            train_indices=all_indices,
            test_indices=Int[],
            test_size=nothing,
        )
    end

    n_observations = length(all_indices)
    n_test = checked_test_size isa Integer ?
             Int(checked_test_size) :
             ceil(Int, Float64(checked_test_size) * n_observations)
    0 < n_test < n_observations ||
        throw(ArgumentError("test_size leaves no observations for one of the CART splits."))

    rng = Random.MersenneTwister(Int(random_state))
    shuffled = copy(all_indices)
    Random.shuffle!(rng, shuffled)
    test_indices = shuffled[1:n_test]
    train_indices = shuffled[(n_test + 1):end]

    return (;
        X_train=Matrix{Float64}(contexts[train_indices, :]),
        X_test=Matrix{Float64}(contexts[test_indices, :]),
        y_train=Matrix{Float64}(targets[train_indices, :]),
        y_test=Matrix{Float64}(targets[test_indices, :]),
        train_indices=Int.(train_indices),
        test_indices=Int.(test_indices),
        test_size=checked_test_size,
    )
end

function _cart_holdout_metrics(regressor, split)
    isempty(split.X_test) && return (; score=nothing, mean_squared_error=nothing)

    prediction = _cart_predict_matrix(regressor, split.X_test)
    residual_sum = sum(abs2, split.y_test .- prediction)
    centered_sum = _cart_centered_sum_of_squares(split.y_test)
    score = iszero(centered_sum) ? (iszero(residual_sum) ? 1.0 : 0.0) :
            1.0 - residual_sum / centered_sum
    return (;
        score=Float64(score),
        mean_squared_error=Float64(residual_sum / length(split.y_test)),
    )
end

function _cart_predict(regressor, context, target_length::Integer)
    context_vector = Float64.(collect(context))
    isempty(context_vector) &&
        throw(ArgumentError("contexts must contain at least one feature."))
    prediction = _cart_predict_matrix(regressor, reshape(context_vector, 1, length(context_vector)))
    target_vector = vec(prediction)
    length(target_vector) == target_length ||
        throw(DimensionMismatch(
            "CART prediction returned length $(length(target_vector)); expected $target_length.",
        ))
    return target_vector
end

function _cart_leaf_ids(regressor, contexts::AbstractMatrix)
    return _cart_apply_matrix(regressor, Matrix{Float64}(contexts))
end

function _cart_leaf_id(regressor, context, feature_count::Integer)
    context_vector = Float64.(collect(context))
    isempty(context_vector) &&
        throw(ArgumentError("contexts must contain at least one feature."))
    length(context_vector) == feature_count ||
        throw(DimensionMismatch(
            "context has length $(length(context_vector)); expected $(Int(feature_count)).",
        ))
    return only(_cart_leaf_ids(regressor, reshape(context_vector, 1, length(context_vector))))
end

function _checked_cart_min_samples_leaf(min_samples_leaf)
    min_samples_leaf isa Integer ||
        throw(ArgumentError("min_samples_leaf must be an integer."))
    checked = Int(min_samples_leaf)
    checked > 0 || throw(ArgumentError("min_samples_leaf must be positive."))
    return checked
end

function _checked_cart_test_size(test_size)
    (test_size === nothing || test_size === false || test_size == 0) && return nothing
    if test_size isa Integer
        checked = Int(test_size)
        checked > 0 || throw(ArgumentError("test_size must be positive."))
        return checked
    elseif test_size isa Real
        checked = Float64(test_size)
        0.0 < checked < 1.0 ||
            throw(ArgumentError("floating-point test_size must be in (0, 1)."))
        return checked
    end

    throw(ArgumentError("test_size must be nothing, false, zero, a positive integer, or a float in (0, 1)."))
end

function _scenario_target_observations(contextual_data_set, target_component)
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
    components = _target_component_tuple(target_component)
    isempty(components) &&
        throw(ArgumentError("target_component must contain at least one component."))
    length(unique(components)) == length(components) ||
        throw(ArgumentError("target_component must not contain duplicate components."))

    for component in components
        component in PARAMETRIC_SCENARIO_COMPONENTS ||
            throw(ArgumentError(
                "target_component must use components from $(PARAMETRIC_SCENARIO_COMPONENTS); got $(repr(component)).",
            ))
    end

    return length(components) == 1 ? only(components) : components
end

function _checked_lex_spo_target_component(target_component)
    checked_target_component = _checked_target_component(target_component)
    components = _target_component_tuple(checked_target_component)
    allowed_components = (:h_eq_xi, :h_ineq_xi)
    unsupported_components = setdiff(components, allowed_components)
    isempty(unsupported_components) ||
        throw(ArgumentError(
            "LexSPOLinearPolicy currently supports only RHS target components " *
            "(:h_eq_xi, :h_ineq_xi); got $(repr(checked_target_component)).",
        ))
    return checked_target_component
end

function _target_component_tuple(target_component)
    if target_component isa Symbol || target_component isa AbstractString
        return (Symbol(target_component),)
    end

    return Tuple(Symbol(component) for component in target_component)
end

function _target_feature_vector(scenario, target_component::Symbol)
    target = getproperty(scenario, target_component)
    vector = _numeric_feature_vector(target; name=target_component)
    isempty(vector) &&
        throw(ArgumentError("target component $(target_component) must not be empty."))
    return vector
end

function _target_feature_vector(scenario, target_component)
    components = _target_component_tuple(target_component)
    vectors = [
        _target_feature_vector(scenario, component)
        for component in components
    ]
    return reduce(vcat, vectors)
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
    target_component,
    target_vector::AbstractVector,
)
    replacements = _scenario_component_replacements(
        scenario_template,
        target_component,
        target_vector,
    )

    return ContextualDFL.ParametricScenario(;
        W_eq_xi=_scenario_component_value(scenario_template, replacements, :W_eq_xi),
        W_ineq_xi=_scenario_component_value(scenario_template, replacements, :W_ineq_xi),
        T_eq_xi=_scenario_component_value(scenario_template, replacements, :T_eq_xi),
        T_ineq_xi=_scenario_component_value(scenario_template, replacements, :T_ineq_xi),
        h_eq_xi=_scenario_component_value(scenario_template, replacements, :h_eq_xi),
        h_ineq_xi=_scenario_component_value(scenario_template, replacements, :h_ineq_xi),
        q_xi=_scenario_component_value(scenario_template, replacements, :q_xi),
    )
end

function _scenario_component_replacements(
    scenario_template,
    target_component,
    target_vector::AbstractVector,
)
    components = _target_component_tuple(target_component)
    replacements = Dict{Symbol,Any}()
    offset = 1

    for component in components
        template_value = getproperty(scenario_template, component)
        component_length = length(_numeric_feature_vector(template_value; name=component))
        next_offset = offset + component_length
        replacements[component] = _reshape_target_vector(
            view(target_vector, offset:(next_offset - 1)),
            template_value,
            component,
        )
        offset = next_offset
    end

    offset == length(target_vector) + 1 ||
        throw(DimensionMismatch(
            "target vector has length $(length(target_vector)); expected $(offset - 1).",
        ))

    return replacements
end

function _scenario_component_value(scenario_template, replacements, component::Symbol)
    return haskey(replacements, component) ?
           replacements[component] :
           _copy_scenario_component(getproperty(scenario_template, component))
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

function _check_fixed_scenario_components(scenario_templates, target_component)
    target_components = _target_component_tuple(target_component)
    base_scenario = first(scenario_templates)
    for scenario in Iterators.drop(scenario_templates, 1)
        for component in PARAMETRIC_SCENARIO_COMPONENTS
            component in target_components && continue
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

function _lex_solve_scenario_collection(
    solver,
    program,
    parametric_decoder,
    scenario_parameters::AbstractVector{<:ContextualDFL.ParametricScenario};
    lex_objective_atol=1e-7,
    lex_objective_rtol=1e-7,
    lex_variable_atol=1e-7,
    kwargs...,
)
    _check_lex_spo_tolerances(lex_objective_atol, lex_objective_rtol, lex_variable_atol)
    _check_lex_spo_solve_kwargs((; kwargs...))
    checked_scenarios = _checked_scenario_collection(scenario_parameters)
    W_eq, W_ineq, T_eq, T_ineq, h_eq, h_ineq, q =
        ContextualDFL.decode_scenario_collection(parametric_decoder, checked_scenarios)
    lp = ContextualDFL.construct_lp(
        program,
        W_eq,
        W_ineq,
        T_eq,
        T_ineq,
        h_eq,
        h_ineq,
        q,
    )

    base_result = _lex_solve_lp_or_throw(solver, lp; kwargs...)
    base_objective = Float64(base_result.objective_value)
    isfinite(base_objective) ||
        throw(ArgumentError("Lex-SPO base LP returned non-finite objective value $(base_objective)."))
    optimal_ub = base_objective +
                 Float64(lex_objective_atol) +
                 Float64(lex_objective_rtol) * abs(base_objective)

    first_stage_count = length(program.first_stage_lp.c)
    first_stage_count == 0 && return Float64[]

    fixed_z = zeros(Float64, first_stage_count)
    n_variables = length(lp.c)
    for coordinate in 1:first_stage_count
        coordinate_objective = zeros(Float64, n_variables)
        coordinate_objective[coordinate] = 1.0
        lex_lp = _lex_lp_with_coordinate_fixes(
            lp,
            coordinate_objective,
            optimal_ub,
            fixed_z,
            coordinate - 1,
            Float64(lex_variable_atol),
        )
        lex_result = _lex_solve_lp_or_throw(solver, lex_lp; kwargs...)
        length(lex_result.z) >= first_stage_count ||
            throw(ArgumentError(
                "Lex-SPO LP returned $(length(lex_result.z)) variables; " *
                "expected at least $(first_stage_count).",
            ))
        fixed_z[coordinate] = Float64(lex_result.z[coordinate])
    end

    return collect(fixed_z)
end

function _lex_solve_lp_or_throw(solver, lp; kwargs...)
    try
        result = ContextualDFL.solve(solver, lp; μ=0, ρ=0, kwargs...)
        status = hasproperty(result, :status) ? string(result.status) : ""
        status in ("OPTIMAL", "LOCALLY_SOLVED") ||
            throw(ArgumentError("Lex-SPO LP solve failed with status $(status)."))
        return result
    catch error
        error isa ArgumentError && rethrow()
        throw(ArgumentError("Lex-SPO LP solve failed: $(sprint(showerror, error))"))
    end
end

function _lex_lp_with_coordinate_fixes(
    lp,
    objective,
    optimal_objective_upper_bound,
    fixed_z,
    fixed_count::Integer,
    lex_variable_atol,
)
    n_variables = length(lp.c)
    objective_row = ContextualDFL.sparse(reshape(Float64.(lp.c), 1, n_variables))
    fix_rows, fix_rhs = _lex_coordinate_fix_rows(
        fixed_z,
        fixed_count,
        n_variables,
        lex_variable_atol,
    )

    return ContextualDFL.LP(
        ContextualDFL.sparse(lp.A_eq),
        ContextualDFL.sparse(vcat(
            ContextualDFL.sparse(lp.A_ineq),
            objective_row,
            fix_rows,
        )),
        Float64.(lp.b_eq),
        vcat(Float64.(lp.b_ineq), [Float64(optimal_objective_upper_bound)], fix_rhs),
        Float64.(objective),
    )
end

function _lex_coordinate_fix_rows(
    fixed_z,
    fixed_count::Integer,
    n_variables::Integer,
    lex_variable_atol,
)
    checked_fixed_count = Int(fixed_count)
    checked_fixed_count >= 0 ||
        throw(ArgumentError("fixed_count must be non-negative."))
    checked_fixed_count <= length(fixed_z) ||
        throw(DimensionMismatch("fixed_count exceeds the number of fixed coordinates."))
    checked_fixed_count == 0 &&
        return ContextualDFL.spzeros(Float64, 0, n_variables), Float64[]

    rows = ContextualDFL.spzeros(Float64, 2 * checked_fixed_count, n_variables)
    rhs = zeros(Float64, 2 * checked_fixed_count)

    for coordinate in 1:checked_fixed_count
        upper_row = 2 * coordinate - 1
        lower_row = 2 * coordinate

        rows[upper_row, coordinate] = 1.0
        rhs[upper_row] = Float64(fixed_z[coordinate]) + Float64(lex_variable_atol)
        rows[lower_row, coordinate] = -1.0
        rhs[lower_row] = -Float64(fixed_z[coordinate]) + Float64(lex_variable_atol)
    end

    return rows, rhs
end

function _check_lex_spo_tolerances(
    lex_objective_atol,
    lex_objective_rtol,
    lex_variable_atol,
)
    for (name, value) in (
        (:lex_objective_atol, lex_objective_atol),
        (:lex_objective_rtol, lex_objective_rtol),
        (:lex_variable_atol, lex_variable_atol),
    )
        checked_value = Float64(value)
        isfinite(checked_value) && checked_value >= 0.0 ||
            throw(ArgumentError("$(name) must be finite and non-negative."))
    end
    return nothing
end

function _check_lex_spo_solve_kwargs(solve_kwargs)
    unsupported = [key for key in keys(solve_kwargs) if key in (:mu, :rho, :μ, :ρ)]
    isempty(unsupported) ||
        throw(ArgumentError(
            "LexSPOLinearPolicy currently supports only unsmoothed LP solves; " *
            "do not pass $(unsupported).",
        ))
    return nothing
end
