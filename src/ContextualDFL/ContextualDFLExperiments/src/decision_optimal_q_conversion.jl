import LinearAlgebra

# Decision-equivalent dataset conversion
#
# This file has a deliberately small public surface and a fairly large internal
# implementation. Experiment code should usually use `make_decision_equivalent_dataset`
# and ignore the helper functions unless debugging the conversion itself.
#
# The intended workflow is:
#
#     payload = make_decision_equivalent_dataset(
#         :q, # or :h / :rhs / :decision_h
#         dataset,
#         solver,
#         program,
#         original_decoder,
#         base_scenario;
#         equality_form=true,
#         constraint_tolerance=1e-10,
#     )
#
# Then pass `payload.converted_dataset` and `payload.decoder` to the downstream
# method. The conversion path is method-agnostic: it does not assume SPO+,
# DFL-Scen, or any particular training loss. The SPO+/DFL-Scen preparation
# helpers later in this file are optional conveniences layered on top.
#
# With `equality_form=true`, the original scenarios are first decoded and
# converted to equality-plus-nonnegativity recourse form. General recourse
# inequalities become equality rows with zero-cost slack variables, while
# explicit nonnegativity rows `-y_i <= 0` are kept as nonnegativity rows. Free
# recourse variables are intentionally rejected because supporting them would
# require variable splitting and would otherwise silently change the model.

const _Q_CONVERSION_UNSUPPORTED_RECOURSE =
    "decision-optimal q conversion currently supports only equality recourse plus nonnegativity."

const _SCENARIO_COMPONENT_FIELDS = (:W_eq, :W_ineq, :T_eq, :T_ineq, :h_eq, :h_ineq, :q)
const _PARAMETRIC_SCENARIO_COMPONENT_FIELDS =
    (:W_eq_xi, :W_ineq_xi, :T_eq_xi, :T_ineq_xi, :h_eq_xi, :h_ineq_xi, :q_xi)

function _normalized_probabilities(K::Integer, probabilities)::Vector{Float64}
    K > 0 || throw(ArgumentError("K must be positive."))

    if probabilities === nothing
        return fill(1.0 / K, K)
    end

    length(probabilities) == K ||
        throw(DimensionMismatch("probabilities must have one entry per scenario."))

    p = Float64.(collect(probabilities))
    all(isfinite, p) || throw(ArgumentError("probabilities must be finite."))
    all(>=(0.0), p) || throw(ArgumentError("probabilities must be nonnegative."))

    total = sum(p)
    total > 0.0 || throw(ArgumentError("probabilities must have positive sum."))

    return p ./ total
end

function _assert_supported_recourse_form(
    W_ineq_array,
    T_ineq_array,
    h_ineq_array;
    atol::Real=1e-10,
)::Nothing
    m_ineq = size(W_ineq_array, 1)
    ny = size(W_ineq_array, 2)
    K = size(W_ineq_array, 3)

    m_ineq == 0 && return nothing
    m_ineq == ny || throw(ArgumentError(_Q_CONVERSION_UNSUPPORTED_RECOURSE))

    nonnegativity_W = -Matrix{Float64}(LinearAlgebra.I, ny, ny)

    for k in 1:K
        if !isapprox(W_ineq_array[:, :, k], nonnegativity_W; atol=atol, rtol=0) ||
           !isapprox(
               T_ineq_array[:, :, k],
               zeros(size(T_ineq_array, 1), size(T_ineq_array, 2));
               atol=atol,
               rtol=0,
           ) ||
           !isapprox(h_ineq_array[:, k], zeros(size(h_ineq_array, 1)); atol=atol, rtol=0)
            throw(ArgumentError(_Q_CONVERSION_UNSUPPORTED_RECOURSE))
        end
    end

    return nothing
end

function _scenario_component_arrays(scenario)
    if all(field -> hasproperty(scenario, field), _SCENARIO_COMPONENT_FIELDS)
        return (;
            W_eq=Matrix(getproperty(scenario, :W_eq)),
            W_ineq=Matrix(getproperty(scenario, :W_ineq)),
            T_eq=Matrix(getproperty(scenario, :T_eq)),
            T_ineq=Matrix(getproperty(scenario, :T_ineq)),
            h_eq=Vector(getproperty(scenario, :h_eq)),
            h_ineq=Vector(getproperty(scenario, :h_ineq)),
            q=Vector(getproperty(scenario, :q)),
        )
    end

    if all(field -> hasproperty(scenario, field), _PARAMETRIC_SCENARIO_COMPONENT_FIELDS)
        return (;
            W_eq=Matrix(getproperty(scenario, :W_eq_xi)),
            W_ineq=Matrix(getproperty(scenario, :W_ineq_xi)),
            T_eq=Matrix(getproperty(scenario, :T_eq_xi)),
            T_ineq=Matrix(getproperty(scenario, :T_ineq_xi)),
            h_eq=Vector(getproperty(scenario, :h_eq_xi)),
            h_ineq=Vector(getproperty(scenario, :h_ineq_xi)),
            q=Vector(getproperty(scenario, :q_xi)),
        )
    end

    throw(ArgumentError(
        "scenario must expose either fields $(_SCENARIO_COMPONENT_FIELDS) or " *
        "$(_PARAMETRIC_SCENARIO_COMPONENT_FIELDS).",
    ))
end

function _scenario_arrays_to_parametric(arrays)::ContextualDFL.ParametricScenario
    return ContextualDFL.ParametricScenario(;
        W_eq_xi=copy(arrays.W_eq),
        W_ineq_xi=copy(arrays.W_ineq),
        T_eq_xi=copy(arrays.T_eq),
        T_ineq_xi=copy(arrays.T_ineq),
        h_eq_xi=copy(arrays.h_eq),
        h_ineq_xi=copy(arrays.h_ineq),
        q_xi=copy(arrays.q),
    )
end

function _decoded_scenario_arrays(arrays, k::Integer)
    return (;
        W_eq=Matrix(arrays.W_eq_array[:, :, k]),
        W_ineq=Matrix(arrays.W_ineq_array[:, :, k]),
        T_eq=Matrix(arrays.T_eq_array[:, :, k]),
        T_ineq=Matrix(arrays.T_ineq_array[:, :, k]),
        h_eq=Vector(arrays.h_eq_array[:, k]),
        h_ineq=Vector(arrays.h_ineq_array[:, k]),
        q=Vector(arrays.q_array[:, k]),
    )
end

function _nonnegative_row_variable(W_row, T_row, h_value; atol::Real)
    isapprox(h_value, 0.0; atol=atol, rtol=0) || return nothing
    all(value -> isapprox(value, 0.0; atol=atol, rtol=0), T_row) || return nothing

    negative_unit_indices = findall(value -> isapprox(value, -1.0; atol=atol, rtol=0), W_row)
    length(negative_unit_indices) == 1 || return nothing

    variable_index = only(negative_unit_indices)
    for j in eachindex(W_row)
        j == variable_index && continue
        isapprox(W_row[j], 0.0; atol=atol, rtol=0) || return nothing
    end

    return variable_index
end

function _recourse_inequality_row_partition(W_ineq, T_ineq, h_ineq; atol::Real)
    ny = size(W_ineq, 2)
    m_ineq = size(W_ineq, 1)

    nonnegative_rows = falses(m_ineq)
    covered_variables = falses(ny)

    for row_index in 1:m_ineq
        variable_index = _nonnegative_row_variable(
            view(W_ineq, row_index, :),
            view(T_ineq, row_index, :),
            h_ineq[row_index];
            atol=atol,
        )
        if variable_index !== nothing
            nonnegative_rows[row_index] = true
            covered_variables[variable_index] = true
        end
    end

    missing_variables = findall(!, covered_variables)
    isempty(missing_variables) || throw(ArgumentError(
        "equality-form conversion requires every original recourse variable to have " *
        "a nonnegativity row -e_i' y <= 0. Missing variable index/indices: " *
        "$(join(missing_variables, ", ")). Free recourse-variable splitting is not supported.",
    ))

    return findall(!, nonnegative_rows)
end

function _convert_scenario_arrays_to_equality_form(arrays; atol::Real=1e-10)
    W_eq = Matrix(arrays.W_eq)
    W_ineq = Matrix(arrays.W_ineq)
    T_eq = Matrix(arrays.T_eq)
    T_ineq = Matrix(arrays.T_ineq)
    h_eq = Vector(arrays.h_eq)
    h_ineq = Vector(arrays.h_ineq)
    q = Vector(arrays.q)

    ny = length(q)
    nz = size(T_eq, 2)
    general_rows = _recourse_inequality_row_partition(W_ineq, T_ineq, h_ineq; atol=atol)
    slack_count = length(general_rows)
    ny_new = ny + slack_count
    m_eq_new = size(W_eq, 1) + slack_count

    T = promote_type(
        eltype(W_eq),
        eltype(W_ineq),
        eltype(T_eq),
        eltype(T_ineq),
        eltype(h_eq),
        eltype(h_ineq),
        eltype(q),
        Float64,
    )

    W_eq_new = zeros(T, m_eq_new, ny_new)
    T_eq_new = zeros(T, m_eq_new, nz)
    h_eq_new = zeros(T, m_eq_new)

    original_eq_rows = axes(W_eq, 1)
    W_eq_new[original_eq_rows, 1:ny] .= W_eq
    T_eq_new[original_eq_rows, :] .= T_eq
    h_eq_new[original_eq_rows] .= h_eq

    for (slack_index, original_row) in enumerate(general_rows)
        converted_row = size(W_eq, 1) + slack_index
        W_eq_new[converted_row, 1:ny] .= view(W_ineq, original_row, :)
        W_eq_new[converted_row, ny + slack_index] = one(T)
        T_eq_new[converted_row, :] .= view(T_ineq, original_row, :)
        h_eq_new[converted_row] = h_ineq[original_row]
    end

    return (;
        W_eq=W_eq_new,
        W_ineq=-Matrix{T}(LinearAlgebra.I, ny_new, ny_new),
        T_eq=T_eq_new,
        T_ineq=zeros(T, ny_new, nz),
        h_eq=h_eq_new,
        h_ineq=zeros(T, ny_new),
        q=vcat(T.(q), zeros(T, slack_count)),
    )
end

function convert_base_scenario_to_equality_form(
    base_scenario;
    atol::Real=1e-10,
)::ContextualDFL.ParametricScenario
    return _scenario_arrays_to_parametric(
        _convert_scenario_arrays_to_equality_form(
            _scenario_component_arrays(base_scenario);
            atol=atol,
        ),
    )
end

function convert_datapoint_to_equality_form(
    datapoint::ContextualDFL.ContextualDataPoint,
    original_decoder;
    atol::Real=1e-10,
)::ContextualDFL.ContextualDataPoint
    decoded = ContextualDFL.decode_scenario_collection(
        original_decoder,
        datapoint.scenario_parameters,
    )
    arrays = (;
        W_eq_array=decoded[1],
        W_ineq_array=decoded[2],
        T_eq_array=decoded[3],
        T_ineq_array=decoded[4],
        h_eq_array=decoded[5],
        h_ineq_array=decoded[6],
        q_array=decoded[7],
    )
    K = size(arrays.q_array, 2)
    converted_scenarios = [
        _scenario_arrays_to_parametric(
            _convert_scenario_arrays_to_equality_form(
                _decoded_scenario_arrays(arrays, k);
                atol=atol,
            ),
        )
        for k in 1:K
    ]

    return ContextualDFL.ContextualDataPoint(datapoint.context, converted_scenarios)
end

function convert_dataset_to_equality_form(
    dataset,
    original_decoder;
    atol::Real=1e-10,
)::Vector{<:ContextualDFL.ContextualDataPoint}
    return [
        convert_datapoint_to_equality_form(dp, original_decoder; atol=atol)
        for dp in dataset
    ]
end

function decode_q_conversion_arrays(
    decoder,
    scenario_collection;
    atol::Real=1e-10,
)
    W_eq_array,
    W_ineq_array,
    T_eq_array,
    T_ineq_array,
    h_eq_array,
    h_ineq_array,
    q_array = ContextualDFL.decode_scenario_collection(decoder, scenario_collection)

    _assert_supported_recourse_form(
        W_ineq_array,
        T_ineq_array,
        h_ineq_array;
        atol=atol,
    )

    return (;
        W_eq_array,
        W_ineq_array,
        T_eq_array,
        T_ineq_array,
        h_eq_array,
        h_ineq_array,
        q_array,
    )
end

function full_base_scenario_arrays(base_scenario; atol::Real=1e-10)
    base = _scenario_component_arrays(base_scenario)

    _assert_supported_recourse_form(
        reshape(base.W_ineq, size(base.W_ineq, 1), size(base.W_ineq, 2), 1),
        reshape(base.T_ineq, size(base.T_ineq, 1), size(base.T_ineq, 2), 1),
        reshape(base.h_ineq, length(base.h_ineq), 1);
        atol=atol,
    )

    return base
end

function _weighted_average_equality_dual(λ_h_eq_array)::Vector
    return vec(sum(λ_h_eq_array; dims=2))
end

function _decision_optimal_q_label(
    solver,
    base,
    z_star,
    λ_bar;
    kwargs...,
)::Vector
    _, λ0_eq, _ = ContextualDFL.G_hat(
        solver,
        z_star,
        base.W_eq,
        base.W_ineq,
        base.T_eq,
        base.T_ineq,
        base.h_eq,
        base.h_ineq,
        base.q;
        return_dual=true,
        kwargs...,
    )

    length(λ_bar) == length(λ0_eq) ||
        throw(DimensionMismatch("base and original equality dual vectors have different lengths."))

    q_star = vec(base.q .+ transpose(base.W_eq) * (λ_bar .- λ0_eq))
    length(q_star) == length(base.q) ||
        throw(DimensionMismatch("converted q label has the wrong length."))
    all(isfinite, q_star) || throw(ArgumentError("converted q label must be finite."))

    return q_star
end

function _converted_q_scenario(base, q_star)::ContextualDFL.ParametricScenario
    return ContextualDFL.ParametricScenario(;
        W_eq_xi=copy(base.W_eq),
        W_ineq_xi=copy(base.W_ineq),
        T_eq_xi=copy(base.T_eq),
        T_ineq_xi=copy(base.T_ineq),
        h_eq_xi=copy(base.h_eq),
        h_ineq_xi=copy(base.h_ineq),
        q_xi=copy(q_star),
    )
end

function convert_datapoint_to_q(
    datapoint::ContextualDFL.ContextualDataPoint,
    solver,
    program,
    original_decoder,
    base_scenario;
    probabilities=nothing,
    atol::Real=1e-10,
    solve_kwargs...,
)::ContextualDFL.ContextualDataPoint
    arrays = decode_q_conversion_arrays(
        original_decoder,
        datapoint.scenario_parameters;
        atol=atol,
    )

    K = size(arrays.q_array, 2)
    p = _normalized_probabilities(K, probabilities)

    z_star,
    _,
    _,
    _,
    λ_h_eq_array,
    _ = ContextualDFL.solve(
        solver,
        program,
        arrays.W_eq_array,
        arrays.W_ineq_array,
        arrays.T_eq_array,
        arrays.T_ineq_array,
        arrays.h_eq_array,
        arrays.h_ineq_array,
        arrays.q_array;
        probabilities=p,
        solve_kwargs...,
    )

    λ_bar = _weighted_average_equality_dual(λ_h_eq_array)
    base = full_base_scenario_arrays(base_scenario; atol=atol)
    q_star = _decision_optimal_q_label(
        solver,
        base,
        z_star,
        λ_bar;
        solve_kwargs...,
    )

    scenario_star = _converted_q_scenario(base, q_star)
    return ContextualDFL.ContextualDataPoint(datapoint.context, [scenario_star])
end

function convert_dataset_to_q(
    dataset,
    solver,
    program,
    original_decoder,
    base_scenario;
    probabilities_by_datapoint=nothing,
    atol::Real=1e-10,
    solve_kwargs...,
)::Vector{<:ContextualDFL.ContextualDataPoint}
    converted = map(enumerate(dataset)) do (i, dp)
        probs =
            probabilities_by_datapoint === nothing ? nothing :
            probabilities_by_datapoint isa Function ? probabilities_by_datapoint(dp) :
            probabilities_by_datapoint[i]

        return convert_datapoint_to_q(
            dp,
            solver,
            program,
            original_decoder,
            base_scenario;
            probabilities=probs,
            atol=atol,
            solve_kwargs...,
        )
    end

    return converted
end

function q_lower_bound_from_converted_dataset(
    converted_dataset;
    margin::Real=1e-6,
)::Vector{Float64}
    margin >= 0 || throw(ArgumentError("margin must be nonnegative."))
    !isempty(converted_dataset) || throw(ArgumentError("converted_dataset must not be empty."))

    q_vectors = [
        Float64.(collect(only(dp.scenario_parameters).q_xi))
        for dp in converted_dataset
    ]
    Q = reduce(hcat, q_vectors)
    q_lb = vec(minimum(Q; dims=2)) .- margin

    all(isfinite, q_lb) || throw(ArgumentError("q lower bound must be finite."))
    return q_lb
end

_softplus_stable(x) = max(x, zero(x)) + log1p(exp(-abs(x)))

struct LowerBoundedQDecoder{TBase,TLower} <: ContextualDFL.VectorDecoder
    base_scenario::TBase
    q_lower_bound::TLower
end

function LowerBoundedQDecoder(
    base_scenario,
    q_lower_bound::AbstractVector;
    atol::Real=1e-10,
)
    base = full_base_scenario_arrays(base_scenario; atol=atol)
    q_lb = Float64.(collect(q_lower_bound))

    length(q_lb) == length(base.q) ||
        throw(DimensionMismatch("q_lower_bound must have one entry per recourse variable."))
    all(isfinite, q_lb) || throw(ArgumentError("q_lower_bound must be finite."))

    return LowerBoundedQDecoder{typeof(base),typeof(q_lb)}(base, q_lb)
end

function (decoder::LowerBoundedQDecoder)(raw_q::AbstractVector)
    length(raw_q) == length(decoder.q_lower_bound) ||
        throw(DimensionMismatch("raw_q must have one entry per recourse variable."))

    q = decoder.q_lower_bound .+ _softplus_stable.(raw_q)

    return (
        decoder.base_scenario.W_eq,
        decoder.base_scenario.W_ineq,
        decoder.base_scenario.T_eq,
        decoder.base_scenario.T_ineq,
        decoder.base_scenario.h_eq,
        decoder.base_scenario.h_ineq,
        q,
    )
end

function make_spoplus_q_loss(
    solver,
    program,
    base_scenario,
    q_lower_bound::AbstractVector;
    atol::Real=1e-10,
)
    input_decoder = LowerBoundedQDecoder(
        base_scenario,
        q_lower_bound;
        atol=atol,
    )
    reference_decoder = ContextualDFL.ParametricDecoder()

    return ContextualDFL.SPOPlusLoss(
        input_decoder,
        reference_decoder,
        solver,
        program;
        nr_scenarios=1,
    )
end

function prepare_spoplus_q_dataset(
    dataset,
    solver,
    program,
    original_decoder,
    base_scenario;
    probabilities_by_datapoint=nothing,
    lower_bound_margin::Real=1e-6,
    atol::Real=1e-10,
    solve_kwargs...,
)
    converted_dataset = convert_dataset_to_q(
        dataset,
        solver,
        program,
        original_decoder,
        base_scenario;
        probabilities_by_datapoint=probabilities_by_datapoint,
        atol=atol,
        solve_kwargs...,
    )
    q_lb = q_lower_bound_from_converted_dataset(
        converted_dataset;
        margin=lower_bound_margin,
    )
    spo_loss = make_spoplus_q_loss(
        solver,
        program,
        base_scenario,
        q_lb;
        atol=atol,
    )

    return (;
        converted_dataset=converted_dataset,
        q_lower_bound=q_lb,
        spo_loss=spo_loss,
    )
end

function _base_scenario_arrays_for_h_conversion(
    base_scenario;
    atol::Real=1e-10,
)
    return full_base_scenario_arrays(base_scenario; atol=atol)
end

function _decision_optimal_h_label(base, z_star)::Vector
    h_star = vec(base.T_eq * z_star)
    length(h_star) == length(base.h_eq) ||
        throw(DimensionMismatch("converted h label has the wrong length."))
    all(isfinite, h_star) || throw(ArgumentError("converted h label must be finite."))
    return h_star
end

function _converted_h_scenario(base, h_star)::ContextualDFL.ParametricScenario
    return ContextualDFL.ParametricScenario(;
        W_eq_xi=copy(base.W_eq),
        W_ineq_xi=copy(base.W_ineq),
        T_eq_xi=copy(base.T_eq),
        T_ineq_xi=copy(base.T_ineq),
        h_eq_xi=copy(h_star),
        h_ineq_xi=copy(base.h_ineq),
        q_xi=copy(base.q),
    )
end

function convert_datapoint_to_h(
    datapoint::ContextualDFL.ContextualDataPoint,
    solver,
    program,
    original_decoder,
    base_scenario;
    probabilities=nothing,
    atol::Real=1e-10,
    solve_kwargs...,
)::ContextualDFL.ContextualDataPoint
    arrays = decode_q_conversion_arrays(
        original_decoder,
        datapoint.scenario_parameters;
        atol=atol,
    )

    K = size(arrays.q_array, 2)
    p = _normalized_probabilities(K, probabilities)

    z_star,
    _,
    _,
    _,
    _,
    _ = ContextualDFL.solve(
        solver,
        program,
        arrays.W_eq_array,
        arrays.W_ineq_array,
        arrays.T_eq_array,
        arrays.T_ineq_array,
        arrays.h_eq_array,
        arrays.h_ineq_array,
        arrays.q_array;
        probabilities=p,
        solve_kwargs...,
    )

    base = _base_scenario_arrays_for_h_conversion(base_scenario; atol=atol)
    h_star = _decision_optimal_h_label(base, z_star)
    scenario_star = _converted_h_scenario(base, h_star)
    return ContextualDFL.ContextualDataPoint(datapoint.context, [scenario_star])
end

function convert_dataset_to_h(
    dataset,
    solver,
    program,
    original_decoder,
    base_scenario;
    probabilities_by_datapoint=nothing,
    atol::Real=1e-10,
    solve_kwargs...,
)::Vector{<:ContextualDFL.ContextualDataPoint}
    converted = map(enumerate(dataset)) do (i, dp)
        probs =
            probabilities_by_datapoint === nothing ? nothing :
            probabilities_by_datapoint isa Function ? probabilities_by_datapoint(dp) :
            probabilities_by_datapoint[i]

        return convert_datapoint_to_h(
            dp,
            solver,
            program,
            original_decoder,
            base_scenario;
            probabilities=probs,
            atol=atol,
            solve_kwargs...,
        )
    end

    return converted
end

function _canonical_decision_equivalent_target(target::Symbol)
    target == :q && return :q
    target in (:h, :rhs, :decision_h) && return :h
    throw(ArgumentError("unsupported decision-equivalent target $(target); use :q or :h."))
end

function _materialized_probabilities_by_datapoint(dataset, probabilities_by_datapoint)
    probabilities_by_datapoint === nothing && return nothing
    probabilities_by_datapoint isa Function &&
        return [probabilities_by_datapoint(dp) for dp in dataset]
    return probabilities_by_datapoint
end

function _parametric_base_scenario(base_scenario)::ContextualDFL.ParametricScenario
    base_scenario isa ContextualDFL.ParametricScenario && return base_scenario
    return _scenario_arrays_to_parametric(_scenario_component_arrays(base_scenario))
end

"""
    make_decision_equivalent_dataset(target, dataset, solver, program, original_decoder, base_scenario; ...)

Build a decision-equivalent training dataset for a downstream experiment.

Use `target=:q` for a q-only dataset, or `target=:h`, `:rhs`, or `:decision_h`
for an h-only dataset. By default, `equality_form=true` first converts the
decoded input scenarios and base scenario to equality-plus-nonnegativity
recourse form, then applies the requested decision-equivalent conversion.

Returns a named tuple with:

- `converted_dataset`: the dataset to train/evaluate on.
- `decoder`: always `ContextualDFL.ParametricDecoder()` for the converted data.
- `base_scenario`: the converted base scenario to use with the converted data.
- `target`: canonicalized to `:q` or `:h`.
- `diagnostics`: small shape/count metadata useful for experiment logs.

The conversion itself is independent of any learning method. After this call,
the experiment runner can choose SPO+, DFL-Scen, a baseline, or another method.
"""
function make_decision_equivalent_dataset(
    target::Symbol,
    dataset,
    solver,
    program,
    original_decoder,
    base_scenario;
    equality_form::Bool=true,
    probabilities_by_datapoint=nothing,
    atol::Real=1e-10,
    solve_kwargs...,
)
    checked_target = _canonical_decision_equivalent_target(target)
    conversion_probabilities =
        _materialized_probabilities_by_datapoint(dataset, probabilities_by_datapoint)

    source_dataset, source_decoder, converted_base_scenario = if equality_form
        (
            convert_dataset_to_equality_form(dataset, original_decoder; atol=atol),
            ContextualDFL.ParametricDecoder(),
            convert_base_scenario_to_equality_form(base_scenario; atol=atol),
        )
    else
        (dataset, original_decoder, _parametric_base_scenario(base_scenario))
    end

    converted_dataset = convert_dataset_to_decision_optimal(
        checked_target,
        source_dataset,
        solver,
        program,
        source_decoder,
        converted_base_scenario;
        probabilities_by_datapoint=conversion_probabilities,
        atol=atol,
        solve_kwargs...,
    )

    base_arrays = _scenario_component_arrays(converted_base_scenario)

    return (;
        converted_dataset=converted_dataset,
        base_scenario=converted_base_scenario,
        decoder=ContextualDFL.ParametricDecoder(),
        target=checked_target,
        diagnostics=(;
            equality_form_applied=equality_form,
            source_datapoints=length(dataset),
            converted_datapoints=length(converted_dataset),
            base_recourse_dimension=length(base_arrays.q),
            base_equality_rows=length(base_arrays.h_eq),
        ),
    )
end

struct DecisionOptimalHDecoder{TBase} <: ContextualDFL.VectorDecoder
    base_scenario::TBase
end

function DecisionOptimalHDecoder(
    base_scenario::ContextualDFL.ParametricScenario;
    atol::Real=1e-10,
)
    base = _base_scenario_arrays_for_h_conversion(base_scenario; atol=atol)
    return DecisionOptimalHDecoder{typeof(base)}(base)
end

function (decoder::DecisionOptimalHDecoder)(h_eq::AbstractVector)
    length(h_eq) == length(decoder.base_scenario.h_eq) ||
        throw(DimensionMismatch(
            "h_eq has length $(length(h_eq)); expected $(length(decoder.base_scenario.h_eq)).",
        ))

    return (
        decoder.base_scenario.W_eq,
        decoder.base_scenario.W_ineq,
        decoder.base_scenario.T_eq,
        decoder.base_scenario.T_ineq,
        collect(h_eq),
        decoder.base_scenario.h_ineq,
        decoder.base_scenario.q,
    )
end

function ChainRulesCore.rrule(
    ::typeof(ContextualDFL.decode_scenario_collection),
    decoder::DecisionOptimalHDecoder,
    h_vector::AbstractVector{<:Number};
    nr_scenarios=nothing,
)
    isnothing(nr_scenarios) &&
        throw(ArgumentError(
            "DecisionOptimalHDecoder rrule requires explicit nr_scenarios.",
        ))
    nr_scenarios isa Integer && nr_scenarios > 0 ||
        throw(ArgumentError("nr_scenarios must be a positive integer."))

    h_dimension = length(decoder.base_scenario.h_eq)
    expected_length = h_dimension * Int(nr_scenarios)
    length(h_vector) == expected_length ||
        throw(DimensionMismatch(
            "h_vector has length $(length(h_vector)); expected " *
            "$expected_length for h_dimension=$h_dimension, " *
            "nr_scenarios=$nr_scenarios.",
        ))

    output = ContextualDFL.decode_scenario_collection(
        decoder,
        h_vector;
        nr_scenarios=nr_scenarios,
    )
    project_h = ChainRulesCore.ProjectTo(h_vector)

    function decision_h_decode_pullback(output_tangent)
        dh_eq_array = ContextualDFL._array_cotangent(
            output_tangent,
            5,
            output[5];
            name=:h_eq_array,
        )
        dh_vector = vec(copy(dh_eq_array))

        return (
            ChainRulesCore.NoTangent(),
            ChainRulesCore.NoTangent(),
            project_h(dh_vector),
        )
    end

    return output, decision_h_decode_pullback
end

function _decision_h_loss_type(loss)
    isnothing(loss) && return ContextualDFL.DflScenLoss
    name = Symbol(loss)
    name == :dfl_scen && return ContextualDFL.DflScenLoss
    throw(ArgumentError(
        "unsupported decision-h loss $(loss); decision-h conversion supports :dfl_scen. " *
        "SPOPlusLoss in ContextualDFL is objective-vector-only; use decision-q for SPO+.",
    ))
end

function make_decision_h_loss(
    solver,
    program,
    base_scenario;
    loss=nothing,
    atol::Real=1e-10,
)
    input_decoder = DecisionOptimalHDecoder(base_scenario; atol=atol)
    reference_decoder = ContextualDFL.ParametricDecoder()
    loss_type = _decision_h_loss_type(loss)

    return loss_type(
        input_decoder,
        reference_decoder,
        solver,
        program;
        nr_scenarios=1,
    )
end

function prepare_decision_h_dataset(
    dataset,
    solver,
    program,
    original_decoder,
    base_scenario;
    probabilities_by_datapoint=nothing,
    loss=nothing,
    atol::Real=1e-10,
    solve_kwargs...,
)
    converted_dataset = convert_dataset_to_h(
        dataset,
        solver,
        program,
        original_decoder,
        base_scenario;
        probabilities_by_datapoint=probabilities_by_datapoint,
        atol=atol,
        solve_kwargs...,
    )
    scenario_decoder = DecisionOptimalHDecoder(base_scenario; atol=atol)
    reference_decoder = ContextualDFL.ParametricDecoder()
    training_loss = make_decision_h_loss(
        solver,
        program,
        base_scenario;
        loss=loss,
        atol=atol,
    )

    return (;
        converted_dataset=converted_dataset,
        h_dimension=length(scenario_decoder.base_scenario.h_eq),
        scenario_decoder=scenario_decoder,
        reference_scenario_decoder=reference_decoder,
        loss=training_loss,
    )
end

function prepare_decision_optimal_h_dataset(args...; kwargs...)
    return prepare_decision_h_dataset(args...; kwargs...)
end

function prepare_decision_optimal_dataset(
    target::Symbol,
    dataset,
    solver,
    program,
    original_decoder,
    base_scenario;
    probabilities_by_datapoint=nothing,
    loss=nothing,
    lower_bound_margin::Real=1e-6,
    atol::Real=1e-10,
    solve_kwargs...,
)
    if target == :q
        q_loss = isnothing(loss) ? :spo_plus : Symbol(loss)
        q_loss in (:spo_plus, :spoplus) || throw(ArgumentError(
            "decision-q preparation supports SPO+ labels; use loss=:spo_plus.",
        ))
        return prepare_spoplus_q_dataset(
            dataset,
            solver,
            program,
            original_decoder,
            base_scenario;
            probabilities_by_datapoint=probabilities_by_datapoint,
            lower_bound_margin=lower_bound_margin,
            atol=atol,
            solve_kwargs...,
        )
    end

    if target in (:h, :rhs, :decision_h)
        h_loss = isnothing(loss) ? :dfl_scen : loss
        return prepare_decision_h_dataset(
            dataset,
            solver,
            program,
            original_decoder,
            base_scenario;
            probabilities_by_datapoint=probabilities_by_datapoint,
            loss=h_loss,
            atol=atol,
            solve_kwargs...,
        )
    end

    throw(ArgumentError("unsupported decision-optimal target $(target); use :q or :h."))
end

function convert_datapoint_to_decision_optimal(
    target::Symbol,
    datapoint::ContextualDFL.ContextualDataPoint,
    args...;
    kwargs...,
)
    target == :q && return convert_datapoint_to_q(datapoint, args...; kwargs...)
    target in (:h, :rhs, :decision_h) &&
        return convert_datapoint_to_h(datapoint, args...; kwargs...)
    throw(ArgumentError("unsupported decision-optimal target $(target); use :q or :h."))
end

function convert_dataset_to_decision_optimal(
    target::Symbol,
    dataset,
    args...;
    kwargs...,
)
    target == :q && return convert_dataset_to_q(dataset, args...; kwargs...)
    target in (:h, :rhs, :decision_h) &&
        return convert_dataset_to_h(dataset, args...; kwargs...)
    throw(ArgumentError("unsupported decision-optimal target $(target); use :q or :h."))
end
