import LinearAlgebra

const _Q_CONVERSION_UNSUPPORTED_RECOURSE =
    "decision-optimal q conversion currently supports only equality recourse plus nonnegativity."

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

function full_base_scenario_arrays(
    base_scenario::ContextualDFL.ParametricScenario;
    atol::Real=1e-10,
)
    arrays = decode_q_conversion_arrays(
        ContextualDFL.ParametricDecoder(),
        [base_scenario];
        atol=atol,
    )

    return (;
        W_eq=Matrix(arrays.W_eq_array[:, :, 1]),
        W_ineq=Matrix(arrays.W_ineq_array[:, :, 1]),
        T_eq=Matrix(arrays.T_eq_array[:, :, 1]),
        T_ineq=Matrix(arrays.T_ineq_array[:, :, 1]),
        h_eq=Vector(arrays.h_eq_array[:, 1]),
        h_ineq=Vector(arrays.h_ineq_array[:, 1]),
        q=Vector(arrays.q_array[:, 1]),
    )
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
    base_scenario::ContextualDFL.ParametricScenario;
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
    base_scenario::ContextualDFL.ParametricScenario;
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
    base_scenario::ContextualDFL.ParametricScenario,
    q_lower_bound::AbstractVector;
    atol::Real=1e-10,
)
    base = full_base_scenario_arrays(base_scenario; atol=atol)
    q_lb = Float64.(collect(q_lower_bound))

    length(q_lb) == length(base.q) ||
        throw(DimensionMismatch("q_lower_bound must have one entry per recourse variable."))
    all(isfinite, q_lb) || throw(ArgumentError("q_lower_bound must be finite."))

    return LowerBoundedQDecoder(base, q_lb)
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
    base_scenario::ContextualDFL.ParametricScenario,
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
    base_scenario::ContextualDFL.ParametricScenario;
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
    base_scenario::ContextualDFL.ParametricScenario;
    atol::Real=1e-10,
)
    return full_base_scenario_arrays(base_scenario; atol=atol)
end

function _base_scenario_arrays_for_h_conversion(base_scenario; atol::Real=1e-10)
    required_fields = (:W_eq, :W_ineq, :T_eq, :T_ineq, :h_eq, :h_ineq, :q)
    missing_fields = [
        field for field in required_fields if !hasproperty(base_scenario, field)
    ]
    isempty(missing_fields) || throw(ArgumentError(
        "base_scenario is missing required field(s): $(join(String.(missing_fields), ", ")).",
    ))

    W_eq = Matrix(base_scenario.W_eq)
    W_ineq = Matrix(base_scenario.W_ineq)
    T_eq = Matrix(base_scenario.T_eq)
    T_ineq = Matrix(base_scenario.T_ineq)
    h_eq = Vector(base_scenario.h_eq)
    h_ineq = Vector(base_scenario.h_ineq)
    q = Vector(base_scenario.q)

    _assert_supported_recourse_form(
        reshape(W_ineq, size(W_ineq, 1), size(W_ineq, 2), 1),
        reshape(T_ineq, size(T_ineq, 1), size(T_ineq, 2), 1),
        reshape(h_ineq, length(h_ineq), 1);
        atol=atol,
    )

    return (;
        W_eq=W_eq,
        W_ineq=W_ineq,
        T_eq=T_eq,
        T_ineq=T_ineq,
        h_eq=h_eq,
        h_ineq=h_ineq,
        q=q,
    )
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

struct DecisionOptimalHDecoder{TBase} <: ContextualDFL.VectorDecoder
    base_scenario::TBase
end

function DecisionOptimalHDecoder(base_scenario; atol::Real=1e-10)
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
    loss=:dfl_scen,
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
    loss=:dfl_scen,
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
