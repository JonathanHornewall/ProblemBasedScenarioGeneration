function construct_lp(
    sp::StochasticProgram,
    W_eq,
    W_in,
    T_eq,
    T_in,
    h_eq,
    h_in,
    q_eq,
    q_in,
)
    scenario = BaseScenario(W_eq, W_in, T_eq, T_in, vcat(vec(h_eq), vec(h_in)), q_eq)
    return construct_lp(sp, [scenario])
end

function construct_lp(sp::StochasticProgram, scenario)
    return construct_lp(sp, _scenario_vector(scenario))
end

function construct_lp(sp::StochasticProgram, scenarios::AbstractVector)
    isempty(scenarios) && error("At least one scenario is required.")
    scen_data = [_scenario_data(sc) for sc in scenarios]
    S = length(scen_data)
    n1 = length(sp.c)
    n2 = length(scen_data[1].q)
    Tnum = promote_type(eltype(sp.c), map(d -> eltype(d.q), scen_data)..., Float64)

    for (idx, data) in enumerate(scen_data)
        length(data.q) == n2 || error("Scenario $idx has inconsistent recourse cost length")
        size(data.T_eq, 2) == n1 || error("Scenario $idx equality T has wrong first-stage dimension")
        size(data.T_in, 2) == n1 || error("Scenario $idx inequality T has wrong first-stage dimension")
        size(data.W_eq, 2) == n2 || error("Scenario $idx equality W has wrong recourse dimension")
        size(data.W_in, 2) == n2 || error("Scenario $idx inequality W has wrong recourse dimension")
    end

    b1_eq, b1_in = _first_stage_rhs(sp)
    A1_eq = Tnum.(sp.A_eq)
    A1_in = Tnum.(sp.A_in)
    c1 = Tnum.(vec(sp.c))
    b1_eq = Tnum.(b1_eq)
    b1_in = Tnum.(b1_in)

    meq1 = size(A1_eq, 1)
    min1 = size(A1_in, 1)
    total_n = n1 + S * n2
    eq_rows = Vector{Matrix{Tnum}}()
    in_rows = Vector{Matrix{Tnum}}()
    beq_parts = Vector{Vector{Tnum}}()
    bin_parts = Vector{Vector{Tnum}}()

    if meq1 > 0
        push!(eq_rows, hcat(A1_eq, zeros(Tnum, meq1, S * n2)))
        push!(beq_parts, b1_eq)
    end
    if min1 > 0
        push!(in_rows, hcat(A1_in, zeros(Tnum, min1, S * n2)))
        push!(bin_parts, b1_in)
    end

    for (s, data_raw) in enumerate(scen_data)
        data = _scenario_data_typed(data_raw, Tnum)
        meq = size(data.W_eq, 1)
        minq = size(data.W_in, 1)
        before = (s - 1) * n2
        after = (S - s) * n2
        if meq > 0
            push!(
                eq_rows,
                hcat(data.T_eq, zeros(Tnum, meq, before), data.W_eq, zeros(Tnum, meq, after)),
            )
            push!(beq_parts, data.h_eq)
        end
        if minq > 0
            push!(
                in_rows,
                hcat(data.T_in, zeros(Tnum, minq, before), data.W_in, zeros(Tnum, minq, after)),
            )
            push!(bin_parts, data.h_in)
        end
    end

    A_eq = isempty(eq_rows) ? zeros(Tnum, 0, total_n) : vcat(eq_rows...)
    A_in = isempty(in_rows) ? zeros(Tnum, 0, total_n) : vcat(in_rows...)
    b_eq = isempty(beq_parts) ? zeros(Tnum, 0) : vcat(beq_parts...)
    b_in = isempty(bin_parts) ? zeros(Tnum, 0) : vcat(bin_parts...)
    c = vcat(c1, [Tnum.(d.q) ./ Tnum(S) for d in scen_data]...)

    return LP(A_eq, A_in, b_eq, b_in, c, (n_first_stage=n1, n_second_stage=n2, n_scenarios=S))
end

function _scenario_vector(scenario)
    if scenario isa AbstractVector && !(scenario isa AbstractVector{<:Number})
        return collect(scenario)
    end
    return [scenario]
end

function _field_or(scenario, names::Tuple, default)
    for name in names
        hasproperty(scenario, name) && return getproperty(scenario, name)
    end
    return default
end

function _scenario_data(scenario)
    W_eq = _field_or(scenario, (:W_eq, :W), nothing)
    T_eq = _field_or(scenario, (:T_eq, :T), nothing)
    h = _field_or(scenario, (:h,), nothing)
    q = _field_or(scenario, (:q,), nothing)
    W_eq === nothing && error("Scenario is missing W_eq/W.")
    T_eq === nothing && error("Scenario is missing T_eq/T.")
    h === nothing && error("Scenario is missing h.")
    q === nothing && error("Scenario is missing q.")

    n2 = size(W_eq, 2)
    n1 = size(T_eq, 2)
    W_in = _field_or(scenario, (:W_ineq, :W_in), zeros(eltype(W_eq), 0, n2))
    T_in = _field_or(scenario, (:T_ineq, :T_in), zeros(eltype(T_eq), 0, n1))
    h_eq_direct = _field_or(scenario, (:h_eq,), nothing)
    h_in_direct = _field_or(scenario, (:h_ineq, :h_in), nothing)

    if h_eq_direct !== nothing || h_in_direct !== nothing
        h_eq = h_eq_direct === nothing ? zeros(eltype(h), size(W_eq, 1)) : vec(h_eq_direct)
        h_in = h_in_direct === nothing ? zeros(eltype(h), size(W_in, 1)) : vec(h_in_direct)
    else
        h_eq, h_in = _split_scenario_rhs(vec(h), size(W_eq, 1), size(W_in, 1))
    end

    return (
        W_eq=W_eq,
        W_in=W_in,
        T_eq=T_eq,
        T_in=T_in,
        h_eq=h_eq,
        h_in=h_in,
        q=vec(q),
    )
end

function _scenario_data_typed(data, ::Type{T}) where {T}
    return (
        W_eq=T.(data.W_eq),
        W_in=T.(data.W_in),
        T_eq=T.(data.T_eq),
        T_in=T.(data.T_in),
        h_eq=T.(data.h_eq),
        h_in=T.(data.h_in),
        q=T.(data.q),
    )
end

function _split_scenario_rhs(h, m_eq::Integer, m_in::Integer)
    if length(h) == m_eq + m_in
        return h[1:m_eq], h[(m_eq + 1):end]
    elseif length(h) == m_eq && m_in == 0
        return h, similar(h, 0)
    elseif length(h) == m_in && m_eq == 0
        return similar(h, 0), h
    else
        error("Scenario RHS length $(length(h)) does not match W rows $(m_eq + m_in)")
    end
end
