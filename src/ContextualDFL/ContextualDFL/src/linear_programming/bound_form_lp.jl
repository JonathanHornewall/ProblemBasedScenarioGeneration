import SparseArrays: findnz, issparse

struct ExtractedBoundRow{T}
    original_row::Int
    variable::Int
    coefficient::T
    rhs::T
    is_lower::Bool
end

struct InequalityBoundMap{T}
    bound_rows::Vector{ExtractedBoundRow{T}}
    general_rows::Vector{Int}
    lower_bounds::Vector{T}
    upper_bounds::Vector{T}
    lower_owner::Vector{Int}
    upper_owner::Vector{Int}
end

struct BoundFormLP{
    TAeq,
    TAineq,
    Tbeq,
    Tbineq,
    Tc,
    Tbounds,
    Tmap,
}
    A_eq::TAeq
    A_ineq::TAineq
    b_eq::Tbeq
    b_ineq::Tbineq
    c::Tc
    lower_bounds::Tbounds
    upper_bounds::Tbounds
    bound_map::Tmap
end

struct BoundFormSolveResult{
    Tz,
    TgeneralSlack,
    TgeneralDualIneq,
    TdualEq,
    TlowerBoundDual,
    TupperBoundDual,
    TobjectiveValue,
    Tstatus,
    Tmetadata,
}
    z::Tz
    general_slack::TgeneralSlack
    general_dual_ineq::TgeneralDualIneq
    dual_eq::TdualEq
    lower_bound_dual::TlowerBoundDual
    upper_bound_dual::TupperBoundDual
    objective_value::TobjectiveValue
    status::Tstatus
    metadata::Tmetadata
end

function _extract_variable_bounds(
    lp::LP;
    μ_vector=nothing,
    slack_lower_bound::Real=1e-9,
    coefficient_atol::Real=0.0,
    convert_zero_barrier_rows::Bool=true,
)
    A = lp.A_ineq
    b = lp.b_ineq
    m, n = size(A)

    if !isnothing(μ_vector) && length(μ_vector) != m
        throw(DimensionMismatch("μ_vector must have one entry per inequality."))
    end

    T = promote_type(
        Float64,
        eltype(A),
        eltype(b),
        eltype(lp.c),
        typeof(slack_lower_bound),
    )

    row_counts = zeros(Int, m)
    row_variables = zeros(Int, m)
    row_coefficients = zeros(T, m)
    coefficient_threshold = T(coefficient_atol)

    if issparse(A)
        I, J, V = findnz(A)
        @inbounds for k in eachindex(V)
            a = T(V[k])
            abs(a) <= coefficient_threshold && continue

            i = I[k]
            row_counts[i] += 1
            if row_counts[i] == 1
                row_variables[i] = J[k]
                row_coefficients[i] = a
            end
        end
    else
        @inbounds for j in 1:n
            for i in 1:m
                a = T(A[i, j])
                abs(a) <= coefficient_threshold && continue

                row_counts[i] += 1
                if row_counts[i] == 1
                    row_variables[i] = j
                    row_coefficients[i] = a
                end
            end
        end
    end

    bound_rows = ExtractedBoundRow{T}[]
    general_rows = Int[]
    lower_bounds = fill(-T(Inf), n)
    upper_bounds = fill(T(Inf), n)
    lower_owner = zeros(Int, n)
    upper_owner = zeros(Int, n)

    @inbounds for i in 1:m
        μ_i = isnothing(μ_vector) ? zero(T) : T(μ_vector[i])
        eligible = row_counts[i] == 1 && (convert_zero_barrier_rows || !iszero(μ_i))

        if !eligible
            push!(general_rows, i)
            continue
        end

        j = row_variables[i]
        a = row_coefficients[i]
        rhs = T(b[i])

        if iszero(a)
            push!(general_rows, i)
            continue
        end

        is_lower = a < zero(T)
        push!(bound_rows, ExtractedBoundRow{T}(i, j, a, rhs, is_lower))

        raw_bound = rhs / a
        effective_bound = if iszero(μ_i)
            raw_bound
        elseif is_lower
            raw_bound + T(slack_lower_bound) / (-a)
        else
            raw_bound - T(slack_lower_bound) / a
        end

        if is_lower
            if effective_bound > lower_bounds[j]
                lower_bounds[j] = effective_bound
                lower_owner[j] = i
            end
        else
            if effective_bound < upper_bounds[j]
                upper_bounds[j] = effective_bound
                upper_owner[j] = i
            end
        end
    end

    @inbounds for j in 1:n
        if lower_bounds[j] > upper_bounds[j]
            throw(
                ArgumentError(
                    "Extracted inconsistent bounds for variable $j: lower bound " *
                    "$(lower_bounds[j]) exceeds upper bound $(upper_bounds[j]).",
                ),
            )
        end
    end

    A_general = A[general_rows, :]
    b_general = b[general_rows]
    bound_map = InequalityBoundMap(
        bound_rows,
        general_rows,
        lower_bounds,
        upper_bounds,
        lower_owner,
        upper_owner,
    )
    bound_lp = BoundFormLP(
        lp.A_eq,
        A_general,
        lp.b_eq,
        b_general,
        lp.c,
        lower_bounds,
        upper_bounds,
        bound_map,
    )

    return bound_lp, bound_map
end

function _extract_variable_bounds_for_solver(solver, lp::LP; kwargs...)
    try
        return _extract_variable_bounds(lp; kwargs...)
    catch error
        if error isa ArgumentError &&
           occursin("Extracted inconsistent bounds", sprint(showerror, error))
            throw(
                ErrorException(
                    string(
                        typeof(solver),
                        " failed to solve the optimization problem: ",
                        "extracted inconsistent variable bounds.",
                    ),
                ),
            )
        end
        rethrow()
    end
end

function _reconstruct_original_inequality_info(
    lp::LP,
    bound_map::InequalityBoundMap,
    raw::BoundFormSolveResult;
    μ_vector=nothing,
)
    n_inequalities = length(lp.b_ineq)
    T = promote_type(
        Float64,
        eltype(lp.A_ineq),
        eltype(lp.b_ineq),
        eltype(raw.z),
        eltype(raw.general_slack),
        eltype(raw.general_dual_ineq),
        eltype(raw.lower_bound_dual),
        eltype(raw.upper_bound_dual),
    )

    slack = Vector{T}(undef, n_inequalities)
    dual_ineq = zeros(T, n_inequalities)

    @inbounds for k in eachindex(bound_map.general_rows)
        original_row = bound_map.general_rows[k]
        slack[original_row] = T(raw.general_slack[k])

        μ_i = isnothing(μ_vector) ? zero(T) : T(μ_vector[original_row])
        dual_ineq[original_row] =
            iszero(μ_i) ? T(raw.general_dual_ineq[k]) : μ_i / slack[original_row]
    end

    @inbounds for rowinfo in bound_map.bound_rows
        i = rowinfo.original_row
        j = rowinfo.variable
        a = T(rowinfo.coefficient)
        rhs = T(rowinfo.rhs)

        s_i = rhs - a * T(raw.z[j])
        slack[i] = s_i

        μ_i = isnothing(μ_vector) ? zero(T) : T(μ_vector[i])
        if !iszero(μ_i)
            dual_ineq[i] = μ_i / s_i
        elseif rowinfo.is_lower
            if bound_map.lower_owner[j] == i
                dual_ineq[i] = T(raw.lower_bound_dual[j]) / (-a)
            end
        elseif bound_map.upper_owner[j] == i
            dual_ineq[i] = T(raw.upper_bound_dual[j]) / a
        end
    end

    return slack, dual_ineq
end

function _reconstruct_original_lp_result(
    lp::LP,
    bound_map::InequalityBoundMap,
    raw::BoundFormSolveResult;
    μ_vector=nothing,
    include_slack::Bool=false,
)
    slack, dual_ineq =
        _reconstruct_original_inequality_info(lp, bound_map, raw; μ_vector=μ_vector)

    if include_slack
        return (;
            z=raw.z,
            slack=slack,
            dual_eq=raw.dual_eq,
            dual_ineq=dual_ineq,
            objective_value=raw.objective_value,
            status=raw.status,
            metadata=raw.metadata,
        )
    end

    return (;
        z=raw.z,
        dual_eq=raw.dual_eq,
        dual_ineq=dual_ineq,
        objective_value=raw.objective_value,
        status=raw.status,
        metadata=raw.metadata,
    )
end
