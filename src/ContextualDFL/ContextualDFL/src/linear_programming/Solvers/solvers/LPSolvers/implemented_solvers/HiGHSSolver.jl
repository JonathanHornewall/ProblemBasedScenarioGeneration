import HiGHS
import JuMP

struct HiGHSSolver <: LPSolver end

function solve(solver::HiGHSSolver, lp::LP; constraint_tolerance=1e-6, kwargs...)
    bound_lp, bound_map = _extract_variable_bounds_for_solver(solver, lp)

    model = JuMP.Model(HiGHS.Optimizer)
    JuMP.set_silent(model)
    _set_optimizer_attributes(model, kwargs)

    n_variables = length(bound_lp.c)
    JuMP.@variable(model, z[1:n_variables])
    _set_variable_bounds!(z, bound_lp.lower_bounds, bound_lp.upper_bounds)

    eq_constraints = JuMP.@constraint(model, bound_lp.A_eq * z .== bound_lp.b_eq)
    ineq_constraints = JuMP.@constraint(model, bound_lp.A_ineq * z .<= bound_lp.b_ineq)

    JuMP.@objective(model, Min, sum(bound_lp.c[j] * z[j] for j in 1:n_variables))
    JuMP.optimize!(model)

    status = _assert_successful_solve(model, solver; accepted_statuses=("OPTIMAL",))
    z_value = JuMP.value.(z)
    _assert_lp_solution_feasible(lp, z_value; atol=constraint_tolerance)

    lower_bound_dual, upper_bound_dual =
        _normalized_variable_bound_duals(z, bound_lp.lower_bounds, bound_lp.upper_bounds)
    raw_result = BoundFormSolveResult(
        z_value,
        bound_lp.b_ineq - bound_lp.A_ineq * z_value,
        -JuMP.dual.(ineq_constraints),
        JuMP.dual.(eq_constraints),
        lower_bound_dual,
        upper_bound_dual,
        JuMP.objective_value(model),
        status,
        (;
            primal_status=JuMP.primal_status(model),
            dual_status=JuMP.dual_status(model),
            raw_status=JuMP.raw_status(model),
            solver=solver,
        ),
    )

    return _reconstruct_original_lp_result(lp, bound_map, raw_result)
end

function _set_optimizer_attributes(model, kwargs)
    for (attribute, value) in kwargs
        JuMP.set_optimizer_attribute(model, String(attribute), value)
    end

    return nothing
end

function _set_variable_bounds!(z, lower_bounds, upper_bounds)
    @inbounds for j in eachindex(z)
        if isfinite(lower_bounds[j])
            JuMP.set_lower_bound(z[j], lower_bounds[j])
        end
        if isfinite(upper_bounds[j])
            JuMP.set_upper_bound(z[j], upper_bounds[j])
        end
    end

    return nothing
end

function _normalized_variable_bound_duals(z, lower_bounds, upper_bounds)
    T = promote_type(Float64, eltype(lower_bounds), eltype(upper_bounds))
    lower_bound_dual = zeros(T, length(z))
    upper_bound_dual = zeros(T, length(z))

    @inbounds for j in eachindex(z)
        if isfinite(lower_bounds[j])
            lower_bound_dual[j] = T(JuMP.dual(JuMP.LowerBoundRef(z[j])))
        end
        if isfinite(upper_bounds[j])
            upper_bound_dual[j] = -T(JuMP.dual(JuMP.UpperBoundRef(z[j])))
        end
    end

    return lower_bound_dual, upper_bound_dual
end
