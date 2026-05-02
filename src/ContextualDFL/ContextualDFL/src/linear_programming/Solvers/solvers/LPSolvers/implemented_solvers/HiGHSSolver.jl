import HiGHS
import JuMP

struct HiGHSSolver <: LPSolver end

function solve(solver::HiGHSSolver, lp::LP; kwargs...)
    model = JuMP.Model(HiGHS.Optimizer)
    JuMP.set_silent(model)
    _set_optimizer_attributes(model, kwargs)

    n_variables = length(lp.c)
    n_equalities = length(lp.b_eq)
    n_inequalities = length(lp.b_ineq)

    JuMP.@variable(model, z[1:n_variables])

    eq_constraints = Vector{JuMP.ConstraintRef}(undef, n_equalities)
    for i in 1:n_equalities
        eq_constraints[i] =
            JuMP.@constraint(model, sum(lp.A_eq[i, j] * z[j] for j in 1:n_variables) == lp.b_eq[i])
    end

    ineq_constraints = Vector{JuMP.ConstraintRef}(undef, n_inequalities)
    for i in 1:n_inequalities
        ineq_constraints[i] = JuMP.@constraint(
            model,
            sum(lp.A_ineq[i, j] * z[j] for j in 1:n_variables) <= lp.b_ineq[i],
        )
    end

    JuMP.@objective(model, Min, sum(lp.c[j] * z[j] for j in 1:n_variables))
    JuMP.optimize!(model)

    return (;
        z=JuMP.value.(z),
        dual_eq=JuMP.dual.(eq_constraints),
        dual_ineq=-JuMP.dual.(ineq_constraints),
        objective_value=JuMP.objective_value(model),
        status=JuMP.termination_status(model),
        metadata=(;
            primal_status=JuMP.primal_status(model),
            dual_status=JuMP.dual_status(model),
            raw_status=JuMP.raw_status(model),
            solver=solver,
        ),
    )
end

function _set_optimizer_attributes(model, kwargs)
    for (attribute, value) in kwargs
        JuMP.set_optimizer_attribute(model, String(attribute), value)
    end

    return nothing
end
