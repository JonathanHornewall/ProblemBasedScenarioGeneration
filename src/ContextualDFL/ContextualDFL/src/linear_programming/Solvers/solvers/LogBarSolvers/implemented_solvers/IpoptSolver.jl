import Ipopt
import JuMP

struct IpoptSolver <: LogBarSolver end

function solve(::IpoptSolver, lp::LP; μ=nothing, slack_lower_bound=1e-9, kwargs...)
    isnothing(μ) && throw(ArgumentError("IpoptSolver requires a positive log-barrier parameter μ."))
    μ > zero(μ) || throw(ArgumentError("IpoptSolver requires μ > 0."))

    model = JuMP.Model(Ipopt.Optimizer)
    JuMP.set_optimizer_attribute(model, "print_level", 0)
    JuMP.set_optimizer_attribute(model, "sb", "yes")
    _set_optimizer_attributes(model, kwargs)

    n_variables = length(lp.c)
    n_equalities = length(lp.b_eq)
    n_inequalities = length(lp.b_ineq)

    JuMP.@variable(model, z[1:n_variables])
    JuMP.@variable(model, s[1:n_inequalities] >= slack_lower_bound)

    eq_constraints = Vector{JuMP.ConstraintRef}(undef, n_equalities)
    for i in 1:n_equalities
        eq_constraints[i] =
            JuMP.@constraint(model, sum(lp.A_eq[i, j] * z[j] for j in 1:n_variables) == lp.b_eq[i])
    end

    slack_constraints = Vector{JuMP.ConstraintRef}(undef, n_inequalities)
    for i in 1:n_inequalities
        slack_constraints[i] = JuMP.@constraint(
            model,
            sum(lp.A_ineq[i, j] * z[j] for j in 1:n_variables) + s[i] == lp.b_ineq[i],
        )
    end

    JuMP.@NLobjective(
        model,
        Min,
        sum(lp.c[j] * z[j] for j in 1:n_variables) -
        μ * sum(log(s[i]) for i in 1:n_inequalities),
    )
    JuMP.optimize!(model)

    status = JuMP.termination_status(model)
    if string(status) != "LOCALLY_SOLVED" && string(status) != "OPTIMAL"
        return (;
            z=fill(NaN, n_variables),
            slack=fill(NaN, n_inequalities),
            dual_eq=fill(NaN, n_equalities),
            dual_ineq=fill(NaN, n_inequalities),
            objective_value=NaN,
            status=status,
            metadata=(;
                primal_status=JuMP.primal_status(model),
                dual_status=JuMP.dual_status(model),
                raw_status=JuMP.raw_status(model),
                solver=IpoptSolver(),
            ),
        )
    end

    return (;
        z=JuMP.value.(z),
        slack=JuMP.value.(s),
        dual_eq=JuMP.dual.(eq_constraints),
        dual_ineq=JuMP.dual.(slack_constraints),
        objective_value=JuMP.objective_value(model),
        status=status,
        metadata=(;
            primal_status=JuMP.primal_status(model),
            dual_status=JuMP.dual_status(model),
            raw_status=JuMP.raw_status(model),
            solver=IpoptSolver(),
        ),
    )
end
