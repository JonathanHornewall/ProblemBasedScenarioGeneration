import Ipopt
import JuMP

struct IpoptSolver <: LogBarSolver end

function solve(
    solver::IpoptSolver,
    lp::LP;
    μ=nothing,
    slack_lower_bound=1e-9,
    constraint_tolerance=1e-6,
    kwargs...,
)
    μ_vector = _barrier_parameter_vector(lp, μ)
    positive_barrier_indices = findall(!iszero, μ_vector)
    isempty(positive_barrier_indices) &&
        throw(ArgumentError("IpoptSolver requires at least one positive log-barrier weight."))
    slack_lower_bound > zero(slack_lower_bound) ||
        throw(ArgumentError("slack_lower_bound must be positive."))

    model = JuMP.Model(Ipopt.Optimizer)
    JuMP.set_optimizer_attribute(model, "print_level", 0)
    JuMP.set_optimizer_attribute(model, "sb", "yes")
    _set_optimizer_attributes(model, kwargs)

    n_variables = length(lp.c)
    n_equalities = length(lp.b_eq)
    n_inequalities = length(lp.b_ineq)

    JuMP.@variable(model, z[1:n_variables])
    JuMP.@variable(model, s[1:n_inequalities] >= 0)
    for i in positive_barrier_indices
        JuMP.set_lower_bound(s[i], slack_lower_bound)
    end

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
        sum(μ_vector[i] * log(s[i]) for i in positive_barrier_indices),
    )
    JuMP.optimize!(model)

    status =
        _assert_successful_solve(model, solver; accepted_statuses=("OPTIMAL", "LOCALLY_SOLVED"))
    z_value = JuMP.value.(z)
    _assert_lp_solution_feasible(lp, z_value; atol=constraint_tolerance)

    return (;
        z=z_value,
        slack=JuMP.value.(s),
        dual_eq=JuMP.dual.(eq_constraints),
        dual_ineq=JuMP.dual.(slack_constraints),
        objective_value=JuMP.objective_value(model),
        status=status,
        metadata=(;
            primal_status=JuMP.primal_status(model),
            dual_status=JuMP.dual_status(model),
            raw_status=JuMP.raw_status(model),
            solver=solver,
        ),
    )
end
