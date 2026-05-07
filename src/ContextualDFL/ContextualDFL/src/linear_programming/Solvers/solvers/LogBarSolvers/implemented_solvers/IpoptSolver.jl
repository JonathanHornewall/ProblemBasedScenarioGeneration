import Ipopt
import JuMP

struct IpoptSolver <: LogBarSolver end

function solve(
    solver::IpoptSolver,
    lp::LP;
    μ=nothing,
    ρ=0,
    rho=ρ,
    slack_lower_bound=1e-12,
    constraint_tolerance=1e-6,
    kwargs...,
)
    μ_value = isnothing(μ) ? zeros(eltype(lp.c), length(lp.b_ineq)) : μ
    μ_vector = _barrier_parameter_vector(lp, μ_value)
    ρ_vector = _quadratic_parameter_vector(lp, rho)
    positive_barrier_indices = findall(!iszero, μ_vector)
    positive_quadratic_indices = findall(!iszero, ρ_vector)
    isempty(positive_barrier_indices) && isempty(positive_quadratic_indices) &&
        throw(ArgumentError("IpoptSolver requires at least one positive smoothing weight."))
    slack_lower_bound > zero(slack_lower_bound) ||
        throw(ArgumentError("slack_lower_bound must be positive."))
    bound_lp, bound_map = _extract_variable_bounds_for_solver(
        solver,
        lp;
        μ_vector=μ_vector,
        slack_lower_bound=slack_lower_bound,
    )

    model = JuMP.Model(Ipopt.Optimizer)
    JuMP.set_optimizer_attribute(model, "print_level", 0)
    JuMP.set_optimizer_attribute(model, "sb", "yes")
    JuMP.set_optimizer_attribute(model, "mu_strategy", "monotone")
    JuMP.set_optimizer_attribute(model, "nlp_scaling_method", "none")
    _set_optimizer_attributes(model, kwargs)

    n_variables = length(bound_lp.c)
    n_general_inequalities = length(bound_lp.b_ineq)
    eq_basis, A_eq_basis = _independent_constraint_rows(bound_lp.A_eq)
    b_eq_basis = bound_lp.b_eq[eq_basis]

    JuMP.@variable(model, z[1:n_variables])
    _set_variable_bounds!(z, bound_lp.lower_bounds, bound_lp.upper_bounds)

    s = Vector{JuMP.VariableRef}(undef, n_general_inequalities)
    slack_constraints = Vector{JuMP.ConstraintRef}(undef, n_general_inequalities)
    if n_general_inequalities > 0
        JuMP.@variable(model, general_slack[1:n_general_inequalities] >= 0)
        s = general_slack
        for k in 1:n_general_inequalities
            if !iszero(μ_vector[bound_map.general_rows[k]])
                JuMP.set_lower_bound(s[k], slack_lower_bound)
            end
        end
        slack_constraints =
            JuMP.@constraint(model, bound_lp.A_ineq * z .+ s .== bound_lp.b_ineq)
    end

    eq_constraints = JuMP.@constraint(model, A_eq_basis * z .== b_eq_basis)

    positive_general_positions = [
        k for k in eachindex(bound_map.general_rows) if !iszero(μ_vector[bound_map.general_rows[k]])
    ]
    positive_bound_rows = [
        row for row in bound_map.bound_rows if !iszero(μ_vector[row.original_row])
    ]
    general_original_rows = bound_map.general_rows
    bound_original_rows = [row.original_row for row in positive_bound_rows]
    bound_variables = [row.variable for row in positive_bound_rows]
    bound_coefficients = [row.coefficient for row in positive_bound_rows]
    bound_rhs = [row.rhs for row in positive_bound_rows]

    JuMP.@NLobjective(
        model,
        Min,
        sum(bound_lp.c[j] * z[j] for j in 1:n_variables) +
        0.5 * sum(ρ_vector[j] * z[j]^2 for j in positive_quadratic_indices) -
        sum(
            μ_vector[general_original_rows[k]] * log(s[k]) for
            k in positive_general_positions
        ) -
        sum(
            μ_vector[bound_original_rows[k]] *
            log(bound_rhs[k] - bound_coefficients[k] * z[bound_variables[k]]) for
            k in eachindex(bound_original_rows)
        ),
    )
    JuMP.optimize!(model)

    status = _assert_successful_solve(
        model,
        solver;
        accepted_statuses=("OPTIMAL", "LOCALLY_SOLVED", "ALMOST_LOCALLY_SOLVED"),
    )
    z_value = JuMP.value.(z)
    _assert_lp_solution_feasible(lp, z_value; atol=constraint_tolerance)

    lower_bound_dual, upper_bound_dual =
        _normalized_variable_bound_duals(z, bound_lp.lower_bounds, bound_lp.upper_bounds)
    dual_eq = zeros(Float64, length(bound_lp.b_eq))
    dual_eq[eq_basis] .= JuMP.dual.(eq_constraints)
    raw_result = BoundFormSolveResult(
        z_value,
        n_general_inequalities == 0 ? similar(z_value, 0) : JuMP.value.(s),
        n_general_inequalities == 0 ? similar(z_value, 0) : -JuMP.dual.(slack_constraints),
        dual_eq,
        lower_bound_dual,
        upper_bound_dual,
        JuMP.objective_value(model),
        status,
        (;
            primal_status=JuMP.primal_status(model),
            dual_status=JuMP.dual_status(model),
            raw_status=JuMP.raw_status(model),
            solver=solver,
            ρ_vector=ρ_vector,
        ),
    )

    return _reconstruct_original_lp_result(
        lp,
        bound_map,
        raw_result;
        μ_vector=μ_vector,
        include_slack=true,
    )
end
