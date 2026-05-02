import JuMP

function _assert_successful_solve(
    model,
    solver;
    accepted_statuses,
    accepted_primal_statuses=("FEASIBLE_POINT", "NEARLY_FEASIBLE_POINT"),
)
    status = JuMP.termination_status(model)
    primal_status = JuMP.primal_status(model)

    status_name = string(status)
    primal_status_name = string(primal_status)
    accepted_status_names = Set(string.(accepted_statuses))
    accepted_primal_status_names = Set(string.(accepted_primal_statuses))

    if !(status_name in accepted_status_names) ||
       !(primal_status_name in accepted_primal_status_names)
        throw(
            ErrorException(
                string(
                    typeof(solver),
                    " failed to solve the optimization problem: ",
                    "termination_status=",
                    status,
                    ", primal_status=",
                    primal_status,
                    ", dual_status=",
                    JuMP.dual_status(model),
                    ", raw_status=",
                    JuMP.raw_status(model),
                    ".",
                ),
            ),
        )
    end

    return status
end

function _assert_lp_solution_feasible(lp::LP, z; atol=1e-6)
    all(isfinite, z) ||
        throw(DomainError(z, "The solver returned non-finite primal values."))

    if !isempty(lp.b_eq)
        equality_residual = lp.A_eq * z - lp.b_eq
        maximum(abs, equality_residual) <= atol ||
            throw(
                DomainError(
                    equality_residual,
                    "The solver returned a solution that violates equality constraints.",
                ),
            )
    end

    if !isempty(lp.b_ineq)
        inequality_violation = lp.A_ineq * z - lp.b_ineq
        maximum(inequality_violation) <= atol ||
            throw(
                DomainError(
                    inequality_violation,
                    "The solver returned a solution that violates inequality constraints.",
                ),
            )
    end

    return nothing
end
