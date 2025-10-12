module RefactoredLPSolvers

using JuMP
using GLPK
using LinearAlgebra

const MOI = JuMP.MOI

using ..RefactoredLogbarLP: InequalityEqualityLP

export solve_general_lp, solve_general_lp_primal

function solve_general_lp(instance::InequalityEqualityLP;
                          solver_tolerance=1e-9,
                          feasibility_margin=1e-8)
    lp = instance
    n = length(lp.c)
    mI = size(lp.A_ineq, 1)
    mE = size(lp.A_eq, 1)

    model = Model(GLPK.Optimizer)
    set_optimizer_attribute(model, "msg_lev", 0)

    @variable(model, x[1:n])

    con_ineq = nothing
    if mI > 0
        con_ineq = @constraint(model, [i=1:mI], dot(lp.A_ineq[i, :], x) <= lp.b_ineq[i])
    end

    con_eq = nothing
    if mE > 0
        con_eq = @constraint(model, [i=1:mE], dot(lp.A_eq[i, :], x) == lp.b_eq[i])
    end

    @objective(model, Min, sum(lp.c[j] * x[j] for j in 1:n))

    optimize!(model)

    ts = termination_status(model)
    if !(ts in (MOI.OPTIMAL, MOI.LOCALLY_SOLVED, MOI.ALMOST_LOCALLY_SOLVED))
        error("No feasible/optimal solution: $(ts) — $(MOI.get(model, MOI.RawStatusString()))")
    end

    x_opt = value.(x)

    if mI > 0
        if maximum(lp.A_ineq * x_opt .- lp.b_ineq) > feasibility_margin
            error("Infeasible inequality solution")
        end
    end
    if mE > 0
        if maximum(abs.(lp.A_eq * x_opt .- lp.b_eq)) > feasibility_margin
            error("Infeasible equality solution")
        end
    end

    λ_ineq = mI > 0 ? dual.(con_ineq) : similar(lp.b_ineq, 0)
    λ_eq = mE > 0 ? dual.(con_eq) : similar(lp.b_eq, 0)

    return x_opt, (λ_ineq, λ_eq)
end

solve_general_lp_primal(instance::InequalityEqualityLP;
                        solver_tolerance=1e-9,
                        feasibility_margin=1e-8) =
    solve_general_lp(instance;
                     solver_tolerance=solver_tolerance,
                     feasibility_margin=feasibility_margin)[1]

end
