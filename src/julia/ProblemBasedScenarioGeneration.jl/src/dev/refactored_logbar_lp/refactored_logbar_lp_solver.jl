module RefactoredLogbarLPSolvers

using JuMP
using Ipopt
using GLPK
using LinearAlgebra

const MOI = JuMP.MOI

using ..RefactoredLogbarLP
using ..RefactoredLPSolvers: solve_general_lp

export find_strictly_feasible_point,
       solve_log_barrier_lp,
       solve_log_barrier_lp_primal

function find_strictly_feasible_point(lp::InequalityEqualityLP; margin=1e-10)
    n = length(lp.c)
    mI = size(lp.A_ineq, 1)
    mE = size(lp.A_eq, 1)

    if mI == 0
        if mE == 0
            return zeros(eltype(lp.c), n)
        else
            return lp.A_eq \ lp.b_eq
        end
    end

    model = Model(GLPK.Optimizer)
    set_optimizer_attribute(model, "msg_lev", 0)

    @variable(model, x[1:n])
    @variable(model, δ >= 0)

    if mE > 0
        @constraint(model, [i=1:mE], dot(lp.A_eq[i, :], x) == lp.b_eq[i])
    end
    @constraint(model, [i=1:mI], dot(lp.A_ineq[i, :], x) + δ <= lp.b_ineq[i])

    @objective(model, Max, δ)
    optimize!(model)

    ts = termination_status(model)
    if !(ts in (MOI.OPTIMAL, MOI.LOCALLY_SOLVED, MOI.ALMOST_LOCALLY_SOLVED))
        error("Unable to find strictly feasible point: $(ts) — $(MOI.get(model, MOI.RawStatusString()))")
    end

    δ_opt = value(δ)
    if δ_opt <= margin
        error("No strictly feasible point exists with margin > $(margin)")
    end

    return value.(x)
end

function solve_log_barrier_lp(instance::LogBarrierLP;
                              solver_tolerance=1e-9,
                              feasibility_margin=1e-8,
                              initial_point=nothing,
                              interior_margin=1e-8)
    lp = instance.lp
    mI = size(lp.A_ineq, 1)
    n = length(lp.c)

    if mI == 0 || all(iszero.(instance.mu))
        x_opt, (_, λ_eq) = solve_general_lp(lp;
                                            solver_tolerance=solver_tolerance,
                                            feasibility_margin=feasibility_margin)
        return x_opt, λ_eq
    end

    x0 = initial_point === nothing ? find_strictly_feasible_point(lp; margin=interior_margin) : initial_point
    if minimum(instance.lp.b_ineq - instance.lp.A_ineq * x0) <= 0
        error("Initial point is not strictly feasible for inequalities")
    end

    model = Model(Ipopt.Optimizer)
    set_optimizer_attribute(model, "tol", solver_tolerance)
    set_optimizer_attribute(model, "print_level", 0)

    @variable(model, x[1:n])
    for j in 1:n
        set_start_value(x[j], x0[j])
    end

    con_eq = nothing
    mE = size(lp.A_eq, 1)
    if mE > 0
        con_eq = @constraint(model, [i=1:mE], dot(lp.A_eq[i, :], x) == lp.b_eq[i])
    end

    @NLobjective(model, Min,
        sum(lp.c[j] * x[j] for j in 1:n) -
        sum(instance.mu[i] * log(lp.b_ineq[i] - sum(lp.A_ineq[i, j] * x[j] for j in 1:n)) for i in 1:mI))

    optimize!(model)
    ts = termination_status(model)
    if !(ts in (MOI.OPTIMAL, MOI.LOCALLY_SOLVED, MOI.ALMOST_LOCALLY_SOLVED))
        error("No feasible/optimal solution: $(ts) — $(MOI.get(model, MOI.RawStatusString()))")
    end

    x_opt = value.(x)
    if minimum(instance.lp.b_ineq - instance.lp.A_ineq * x_opt) <= 0
        error("Solution violates inequality interior")
    end
    if mE > 0 && maximum(abs.(lp.A_eq * x_opt .- lp.b_eq)) > feasibility_margin
        error("Equality constraints violated beyond tolerance")
    end

    λ_eq = mE > 0 ? dual.(con_eq) : similar(lp.b_eq, 0)

    return x_opt, λ_eq
end

solve_log_barrier_lp_primal(instance::LogBarrierLP;
                            solver_tolerance=1e-9,
                            feasibility_margin=1e-8,
                            initial_point=nothing,
                            interior_margin=1e-8) =
    solve_log_barrier_lp(instance;
                         solver_tolerance=solver_tolerance,
                         feasibility_margin=feasibility_margin,
                         initial_point=initial_point,
                         interior_margin=interior_margin)[1]

end
