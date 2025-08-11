using JuMP, Ipopt, SparseArrays

"""
    ipot_solver(instance::LogBarCanLP, solver_tolerance=1e-9, feasibility_margin=0)
Solves a log-barrier regularized linear program in canonical form up to specified optimality tolerance
"""
function ipot_solver(instance::LogBarCanLP, solver_tolerance=1e-9, feasibility_margin=1e-8)
    # data 
    A   = instance.linear_program.constraint_matrix        
    b   = instance.linear_program.constraint_vector
    c   = instance.linear_program.cost_vector
    mu   = instance.regularization_parameters

    n = length(c)  # number of decision variables

    # model 
    model = Model(Ipopt.Optimizer)
    set_optimizer_attribute(model, "tol", solver_tolerance)   # KKT tolerance
    set_optimizer_attribute(model, "print_level",  0)      # silent output

    @variable(model, x[1:n] >= 0, start = 1.0)  # ensure strictly interior start
    con = @constraint(model, A * x .== b)  # Ax = b

    @NLobjective(model, Min,
        sum(c[i] * x[i] for i in 1:n) -
        sum(mu[i] * log(x[i]) for i in 1:n))

    optimize!(model)

    ts = termination_status(model)
    if !(ts in (MOI.OPTIMAL, MOI.LOCALLY_SOLVED, MOI.ALMOST_LOCALLY_SOLVED))
        error("No feasible/optimal solution: $(ts) — $(MOI.get(model, MOI.RawStatusString()))")
    end

    xv = value.(x)
    if maximum(abs.(A * xv .- b)) > feasibility_margin
        error("Infeasible: max |Ax - b| = $(maximum(abs.(A * xv .- b)))")
    end

    x_opt = value.(x)                      # optimal decision vector
    lambda_opt = - dual.(con)                     # Lagrange multipliers  
    #lambda_opt = A' \ (mu ./ x_opt .- c)
    return x_opt, lambda_opt
end

"""
    RegCanLP_standard_solver(instance::LogBarCanLP)
Defines the standard choice of solver when differentiating log barrier regularized canonical form linear programs
"""
function LogBarCanLP_standard_solver(instance::LogBarCanLP)
    return ipot_solver(instance::LogBarCanLP)
end

"""
    optimal_value(instance::LogBarCanLP, decision, solver=LogBarCanLP_standard_solver)
returns the optimal value of a log-barrier regularized linaer program
"""
function optimal_value(instance::LogBarCanLP, solver=LogBarCanLP_standard_solver)
    optimal_solution, optimal_dual = solver(instance)
    return cost(instance, optimal_solution)
end