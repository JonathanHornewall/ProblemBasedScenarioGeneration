using JuMP, GLPK, SparseArrays

"""
    solve_canonical_lp(A, b, c; solver_tolerance=1e-9, feasibility_margin=1e-8)

Solves a linear program in canonical form:
    min c'x
    s.t. Ax = b, x ≥ 0

Inputs:
    A: constraint matrix (m × n)
    b: right-hand side vector (m)
    c: cost vector (n)
    solver_tolerance: optimality tolerance for the solver (default: 1e-9)
    feasibility_margin: tolerance for constraint violation (default: 1e-8)

Outputs:
    x_opt: optimal primal solution
    lambda_opt: optimal dual solution (Lagrange multipliers)
"""
function solve_canonical_lp(instance::CanLP; solver_tolerance=1e-9, feasibility_margin=1e-8)
    A = instance.constraint_matrix
    b = instance.constraint_vector
    c = instance.cost_vector

    # Get dimensions
    m, n = size(A)
    
    # Validate inputs
    if length(b) != m
        error("Dimension mismatch: A is $(m)×$(n), but b has length $(length(b))")
    end
    if length(c) != n
        error("Dimension mismatch: A is $(m)×$(n), but c has length $(length(c))")
    end
    
    # Create optimization model
    model = Model(GLPK.Optimizer)
    set_optimizer_attribute(model, "msg_lev", 0)  # silent output
    
    # Variables: x ≥ 0
    @variable(model, x[1:n] >= 0)
    
    # Constraints: Ax = b
    con = @constraint(model, A * x .== b)
    
    # Objective: min c'x
    @objective(model, Min, dot(c, x))
    
    # Solve the problem
    optimize!(model)
    
    # Check termination status
    ts = termination_status(model)
    if !(ts in (MOI.OPTIMAL, MOI.LOCALLY_SOLVED, MOI.ALMOST_LOCALLY_SOLVED))
        error("No feasible/optimal solution: $(ts) — $(MOI.get(model, MOI.RawStatusString()))")
    end
    
    # Get solution
    x_opt = value.(x)
    lambda_opt = dual.(con)
    
    # Check feasibility
    if maximum(abs.(A * x_opt .- b)) > feasibility_margin
        error("Infeasible: max |Ax - b| = $(maximum(abs.(A * x_opt .- b)))")
    end
    
    return x_opt, lambda_opt
end

"""
    optimal_value(instance::CanLP, solver=LogBarCanLP_standard_solver)
returns the optimal value of a linear program
"""
function optimal_value(instance::CanLP, solver=solve_canonical_lp)
    optimal_solution, optimal_dual = solver(instance)
    return cost(instance, optimal_solution)
end