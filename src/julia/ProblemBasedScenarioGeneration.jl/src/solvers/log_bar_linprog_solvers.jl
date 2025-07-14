using JuMP, Ipopt, SparseArrays

"""
    ipot_solver(instance::LogBarCanLP, solver_tolerance=1e-9, feasibility_margin=0)
Solves a log-barrier regularized linear program in canonical form up to specified optimality tolerance
"""
function ipot_solver(instance::LogBarCanLP, solver_tolerance=1e-9, feasibility_margin=0)
    
# data 
A   = instance.linear_program.constraint_matrix        
b   = instance.linear_program.constraint_vector
c   = instance.linear_program.cost_vector
mu   = instance.regularization_parameter

# model 
model = Model(Ipopt.Optimizer)
set_optimizer_attribute(model, "tol", solver_tolerance)   # KKT tolerance
set_optimizer_attribute(model, "print_level",  0)      # silent output

@variable(model, x[1:n] >= feasibility_margin, start = 1.0)            # ensure strictly interior start
con = @constraint(model, A * x .== b)                 # Ax = b

@NLobjective(model, Min,
    sum(c[i] * x[i] for i in 1:n) -
    mu * sum(log(x[i])      for i in 1:n))

optimize!(model)

x_opt = value.(x)                      # optimal decision vector
#lambda_opt = dual.(con)                     # Lagrange multipliers  
lambda_opt = A' \ (c - [mu/xi for xi in x_opt])  # Lagrange multipliers
return x_opt, lambda_opt
end

"""
    standard_solver(instance::LogBarCanLP)
Defines the standard choice of solver when differentiating log barrier regularized canonical form linear programs
"""
function standard_solver(instance::LogBarCanLP)
    return ipot_solver(instance::LogBarCanLP)
end