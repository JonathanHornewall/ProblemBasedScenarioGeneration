"""
    convert_standard_to_canonical_form(A, b, c)
Converts a standard linear program in the form min c^T x s.t. Ax = b, to a canonical form
by adding slack variables and extending the cost vector.
"""
function convert_standard_to_canonical_form(A, b, c; p = 1e-7, rescale=true)
    A = float(A); b = float(b); c = float(c)
    m, n = size(A)
    if rescale # Rescale p based on the cost vector to ensure numerical stability. 
        p = p * ifelse(maximum(abs, c)==0, one(eltype(c)), maximum(abs, c))
    end 
    A = hcat(A, -A, Matrix{eltype(A)}(I, m, m))  # add slack variables
    c = vcat(c, -c, zeros(eltype(c), m))  # extend cost vector with zeros for slack variables
    c[1:2*n] .+= p   # ensure bounds on split variables
    return A, b, c
end

function convert_decision_standard_to_canonical(constraint_matrix, constraint_vector, decision)
    A = constraint_matrix
    b = constraint_vector
    x = decision

    slack = b .- A * x  # slack variables
    all(slack .>= 0) || error("slack variables must be non-negative")
    x_new = [max.(x, 0); max.(-x, 0); slack]  # split x into non-negative and non-positive parts, and add slack variables
    x_new[iszero.(x_new)] .= 1.0  # ensure no zero entries to avoid division by zero in log barrier
    return x_new
end

"""
        KKT(instance::LogBarCanLP, state, dual_state)
Checks the KKT conditions for optimality of a log-barrier regularized linear program in canonical form
"""
function KKT(instance::LogBarCanLP, state, dual_state)
    A = instance.linear_program.constraint_matrix
    μ = instance.regularization_parameters
    b = instance.linear_program.constraint_vector
    c = instance.linear_program.cost_vector
    x = state
    λ = dual_state
    if length(x) != length(μ) || length(λ) != size(A, 1)
        error("State and dual state dimensions do not match the problem instance")
    end
    return [c - μ ./ x - A'*λ; A * x - b]
end