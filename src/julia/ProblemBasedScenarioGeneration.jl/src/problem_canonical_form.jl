# We want to build the problem in canonical form from the scenario ξ
# This code is specific to the ressource allocation problem

include("parameters.jl")

I = 20 # number of resources
J = 30 # number of clients 

#define c
c = cz

# define q
q = zeros(J + I*J + I + J)
q[1:J] .= qw[:]

#define W
W = zeros(I+J, J + I*J + I + J)

for i in 1:I
    for j in 1:J
        W[i,J + J*(i-1) +j] = 1
    end
    W[i, J + I*J + i] = 1
end 

for j in 1:J
    W[j,j] = 1
    for i in 1:I
        W[j,J + J*(i-1) +j] = μᵢⱼ[i,j]
    end
    W[j, J + I*J + I + j] = -1
end 

#define T
T = zeros(I+J,I)
for i in 1:I
    T[i,i] = -ρᵢ[i]
end

function second_stage_problem(ξ)
    h = vcat(zeros(I), ξ) # avoid in space mutations that are not allowed if we want to derivate using Zygote 
    return q, W, h, T
end
