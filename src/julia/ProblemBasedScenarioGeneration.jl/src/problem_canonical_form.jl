# We want to build the problem in canonical form from the scenario ξ
# This code is specific to the ressource allocation problem

include("parameters.jl")

c = cz

I = 20
J = 30

function second_stage_problem(ξ)

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

    #define h
    h = zeros(I+J)
    h[I+1:I+J] .= ξ

    #define T
    T = zeros(I+J,I)
    for i in 1:I
        T[i,i] = -ρᵢ[i]
    end

    return q, W, h, T
end

function second_stage_derivatives(ξ)
    # compute the derivatives of the second stage matrices with respect to the scenario ξ

    Dq = zeros(J + I*J + I + J,J)
    DW = zeros(I+J, J + I*J + I + J,J)

    Dh = zeros(I+J,J)
    for j in 1:J
        Dh[I+j,j] = 1
    end

    DT = zeros(I+J,I,J)

    return Dq, DW, Dh, DT
end
    
