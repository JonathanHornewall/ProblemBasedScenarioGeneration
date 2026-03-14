N = 10e7

include("optimality_test.jl")

using Random, Statistics

list_gaps = []
list_ants = []
list_gaps_2 = []


for _ in 1:N
    D = b*rand()
    U = rand()
    pos = max(U*z_star - D,0)
    neg = max(D -U*z_star,0)
    ant = (c-p)*D 
    non_ant = (c-p)*U*z_star + (p+η)*pos + π*neg
    gap = non_ant - ant
    push!(list_gaps,gap)
    push!(list_gaps_2,gap/ant)
    push!(list_ants,ant)
end


gaps1 = abs(mean(list_gaps)/mean(list_ants))
gaps2 = abs(mean(list_gaps_2))

println("gap 1: ", gaps1)
println("gap 2: ", gaps2)

# gap bigger if we divide by point anticipative value instead of mean