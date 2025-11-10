
function fullInfoSAA(K,J,I,yⱼₖ,cz,qw,ρ,μᵢⱼ)
    model_saa2 = Model(() -> Gurobi.Optimizer(GRB_ENV))
    set_optimizer_attribute(model_saa2, "OutputFlag", 0)

    #model_saa2 = Model(GLPK.Optimizer)
    #set_optimizer_attribute(model_saa2, "msg_lev", 0)  

    @variables(model_saa2,
    begin
        zᵢ[1:I] >= 0
        vᵢⱼ[1:I,1:J,1:K] >= 0
        wⱼ[1:J,1:K] >= 0
    end)
 

    @objective(model_saa2,Min,
        (1/K).*sum(  sum(zᵢ[i]*cz[i] for i in 1:I)  + sum(qw[j]*wⱼ[j,k] for j in 1:J)  for k in 1:K)
    )

    @constraints(model_saa2,
    begin
        [i in 1:I,k in 1:K], sum(vᵢⱼ[i,j,k] for j in 1:J) ≤ ρ[i]*zᵢ[i]
        [j in 1:J,k in 1:K], sum(μᵢⱼ[i,j]*vᵢⱼ[i,j,k] for i in 1:I) + wⱼ[j,k] ≥ yⱼₖ[j,k]
    end)
    
    optimize!(model_saa2)
    #println(model_saa)
    return objective_value(model_saa2)

end

function solution_cost(zᵢ,K,J,I,yⱼₖ,cz,qw,ρ,μᵢⱼ)
    model_saa3 = Model(() -> Gurobi.Optimizer(GRB_ENV))
    set_optimizer_attribute(model_saa3, "OutputFlag", 0)

    #model_saa3 = Model(GLPK.Optimizer)
    #set_optimizer_attribute(model_saa3, "msg_lev", 0)  

    @variables(model_saa3,
    begin
        vᵢⱼ[1:I,1:J,1:K] >= 0
        wⱼ[1:J,1:K] >= 0
    end)
 

    @objective(model_saa3,Min,
        (1/K).*sum(  sum(zᵢ[i]*cz[i] for i in 1:I)  + sum(qw[j]*wⱼ[j,k] for j in 1:J)  for k in 1:K)
    )

    @constraints(model_saa3,
    begin
        [i in 1:I,k in 1:K], sum(vᵢⱼ[i,j,k] for j in 1:J) ≤ ρ[i]*zᵢ[i]
        [j in 1:J,k in 1:K], sum(μᵢⱼ[i,j]*vᵢⱼ[i,j,k] for i in 1:I) + wⱼ[j,k] ≥ yⱼₖ[j,k]
    end)
    
    optimize!(model_saa3)
    #println(model_saa)
    return objective_value(model_saa3)

end




function algorithm(ẑₙ,OoS,J,I,cz,qw,ρ,μᵢⱼ)
    list_G = Float64[]
    list_v = Float64[]
    for k in 1:30
        print(".")
        ending = k*100    
        start = ending - 99
        #@show start
        #@show ending

        data_Dᵏ = OoS[:,start:ending]

        #v̄ᵏ
        v̄ᵏ =  fullInfoSAA(100,J,I,data_Dᵏ,cz,qw,ρ,μᵢⱼ)

        #v̂ᵏ
        v̂ᵏ = solution_cost(ẑₙ,100,J,I,data_Dᵏ,cz,qw,ρ,μᵢⱼ)

        #Ĝᵏ = v̂ᵏ - v̄ᵏ
        Ĝᵏ = v̂ᵏ - v̄ᵏ

        push!(list_G,Ĝᵏ)
        push!(list_v,v̄ᵏ)
    end

    #Construct normalized estimate of the 99% UCB on the optimal_gap
    v̄ = mean(list_v)
    B̂ = (100/abs(v̄))*(  (1/30)*sum( list_G[k] + 2.462*sqrt( (var(list_G)/30 ))  for k in 1:30) )
    # compute a confidence interval for the gap and express it as a percentage of value in step 3
    println("")
    return B̂
end


