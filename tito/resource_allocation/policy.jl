function kannanOpt(J,I,Y_scenario,cz,qw,ρ,μᵢⱼ)
    modelK1 = Model(() -> Gurobi.Optimizer(GRB_ENV))
    set_optimizer_attribute(modelK1, "OutputFlag", 0)

    @variables(modelK1,
    begin
        zᵢ[1:I] >= 0
        vᵢⱼ[1:I,1:J] >= 0
        wⱼ[1:J] >= 0
    end)

    @objective(modelK1,Min,
        sum(zᵢ[i]*cz[i] for i in 1:I) + sum(qw[j]*wⱼ[j] for j in 1:J)
    )

    @constraints(modelK1,
    begin
        [i in 1:I], sum(vᵢⱼ[i,j] for j in 1:J) ≤ ρ[i]*zᵢ[i]
        [j in 1:J], sum(μᵢⱼ[i,j]*vᵢⱼ[i,j] for i in 1:I) + wⱼ[j] ≥ Y_scenario[j]
    end)

    optimize!(modelK1)
    #println(objective_value(modelK1))
    return value.(zᵢ)

end

function SAA_kannan(K,J,I,yⱼₖ,cz,qw,ρ,μᵢⱼ)
    model_saa = Model(() -> Gurobi.Optimizer(GRB_ENV))
    set_optimizer_attribute(model_saa, "OutputFlag", 0)

    @variables(model_saa,
    begin
        zᵢ[1:I] >= 0
        vᵢⱼ[1:I,1:J,1:K] >= 0
        wⱼ[1:J,1:K] >= 0
    end)
 

    @objective(model_saa,Min,
        (1/K).*sum(  sum(zᵢ[i]*cz[i] for i in 1:I)  + sum(qw[j]*wⱼ[j,k] for j in 1:J)  for k in 1:K)
    )

    @constraints(model_saa,
    begin
        [i in 1:I,k in 1:K], sum(vᵢⱼ[i,j,k] for j in 1:J) ≤ ρ[i]*zᵢ[i]
        [j in 1:J,k in 1:K], sum(μᵢⱼ[i,j]*vᵢⱼ[i,j,k] for i in 1:I) + wⱼ[j,k] ≥ yⱼₖ[j,k]
    end)
    
    optimize!(model_saa)
    #println(model_saa)
    return value.(zᵢ)

end

