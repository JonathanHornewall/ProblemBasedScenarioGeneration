

function kannanPlanPolicy(J,I,ŷₜ ,cz,qw,ρ,μᵢⱼ)
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
        [j in 1:J], sum(μᵢⱼ[i,j]*vᵢⱼ[i,j] for i in 1:I) + wⱼ[j] ≥ ŷₜ[j]
    end)

    optimize!(modelK1)
    return value.(zᵢ)

end


function kannanCostAssessment(zᵢ,J,I,yₜ ,cz,qw,ρ,μᵢⱼ,δ)
    modelK2 = Model(() -> Gurobi.Optimizer(GRB_ENV))
    set_optimizer_attribute(modelK2, "OutputFlag", 0)

    @variables(modelK2,
    begin
        vᵢⱼ[1:I,1:J] >= 0
        wⱼ[1:J] >= 0
    end)

    @objective(modelK2,Min,
        sum(zᵢ[i]*cz[i] for i in 1:I) + sum(qw[j]*wⱼ[j] for j in 1:J) +1000*sum(δ)
    )

    @constraints(modelK2,
    begin
        [i in 1:I], sum(vᵢⱼ[i,j] for j in 1:J) ≤ ρ[i]*zᵢ[i]
        [j in 1:J], sum(μᵢⱼ[i,j]*vᵢⱼ[i,j] for i in 1:I) + wⱼ[j] ≥ yₜ[j]
    end)

    optimize!(modelK2)
    return objective_value(modelK2)

end

function forecast(θ,Xₜ)

    x,y = size(θ) #x-> filas , y = 2 por t1 y t2
    ŷⱼₜ = sum(θ[:,l].*Xₜ[l] for l in 1:3) .+ θ[:,3+1]

    d_list = Float64[]
    for j in 1:x
       if  ŷⱼₜ[j]<0
        push!(d_list,abs(ŷⱼₜ[j]))
        ŷⱼₜ[j] = 0
       else
        push!(d_list,0)
       end 
    end
    #@show ŷⱼₜ
    #@show d_list
    
    return ŷⱼₜ,d_list
end

function heuristicAD_par(θ,T,data_yⱼₜ,data_xₜ,J,I,cz,qw,ρ,μᵢⱼ)

    #println("solving NM paralelo....")
    cost_θ = [Float64[] for _ in 1:Threads.nthreads()] #Float64[]
    
    Threads.@threads for t in 1:T
        #----- forecast, ŷₜ = ψ(Θ,xₜ)
        #println("forecasting....")
        #@show data_xₜ[t,:]     
        ŷⱼₜ,δ = forecast(θ,data_xₜ[t,:])   #Vector 3x1 con demanda forcasteada para el cliente j
        

        
        #------ Plan Policy, zₜ* ∈ arg min Gₚ(z,ŷₜ)
        #println("Plan Policy....")
        zₜ = kannanPlanPolicy(J,I,ŷⱼₜ ,cz,qw,ρ,μᵢⱼ)
        
        
        #------ Cost assesmet costₜ ∈ Gₐ(z,yₜ)
        #print(".")
        #println("Cost assessment...")
        #costₜ = costAssessment(zₜ,data_yⱼₜ[:,t],CostMatrix,δ)
        costₜ = kannanCostAssessment(zₜ,J,I,data_yⱼₜ[:,t],cz,qw,ρ,μᵢⱼ,δ)
        
        #@show costₜ
        
        push!(cost_θ[Threads.threadid()], costₜ)

        #append!(cost_θ,costₜ)
        
    end

    costTotal = sum(reduce(vcat, cost_θ))
    
    return costTotal
    
end

#=
function forecast(θ,Dⱼₜ₋₁)
    
    x,y = size(θ) #x-> filas , y = 2 por t1 y t2
    ŷⱼₜ = ones(x)*999
    # ŷₜ tiene tamañp x*1

    for j in 1:x
        ŷⱼₜ[j] = θ[j,1]*Dⱼₜ₋₁[j] + θ[j,2]
    end

    #@show ŷⱼₜ

    d_list = Float64[]
    for j in 1:x
       if  ŷⱼₜ[j]<0
        push!(d_list,abs(ŷⱼₜ[j]))
        ŷⱼₜ[j] = 0
       else
        push!(d_list,0)
       end 
    end
    #@show ŷⱼₜ
    #@show d_list
    
    return ŷⱼₜ,d_list
end


function planPolicy(ŷⱼₜ,CostMatrix)

    dy = 3
    dz = 4
    p₁ = 5  #Costo de producir antes
    p₂ = 100 #costo de producción en el ultimo minuto
    #CostMatrix = CostMatrix[:,1:dy]
    
    model = Model(() -> Gurobi.Optimizer(GRB_ENV))
    set_optimizer_attribute(model, "OutputFlag", 0)

    @variables(model,
    begin
        zᵢ[1:dz] >=0
        tᵢ[1:dz] >=0
        sᵢⱼ[1:dz,1:dy] >=0
    end)

    @objective(model,Min,
        p₁*sum(zᵢ[i] for i in 1:dz) +  p₂*sum(tᵢ[i] for i in 1:dz) + sum(CostMatrix[i,j]*sᵢⱼ[i,j] for i in 1:dz,j in 1:dy )  
    )


    @constraints(model,
    begin
        [j in 1:dy], sum(sᵢⱼ[i,j] for i in 1:dz) >= ŷⱼₜ[j]
        [i in 1:dz], sum(sᵢⱼ[i,j] for j in 1:dy ) <= zᵢ[i] + tᵢ[i]     
    end)

    JuMP.optimize!(model)
    return value.(zᵢ)
end



function costAssessment(zₜ,yₜ,Cᵢⱼ,δ)
    dz = 4
    dy = 3
    p₁ = 5  #Costo de producir antes
    p₂ = 100 #costo de producción en el ultimo minuto
    CostMatrix = Cᵢⱼ
    
    model = Model(() -> Gurobi.Optimizer(GRB_ENV))
    set_optimizer_attribute(model, "OutputFlag", 0)

    @variables(model,
    begin
        tᵢ[1:dz] >=0
        sᵢⱼ[1:dz,1:dy] >=0
    end)

    @objective(model,Min,
        p₁*sum(zₜ) + ( p₂*sum(tᵢ[i] for i in 1:dz) + sum(CostMatrix[i,j]*sᵢⱼ[i,j] for i in 1:dz,j in 1:dy )  ) +1000*sum(δ)
    )


    @constraints(model,
    begin
        [j in 1:dy], sum(sᵢⱼ[i,j] for i in 1:dz) >= yₜ[j]
        [i in 1:dz], sum(sᵢⱼ[i,j] for j in 1:dy ) <= zₜ[i] + tᵢ[i]     
    end)

    JuMP.optimize!(model)

    return objective_value(model)
    
end

function forecast(θ,Xₜ)
    
    
    x,y = size(θ) #x-> filas , y = 2 por t1 y t2
    #=
    ŷⱼₜ = ones(x)*999
    # ŷₜ tiene tamañp x*1

    for j in 1:x
        ŷⱼₜ[j] = θ[j,1]*Dⱼₜ₋₁[j] + θ[j,2]
    end
=#
    #sum(θₗₛ[:,l].*X_hoy[l] for l in 1:L) .+ θₗₛ[:,L+1]
    #@show ŷⱼₜ
    ŷⱼₜ = sum(θ[:,l].*Xₜ[l] for l in 1:3) .+ θ[:,3+1]

    d_list = Float64[]
    for j in 1:x
       if  ŷⱼₜ[j]<0
        push!(d_list,abs(ŷⱼₜ[j]))
        ŷⱼₜ[j] = 0
       else
        push!(d_list,0)
       end 
    end
    #@show ŷⱼₜ
    #@show d_list
    
    return ŷⱼₜ,d_list
end

function heuristicAD_par(θ,T,data_yⱼₜ,data_xₜ,CostMatrix)

    println("solving NM paralelo....")
    cost_θ = [Float64[] for _ in 1:Threads.nthreads()] #Float64[]
    
    Threads.@threads for t in 1:T
        #----- forecast, ŷₜ = ψ(Θ,xₜ)
        #println("forecasting....")
        #@show data_xₜ[t,:]     
        ŷⱼₜ,δ = forecast(θ,data_xₜ[t,:])   #Vector 3x1 con demanda forcasteada para el cliente j
        

        
        #------ Plan Policy, zₜ* ∈ arg min Gₚ(z,ŷₜ)
        #println("Plan Policy....")
        zₜ = planPolicy(ŷⱼₜ,CostMatrix)
        
        
        #------ Cost assesmet costₜ ∈ Gₐ(z,yₜ)
        #print(".")
        #println("Cost assessment...")
        costₜ = costAssessment(zₜ,data_yⱼₜ[:,t],CostMatrix,δ)
        #@show costₜ
        
        push!(cost_θ[Threads.threadid()], costₜ)

        #append!(cost_θ,costₜ)
        
    end

    costTotal = sum(reduce(vcat, cost_θ))
    
    return costTotal
    
end

#=

function forecast(θ,Dⱼₜ₋₁)
    
    x,y = size(θ) #x-> filas , y = 2 por t1 y t2
    ŷⱼₜ = ones(x)*999
    # ŷₜ tiene tamañp x*1

    for j in 1:x
        ŷⱼₜ[j] = θ[j,1]*Dⱼₜ₋₁[j] + θ[j,2]
    end

    #@show ŷⱼₜ

    d_list = Float64[]
    for j in 1:x
       if  ŷⱼₜ[j]<0
        push!(d_list,abs(ŷⱼₜ[j]))
        ŷⱼₜ[j] = 0
       else
        push!(d_list,0)
       end 
    end
    #@show ŷⱼₜ
    #@show d_list
    
    return ŷⱼₜ,d_list
end

function heuristicAD_par(θ,T,data_yⱼₜ,data_yⱼₜ₋₁,CostMatrix)

    println("solving NM paralelo....")
    cost_θ = [Float64[] for _ in 1:Threads.nthreads()] #Float64[]
    
    Threads.@threads for t in 1:T
        #----- forecast, ŷₜ = ψ(Θ,xₜ)
        #println("forecasting....")
        #@show data_yⱼₜ₋₁[:,t]       
        ŷⱼₜ,δ = forecast(θ,data_yⱼₜ₋₁[:,t])   #Vector 3x1 con demanda forcasteada para el cliente j
        

        
        #------ Plan Policy, zₜ* ∈ arg min Gₚ(z,ŷₜ)
        #println("Plan Policy....")
        zₜ = planPolicy(ŷⱼₜ,CostMatrix)
        
        
        #------ Cost assesmet costₜ ∈ Gₐ(z,yₜ)
        #print(".")
        #println("Cost assessment...")
        costₜ = costAssessment(zₜ,data_yⱼₜ[:,t],CostMatrix,δ)
        #@show costₜ
        
        push!(cost_θ[Threads.threadid()], costₜ)

        #append!(cost_θ,costₜ)
        
    end

    costTotal = sum(reduce(vcat, cost_θ))
    
    return costTotal
    
end
=#
=#