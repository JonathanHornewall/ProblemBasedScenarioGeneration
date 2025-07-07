

function parent_node(t)
    #Funcion que encuentra el parent node del nodo t
    #Si el nodo es par entonces aplico formula para el parent
    if t%2 == 0
        parent_t = t/2
        return Int(parent_t)
    else
        parent_t = (t-1)/2
        return Int(parent_t)
    end
    
end

function left_and_right_nodes(t)
    #Entra el nodo t
    AR = Int64[]
    AL = Int64[]
    #Obtengo todos los notos padres del notos T

    query_t = t
    while query_t != 1

        if query_t%2==0
            push!(AR,parent_node(query_t))
        else
            push!(AL,parent_node(query_t))
        end
        query_t = parent_node(query_t)
    end

    #si el nodo es impar
    
    #println("El conjunto AR(",t,") es: ",AR)
    #println("El conjunto AL(",t,") es: ",AL)
    return AR,AL
end

function AD_tree_model(N,Tₗ,Tᵦ ,Rᵢⱼ,α,Nₘᵢₙ,P,ϵⱼ,ϵₘₐₓ,xᵢ)
    
    model_tree = Model(() -> Gurobi.Optimizer(GRB_ENV))
    set_optimizer_attribute(model_tree, "OutputFlag", 0)

    @variables(model_tree,
    begin
        zᵢₜ[1:N, t in Tₗ], Bin
        uⱼₜ[1:N,t in Tₗ], Bin
        wᵢⱼₜ[1:N,1:N,t in Tₗ],Bin
        lₜ[t in Tₗ], Bin
        aₜₚ[t in Tᵦ,1:P], Bin #Acá es a(p,t) por los features p
        dₜ[t in Tᵦ], Bin
        0 ≤ bₜ[t in Tᵦ] ≤ 1

    end) 
    
    
    @objective(model_tree,Min,
        sum(Rᵢⱼ[i,j]*wᵢⱼₜ[i,j,t] for i in 1:N,j in 1:N, t in Tₗ) + α*sum(dₜ[t] for t in Tᵦ)
    )
    
    @constraints(model_tree,
    begin
        #=
        #(1)
        [i in 1:N, j in 1:N,t in Tₗ], wᵢⱼₜ[i,j,t] ≤ uⱼₜ[j,t]
        #[i in 1:N, j in 1:J,t in 1:Tₗ], 0 ≤ wᵢⱼₜ[i,j,t]
        #(2)
        [i in 1:N, j in 1:N,t in Tₗ], zᵢₜ[i,t] + uⱼₜ[j,t] ≤ wᵢⱼₜ[i,j,t] +1
        #(3)
        [j in 1:N], sum(uⱼₜ[j,t] for t in Tₗ) ≤ 1
        #(4) Ojo aca que z esta indexado por j
        [j in 1:N,t in Tₗ], uⱼₜ[j,t] ≤ zᵢₜ[j,t]
        #(5)
        [i in 1:N], sum( zᵢₜ[i,t] for t in Tₗ) == 1
        #(6)
        [i in 1:N,t in Tₗ], zᵢₜ[i,t] ≤ lₜ[t]
        #(7)
        [t in Tₗ], sum(zᵢₜ[i,t] for i in 1:N) ≥ Nₘᵢₙ*lₜ[t]
        #(8)
        =#
        [i in 1:N ,t in Tₗ, m in left_and_right_nodes(t)[2]], sum(aₜₚ[m,p]*xᵢ[i,p] for p in 1:P ) ≥ bₜ[m] - (1 - zᵢₜ[i,t])
        #(9)
        #=
        [i in 1:N ,t in Tₗ, m in left_and_right_nodes(t)[1]], sum(aₜₚ[m,p]*(xᵢ[i,p] + ϵⱼ[p]) for p in 1:P ) ≤ bₜ[m] + (1+ϵₘₐₓ)*(1 - zᵢₜ[i,t])
        #(10)
        
        [t in Tᵦ], sum(aₜₚ[t,p] for p in 1:P) == dₜ[t]
        #(11)
        [t in Tᵦ], bₜ[t] ≤ dₜ[t]
        #(12) dₜ ≤ dₜ(parent), no incluye el 1
        [t in Tᵦ[2:end]], dₜ[t] ≤ dₜ[parent_node(t)]
        #(13) extra agregada
        [t in Tₗ], sum(uⱼₜ[j, t] for j in 1:N) == 1
        =#
    end)
    
    optimize!(model_tree)
    @show value.(zᵢₜ)
    @show value.(uⱼₜ)
    @show value.(wᵢⱼₜ)
    @show value.(dₜ)
    @show value.(aₜₚ)
    @show value.(bₜ)
    #println(model_tree)
    #return value.(wᵢⱼₜ)
end

function normalize(X)
    dt = fit(UnitRangeTransform, X,dims=1)
    X_normalized = StatsBase.transform(dt, X)
    return X_normalized,dt
end

function recover_normalize(X,dt)
    X_norm = StatsBase.reconstruct!(dt,X)
    return X_norm
end

function compute_ϵ(X,P)
    ϵⱼ = Float64[]
    for p in 1:P
        lista_diffs_feature = Float64[]
        xᵢ = X[:,p]
        xᵢ_sorted = sort(xᵢ)
        #@show xᵢ_sorted
        for l in 2:length(xᵢ)
            diff = xᵢ_sorted[l] - xᵢ_sorted[l-1]
            push!(lista_diffs_feature,diff)
        end
        push!(ϵⱼ,minimum(lista_diffs_feature))
    end  

    ϵₘₐₓ = maximum(ϵⱼ)

    return ϵⱼ,ϵₘₐₓ
end


function compute_Rᵢⱼ(Y,J,N,cz,qw,ρ,μᵢⱼ,T)
    println("computo Rij")
    #Agarro la observacion i
    Rᵢⱼ = ones(T,T)*9999
    for i in 1:T
        println(i)
        yᵢ = Y[:,i]
        zᵢ = kannanOpt(J,N,yᵢ,cz,qw,ρ,μᵢⱼ)
        #@show zᵢ
        for j in 1:T
            yⱼ = Y[:,j]
            Rᵢⱼ[i,j] = kannanCostAssessment(zᵢ,J,N,yⱼ ,cz,qw,ρ,μᵢⱼ,0)
            #@show Gᵢⱼ
        end
    end
    #@show Rᵢⱼ
    return Rᵢⱼ
end

#=
D = 2
T = 2^(D+1) -1 
N = 10 # T
J = 10     # T
Tᵦ = Int.(collect(1:floor(T/2))) #Branch nodes
Tₗ =  Int.(collect(floor(T/2)+1:T)) #Leaf nodes
P = 3
xᵢ = rand(N,P)
Rᵢⱼ = rand(N,J)
α = 0.01
Nₘᵢₙ = 5
ϵⱼ = rand(3)
ϵₘₐₓ = maximum(ϵⱼ)

AD_tree_model(N,J,Tₗ,Tᵦ,Rᵢⱼ,α,Nₘᵢₙ,P,ϵⱼ,ϵₘₐₓ,xᵢ)

=#
