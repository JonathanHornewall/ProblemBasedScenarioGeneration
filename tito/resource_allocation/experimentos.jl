include("imports.jl")
include("data.jl")
include("LS.jl")
include("residuals.jl")
include("optimalityGap.jl")
include("parameters.jl")
include("policy.jl")
include("NM_para.jl")
include("CART.jl")
include("AD_tree.jl")

#----- Parámetros
J = 30
I_prods = 20
ω = 1
L = 3
σ = 5
p = 2
rep = 1
N_insample = 10000
N_outofsample = 1000*30
#------ generamos matrices para crear data.
ϕ,ζ,Σ = sampleParameters(J,σ,ω)
Y,Yₒₒₛ,X,X_new =  dataGeneration(N_insample,N_outofsample,ϕ,ζ,σ,J,p,L,Σ)

#-------------- Obtencion de data de entrenamiento
T = 20

#--- REVISAR SI LOS DATOS CORRESPONDEN!
train_yₜ = Y[:,1:T] 
train_xₜ =  X[1:T,:]

        
normalized_x_train,dt = normalize(train_xₜ)
ϵⱼ,ϵₘₐₓ = compute_ϵ(normalized_x_train,L)
Rᵢⱼ = compute_Rᵢⱼ(train_yₜ,J,I_prods,cz,qw,ρᵢ ,μᵢⱼ,T)
D = 2
Td = 2^(D+1) -1 
Tᵦ = Int.(collect(1:floor(Td/2))) #Branch nodes
Tₗ =  Int.(collect(floor(Td/2)+1:Td)) #Leaf nodes
Nₘᵢₙ = 2
α = 0.001

AD_tree_model(T,Tₗ,Tᵦ,Rᵢⱼ,α,Nₘᵢₙ,L,ϵⱼ,ϵₘₐₓ,normalized_x_train)