include("imports.jl")
include("data.jl")
include("LS.jl")
include("residuals.jl")
include("optimalityGap.jl")
include("parameters.jl")
include("policy.jl")
include("NM_para.jl")
include("CART.jl")
include("kNN.jl")


#----- Parámetros
NScenarios = [100,1000,10000]
J = 30
I_prods = 20
ω = 1
L = 3
σ = 5
p = 2
rep = 30
N_insample = 10000
N_outofsample = 1000*30
#------ generamos matrices para crear data.
ϕ,ζ,Σ = sampleParameters(J,σ,ω)

#------ DF para guardar data
df_final= DataFrame()
df_inSample = DataFrame()
thetas_dict = Dict()

for nRep in 1:rep
    println("**************************************")
    println("Repeticion: "*string(nRep))
    #Genero la data para el experimento
    Y,Yₒₒₛ,X,X_new =  dataGeneration(N_insample,N_outofsample,ϕ,ζ,σ,J,p,L,Σ)

    #Para cada escenario
    for t_idx in eachindex(NScenarios)
        println("Cantidad de escenarios: "*string(NScenarios[t_idx]))

        #-------------- Obtencion de data de entrenamiento
        T = NScenarios[t_idx]  

        #--- REVISAR SI LOS DATOS CORRESPONDEN!
        train_yₜ = Y[:,1:T] 
        train_xₜ =  X[1:T,:]
        
        
        println("*** Least squares")
        θₗₛ = LS(train_yₜ,train_xₜ,J,L)
        
        #εin := (yi − fn(xi)),
        println("***ER")
        ls_fₙ = forecast_cesgado(θₗₛ,train_xₜ,L,J,T) # ESTE ME PERMITE CALCULAR LA PREDICCION fn(Xi)
        ls_ε = residuos(ls_fₙ,train_yₜ)
        ŷₗₛ = pointPred(X_new,θₗₛ,J)
        Zₗₛ = kannanOpt(J,I_prods,ŷₗₛ ,cz,qw,ρᵢ,μᵢⱼ)

        #---- ER - SAA - OLS
        println("*** SAA-ER")
        saa_zₗₛ = SAA_kannan(T,J,I_prods,ŷₗₛ .+ls_ε ,cz,qw,ρᵢ,μᵢⱼ)
        

        #------------ CART
        println("*** CART")
        X2 = train_xₜ
        y2 = transpose(train_yₜ)
        X_train, X_test, y_train, y_test = py"train_test_split"(X2, y2, test_size=0.2,random_state=42)
        tree = py"getRegressor"(X_train,y_train,X_test, y_test)
        getLeav = py"getLeav"(tree,X_new,X_train,y_train)

        #Transformo la formato del AD
        println("*** M5+AD")
        T_hojas = length(getLeav[1][:,:,1])
        X_hoja = reshape(getLeav[1][:,:,:],(length(getLeav[1][:,:,1]),L))
        Y_hoja = transpose(reshape(getLeav[2][:,:,:],(length(getLeav[2][:,:,1]),J)) )

        #Hago AD con los samples de las hojas
        cart_res = Optim.optimize(θ->heuristicAD_par(θ,T_hojas ,Y_hoja,X_hoja,J,I_prods,cz,qw,ρᵢ,μᵢⱼ),θₗₛ  , NelderMead(), Optim.Options(f_tol=0.0001))
        cart_θₙₘ = Optim.minimizer(cart_res)
        cart_ŷ = pointPred(X_new,cart_θₙₘ,J)
        cart_Z = kannanOpt(J,I_prods,cart_ŷ ,cz,qw,ρᵢ,μᵢⱼ)
        
        #------ cart normal
        
        normal_cart_ŷ = getLeav[3]
        normal_cart_Z = kannanOpt(J,I_prods,normal_cart_ŷ ,cz,qw,ρᵢ,μᵢⱼ)
        


        println("")
        #---- NELDER MEAD 
        println("*** Nelder Mead")
        res = Optim.optimize(θ->heuristicAD_par(θ,T,train_yₜ,train_xₜ,J,I_prods,cz,qw,ρᵢ,μᵢⱼ),θₗₛ  , NelderMead(), Optim.Options(f_tol=0.0001))
        θₙₘ = Optim.minimizer(res)
        nm_fₙ = forecast_cesgado(θₙₘ,train_xₜ,L,J,T)
        nm_ε = residuos(nm_fₙ,train_yₜ)
        ŷₙₘ  = pointPred(X_new,θₙₘ,J)

        Zₙₘ = kannanOpt(J,I_prods,ŷₙₘ ,cz,qw,ρᵢ,μᵢⱼ)

        #----- knn
        println("KNN")
        k = minimum((Int(round(5*(T^0.4))),T-1)) 
        yₖₙₙ = generateDemandKNN(train_xₜ,train_yₜ,k,X_new)
        zₖₙₙ = SAA_kannan(k,J,I_prods,yₖₙₙ ,cz,qw,ρᵢ,μᵢⱼ)
        
        z_saa = SAA_kannan(T,J,I_prods,train_yₜ ,cz,qw,ρᵢ,μᵢⱼ)

        #---------------------------
        
        TDₗₛ = algorithm(Zₗₛ,Yₒₒₛ,J,I_prods,cz,qw,ρᵢ,μᵢⱼ)
        @show TDₗₛ

        TDₖₙₙ = algorithm(zₖₙₙ,Yₒₒₛ,J,I_prods,cz,qw,ρᵢ,μᵢⱼ)
        @show TDₖₙₙ

        cart_TD = algorithm(cart_Z,Yₒₒₛ,J,I_prods,cz,qw,ρᵢ,μᵢⱼ)
        @show cart_TD
        TD_cart_normal = algorithm(normal_cart_Z,Yₒₒₛ,J,I_prods,cz,qw,ρᵢ,μᵢⱼ)
        @show TD_cart_normal

        saa_TDₗₛ = algorithm(saa_zₗₛ,Yₒₒₛ,J,I_prods,cz,qw,ρᵢ,μᵢⱼ)
        @show saa_TDₗₛ
        
        TDₙₘ = algorithm(Zₙₘ,Yₒₒₛ,J,I_prods,cz,qw,ρᵢ,μᵢⱼ)
        @show TDₙₘ
        

        TD_saa = algorithm(z_saa,Yₒₒₛ,J,I_prods,cz,qw,ρᵢ,μᵢⱼ)
        @show TD_saa

        append!(df_final, DataFrame(T = string(T), OoS = TD_saa , method = "SAA"))
        append!(df_final, DataFrame(T = string(T), OoS = TDₗₛ , method = "LS"))
        append!(df_final, DataFrame(T = string(T), OoS = saa_TDₗₛ , method = "ER-SAA"))
        append!(df_final, DataFrame(T = string(T), OoS = TDₙₘ , method = "AD"))
        append!(df_final, DataFrame(T = string(T), OoS = TDₖₙₙ , method = "KNN"))
        append!(df_final, DataFrame(T = string(T), OoS = cart_TD , method = "M5 + AD"))
        append!(df_final, DataFrame(T = string(T), OoS = TD_cart_normal , method = "CART"))
        

        jldopen("./results/df_results_"*string(p)*".jld2", "w") do file
            write(file, "df", df_final)  
        end
        

    end
end
