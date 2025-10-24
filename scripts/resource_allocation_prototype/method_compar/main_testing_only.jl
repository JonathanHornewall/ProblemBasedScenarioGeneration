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
include("NN.jl")

include(joinpath(@__DIR__, "..", "custom_code", "neural_net.jl"))

using Serialization

using JLD2

include("parameters.jl")   
cz, qw, ρᵢ, = vec(cz), vec(qw), vec(ρᵢ)    
problem_data = ResourceAllocationProblemData(μᵢⱼ, cz, qw, ρᵢ)         
problem_instance = ResourceAllocationProblem(problem_data) 


# ============================================================
# Parameters - can be changed
# ============================================================

p = 2 # degree of the data generation procedure

# ============================================================
# Parameters - do not change
# ============================================================
NScenarios = [100]
J = 30
I_prods = 20
ω = 1
L = 3
σ = 5
rep = 30 #to replicate the experiment in tito's paper, do not change
N_insample = 100
N_outofsample = 100*30 # 30 iterations with the same x to compute a bound for the boxplot, and 
#each requires 1000 samples

# ============================================================
# Generating data and DF to store results
# ============================================================

#generate matrices to create data
ϕ,ζ,Σ = sampleParameters(J,σ,ω) #A,B and convariance matrix are generated in advance

#DF to store data
df_final= DataFrame()
df_inSample = DataFrame()
thetas_dict = Dict()

#load training and testing data
dict_train, dict_test = deserialize("dataset_seed_145_trial_03_Ntrain_100_Ntest_30_Nxi_100_sigma_5_00_p_2_L_3.jls")


# ============================================================
# Training and testing for different number of scenarios
# ============================================================

for t_idx in eachindex(NScenarios) #iterate over number of scenarios in training set 
    println("**************************************")
    println("Number of scenarios: "*string(NScenarios[t_idx]))

    #get training data
    T = NScenarios[t_idx]  
    train_yₜ = hcat(values(dict_train)...)
    keys_vec = collect(keys(dict_train)) 
    train_xₜ = zeros(100,3)
    for scen in 1:100
        train_xₜ[scen,:] = keys_vec[scen]
    end
    
    # ============================================================
    # Training models
    # ============================================================
    model = deserialize("stage_12_model.jls")# use pretrained model

    println("train least squares")
    θₗₛ = LS(train_yₜ,train_xₜ,J,L)
    ls_fₙ = forecast_cesgado(θₗₛ,train_xₜ,L,J,T) # to compute prediction
    ls_ε = residuos(ls_fₙ,train_yₜ)
    

    #for each covariate / iteration of the algorithm to compute gap
    for nRep in 1:rep #30 iterations in algorithm for computing gap
        
        println("Iteration: "*string(nRep))

        # load data for testing
        keys_list = collect(keys(dict_test))
        key = keys_list[nRep]

        X_new  = zeros(1,3)
        for i in 1:3
            X_new[i] = key[i]
        end

        vec_list = dict_test[key]
        Yₒₒₛ = zeros(30,3000)
        for scen in 1:100
            for j in 1:J
                Yₒₒₛ[j,(scen-1)*30+1:(scen-1)*30+30] = vec_list[:,scen,j]
            end
        end
        


        # ============================================================
        # Computing first stage decisions
        # ============================================================

        println("**************************************")
        println("Computing first stage decisions")

        #Neural network
        println("*** Neural network")
        ξ_hat = model(vec(X_new))
        z_NN = surrogate_solution(problem_instance, 0.01, ξ_hat)
        
        #Least squares
        println("*** Least squares")
        
        ŷₗₛ = pointPred(X_new,θₗₛ,J)
        z_LS = kannanOpt(J,I_prods,ŷₗₛ ,cz,qw,ρᵢ,μᵢⱼ)
        
        #εin := (yi − fn(xi)),
        println("*** ER")
    

        #ER - SAA - OLS
        println("*** SAA-ER")
        z_ERSAA = SAA_kannan(T,J,I_prods,ŷₗₛ .+ls_ε ,cz,qw,ρᵢ,μᵢⱼ)
        

        #CART
        println("*** CART")
        X2 = train_xₜ
        y2 = transpose(train_yₜ)
        #compute the structure of the tree and the prediction
        X_train, X_test, y_train, y_test = py"train_test_split"(X2, y2, test_size=0.2,random_state=42) #it could be good to slit the function into two, training and testing, so that we train only once
        tree = py"getRegressor"(X_train,y_train,X_test, y_test)
        getLeav = py"getLeav"(tree,X_new,X_train,y_train)

        #Transform in right format for AD
        println("*** M5+AD")
        T_leafs = length(getLeav[1][:,:,1])
        X_leaf = reshape(getLeav[1][:,:,:],(length(getLeav[1][:,:,1]),L))
        Y_leaf = transpose(reshape(getLeav[2][:,:,:],(length(getLeav[2][:,:,1]),J)) )

        #AD from samples
        cart_res = Optim.optimize(θ->heuristicAD_par(θ,T_leafs ,Y_leaf,X_leaf,J,I_prods,cz,qw,ρᵢ,μᵢⱼ),θₗₛ  , NelderMead(), Optim.Options(f_tol=0.0001))
        θ_cart = Optim.minimizer(cart_res)
        cart_ŷ = pointPred(X_new,θ_cart,J)
        z_M5AD = kannanOpt(J,I_prods,cart_ŷ ,cz,qw,ρᵢ,μᵢⱼ)
        
        #cart normal
        normal_cart_ŷ = getLeav[3]
        z_CART = kannanOpt(J,I_prods,normal_cart_ŷ ,cz,qw,ρᵢ,μᵢⱼ)
        
        #NELDER MEAD 
        res = Optim.optimize(θ->heuristicAD_par(θ,T,train_yₜ,train_xₜ,J,I_prods,cz,qw,ρᵢ,μᵢⱼ),θₗₛ  , NelderMead(), Optim.Options(f_tol=0.0001))
        θₙₘ = Optim.minimizer(res)
        nm_fₙ = forecast_cesgado(θₙₘ,train_xₜ,L,J,T)
        nm_ε = residuos(nm_fₙ,train_yₜ)
        ŷₙₘ  = pointPred(X_new,θₙₘ,J)

        z_AD = kannanOpt(J,I_prods,ŷₙₘ ,cz,qw,ρᵢ,μᵢⱼ)

        #KNN (SAA with scenarios associated to k nearest covariates in the training set)
        k = minimum((Int(round(5*(T^0.4))),T-1)) 
        yₖₙₙ = generateDemandKNN(train_xₜ,train_yₜ,k,X_new) # find scenarios associated to k nearest neighbords
        # of X_new in the training set 
        z_KNN = SAA_kannan(k,J,I_prods,yₖₙₙ ,cz,qw,ρᵢ,μᵢⱼ) # computed first stage decision
        
        # SAA (with full set of scenarios in training set)
        z_SAA = SAA_kannan(T,J,I_prods,train_yₜ ,cz,qw,ρᵢ,μᵢⱼ)

        # ============================================================
        # Testing
        # ============================================================

        println("**************************************")
        println("Computing gaps")

        TD_NN = algorithm(z_NN,Yₒₒₛ,J,I_prods,cz,qw,ρᵢ,μᵢⱼ)
        @show TD_NN
        
        TD_LS = algorithm(z_LS,Yₒₒₛ,J,I_prods,cz,qw,ρᵢ,μᵢⱼ)
        @show TD_LS

        TD_KNN = algorithm(z_KNN,Yₒₒₛ,J,I_prods,cz,qw,ρᵢ,μᵢⱼ)
        @show TD_KNN

        TD_M5AD = algorithm(z_M5AD,Yₒₒₛ,J,I_prods,cz,qw,ρᵢ,μᵢⱼ)
        @show TD_M5AD

        TD_CART = algorithm(z_CART,Yₒₒₛ,J,I_prods,cz,qw,ρᵢ,μᵢⱼ)
        @show TD_CART

        TD_ERSAA = algorithm(z_ERSAA,Yₒₒₛ,J,I_prods,cz,qw,ρᵢ,μᵢⱼ)
        @show TD_ERSAA
        
        TD_AD = algorithm(z_AD,Yₒₒₛ,J,I_prods,cz,qw,ρᵢ,μᵢⱼ)
        @show TD_AD
        
        TD_SAA = algorithm(z_SAA,Yₒₒₛ,J,I_prods,cz,qw,ρᵢ,μᵢⱼ)
        @show TD_SAA

        # store gaps in DF

        append!(df_final, DataFrame(T = string(T), OoS = TD_SAA , method = "SAA"))
        append!(df_final, DataFrame(T = string(T), OoS = TD_LS , method = "LS"))
        append!(df_final, DataFrame(T = string(T), OoS = TD_ERSAA , method = "ER-SAA"))
        append!(df_final, DataFrame(T = string(T), OoS = TD_AD , method = "AD"))
        append!(df_final, DataFrame(T = string(T), OoS = TD_KNN , method = "KNN"))
        append!(df_final, DataFrame(T = string(T), OoS = TD_M5AD , method = "M5 + AD"))
        append!(df_final, DataFrame(T = string(T), OoS = TD_CART , method = "CART"))
        append!(df_final, DataFrame(T = string(T), OoS = TD_NN , method = "NN"))
        

        jldopen("./results/df_results_"*string(p)*".jld2", "w") do file
            write(file, "df", df_final)  
        end
        

    end
end

