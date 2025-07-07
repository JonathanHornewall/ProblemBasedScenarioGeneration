
#------ funciones
#forecast_cesgado(θ,X,L) = sum(θ[:,l].*X[:,l] for l in 1:L) .+ θ[:,L+1] #θ₁.*data .+ θ₂

function forecast_cesgado(θ,X,L,J,T)
    Ŷ = ones(J,T)
    for j in 1:J
        #Obtengo thetas del cliente j
        θⱼ = θ[j,:]
        #Obtengo las predicciones para todos los samples i
        ŷⱼ = sum(θⱼ[l].*X[:,l] for l in 1:L) .+ θⱼ[L+1]
        Ŷ[j,:] = ŷⱼ
    end
    return Ŷ
end

residuos(forecast,data_real)  = data_real .- forecast

function residual(θ,data_yₜ,data_xₜ,L,T,J)
    residuos = zeros(J,T)
    for i in 1:T
        #Obtengo la data del periodo idea
        y_obs = data_yₜ[:,i]
        #@show y_obs


        #obtengo los features del periodo i 
        x_obs = data_xₜ[i,:]
        y_pred = sum(θ[:,l].*x_obs[l] for l in 1:L) .+ θ[:,L+1]
        #@show y_pred

        arg = y_obs .- y_pred #y_pred
        residuos[:,i] = arg
    end

    return residuos
end

function pointPred(X,θ,J)
    ŷ_pred = zeros(J)
    for j in 1:J
        ŷ_pred[j] = sum(θ[j,l].*X[l] for l in 1:L) .+ θ[j,L+1]
    end


    return ŷ_pred
end