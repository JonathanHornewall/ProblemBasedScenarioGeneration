function sampleParameters(J,σ,ω)
    #Obtengo todos los parámetros para cada cliente j!
    ϕⱼ = 50 .+ 5 .*rand(Normal(0,1),J)
    ζⱼ₁ = 10 .+ rand(Uniform(-4,4),J)
    ζⱼ₂ = 5 .+ rand(Uniform(-4,4),J)
    ζⱼ₃ = 2 .+ rand(Uniform(-4,4),J)
    ζ = hcat(ζⱼ₁,ζⱼ₂,ζⱼ₃)
    #πⱼₗ = 2*(ω-1)^2 *rand(Uniform(0,1),J,L)
    #ξⱼ = rand(Normal(0,σ),J)

    corr_matrix = generateRandomCorrMat(3)
    #cov_matrix = covMatrix(corr_matrix,5,3)

    return ϕⱼ,ζ,corr_matrix#cov_matrix
end

function dataGeneration(Nsamples,Noutofsamples,ϕ,ζ,σ,J,p,L,Σ)
    
    #---- Genero los x
    #X = abs.(rand(Normal(0, σ),Nsamples,L))
    #X_hoy = abs.(rand(Normal(0, σ),L))
    
    μ = zeros(3)
    X = transpose(abs.(rand(MvNormal(μ,Σ),Nsamples)))
    X_hoy = transpose(abs.(rand(MvNormal(μ,Σ),1)))


    Y = zeros(J,Nsamples)
    Yoos = zeros(J,Noutofsamples)    

    for j in 1:J
        ϕⱼ = ϕ[j]
        ζⱼ = ζ[j,:]
        #ξⱼ = ξ[j]

        for i in 1:Nsamples
            #@show X[i,:]
            Yⱼᵢ = ϕⱼ .+ sum(ζⱼ[l].*(X[i,l]).^p for l in 1:L) .+ rand(Normal(0,σ)) #ξⱼ
            Y[j,i] = Yⱼᵢ
            #@show Yⱼᵢ
        end
       
        #data in sample
        for n in 1:Noutofsamples
            oos_ξⱼ = rand(Normal(0,σ))
            oos_Yⱼ = ϕⱼ .+ sum(ζⱼ[l].*(X_hoy[l]).^p for l in 1:L) .+ oos_ξⱼ
            Yoos[j,n] = oos_Yⱼ
        end
        
    end
    
    return Y,Yoos,X,X_hoy
end

#σ = 5
#ϕ,ζ,ξ = sampleParameters(10,σ,1)
#Y,Yₒₒₛ,X,X_new =  dataGeneration(10,10,ϕ,ζ,ξ,σ,10,1,3)


function generateRandomCorrMat(dim)

	betaparam = 2.0

	partCorr = zeros(Float64,dim,dim)
	corrMat =  Matrix{Float64}(I, dim, dim) #eye(dim)

	for k = 1:dim-1
		for i = k+1:dim
			partCorr[k,i] = ((rand(Distributions.Beta(betaparam,betaparam),1))[1] - 0.5)*2.0
			p = partCorr[k,i]
            println(".")
			for j = (k-1):-1:1
				p = p*sqrt((1-partCorr[j,i]^2)*(1-partCorr[j,k]^2)) + partCorr[j,i]*partCorr[j,k]
			end
			corrMat[k,i] = p
			corrMat[i,k] = p
		end
	end

	permut = Random.randperm(dim)
	corrMat = corrMat[permut, permut]

	return corrMat
end
#=
function covMatrix(corrM,σ,dim)
    D = Matrix{Float64}(I, dim, dim).* σ
    cov_matrix = D*corrM*D
    return cov_matrix
end
=#