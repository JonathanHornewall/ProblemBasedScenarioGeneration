function dataGeneration(instance::ResourceAllocationProblem, Nsamples, Noutofsamples, N_xi_per_x, σ, p, L)

    function generateRandomCorrMat(dim)

        betaparam = 2.0
    
        partCorr = zeros(Float64,dim,dim)
        corrMat =  Matrix{Float64}(I, dim, dim) #eye(dim)
    
        for k = 1:dim-1
            for i = k+1:dim
                partCorr[k,i] = ((rand(Distributions.Beta(betaparam,betaparam),1))[1] - 0.5)*2.0
                p = partCorr[k,i]
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

    corrMat = generateRandomCorrMat(3)


    function sampleParameters(J)
        #returns parameters A and B in the data generation procedure for each client
        A = 50 .+ 5 .*rand(Normal(0,1),J)
        B₁ = 10 .+ rand(Uniform(-4,4),J)
        B₂ = 5 .+ rand(Uniform(-4,4),J)
        B₃ = 2 .+ rand(Uniform(-4,4),J)
        B = hcat(B₁,B₂,B₃)
        return A,B
    end
    J = size(instance.problem_data.service_rate_parameters, 2)  # Number of clients

    A, B = sampleParameters(J)  # Sample parameters A and B
    
    μ = zeros(3)
    x = transpose(abs.(rand(MvNormal(μ,corrMat),Nsamples)))
    xoos = transpose(abs.(rand(MvNormal(μ,corrMat),Noutofsamples)))


    ξ = zeros(J,Nsamples)
    ξoos = zeros(30, N_xi_per_x, J, Noutofsamples)    

    for j in 1:J
        Aⱼ = A[j]
        Bⱼ = B[j,:]

        #data in samples
        for i in 1:Nsamples
            ξ_ji = Aⱼ .+ sum(Bⱼ[l].*(x[i,l]).^p for l in 1:L) .+ rand(Normal(0,σ)) 
            ξ[j,i] = ξ_ji
        end
        
        #data out of samples
    
        for n in 1:Noutofsamples
            for k in 1:N_xi_per_x
                for l in 1:30
                    ξoos_lkjn = Aⱼ .+ sum(Bⱼ[l].*(xoos[n,l]).^p for l in 1:L) .+ rand(Normal(0,σ))
                    ξoos[l,k,j,n] = ξoos_lkjn
                end
            end
        end
        
    end

    in_sample=[]
    for i in 1:Nsamples
        push!(in_sample, (x[i,:], ξ[:,i]))
    end
    out_of_sample=[]
    for n in 1:Noutofsamples
        push!(out_of_sample, (xoos[n,:], ξoos[:,:,:,n]))
    end
    in_sample, out_of_sample = Dict(in_sample), Dict(out_of_sample)  # Convert to dictionaries for easier access

    return in_sample, out_of_sample, A, B
end


#σ = 5
#J = 30
#A,B = sampleParameters(30,σ,1)
#ξ,ξoos,x,xoos=  dataGeneration(10,10,A,B,σ,30,1,3,3)
