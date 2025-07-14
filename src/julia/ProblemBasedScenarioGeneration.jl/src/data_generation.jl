function sampleParameters(J,σ,ω)
    #returns parameters A and B in the data generation procedure for each client
    A = 50 .+ 5 .*rand(Normal(0,1),J)
    B₁ = 10 .+ rand(Uniform(-4,4),J)
    B₂ = 5 .+ rand(Uniform(-4,4),J)
    B₃ = 2 .+ rand(Uniform(-4,4),J)
    B = hcat(B₁,B₂,B₃)

    return A,B
end

function dataGeneration(Nsamples,Noutofsamples,A,B,σ,J,p,L,Σ)

    # generate x and ξ, in sample and out of sample
    
    μ = zeros(3)
    x = transpose(abs.(rand(MvNormal(μ,Σ),Nsamples)))
    xoos = transpose(abs.(rand(MvNormal(μ,Σ),Noutofsamples)))


    ξ = zeros(J,Nsamples)
    ξoos = zeros(J,Noutofsamples)    

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
            ξoos_jn = Aⱼ .+ sum(Bⱼ[l].*(xoos[n,l]).^p for l in 1:L) .+ rand(Normal(0,σ))
            ξoos[j,n] = ξoos_jn
        end
        
    end
    
    return ξ,ξoos,x,xoos
end

#σ = 5
#J = 30
#A,B = sampleParameters(30,σ,1)
#ξ,ξoos,x,xoos=  dataGeneration(10,10,A,B,σ,30,1,3,3)

