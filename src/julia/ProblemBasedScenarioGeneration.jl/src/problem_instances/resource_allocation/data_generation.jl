function dataGeneration(instance::ResourceAllocationProblem, Nsamples, Noutofsamples, σ, p, L, Σ)
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

    in_sample=[]
    for i in 1:Nsamples
        push!(in_sample, (x[i,:], ξ[:,i]))
    end
    out_of_sample=[]
    for n in 1:Noutofsamples
        push!(out_of_sample, (xoos[n,:], ξoos[:,n]))
    end
    in_sample, out_of_sample = dict(in_sample), dict(out_of_sample)  # Convert to dictionaries for easier access

    return in_sample, out_of_sample
end

#σ = 5
#J = 30
#A,B = sampleParameters(30,σ,1)
#ξ,ξoos,x,xoos=  dataGeneration(10,10,A,B,σ,30,1,3,3)