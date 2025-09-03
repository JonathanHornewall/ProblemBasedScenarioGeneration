function dataGeneration(instance::UnreliableNewsvendorProblem, Nsamples, Noutofsamples, N_xi_per_x)

    # parameter for the low of D, can be changed
    b = 1.0
   
    x = (1.0e-6)*rand(Nsamples,1) .+= 1
    xoos = (1.0e-6)*rand(Noutofsamples,1) .+= 1


    ξ = zeros(2,Nsamples)
    ξoos = zeros(30, N_xi_per_x, 2, Noutofsamples)    

    #data in samples
    for i in 1:Nsamples
        U = rand()
        D = b*rand() 
        ξ[:,i] = [D,U]
    end
        
    #data out of samples
    
    for n in 1:Noutofsamples
        for k in 1:N_xi_per_x
            for l in 1:30
                U = rand()
                D = b*rand()
                ξoos_lkn = [D,U]
                ξoos[l,k,:,n] = ξoos_lkn
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

    return in_sample, out_of_sample
end


