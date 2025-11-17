"""
    dataGeneration(instance::ShipmentPlanningProblem, Nsamples, Noutofsamples, N_xi_per_x,
                   σ, seasonal_scale, trend_decay; collections_per_sample = 30)

Synthetic contextual data generator for the shipment planning benchmark. Mirrors the signature used by
the resource allocation prototype so that experimentation scripts can be re-used.
"""
function dataGeneration(instance::ShipmentPlanningProblem,
                        Nsamples,
                        Noutofsamples,
                        N_xi_per_x,
                        σ,
                        seasonal_scale,
                        trend_decay;
                        collections_per_sample::Int = 30,
                        p::Float64 = 1.0)
    _, n_locations = size(instance.problem_data.shipment_costs)
    context_dim = instance.problem_data.context_dimension

    function generateRandomCorrMat(dim)
        betaparam = 2.0
        partCorr = zeros(Float64, dim, dim)
        corrMat = Matrix{Float64}(I, dim, dim)
        for k = 1:dim-1
            for i = k+1:dim
                partCorr[k, i] = ((rand(Beta(betaparam, betaparam), 1))[1] - 0.5) * 2.0
                val = partCorr[k, i]
                for j = (k-1):-1:1
                    val = val * sqrt((1 - partCorr[j, i]^2) * (1 - partCorr[j, k]^2)) + partCorr[j, i] * partCorr[j, k]
                end
                corrMat[k, i] = val
                corrMat[i, k] = val
            end
        end
        perm = randperm(dim)
        corrMat[perm, perm]
    end

    function sampleParameters(J)
        A = 50 .+ 5 .* rand(Normal(0, 1), J)
        B1 = 10 .+ rand(Uniform(-4, 4), J)
        B2 = 5 .+ rand(Uniform(-4, 4), J)
        B3 = 2 .+ rand(Uniform(-4, 4), J)
        hcat(B1, B2, B3), A
    end

    corrMat = generateRandomCorrMat(context_dim)
    base_contexts = rand(MvNormal(zeros(context_dim), corrMat), Nsamples + Noutofsamples)
    base_contexts = abs.(base_contexts)

    B, A = sampleParameters(n_locations)

    function polynomial_demand(context_vec)
        demand = similar(A)
        for j in 1:n_locations
            demand[j] = A[j] + sum(B[j, l] * (context_vec[l])^p for l in 1:context_dim) +
                        rand(Normal(0, σ))
        end
        return max.(demand, 0.0)
    end

    ξ = zeros(Float64, n_locations, Nsamples)
    for i in 1:Nsamples
        ξ[:, i] = polynomial_demand(base_contexts[:, i])
    end

    ξoos = zeros(Float64, collections_per_sample, N_xi_per_x, n_locations, Noutofsamples)
    for n in 1:Noutofsamples
        context = base_contexts[:, Nsamples + n]
        for k in 1:N_xi_per_x
            for l in 1:collections_per_sample
                ξoos[l, k, :, n] = polynomial_demand(context)
            end
        end
    end

    in_sample = Dict((collect(base_contexts[:, i]), ξ[:, i]) for i in 1:Nsamples)
    out_of_sample = Dict((collect(base_contexts[:, Nsamples + n]), ξoos[:, :, :, n]) for n in 1:Noutofsamples)

    return in_sample, out_of_sample
end
