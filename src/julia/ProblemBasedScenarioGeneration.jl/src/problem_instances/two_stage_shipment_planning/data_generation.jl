"""
    dataGeneration(instance::ShipmentPlanningProblem, Nsamples, Noutofsamples, N_xi_per_x, σ, seasonal_scale, trend_decay;
                   collections_per_sample=30)

Creates synthetic contextual features and demand scenarios for the shipment planning problem.  The function mirrors the signature
of the resource allocation generator so it can be dropped into existing experimentation code with minimal friction.

Arguments:
- `instance`: shipment planning problem instance.
- `Nsamples`: number of in-sample observations.
- `Noutofsamples`: number of out-of-sample contexts.
- `N_xi_per_x`: number of scenario draws per out-of-sample context.
- `σ`: standard deviation of the demand noise.
- `seasonal_scale`: amplitude of the sinusoidal seasonal component applied to the contextual signal.
- `trend_decay`: controls a mild linear trend injected into the baseline demand.
- `collections_per_sample`: replicates of each out-of-sample draw (defaults to 30 to mirror the resource allocation helper).

Returns:
- `in_sample`: dictionary mapping feature vectors to demand realizations.
- `out_of_sample`: dictionary mapping feature vectors to 4-D tensors of scenario draws.
- `baseline_demand`: vector describing the stationary demand component used in the generator.
- `feature_sensitivity`: matrix capturing how each contextual feature perturbs demand.
"""
function dataGeneration(instance::ShipmentPlanningProblem,
                        Nsamples,
                        Noutofsamples,
                        N_xi_per_x,
                        σ,
                        seasonal_scale,
                        trend_decay;
                        collections_per_sample::Int = 30)
    markets = length(instance.problem_data.first_stage_costs)
    context_dim = instance.problem_data.context_dimension

    # Placeholder coefficients – replace once calibrated numbers from the paper become available.
    baseline_demand = 40 .+ 5 .* collect(0:(markets - 1))
    feature_sensitivity = ones(Float64, markets, context_dim)
    feature_sensitivity .*= reshape(range(0.8, 1.2; length=markets), markets, 1)

    cov = Matrix{Float64}(I, context_dim, context_dim)
    base_contexts = rand(MvNormal(zeros(context_dim), cov), Nsamples + Noutofsamples)
    base_contexts .= clamp.(base_contexts, -2.5, 2.5)

    function synthetic_demand(context, idx)
        seasonal = seasonal_scale .* sin.(range(0, stop=2π, length=markets) .+ context[1])
        trend = trend_decay .* idx ./ max(Nsamples, 1)
        demand = baseline_demand .+ seasonal .+ trend .+ feature_sensitivity * context
        demand .+= σ .* randn(markets)
        return max.(demand, 0.0)
    end

    ξ = zeros(Float64, markets, Nsamples)
    for i in 1:Nsamples
        ξ[:, i] = synthetic_demand(base_contexts[:, i], i)
    end

    ξoos = zeros(Float64, collections_per_sample, N_xi_per_x, markets, Noutofsamples)
    for n in 1:Noutofsamples
        context = base_contexts[:, Nsamples + n]
        for k in 1:N_xi_per_x
            for l in 1:collections_per_sample
                ξoos[l, k, :, n] = synthetic_demand(context, Nsamples + n)
            end
        end
    end

    in_sample_pairs = Vector{Tuple{Vector{Float64}, Vector{Float64}}}(undef, Nsamples)
    for i in 1:Nsamples
        in_sample_pairs[i] = (collect(base_contexts[:, i]), ξ[:, i])
    end
    in_sample = Dict(in_sample_pairs)

    out_of_sample_pairs = Vector{Tuple{Vector{Float64}, Array{Float64, 3}}}(undef, Noutofsamples)
    for n in 1:Noutofsamples
        out_of_sample_pairs[n] = (collect(base_contexts[:, Nsamples + n]), ξoos[:, :, :, n])
    end
    out_of_sample = Dict(out_of_sample_pairs)

    return in_sample, out_of_sample, baseline_demand, feature_sensitivity
end
