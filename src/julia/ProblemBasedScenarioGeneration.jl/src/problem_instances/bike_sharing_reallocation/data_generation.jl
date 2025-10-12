"""
    dataGeneration(instance::BikeSharingReallocationProblem, Nsamples, Noutofsamples, N_xi_per_x, σ,
                   commuter_peak_factor, weather_volatility; collections_per_sample=30)

Synthetic data generator for the bike sharing reallocation problem.  The signature mirrors the resource allocation helper to
simplify experimentation code reuse.

Arguments:
- `instance`: bike sharing problem instance.
- `Nsamples`: number of in-sample observations.
- `Noutofsamples`: number of contextual observations reserved for evaluation.
- `N_xi_per_x`: number of demand draws per out-of-sample context.
- `σ`: standard deviation of the demand noise.
- `commuter_peak_factor`: controls the amplitude of peak-hour surges driven by commuting.
- `weather_volatility`: scales the impact of weather uncertainty on demand.
- `collections_per_sample`: number of repeated scenario collections for each context (default 30).

Returns:
- `in_sample`: dictionary from context vectors to demand realizations.
- `out_of_sample`: dictionary from context vectors to 4-D tensors containing scenario collections.
- `baseline_demand`: nominal bikes requested per station before context adjustments.
- `context_effects`: matrix describing how each contextual feature perturbs demand.
"""
function dataGeneration(instance::BikeSharingReallocationProblem,
                        Nsamples,
                        Noutofsamples,
                        N_xi_per_x,
                        σ,
                        commuter_peak_factor,
                        weather_volatility;
                        collections_per_sample::Int = 30)
    stations = length(instance.problem_data.first_stage_costs)
    context_dim = instance.problem_data.context_dimension

    # Placeholder signal components.
    baseline_demand = 120 .+ 10 .* sin.(range(0, stop=π, length=stations))
    context_effects = zeros(Float64, stations, context_dim)
    context_effects[:, 1] .= commuter_peak_factor .* range(0.6, 1.2; length=stations)
    context_effects[:, 2] .= -weather_volatility .* range(0.3, 0.9; length=stations)
    if context_dim >= 3
        context_effects[:, 3] .= 5 .* range(1.0, 0.5; length=stations)
    end
    if context_dim >= 4
        context_effects[:, 4] .= range(-2.0, 2.0; length=stations)
    end

    cov = 0.5 .* ones(Float64, context_dim, context_dim)
    cov += 0.5 .* Matrix{Float64}(I, context_dim, context_dim)
    contexts = rand(MvNormal(zeros(context_dim), Symmetric(cov)), Nsamples + Noutofsamples)

    function demand_from_context(context)
        seasonal_profile = commuter_peak_factor .* cos.(range(0, stop=2π, length=stations) .+ context[1])
        weather_shock = weather_volatility .* randn(stations)
        demand = baseline_demand .+ seasonal_profile .+ context_effects * context + weather_shock
        demand .+= σ .* randn(stations)
        return max.(demand, 0.0)
    end

    ξ = zeros(Float64, stations, Nsamples)
    for i in 1:Nsamples
        ξ[:, i] = demand_from_context(contexts[:, i])
    end

    ξoos = zeros(Float64, collections_per_sample, N_xi_per_x, stations, Noutofsamples)
    for n in 1:Noutofsamples
        context = contexts[:, Nsamples + n]
        for k in 1:N_xi_per_x
            for l in 1:collections_per_sample
                ξoos[l, k, :, n] = demand_from_context(context)
            end
        end
    end

    in_sample_pairs = Vector{Tuple{Vector{Float64}, Vector{Float64}}}(undef, Nsamples)
    for i in 1:Nsamples
        in_sample_pairs[i] = (collect(contexts[:, i]), ξ[:, i])
    end
    in_sample = Dict(in_sample_pairs)

    out_of_sample_pairs = Vector{Tuple{Vector{Float64}, Array{Float64, 3}}}(undef, Noutofsamples)
    for n in 1:Noutofsamples
        out_of_sample_pairs[n] = (collect(contexts[:, Nsamples + n]), ξoos[:, :, :, n])
    end
    out_of_sample = Dict(out_of_sample_pairs)

    return in_sample, out_of_sample, baseline_demand, context_effects
end
