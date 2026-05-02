import Distributions
import Random

struct ResourceAllocationScenarioDataGenerator{TIntercepts,TSlopes,TSigma,TPower,TRng} <:
       ScenarioDataGenerator
    intercepts::TIntercepts
    slopes::TSlopes
    sigma::TSigma
    p::TPower
    L::Int
    rng::TRng
end

function sample_resource_allocation_demand_parameters(
    rng::Random.AbstractRNG,
    demand_count::Integer,
)
    intercepts = 50 .+ 5 .* rand(rng, Distributions.Normal(0, 1), demand_count)
    B1 = 10 .+ rand(rng, Distributions.Uniform(-4, 4), demand_count)
    B2 = 5 .+ rand(rng, Distributions.Uniform(-4, 4), demand_count)
    B3 = 2 .+ rand(rng, Distributions.Uniform(-4, 4), demand_count)
    return intercepts, hcat(B1, B2, B3)
end

function ResourceAllocationScenarioDataGenerator(
    problem::ResourceAllocationProblem;
    sigma,
    p,
    L,
    rng::Random.AbstractRNG=Random.default_rng(),
    intercepts=nothing,
    slopes=nothing,
)
    L <= 3 || throw(ArgumentError("resource-allocation data generation has three context terms."))
    demand_count = size(problem.problem_data.service_rate_parameters, 2)
    sampled_intercepts, sampled_slopes = if isnothing(intercepts) || isnothing(slopes)
        sample_resource_allocation_demand_parameters(rng, demand_count)
    else
        intercepts, slopes
    end
    length(sampled_intercepts) == demand_count ||
        throw(DimensionMismatch("intercepts must have one entry per demand."))
    size(sampled_slopes, 1) == demand_count && size(sampled_slopes, 2) >= L ||
        throw(DimensionMismatch("slopes must have demand_count rows and at least L columns."))

    return ResourceAllocationScenarioDataGenerator(
        Vector{Float64}(sampled_intercepts),
        Matrix{Float64}(sampled_slopes),
        sigma,
        p,
        Int(L),
        rng,
    )
end

function (generator::ResourceAllocationScenarioDataGenerator)(context)
    length(context) >= generator.L ||
        throw(DimensionMismatch("context must have at least $(generator.L) entries."))

    demand = zeros(Float64, length(generator.intercepts))
    for demand_index in eachindex(demand)
        signal = generator.intercepts[demand_index]
        for term in 1:generator.L
            signal += generator.slopes[demand_index, term] * context[term]^generator.p
        end
        demand[demand_index] = signal + rand(generator.rng, Distributions.Normal(0, generator.sigma))
    end

    return ContextualDFL.ParametricScenario(;
        W_eq_xi=Float64[],
        W_ineq_xi=Float64[],
        T_eq_xi=Float64[],
        T_ineq_xi=Float64[],
        h_eq_xi=demand,
        h_ineq_xi=Float64[],
        q_xi=Float64[],
    )
end
