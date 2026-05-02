import Distributions
import LinearAlgebra
import Random

struct ResourceAllocationContextDataGenerator{TCorrelation,TRng} <: ContextDataGenerator
    correlation_matrix::TCorrelation
    rng::TRng
end

function sample_resource_allocation_correlation_matrix(
    rng::Random.AbstractRNG,
    dimension::Integer=3,
)
    beta_parameter = 2.0
    partial_correlation = zeros(Float64, dimension, dimension)
    correlation = Matrix{Float64}(LinearAlgebra.I, dimension, dimension)

    for k in 1:(dimension - 1)
        for i in (k + 1):dimension
            partial_correlation[k, i] =
                (rand(rng, Distributions.Beta(beta_parameter, beta_parameter)) - 0.5) * 2.0
            rho = partial_correlation[k, i]
            for j in (k - 1):-1:1
                rho =
                    rho *
                    sqrt((1 - partial_correlation[j, i]^2) * (1 - partial_correlation[j, k]^2)) +
                    partial_correlation[j, i] * partial_correlation[j, k]
            end
            correlation[k, i] = rho
            correlation[i, k] = rho
        end
    end

    permutation = Random.randperm(rng, dimension)
    return correlation[permutation, permutation]
end

function ResourceAllocationContextDataGenerator(;
    rng::Random.AbstractRNG=Random.default_rng(),
    correlation_matrix=sample_resource_allocation_correlation_matrix(rng, 3),
)
    return ResourceAllocationContextDataGenerator(Matrix{Float64}(correlation_matrix), rng)
end

function (generator::ResourceAllocationContextDataGenerator)()
    distribution = Distributions.MvNormal(
        zeros(size(generator.correlation_matrix, 1)),
        LinearAlgebra.Symmetric(generator.correlation_matrix + 1e-8LinearAlgebra.I),
    )
    return abs.(rand(generator.rng, distribution))
end
