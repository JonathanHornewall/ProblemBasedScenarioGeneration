import ChainRulesCore
import Distributions
import Flux
import LinearAlgebra
import Random
import Statistics

module ResourceAllocationLegacyParameters
include(
    normpath(
        joinpath(
            @__DIR__,
            "..",
            "..",
            "..",
            "..",
            "ProblemBasedScenarioGeneration",
            "src",
            "problem_instances",
            "resource_allocation",
            "parameters.jl",
        ),
    ),
)
end

struct ResourceAllocationProblemData
    service_rate_parameters::Matrix{Float64}
    first_stage_costs::Vector{Float64}
    second_stage_costs::Vector{Float64}
    yield_parameters::Vector{Float64}
end

struct ResourceAllocationTestInstance
    problem_data::ResourceAllocationProblemData
    legacy_first_stage::NamedTuple
    stochastic_program::StochasticProgram
    base_scenario::NamedTuple
end

struct ResourceAllocationDemandDecoder{TB} <: ScenarioDecoder
    base_scenario::TB
    resource_count::Int
    demand_count::Int
end

function imported_resource_allocation_data()
    data = ResourceAllocationProblemData(
        Matrix{Float64}(ResourceAllocationLegacyParameters.μᵢⱼ),
        vec(Float64.(ResourceAllocationLegacyParameters.cz)),
        vec(Float64.(ResourceAllocationLegacyParameters.qw)),
        vec(Float64.(ResourceAllocationLegacyParameters.ρᵢ)),
    )

    resource_count, demand_count = size(data.service_rate_parameters)
    length(data.first_stage_costs) == resource_count ||
        throw(DimensionMismatch("first-stage costs must match resource count."))
    length(data.second_stage_costs) == demand_count ||
        throw(DimensionMismatch("second-stage costs must match demand count."))
    length(data.yield_parameters) == resource_count ||
        throw(DimensionMismatch("yield parameters must match resource count."))

    return data
end

function resource_allocation_instance(; resource_indices=:, demand_indices=:)
    imported = imported_resource_allocation_data()
    service_rates = Matrix{Float64}(imported.service_rate_parameters[resource_indices, demand_indices])
    first_costs = vec(Float64.(imported.first_stage_costs[resource_indices]))
    second_costs = vec(Float64.(imported.second_stage_costs[demand_indices]))
    yields = vec(Float64.(imported.yield_parameters[resource_indices]))
    data = ResourceAllocationProblemData(service_rates, first_costs, second_costs, yields)

    resource_count, demand_count = size(service_rates)
    recourse_variables = demand_count + resource_count * demand_count + resource_count + demand_count
    recourse_rows = resource_count + demand_count

    W_eq = zeros(Float64, recourse_rows, recourse_variables)
    for resource_index in 1:resource_count
        for demand_index in 1:demand_count
            allocation_index = demand_count + demand_count * (resource_index - 1) + demand_index
            W_eq[resource_index, allocation_index] = 1.0
        end
        W_eq[resource_index, demand_count + resource_count * demand_count + resource_index] = 1.0
    end

    for demand_index in 1:demand_count
        row = resource_count + demand_index
        W_eq[row, demand_index] = 1.0
        for resource_index in 1:resource_count
            allocation_index = demand_count + demand_count * (resource_index - 1) + demand_index
            W_eq[row, allocation_index] = service_rates[resource_index, demand_index]
        end
        slack_index = demand_count + resource_count * demand_count + resource_count + demand_index
        W_eq[row, slack_index] = -1.0
    end

    T_eq = zeros(Float64, recourse_rows, resource_count)
    for resource_index in 1:resource_count
        T_eq[resource_index, resource_index] = -yields[resource_index]
    end

    q = zeros(Float64, recourse_variables)
    q[1:demand_count] .= second_costs

    first_stage_nonnegativity = -Matrix{Float64}(LinearAlgebra.I, resource_count, resource_count)
    recourse_nonnegativity = -Matrix{Float64}(LinearAlgebra.I, recourse_variables, recourse_variables)

    program = StochasticProgram(
        A_eq=zeros(Float64, 0, resource_count),
        A_ineq=first_stage_nonnegativity,
        b_eq=Float64[],
        b_ineq=zeros(Float64, resource_count),
        c=first_costs,
    )

    base_scenario = (;
        W_eq=W_eq,
        W_ineq=recourse_nonnegativity,
        T_eq=T_eq,
        T_ineq=zeros(Float64, recourse_variables, resource_count),
        h_ineq=zeros(Float64, recourse_variables),
        q=q,
    )

    legacy_first_stage = (;
        A=zeros(Float64, 1, resource_count),
        b=[0.0],
        c=first_costs,
    )

    return ResourceAllocationTestInstance(data, legacy_first_stage, program, base_scenario)
end

ResourceAllocationDemandDecoder(instance::ResourceAllocationTestInstance) =
    ResourceAllocationDemandDecoder(
        instance.base_scenario,
        size(instance.problem_data.service_rate_parameters, 1),
        size(instance.problem_data.service_rate_parameters, 2),
    )

function (decoder::ResourceAllocationDemandDecoder)(scenario_parameter)
    raw = _resource_allocation_demand_or_rhs(decoder, scenario_parameter)
    h_eq = if length(raw) == decoder.demand_count
        vcat(zeros(eltype(raw), decoder.resource_count), raw)
    elseif length(raw) == decoder.resource_count + decoder.demand_count
        raw
    else
        throw(
            DimensionMismatch(
                "resource allocation scenario parameter has length $(length(raw)); " *
                "expected $(decoder.demand_count) or $(decoder.resource_count + decoder.demand_count).",
            ),
        )
    end

    return (
        decoder.base_scenario.W_eq,
        decoder.base_scenario.W_ineq,
        decoder.base_scenario.T_eq,
        decoder.base_scenario.T_ineq,
        h_eq,
        decoder.base_scenario.h_ineq,
        decoder.base_scenario.q,
    )
end

function _resource_allocation_demand_or_rhs(decoder, scenario_parameter)
    value = if scenario_parameter isa AbstractVector
        scenario_parameter
    elseif hasproperty(scenario_parameter, :h_eq)
        getproperty(scenario_parameter, :h_eq)
    elseif hasproperty(scenario_parameter, :h)
        getproperty(scenario_parameter, :h)
    else
        throw(ArgumentError("scenario parameter must be a demand vector or have field `h_eq`/`h`."))
    end

    return vec(value)
end

function demand_parameter_collection(demand_matrix::AbstractMatrix)
    return [(; h_eq=view(demand_matrix, :, k)) for k in axes(demand_matrix, 2)]
end

function demand_matrix(scenario_collection)
    return hcat((_resource_allocation_demand_or_rhs(nothing, scenario) for scenario in scenario_collection)...)
end

function decoded_resource_allocation_arrays(
    decoder::ResourceAllocationDemandDecoder,
    scenario_collection,
)
    return decode_scenario_collection(decoder, scenario_collection)
end

function ChainRulesCore.rrule(
    ::typeof(decode_scenario_collection),
    decoder::ResourceAllocationDemandDecoder,
    scenario_parameter_collection::AbstractVector,
)
    output = decode_scenario_collection(decoder, scenario_parameter_collection)

    function resource_allocation_decode_pullback(output_tangent)
        output_tangent = ChainRulesCore.unthunk(output_tangent)
        dh_eq_array = try
            ChainRulesCore.unthunk(output_tangent[5])
        catch
            return (
                ChainRulesCore.NoTangent(),
                ChainRulesCore.NoTangent(),
                ChainRulesCore.NoTangent(),
            )
        end

        dh_eq_array isa AbstractArray ||
            return (
                ChainRulesCore.NoTangent(),
                ChainRulesCore.NoTangent(),
                ChainRulesCore.NoTangent(),
            )

        scenario_parameter_tangents = map(enumerate(scenario_parameter_collection)) do (k, scenario_parameter)
            names = propertynames(scenario_parameter)
            raw = _resource_allocation_demand_or_rhs(decoder, scenario_parameter)
            h_tangent = if length(raw) == decoder.demand_count
                view(dh_eq_array, (decoder.resource_count + 1):(decoder.resource_count + decoder.demand_count), k)
            else
                view(dh_eq_array, :, k)
            end

            values = map(names) do name
                name in (:h_eq, :h) ? h_tangent : ChainRulesCore.NoTangent()
            end
            NamedTuple{names}(values)
        end

        return (
            ChainRulesCore.NoTangent(),
            ChainRulesCore.NoTangent(),
            scenario_parameter_tangents,
        )
    end

    return output, resource_allocation_decode_pullback
end

function generate_random_correlation_matrix(rng::Random.AbstractRNG, dimension::Int)
    beta_parameter = 2.0
    partial_correlation = zeros(Float64, dimension, dimension)
    correlation = Matrix{Float64}(LinearAlgebra.I, dimension, dimension)

    for k in 1:(dimension - 1)
        for i in (k + 1):dimension
            partial_correlation[k, i] = (rand(rng, Distributions.Beta(beta_parameter, beta_parameter)) - 0.5) * 2.0
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

function sample_resource_allocation_demand_parameters(
    rng::Random.AbstractRNG,
    demand_count::Int,
)
    intercept = 50 .+ 5 .* rand(rng, Distributions.Normal(0, 1), demand_count)
    B1 = 10 .+ rand(rng, Distributions.Uniform(-4, 4), demand_count)
    B2 = 5 .+ rand(rng, Distributions.Uniform(-4, 4), demand_count)
    B3 = 2 .+ rand(rng, Distributions.Uniform(-4, 4), demand_count)
    return intercept, hcat(B1, B2, B3)
end

function generate_resource_allocation_context_scenarios(
    instance::ResourceAllocationTestInstance;
    n_contexts::Int,
    n_scenarios::Int,
    sigma::Real,
    p::Real,
    L::Int,
    rng::Random.AbstractRNG=Random.default_rng(),
)
    L <= 3 || throw(ArgumentError("The legacy resource allocation generator has three context terms."))

    demand_count = size(instance.problem_data.service_rate_parameters, 2)
    correlation = generate_random_correlation_matrix(rng, 3)
    distribution = Distributions.MvNormal(zeros(3), LinearAlgebra.Symmetric(correlation + 1e-8LinearAlgebra.I))
    x_array = abs.(rand(rng, distribution, n_contexts))
    intercept, slopes = sample_resource_allocation_demand_parameters(rng, demand_count)

    scenario_collections = Vector{Vector{NamedTuple}}(undef, n_contexts)
    for context_index in 1:n_contexts
        collection = NamedTuple[]
        context = view(x_array, :, context_index)
        for _ in 1:n_scenarios
            demand = zeros(Float64, demand_count)
            for demand_index in 1:demand_count
                signal = intercept[demand_index]
                for term in 1:L
                    signal += slopes[demand_index, term] * context[term]^p
                end
                demand[demand_index] = signal + rand(rng, Distributions.Normal(0, sigma))
            end
            push!(collection, (; h_eq=demand))
        end
        scenario_collections[context_index] = collection
    end

    data = [
        (copy(view(x_array, :, context_index)), scenario_collections[context_index])
        for context_index in 1:n_contexts
    ]

    return (;
        x_array=x_array,
        scenario_collections=scenario_collections,
        data=data,
        demand_intercepts=intercept,
        demand_slopes=slopes,
        correlation_matrix=correlation,
    )
end

function construct_resource_allocation_neural_net(
    instance::ResourceAllocationTestInstance;
    n_scenarios::Int=1,
)
    demand_count = size(instance.problem_data.service_rate_parameters, 2)
    output_dim = demand_count * n_scenarios
    return Flux.Chain(
        Flux.Dense(3, 128, Flux.relu),
        Flux.Dense(128, 128, Flux.relu),
        Flux.Dense(128, 128, Flux.relu),
        Flux.Dense(128, output_dim, Flux.relu),
        x -> reshape(x, demand_count, n_scenarios),
    ) |> Flux.f64
end

function resource_allocation_training_loss(
    predicted_demands,
    reference_collection,
    mu_in=0.0,
    mu_ref=0.0;
    kwargs...,
)
    target = ChainRulesCore.ignore_derivatives() do
        demand_matrix(reference_collection)
    end
    size(predicted_demands) == size(target) ||
        throw(DimensionMismatch("predicted demand matrix and target matrix have different sizes."))
    return Statistics.mean(abs2, predicted_demands .- target)
end

function relative_resource_allocation_training_loss(
    predicted_demands,
    reference_collection,
    mu_in=0.0,
    mu_ref=0.0;
    kwargs...,
)
    target = ChainRulesCore.ignore_derivatives() do
        demand_matrix(reference_collection)
    end
    denominator = max(Statistics.mean(abs2, target), eps(Float64))
    return resource_allocation_training_loss(predicted_demands, reference_collection) / denominator
end

function mean_resource_allocation_training_loss(model, data)
    return Statistics.mean(resource_allocation_training_loss(model(x), scenarios) for (x, scenarios) in data)
end

function resource_allocation_scenario_arrays(instance::ResourceAllocationTestInstance, scenario_collection)
    decoder = ResourceAllocationDemandDecoder(instance)
    return decoded_resource_allocation_arrays(decoder, scenario_collection)
end

function status_is_optimal(status)
    return string(status) in ("OPTIMAL", "LOCALLY_SOLVED")
end

function assert_resource_allocation_feasible(lp::LP, z; atol=1e-6)
    isempty(lp.b_eq) || @test LinearAlgebra.norm(lp.A_eq * z - lp.b_eq, Inf) <= atol
    isempty(lp.b_ineq) || @test maximum(lp.A_ineq * z - lp.b_ineq) <= atol
end

function deterministic_resource_allocation_direction(shape; scale=1.0, phase=0.0)
    values = [scale * sin(index + phase) for index in 1:prod(shape)]
    return reshape(values, shape)
end
