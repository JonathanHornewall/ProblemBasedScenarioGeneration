import LinearAlgebra
import Random

struct RandomYieldProblem <: ProgramInstance
    product_count::Int
    activity_count::Int
    context_dim::Int
    support_count::Int
    sigma_W::Float64
    demand_mean::Vector{Float64}
    B::Float64
    alpha::Vector{Float64}
    beta::Matrix{Float64}
    W_support::Vector{Matrix{Float64}}
    stochastic_program::ContextualDFL.StochasticProgram
    base_scenario::NamedTuple
end

function RandomYieldProblem(;
    r=20,
    a=40,
    context_dim=3,
    K_support=20,
    sigma_W=0.25,
    parameter_seed=1,
    demand_mean=nothing,
    B=nothing,
    alpha=nothing,
    beta=nothing,
)
    product_count = _checked_positive_integer(r, :r)
    activity_count = _checked_positive_integer(a, :a)
    checked_context_dim = _checked_positive_integer(context_dim, :context_dim)
    support_count = _checked_positive_integer(K_support, :K_support)
    checked_sigma_W = Float64(sigma_W)
    checked_sigma_W >= 0.0 || throw(ArgumentError("sigma_W must be nonnegative."))

    rng = Random.MersenneTwister(parameter_seed)
    checked_demand_mean = _checked_vector_or_default(
        demand_mean,
        fill(2.0, product_count),
        product_count,
        :demand_mean,
    )
    checked_B = isnothing(B) ? 0.5 * sum(checked_demand_mean) : Float64(B)
    checked_B > 0.0 || throw(ArgumentError("B must be positive."))

    checked_alpha = _checked_vector_or_default(
        alpha,
        0.2 .* randn(rng, support_count),
        support_count,
        :alpha,
    )
    checked_beta = _checked_matrix_or_default(
        beta,
        0.6 .* randn(rng, support_count, checked_context_dim),
        support_count,
        checked_context_dim,
        :beta,
    )

    W_support = _sample_random_yield_support(
        rng,
        product_count,
        activity_count,
        support_count,
        checked_sigma_W,
    )
    program, scenario = _random_yield_program_and_scenario(
        product_count,
        activity_count,
        checked_B,
        checked_demand_mean,
        first(W_support),
    )

    return RandomYieldProblem(
        product_count,
        activity_count,
        checked_context_dim,
        support_count,
        checked_sigma_W,
        checked_demand_mean,
        checked_B,
        checked_alpha,
        checked_beta,
        W_support,
        program,
        scenario,
    )
end

stochastic_program(problem::RandomYieldProblem) = problem.stochastic_program

base_scenario(problem::RandomYieldProblem) = problem.base_scenario

function _random_yield_program_and_scenario(
    product_count,
    activity_count,
    budget,
    demand_mean,
    base_W_eq,
)
    recourse_count = activity_count + 2 * product_count

    program = ContextualDFL.StochasticProgram(
        A_eq=ones(Float64, 1, product_count),
        A_ineq=-Matrix{Float64}(LinearAlgebra.I, product_count, product_count),
        b_eq=[Float64(budget)],
        b_ineq=zeros(Float64, product_count),
        c=fill(1.0, product_count),
    )

    scenario = (;
        W_eq=base_W_eq,
        W_ineq=-Matrix{Float64}(LinearAlgebra.I, recourse_count, recourse_count),
        T_eq=Matrix{Float64}(LinearAlgebra.I, product_count, product_count),
        T_ineq=zeros(Float64, recourse_count, product_count),
        h_eq=copy(demand_mean),
        h_ineq=zeros(Float64, recourse_count),
        q=vcat(fill(2.0, activity_count), fill(50.0, product_count), fill(0.1, product_count)),
    )

    return program, scenario
end

function _sample_random_yield_support(
    rng::Random.AbstractRNG,
    product_count,
    activity_count,
    support_count,
    sigma_W,
)
    mask = rand(rng, product_count, activity_count) .< 0.25
    for product in 1:product_count
        mask[product, rand(rng, 1:activity_count)] = true
    end
    for activity in 1:activity_count
        mask[rand(rng, 1:product_count), activity] = true
    end

    Y_bar = zeros(Float64, product_count, activity_count)
    for index in eachindex(Y_bar)
        if mask[index]
            Y_bar[index] = 0.5 + rand(rng)
        end
    end

    support = Matrix{Float64}[]
    push!(support, _random_yield_W_eq(Y_bar))
    for _ in 2:support_count
        Y = copy(Y_bar)
        for index in eachindex(Y)
            if mask[index]
                Y[index] *= exp(sigma_W * randn(rng))
            end
        end
        push!(support, _random_yield_W_eq(Y))
    end
    return support
end

function _random_yield_W_eq(Y::AbstractMatrix)
    product_count, _ = size(Y)
    return hcat(
        Matrix{Float64}(Y),
        Matrix{Float64}(LinearAlgebra.I, product_count, product_count),
        -Matrix{Float64}(LinearAlgebra.I, product_count, product_count),
    )
end
