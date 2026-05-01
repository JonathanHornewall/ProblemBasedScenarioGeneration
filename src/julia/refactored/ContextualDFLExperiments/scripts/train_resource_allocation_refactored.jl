using ContextualDFL
using Flux
using LinearAlgebra
using Random
using Statistics

include(joinpath(@__DIR__, "..", "..", "..", "scripts", "resource_allocation_prototype", "parameters.jl"))
const RA_CZ = Float64.(vec(cz))
const RA_QW = Float64.(vec(qw))
const RA_RHO = Float64.(vec(getfield(@__MODULE__, Symbol("ρᵢ"))))
const RA_MU_IJ = Float64.(getfield(@__MODULE__, Symbol("μᵢⱼ")))

struct DemandHDecoder <: ComponentDecoder
    first_stage_rows::Int
end

function (decoder::DemandHDecoder)(demand)
    return vcat(zeros(eltype(demand), decoder.first_stage_rows), vec(demand))
end

load_resource_allocation_parameters() = copy(RA_CZ), copy(RA_QW), copy(RA_RHO), copy(RA_MU_IJ)

function build_resource_allocation_instance()
    cz, qw, rho, mu_ij = load_resource_allocation_parameters()
    I, J = size(mu_ij)
    A = zeros(1, I)
    b = [0.0]
    c = cz

    n2 = J + I * J + I + J
    W = zeros(I + J, n2)
    for i in 1:I
        for j in 1:J
            W[i, J + J * (i - 1) + j] = 1.0
        end
        W[i, J + I * J + i] = 1.0
    end
    for j in 1:J
        W[I + j, j] = 1.0
        for i in 1:I
            W[I + j, J + J * (i - 1) + j] = mu_ij[i, j]
        end
        W[I + j, J + I * J + I + j] = -1.0
    end

    T = zeros(I + J, I)
    for i in 1:I
        T[i, i] = -rho[i]
    end

    q = zeros(n2)
    q[1:J] .= qw

    base_scenario = BaseScenario(
        W,
        zeros(0, n2),
        T,
        zeros(0, I),
        zeros(I + J),
        q,
    )
    program = StochasticProgram(A, zeros(0, I), b, c)
    return (program=program, base_scenario=base_scenario, I=I, J=J)
end

function build_dataset(J::Integer; n_samples::Integer=5, seed::Integer=11, sigma::Real=0.5)
    Random.seed!(seed)
    context_dim = 3
    x_data = abs.(randn(Float32, context_dim, n_samples))
    intercept = Float32.(50 .+ 5 .* randn(J))
    weights = Float32.(hcat(
        10 .+ 4 .* rand(J),
        5 .+ 4 .* rand(J),
        2 .+ 4 .* rand(J),
    ))
    h_data = zeros(Float32, J, n_samples)
    for k in 1:n_samples
        x = x_data[:, k]
        h_data[:, k] .= intercept .+ weights * x .+ Float32(sigma) .* randn(Float32, J)
    end
    return DataSet(x_data, nothing, nothing, max.(h_data, Float32(1.0)), nothing)
end

function main(; n_samples::Integer=parse(Int, get(ENV, "RA_REFACTORED_SAMPLES", "5")),
    epochs::Integer=parse(Int, get(ENV, "RA_REFACTORED_EPOCHS", "2")),
    seed::Integer=parse(Int, get(ENV, "RA_REFACTORED_SEED", "11")))

    instance = build_resource_allocation_instance()
    decoder = DataSetScenarioDecoder(
        DecoderStrategy(h_decoder=DemandHDecoder(instance.I)),
        instance.base_scenario,
        (:h,),
    )
    data_set = build_dataset(instance.J; n_samples=n_samples, seed=seed)
    model = Chain(Dense(3 => 32, relu), Dense(32 => instance.J, softplus))
    solver = GLPKSolver()
    generator = DFLScenarioGenerator(decoder, solver, model, instance.program)

    training = train(
        generator,
        MSEScenLoss(),
        data_set,
        decoder,
        constant_schedule(0.0; length=epochs),
        constant_schedule(0.0; length=epochs),
        constant_schedule(1; length=epochs),
        constant_schedule(1e-3; length=epochs),
        epochs=epochs,
    )

    row = data_set[1]
    actual = decoder(row)
    predicted = generator(row.x)
    result = solve(instance.program, solver, predicted)
    final_value = cost_function(instance.program, result.first_stage_decision, actual; solver=solver)
    println("final_training_loss = ", training.loss_history[end])
    println("final_downstream_value = ", final_value)
    return final_value
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
