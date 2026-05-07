using ContextualDFL
using ContextualDFLExperiments
using Dates
using LinearAlgebra
using Printf
using Random
using Sockets

function logmsg(message)
    println(Dates.format(now(Dates.UTC), dateformat"yyyy-mm-ddTHH:MM:SS"), "Z ", message)
    flush(stdout)
end

function unit_demand_base_scenario(problem, base)
    h_eq = copy(base.h_eq)
    resource_count = length(problem.problem_data.first_stage_costs)
    demand_count = length(problem.problem_data.second_stage_costs)
    h_eq[1:resource_count] .= 0.0
    h_eq[(resource_count + 1):(resource_count + demand_count)] .= 1.0

    return ContextualDFL.ParametricScenario(;
        W_eq_xi=copy(base.W_eq),
        W_ineq_xi=copy(base.W_ineq),
        T_eq_xi=copy(base.T_eq),
        T_ineq_xi=copy(base.T_ineq),
        h_eq_xi=h_eq,
        h_ineq_xi=copy(base.h_ineq),
        q_xi=copy(base.q),
    )
end

function converted_q_scenario(base, q_star)
    return ContextualDFL.ParametricScenario(;
        W_eq_xi=copy(base.W_eq),
        W_ineq_xi=copy(base.W_ineq),
        T_eq_xi=copy(base.T_eq),
        T_ineq_xi=copy(base.T_ineq),
        h_eq_xi=copy(base.h_eq),
        h_ineq_xi=copy(base.h_ineq),
        q_xi=copy(q_star),
    )
end

function decode_arrays(decoder, scenario_collection)
    return ContextualDFL.decode_scenario_collection(decoder, scenario_collection)
end

function solve_arrays(solver, program, arrays; mu, rho, constraint_tolerance)
    return ContextualDFL.solve(
        solver,
        program,
        arrays...;
        μ=mu,
        ρ=rho,
        constraint_tolerance=constraint_tolerance,
    )
end

function build_raw_prediction(q_dim; seed)
    rng = Random.MersenneTwister(seed)
    return 0.03 .* randn(rng, q_dim)
end

function q_variant(base, lambda_bar, lambda0; orientation::Symbol, dual_sign::Float64)
    signed_bar = dual_sign .* lambda_bar
    signed_0 = dual_sign .* lambda0
    delta =
        orientation === :bar_minus_base ? signed_bar .- signed_0 :
        orientation === :base_minus_bar ? signed_0 .- signed_bar :
        throw(ArgumentError("unknown orientation $(orientation)"))
    return vec(base.q .+ transpose(base.W_eq) * delta)
end

function base_dual_at_z(solver, base, z; mu, rho, constraint_tolerance)
    _, lambda0_eq, _ = ContextualDFL.G_hat(
        solver,
        z,
        base.W_eq,
        base.W_ineq,
        base.T_eq,
        base.T_ineq,
        base.h_eq,
        base.h_ineq,
        base.q;
        μ=mu,
        ρ=rho,
        return_dual=true,
        constraint_tolerance=constraint_tolerance,
    )
    return lambda0_eq
end

softplus_stable(x) = max(x, zero(x)) + log1p(exp(-abs(x)))
sigmoid_stable(x) = x >= 0 ? inv(1 + exp(-x)) : begin
    ex = exp(x)
    ex / (1 + ex)
end

function spo_objective(program, z, y, q_array)
    K = size(q_array, 2)
    value = sum(program.first_stage_lp.c .* z)
    for k in 1:K
        value += (1.0 / K) * sum(view(q_array, :, k) .* view(y, :, k))
    end
    return value
end

function spo_random_metrics(
    solver,
    program,
    base_scenario,
    q_star,
    converted_scenario,
    raw_prediction;
    lower_bound_margin,
    constraint_tolerance,
)
    q_lb = q_star .- lower_bound_margin
    q_pred = q_lb .+ softplus_stable.(raw_prediction)
    q_perturbed = 2 .* q_pred .- q_star

    reference_arrays = decode_arrays(ContextualDFL.ParametricDecoder(), [converted_scenario])
    perturbed_arrays = (
        reference_arrays[1],
        reference_arrays[2],
        reference_arrays[3],
        reference_arrays[4],
        reference_arrays[5],
        reference_arrays[6],
        reshape(q_perturbed, :, 1),
    )

    reference_z, reference_y, _, _, _, _ = solve_arrays(
        solver,
        program,
        reference_arrays;
        mu=0.0,
        rho=0.0,
        constraint_tolerance=constraint_tolerance,
    )
    perturbed_z, perturbed_y, _, _, _, _ = solve_arrays(
        solver,
        program,
        perturbed_arrays;
        mu=0.0,
        rho=0.0,
        constraint_tolerance=constraint_tolerance,
    )

    q_perturbed_array = reshape(q_perturbed, :, 1)
    value =
        spo_objective(program, reference_z, reference_y, q_perturbed_array) -
        spo_objective(program, perturbed_z, perturbed_y, q_perturbed_array)

    dq_pred = 2 .* vec(reference_y[:, 1] .- perturbed_y[:, 1])
    draw = dq_pred .* sigmoid_stable.(raw_prediction)
    return Float64(value), Float64(norm(draw))
end

function write_rows_csv(path, rows)
    header = [
        "mu",
        "rho",
        "variant",
        "orientation",
        "dual_sign",
        "decoded_q_match",
        "decoded_q_max_abs_diff",
        "z_diff_norm",
        "z_relative_diff",
        "z_converted_norm",
        "z_converted_sum",
        "q_min",
        "q_max",
        "q_negative_count",
        "spo_loss_random_init_mu0",
        "gradient_norm_mu0",
        "reproduces_z",
        "status",
    ]
    open(path, "w") do io
        println(io, join(header, ","))
        for row in rows
            values = [string(getproperty(row, Symbol(col))) for col in header]
            println(io, join(replace.(values, "," => ";"), ","))
        end
    end
end

function main()
    seed = 20260507
    rho = 0.0
    constraint_tolerance = 1e-8
    lower_bound_margin = 1e-4
    z_atol = 1e-5
    z_rtol = 1e-6
    mus = (0.0, 1e-4, 1e-3)
    variants = (
        (; name="q0_plus_Wt_lambda_bar_minus_lambda0", orientation=:bar_minus_base, dual_sign=1.0),
        (; name="q0_plus_Wt_lambda0_minus_lambda_bar", orientation=:base_minus_bar, dual_sign=1.0),
        (; name="q0_plus_Wt_neg_lambda_bar_minus_neg_lambda0", orientation=:bar_minus_base, dual_sign=-1.0),
        (; name="q0_plus_Wt_neg_lambda0_minus_neg_lambda_bar", orientation=:base_minus_bar, dual_sign=-1.0),
    )

    logmsg("host=$(Sockets.gethostname()) pid=$(getpid()) seed=$(seed)")
    solver = ContextualDFL.Solver(ContextualDFL.IpoptSolver(), ContextualDFL.HiGHSSolver())
    problem = ContextualDFLExperiments.ResourceAllocationProblem(
        ContextualDFLExperiments.default_resource_allocation_problem_data(),
    )
    program = ContextualDFLExperiments.stochastic_program(problem)
    original_decoder = ContextualDFLExperiments.ResourceAllocationDemandParametricDecoder(problem)
    base_scenario = unit_demand_base_scenario(
        problem,
        ContextualDFLExperiments.base_scenario(problem),
    )
    base = ContextualDFLExperiments.full_base_scenario_arrays(base_scenario)

    data = ContextualDFLExperiments.generate_benchmark_dataset(
        problem;
        n_contexts=1,
        scenarios_per_context=1,
        seed=seed,
    )
    datapoint = only(data)
    original_arrays = decode_arrays(original_decoder, datapoint.scenario_parameters)
    raw_prediction = build_raw_prediction(length(base.q); seed=seed + 99)

    resource_count = length(problem.problem_data.first_stage_costs)
    demand_count = length(problem.problem_data.second_stage_costs)
    logmsg("resource_count=$(resource_count) demand_count=$(demand_count) q_dim=$(length(base.q))")
    logmsg(
        "base_h_resource_min=$(minimum(base.h_eq[1:resource_count])) " *
        "base_h_resource_max=$(maximum(base.h_eq[1:resource_count])) " *
        "base_h_demand_min=$(minimum(base.h_eq[(resource_count + 1):(resource_count + demand_count)])) " *
        "base_h_demand_max=$(maximum(base.h_eq[(resource_count + 1):(resource_count + demand_count)]))",
    )
    logmsg(
        "original_h_min=$(minimum(original_arrays[5])) original_h_max=$(maximum(original_arrays[5])) " *
        "raw_prediction_norm=$(norm(raw_prediction))",
    )

    rows = NamedTuple[]
    for mu in mus
        logmsg("solving original datapoint with mu=$(mu) rho=$(rho)")
        z_star, _, _, _, lambda_h_eq_array, _ = solve_arrays(
            solver,
            program,
            original_arrays;
            mu=mu,
            rho=rho,
            constraint_tolerance=constraint_tolerance,
        )
        lambda_bar = vec(sum(lambda_h_eq_array; dims=2))
        lambda0 = base_dual_at_z(
            solver,
            base,
            z_star;
            mu=mu,
            rho=rho,
            constraint_tolerance=constraint_tolerance,
        )
        logmsg(
            "mu=$(mu) z_star_norm=$(norm(z_star)) z_star_sum=$(sum(z_star)) " *
            "lambda_bar_norm=$(norm(lambda_bar)) lambda0_norm=$(norm(lambda0))",
        )

        for variant in variants
            q_star = q_variant(
                base,
                lambda_bar,
                lambda0;
                orientation=variant.orientation,
                dual_sign=variant.dual_sign,
            )
            scenario_star = converted_q_scenario(base, q_star)
            decoded = decode_arrays(ContextualDFL.ParametricDecoder(), [scenario_star])
            decoded_q = vec(decoded[7][:, 1])
            decoded_q_max_abs_diff = maximum(abs.(decoded_q .- q_star))
            decoded_q_match = isapprox(decoded_q, q_star; atol=1e-10, rtol=0.0)
            decoded_q_match || error(
                "ParametricDecoder failed q_xi round trip for $(variant.name); " *
                "max abs diff $(decoded_q_max_abs_diff)",
            )

            status = "ok"
            z_converted = similar(z_star)
            z_diff_norm = NaN
            z_relative_diff = NaN
            z_converted_norm = NaN
            z_converted_sum = NaN
            spo_loss_value = NaN
            grad_norm = NaN
            reproduces_z = false

            try
                z_converted, _, _, _, _, _ = solve_arrays(
                    solver,
                    program,
                    decoded;
                    mu=mu,
                    rho=rho,
                    constraint_tolerance=constraint_tolerance,
                )
                z_diff_norm = norm(z_converted .- z_star)
                z_relative_diff = z_diff_norm / max(norm(z_star), eps(Float64))
                z_converted_norm = norm(z_converted)
                z_converted_sum = sum(z_converted)
                reproduces_z = isapprox(z_converted, z_star; atol=z_atol, rtol=z_rtol)

                spo_loss_value, grad_norm = spo_random_metrics(
                    solver,
                    program,
                    base_scenario,
                    q_star,
                    scenario_star,
                    raw_prediction;
                    lower_bound_margin=lower_bound_margin,
                    constraint_tolerance=constraint_tolerance,
                )
            catch err
                status = sprint(showerror, err)
            end

            row = (;
                mu=mu,
                rho=rho,
                variant=variant.name,
                orientation=variant.orientation,
                dual_sign=variant.dual_sign,
                decoded_q_match=decoded_q_match,
                decoded_q_max_abs_diff=decoded_q_max_abs_diff,
                z_diff_norm=z_diff_norm,
                z_relative_diff=z_relative_diff,
                z_converted_norm=z_converted_norm,
                z_converted_sum=z_converted_sum,
                q_min=minimum(q_star),
                q_max=maximum(q_star),
                q_negative_count=count(<(0.0), q_star),
                spo_loss_random_init_mu0=spo_loss_value,
                gradient_norm_mu0=grad_norm,
                reproduces_z=reproduces_z,
                status=status,
            )
            push!(rows, row)
            @printf(
                "mu=%8.1e variant=%-45s decoded_q=%5s z_diff=%12.6g rel=%12.6g spo_loss=%12.6g grad=%12.6g q_min=%12.6g q_max=%12.6g reproduces_z=%5s status=%s\n",
                row.mu,
                row.variant,
                string(row.decoded_q_match),
                row.z_diff_norm,
                row.z_relative_diff,
                row.spo_loss_random_init_mu0,
                row.gradient_norm_mu0,
                row.q_min,
                row.q_max,
                string(row.reproduces_z),
                row.status,
            )
            flush(stdout)
        end
    end

    output_path = joinpath(@__DIR__, "q_conversion_invariant_report.csv")
    write_rows_csv(output_path, rows)
    logmsg("wrote $(output_path)")

    if any(row -> row.reproduces_z, rows)
        logmsg("at least one conversion variant reproduces z_star")
    else
        logmsg("no conversion variant reproduced z_star")
    end

    return nothing
end

main()
