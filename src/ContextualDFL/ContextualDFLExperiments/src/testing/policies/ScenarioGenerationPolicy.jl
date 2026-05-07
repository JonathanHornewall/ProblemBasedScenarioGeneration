struct ScenarioGenerationPolicy{
    TGenerator<:ContextualDFL.ScenarioGenerator,
    TSolver,
    TProgram,
    TMu,
    TRho,
} <: Policy
    scenario_generator::TGenerator
    solver::TSolver
    program::TProgram
    mu::TMu
    rho::TRho
    nr_scenarios::Int
end

function ScenarioGenerationPolicy(scenario_generator, solver, program; mu=0, rho=0, nr_scenarios=1)
    nr_scenarios isa Integer && nr_scenarios > 0 ||
        throw(ArgumentError("nr_scenarios must be a positive integer."))
    return ScenarioGenerationPolicy(
        scenario_generator,
        solver,
        program,
        mu,
        rho,
        Int(nr_scenarios),
    )
end

function infer(policy::ScenarioGenerationPolicy, context)
    scenario_parameters = policy.scenario_generator.neural_net(context)
    W_eq, W_ineq, T_eq, T_ineq, h_eq, h_ineq, q = ContextualDFL.decode_scenario_collection(
        policy.scenario_generator.scenario_decoder,
        scenario_parameters;
        nr_scenarios=policy.nr_scenarios,
    )

    z, _, _, _, _, _ = ContextualDFL.solve(
        policy.solver,
        policy.program,
        W_eq,
        W_ineq,
        T_eq,
        T_ineq,
        h_eq,
        h_ineq,
        q;
        μ=policy.mu,
        ρ=policy.rho,
    )

    return z
end
