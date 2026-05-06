import LinearAlgebra

struct UnreliableNewsvendorProblemData
    p::Float64
    c::Float64
    pi::Float64
    eta::Float64

    function UnreliableNewsvendorProblemData(
        p::Real,
        c::Real,
        pi::Real,
        eta::Real,
    )
        values = Float64.((p, c, pi, eta))
        all(isfinite, values) ||
            throw(ArgumentError("newsvendor cost parameters must be finite."))
        return new(values...)
    end

    function UnreliableNewsvendorProblemData(; p=5.0, c=1.0, pi=5.0, eta=0.5)
        return UnreliableNewsvendorProblemData(p, c, pi, eta)
    end
end

struct UnreliableNewsvendorProblem <: ProgramInstance
    problem_data::UnreliableNewsvendorProblemData
    context_dim::Int
    demand_upper_bound::Float64
    stochastic_program::ContextualDFL.StochasticProgram
    base_scenario::NamedTuple
end

function UnreliableNewsvendorProblem(;
    problem_data::UnreliableNewsvendorProblemData=UnreliableNewsvendorProblemData(),
    context_dim=1,
    demand_upper_bound=1.0,
)
    checked_context_dim = _checked_positive_integer(context_dim, :context_dim)
    checked_demand_upper_bound = Float64(demand_upper_bound)
    isfinite(checked_demand_upper_bound) && checked_demand_upper_bound > 0.0 ||
        throw(ArgumentError("demand_upper_bound must be positive and finite."))

    program, scenario = _unreliable_newsvendor_program_and_scenario(problem_data)

    return UnreliableNewsvendorProblem(
        problem_data,
        checked_context_dim,
        checked_demand_upper_bound,
        program,
        scenario,
    )
end

stochastic_program(problem::UnreliableNewsvendorProblem) = problem.stochastic_program

base_scenario(problem::UnreliableNewsvendorProblem) = problem.base_scenario

function _unreliable_newsvendor_program_and_scenario(
    problem_data::UnreliableNewsvendorProblemData,
)
    W_eq = [1.0 -1.0 -1.0; 0.0 0.0 1.0]
    q = [problem_data.p + problem_data.eta, problem_data.pi, problem_data.c - problem_data.p]

    program = ContextualDFL.StochasticProgram(
        A_eq=zeros(Float64, 0, 1),
        A_ineq=-Matrix{Float64}(LinearAlgebra.I, 1, 1),
        b_eq=Float64[],
        b_ineq=[0.0],
        c=[0.0],
    )

    scenario = (;
        W_eq=W_eq,
        W_ineq=-Matrix{Float64}(LinearAlgebra.I, 3, 3),
        T_eq=zeros(Float64, 2, 1),
        T_ineq=zeros(Float64, 3, 1),
        h_eq=zeros(Float64, 2),
        h_ineq=zeros(Float64, 3),
        q=q,
    )

    return program, scenario
end
