import Dates
import Serialization

const _STOCHASTIC_CRASH_DEFAULT_ROOT = "/tmp/contextual-dfl"
const _STOCHASTIC_CRASH_ROOT = Ref{String}(_STOCHASTIC_CRASH_DEFAULT_ROOT)

struct StochasticProgramFailure <: Exception
    location::Symbol
    crash_file::String
    scenario_index::Union{Nothing,Int}
end

function Base.showerror(io::IO, error::StochasticProgramFailure)
    if error.location === :single_scenario_solve
        print(io, "single-scenario problem failed.")
    elseif error.location === :stochastic_program_solve
        print(io, "stochastic program solve failed.")
    elseif error.location === :second_stage_cost
        print(io, "second-stage problem failed in scenario $(error.scenario_index).")
    else
        print(io, "stochastic program failure.")
    end

    print(io, " Crash data serialized at ", error.crash_file)
end

function _set_stochastic_crash_root!(root::AbstractString)
    previous = _STOCHASTIC_CRASH_ROOT[]
    _STOCHASTIC_CRASH_ROOT[] = String(root)
    return previous
end

function _reset_stochastic_crash_root!()
    return _set_stochastic_crash_root!(_STOCHASTIC_CRASH_DEFAULT_ROOT)
end

_stochastic_crash_root() = _STOCHASTIC_CRASH_ROOT[]

_crash_copy(value) = value
_crash_copy(value::AbstractArray) = copy(value)

function _stochastic_failure_location(W_eq_array)
    return size(W_eq_array, 3) == 1 ? :single_scenario_solve : :stochastic_program_solve
end

function _stochastic_crash_file()
    root = _stochastic_crash_root()
    mkpath(root)

    timestamp = Dates.format(Dates.now(), "yyyymmddTHHMMSSsss")
    crash_dir = mktempdir(root; prefix="crashed_$(timestamp)_")
    return joinpath(crash_dir, "stochastic_program_failure.jls")
end

function _first_stage_crash_payload(sp::StochasticProgram)
    first_stage_lp = sp.first_stage_lp
    return (;
        A_eq=_crash_copy(first_stage_lp.A_eq),
        A_ineq=_crash_copy(first_stage_lp.A_ineq),
        b_eq=_crash_copy(first_stage_lp.b_eq),
        b_ineq=_crash_copy(first_stage_lp.b_ineq),
        c=_crash_copy(first_stage_lp.c),
    )
end

function _scenario_crash_payload(
    W_eq_array,
    W_ineq_array,
    T_eq_array,
    T_ineq_array,
    h_eq_array,
    h_ineq_array,
    q_array,
)
    return (;
        W_eq_array=_crash_copy(W_eq_array),
        W_ineq_array=_crash_copy(W_ineq_array),
        T_eq_array=_crash_copy(T_eq_array),
        T_ineq_array=_crash_copy(T_ineq_array),
        h_eq_array=_crash_copy(h_eq_array),
        h_ineq_array=_crash_copy(h_ineq_array),
        q_array=_crash_copy(q_array),
    )
end

function _record_stochastic_program_failure(
    error,
    location::Symbol,
    solver::Solver,
    sp::StochasticProgram,
    W_eq_array,
    W_ineq_array,
    T_eq_array,
    T_ineq_array,
    h_eq_array,
    h_ineq_array,
    q_array;
    μ,
    ρ=0,
    effective_μ=nothing,
    effective_ρ=nothing,
    probabilities=nothing,
    kwargs=(;),
    z=nothing,
    scenario_index=nothing,
    scenario_μ=nothing,
    scenario_ρ=nothing,
)
    crash_file = _stochastic_crash_file()
    payload = (;
        location=location,
        timestamp=Dates.now(),
        first_stage=_first_stage_crash_payload(sp),
        scenario_data=_scenario_crash_payload(
            W_eq_array,
            W_ineq_array,
            T_eq_array,
            T_ineq_array,
            h_eq_array,
            h_ineq_array,
            q_array,
        ),
        μ=_crash_copy(μ),
        ρ=_crash_copy(ρ),
        effective_μ=_crash_copy(effective_μ),
        effective_ρ=_crash_copy(effective_ρ),
        scenario_μ=_crash_copy(scenario_μ),
        scenario_ρ=_crash_copy(scenario_ρ),
        probabilities=_crash_copy(probabilities),
        solver_type=string(typeof(solver)),
        kwargs=kwargs,
        z=_crash_copy(z),
        scenario_index=scenario_index,
        original_error_type=string(typeof(error)),
        original_error_text=sprint(showerror, error),
    )

    Serialization.serialize(crash_file, payload)
    return crash_file
end

function _throw_stochastic_program_failure(
    error,
    location::Symbol,
    solver::Solver,
    sp::StochasticProgram,
    W_eq_array,
    W_ineq_array,
    T_eq_array,
    T_ineq_array,
    h_eq_array,
    h_ineq_array,
    q_array;
    μ,
    ρ=0,
    effective_μ=nothing,
    effective_ρ=nothing,
    probabilities=nothing,
    kwargs=(;),
    z=nothing,
    scenario_index=nothing,
    scenario_μ=nothing,
    scenario_ρ=nothing,
)
    crash_file = _record_stochastic_program_failure(
        error,
        location,
        solver,
        sp,
        W_eq_array,
        W_ineq_array,
        T_eq_array,
        T_ineq_array,
        h_eq_array,
        h_ineq_array,
        q_array;
        μ=μ,
        ρ=ρ,
        effective_μ=effective_μ,
        effective_ρ=effective_ρ,
        probabilities=probabilities,
        kwargs=kwargs,
        z=z,
        scenario_index=scenario_index,
        scenario_μ=scenario_μ,
        scenario_ρ=scenario_ρ,
    )
    throw(StochasticProgramFailure(location, crash_file, scenario_index))
end
