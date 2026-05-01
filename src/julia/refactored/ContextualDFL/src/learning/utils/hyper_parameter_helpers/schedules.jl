struct FunctionalSchedule{F}
    f::F
    len::Union{Nothing,Int}
end

(schedule::FunctionalSchedule)(i::Integer) = schedule.f(i)
Base.getindex(schedule::FunctionalSchedule, i::Integer) = schedule(i)
function Base.length(schedule::FunctionalSchedule)
    schedule.len === nothing && error("This schedule has no finite length.")
    return schedule.len
end

function constant_schedule(value; length=nothing)
    return FunctionalSchedule(_ -> value, length)
end

function linear_schedule(start_value, stop_value; steps)
    steps >= 1 || error("linear_schedule requires at least one step.")
    return FunctionalSchedule(steps) do i
        idx = clamp(i, 1, steps)
        steps == 1 && return stop_value
        weight = (idx - 1) / (steps - 1)
        return (1 - weight) * start_value + weight * stop_value
    end
end

function geometric_schedule(start_value, stop_value; steps)
    steps >= 1 || error("geometric_schedule requires at least one step.")
    start_value > 0 && stop_value > 0 || error("geometric schedules require positive endpoints.")
    ratio = steps == 1 ? 1.0 : (stop_value / start_value)^(1 / (steps - 1))
    return FunctionalSchedule(i -> start_value * ratio^(clamp(i, 1, steps) - 1), steps)
end

function make_mu_schedule(args...; kwargs...)
    isempty(args) && return constant_schedule(0.0; kwargs...)
    length(args) == 1 && return constant_schedule(args[1]; kwargs...)
    return geometric_schedule(args...; kwargs...)
end

function make_rho_schedule(args...; kwargs...)
    isempty(args) && return constant_schedule(0.0; kwargs...)
    length(args) == 1 && return constant_schedule(args[1]; kwargs...)
    return geometric_schedule(args...; kwargs...)
end

function make_batch_size_schedule(args...; kwargs...)
    return isempty(args) ? constant_schedule(1; kwargs...) : constant_schedule(args[1]; kwargs...)
end

function make_step_size_schedule(args...; kwargs...)
    return isempty(args) ? constant_schedule(1e-3; kwargs...) : constant_schedule(args[1]; kwargs...)
end
