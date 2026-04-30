function constant_schedule(value; length=nothing)
    return not_implemented(:constant_schedule)
end

function linear_schedule(start_value, stop_value; steps)
    return not_implemented(:linear_schedule)
end

function geometric_schedule(start_value, stop_value; steps)
    return not_implemented(:geometric_schedule)
end

function make_mu_schedule(args...; kwargs...)
    return not_implemented(:make_mu_schedule)
end

function make_rho_schedule(args...; kwargs...)
    return not_implemented(:make_rho_schedule)
end

function make_batch_size_schedule(args...; kwargs...)
    return not_implemented(:make_batch_size_schedule)
end

function make_step_size_schedule(args...; kwargs...)
    return not_implemented(:make_step_size_schedule)
end
