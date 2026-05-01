abstract type LossFunction end

function (loss::LossFunction)(program, xi, xi_tilde, mu, rho)
    return not_implemented(:LossFunction)
end

function _scenario_numeric_parts(x)
    if x isa BaseScenario
        return (x.W_eq, x.W_ineq, x.T_eq, x.T_ineq, x.h, x.q)
    elseif x isa NamedTuple
        return values(x)
    elseif x isa Tuple
        return x
    else
        return (x,)
    end
end

function _flatten_numeric(x)
    parts = [
        part isa Number ? [part] : vec(part)
        for part in _scenario_numeric_parts(x)
        if part !== nothing && (part isa Number || part isa AbstractArray)
    ]
    isempty(parts) && return Float64[]
    return vcat(parts...)
end

function _scenario_mse(a, b)
    av = _flatten_numeric(a)
    bv = _flatten_numeric(b)
    length(av) == length(bv) || error("Cannot compare scenario parameters of lengths $(length(av)) and $(length(bv))")
    isempty(av) && return 0.0
    return sum(abs2, av .- bv) / length(av)
end

function _scenario_mse(a::BaseScenario, b::BaseScenario)
    size(a.W_eq) == size(b.W_eq) || error("W_eq sizes do not match")
    size(a.W_ineq) == size(b.W_ineq) || error("W_ineq sizes do not match")
    size(a.T_eq) == size(b.T_eq) || error("T_eq sizes do not match")
    size(a.T_ineq) == size(b.T_ineq) || error("T_ineq sizes do not match")
    length(a.h) == length(b.h) || error("h lengths do not match")
    length(a.q) == length(b.q) || error("q lengths do not match")
    numerator =
        sum(abs2, a.W_eq .- b.W_eq) +
        sum(abs2, a.W_ineq .- b.W_ineq) +
        sum(abs2, a.T_eq .- b.T_eq) +
        sum(abs2, a.T_ineq .- b.T_ineq) +
        sum(abs2, a.h .- b.h) +
        sum(abs2, a.q .- b.q)
    denominator =
        length(a.W_eq) +
        length(a.W_ineq) +
        length(a.T_eq) +
        length(a.T_ineq) +
        length(a.h) +
        length(a.q)
    return numerator / max(denominator, 1)
end

_loss_solver(loss) = hasproperty(loss, :solver) ? getproperty(loss, :solver) : GLPKSolver()
_loss_program(loss, program) = hasproperty(loss, :program) ? getproperty(loss, :program) : program
