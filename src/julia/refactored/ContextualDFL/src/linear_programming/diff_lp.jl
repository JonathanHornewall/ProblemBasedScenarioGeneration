function differentiate_solve(
    solver::Solver,
    lp::LP;
    with_respect_to=(:b, :c),
    config=nothing,
)
    result = solve(solver, lp, config)
    cache = result.cache
    cache isa BarrierCache || error("differentiate_solve requires a positive barrier parameter `mu`.")
    K = _barrier_kkt_matrix(cache)
    n = length(cache.x)
    m = length(cache.b)
    sensitivities = Dict{Symbol,Any}()

    if :b in with_respect_to
        rhs_b = vcat(zeros(eltype(cache.x), n, m), Matrix{eltype(cache.x)}(I, m, m))
        sensitivities[:b] = (K \ rhs_b)[1:n, :]
    end
    if :c in with_respect_to
        rhs_c = vcat(-Matrix{eltype(cache.x)}(I, n, n), zeros(eltype(cache.x), m, n))
        sensitivities[:c] = (K \ rhs_c)[1:n, :]
    end
    return (result=result, sensitivities=(; sensitivities...))
end

function _barrier_kkt_matrix(cache::BarrierCache{T}) where {T}
    m, n = size(cache.A)
    D = cache.mu ./ (cache.x .^ 2)
    return vcat(
        hcat(Diagonal(D), cache.A'),
        hcat(cache.A, zeros(T, m, m)),
    )
end
