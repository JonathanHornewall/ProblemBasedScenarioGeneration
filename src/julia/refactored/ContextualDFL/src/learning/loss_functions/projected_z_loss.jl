struct ProjectedZLoss{S,P} <: LossFunction
    solver::S
    program::P
end

function (loss::ProjectedZLoss)(program, xi, xi_tilde, mu, rho)
    return not_implemented(:ProjectedZLoss)
end
