struct DflCLoss{S,P,R,M} <: LossFunction
    solver::S
    program::P
    rho::R
    mu::M
end

const DFLCLoss = DflCLoss

function (loss::DflCLoss)(program, xi, xi_tilde, mu, rho)
    return not_implemented(:DflCLoss)
end
