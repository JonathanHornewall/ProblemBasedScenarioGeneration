abstract type LossFunction end

function (loss::LossFunction)(program, xi, xi_tilde, mu, rho)
    return not_implemented(:LossFunction)
end
