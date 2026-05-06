_decoder_softplus(x) = log1p(exp(-abs(x))) + max(x, zero(x))
