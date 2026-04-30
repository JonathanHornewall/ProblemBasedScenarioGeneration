struct DecoderStrategy{WEQ,WIN,TEQ,TIN,H,Q}
    W_eq_decoder::WEQ
    W_ineq_decoder::WIN
    T_eq_decoder::TEQ
    T_ineq_decoder::TIN
    h_decoder::H
    q_decoder::Q
end

function DecoderStrategy(;
    W_eq_decoder=EmptyComponentDecoder(:W_eq),
    W_ineq_decoder=EmptyComponentDecoder(:W_ineq),
    T_eq_decoder=EmptyComponentDecoder(:T_eq),
    T_ineq_decoder=EmptyComponentDecoder(:T_ineq),
    h_decoder=EmptyComponentDecoder(:h),
    q_decoder=EmptyComponentDecoder(:q),
)
    return DecoderStrategy(
        W_eq_decoder,
        W_ineq_decoder,
        T_eq_decoder,
        T_ineq_decoder,
        h_decoder,
        q_decoder,
    )
end
