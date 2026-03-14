"""
Output head layers that map raw network output to scenario parameters,
based on the `NoisePattern` of the problem.
"""

using Flux

"""
    build_output_head(pattern::NoisePattern, prob::ProblemInstance;
                      nr_of_scenarios=1) -> Function

Return a function that takes the raw network output vector and returns
a `Vector{Scenario}` of length `nr_of_scenarios`.

The head implements any problem-specific output transformation,
e.g., ensuring positivity or reshaping.
"""
function build_output_head(
    pattern::NoisePattern,
    prob::ProblemInstance;
    nr_of_scenarios::Int = 1
)
    if pattern == H_ONLY
        return _h_only_head(prob, nr_of_scenarios)
    elseif pattern == Q_ONLY
        return _q_only_head(prob, nr_of_scenarios)
    elseif pattern == WH
        return _wh_head(prob, nr_of_scenarios)
    else
        error("Output head not implemented for pattern $pattern")
    end
end

# -----------------------------------------------------------------------
# H_ONLY head: output is directly h (demand vector)
# -----------------------------------------------------------------------
function _h_only_head(prob::ProblemInstance, nr_of_scenarios::Int)
    sc_dim = _scenario_param_dim(prob)
    function head(raw_output::AbstractVector)
        # raw_output has length sc_dim * nr_of_scenarios
        params = reshape(raw_output, sc_dim, nr_of_scenarios)
        return [scenario_realization(prob, params[:, s]) for s in 1:nr_of_scenarios]
    end
    return head
end

# -----------------------------------------------------------------------
# Q_ONLY head: output is directly q (cost vector)
# -----------------------------------------------------------------------
function _q_only_head(prob::ProblemInstance, nr_of_scenarios::Int)
    sc_dim = _scenario_param_dim(prob)
    function head(raw_output::AbstractVector)
        params = reshape(raw_output, sc_dim, nr_of_scenarios)
        return [scenario_realization(prob, params[:, s]) for s in 1:nr_of_scenarios]
    end
    return head
end

# -----------------------------------------------------------------------
# WH head: output is [W_params; h_params]
# -----------------------------------------------------------------------
function _wh_head(prob::ProblemInstance, nr_of_scenarios::Int)
    sc_dim = _scenario_param_dim(prob)
    function head(raw_output::AbstractVector)
        params = reshape(raw_output, sc_dim, nr_of_scenarios)
        return [scenario_realization(prob, params[:, s]) for s in 1:nr_of_scenarios]
    end
    return head
end

# -----------------------------------------------------------------------
# Full model: generator + output head
# -----------------------------------------------------------------------

"""
    build_full_model(prob::ProblemInstance; nr_of_scenarios=1, kw...) -> Function

Build a callable `x -> Vector{Scenario}` that combines a neural network
generator with the appropriate output head for `prob`.
"""
function build_full_model(
    prob::ProblemInstance;
    nr_of_scenarios::Int = 1,
    hidden_dim::Int = 128,
    n_layers::Int = 3
)
    gen  = build_generator(prob; nr_of_scenarios=nr_of_scenarios,
                            hidden_dim=hidden_dim, n_layers=n_layers)
    head = build_output_head(noise_pattern(prob), prob; nr_of_scenarios=nr_of_scenarios)

    function model(x::AbstractVector)
        raw = gen(x)
        return head(raw)
    end

    return gen, head, model
end
