"""
    make_pyramidal_net(input_dim, M, N, trunk_length, head_length;
                    act = relu, growth = 2.0)

Return a `Flux.Chain` that

* takes a vector of length `input_dim`;
* builds `trunk_length` shared Dense layers whose width is multiplied by
  `growth` at every step;
* branches into `N` identical heads, each with `head_length` further
  growth layers and a final `Dense(_, M)` layer;
* produces an `M x N` matrix whose `[:, i]` column is the i-th head's
  prediction.
* Converts the output into a scenario realization format.

Keyword arguments
-----------------
* `act' Activation function (default `relu`).
* `growth` Width multiplier between successive layers (default 2.0).
            Use `growth < 1` if you ever want the shrinking variant again.
"""
function make_pyramidal_net(input_dim::Integer, output_dim::Integer, nr_of_heads::Integer, trunk_length::Integer, head_length::Integer; 
                            act = relu, growth = nothing)

    function suggested_growth(inp, outp, depth; min_g = 1.2, max_g = 3.0)
    depth == 0 && return 1.0
    g = (outp / inp) ^ (1 / depth)
    return clamp(g, min_g, max_g)           # stay in a sane range
    end

    growth === nothing &&        # compute a sensible default
        (growth = suggested_growth(input_dim, output_dim, total_depth))

    ##################################################################
    # helper that returns (vector of Dense layers, final width)
    ##################################################################
    function growth_layers(in_dim, depth)
        layers = Vector{Any}(undef, depth)
        w = in_dim
        for i in 1:depth
            w_next = max(1, Int(round(w * growth)))
            layers[i] = Dense(w, w_next, act)
            w = w_next
        end
        return layers, w
    end

    # -------- shared trunk -------------------------------------------------
    trunk_layers, trunk_out = growth_layers(input_dim, trunk_length)
    trunk = Chain(trunk_layers...)

    # -------- individual heads --------------------------------------------
    head_layers, head_out = growth_layers(trunk_out, head_length)
    push!(head_layers, Dense(head_out, output_dim))   # no activation on outputs
    head_template = Chain(head_layers...)

    # deep‑copy the head for each output branch
    heads = [Flux.deepcopy(head_template) for _ in 1:nr_of_heads]

    return heads

    function convert_to_scenario_data(problem_instance, neural_output)
        Ws = Vector{Matrix{Float64}}()
        Ts = Vector{Matrix{Float64}}()
        hs = Vector{Matrix{Float64}}()
        qs = Vector{Matrix{Float64}}()
        for param in eachcol(neural_output)
            W, T, h, q = scenario_realization(problem_instance, param)
            push!(Ws, W)
            push!(Ts, T)
            push!(hs, h)
            push!(qs, q)
        end
        return Ws, Ts, hs, qs
    end

    # -------- wrap everything into one model ------------------------------
    model = Chain(
        trunk,
        x -> hcat((h(x) for h in heads)...),
        x -> convert_to_scenario_data(problem_instance, x)   # M × N matrix
    )
    return model
end

"""
    construct_neural_network(problem_instance, number_of_scenarios, trunk_length, head_length;
                            act = relu, growth = nothing)      
Construct a neural network for the given problem instance.
This function creates a pyramidal neural network with a shared trunk and multiple heads,
where each head corresponds to a scenario in the problem instance.
"""
function construct_neural_network(problem_instance, number_of_scenarios, trunk_length, head_length;
                                act = relu, growth = nothing)
    input_dim = generate_data_set(problem_instance)[1][1]
    output_dimension = return_scenario_parameter_dimension(problem_instance)  # number of outputs per head
    nr_of_heads = number_of_scenarios  # number of heads
    # Construct the neural network
    return make_pyramidal_net(input_dim, output_dimension, nr_of_heads, trunk_length, head_length; 
                            act = act, growth = growth)
end

#=
"""
    loss(model, problem_instance, regularization_parameter, context_variable, number_of_scenarios)
"""
function loss(model, problem_instance, regularization_parameter, context_variable, data_set)
    scenario_parameter = model(context_variable)
    Ws_simulated, Ts_simulated, hs_simulated, qs_simulated = scenario_realization(problem_instance, scenario_parameter)
    surrogate_decision = surrogate_solution(problem_instance, Ws_simulated, Ts_simulated, hs_simulated, qs_simulated, 
    regularization_parameter)
    Ws_actual, Ts_actual, hs_actual, qs_actual = nothing, nothing, nothing, nothing
    for data in data_set
        if data[1] == context_variable
            Ws_actual, Ts_actual, hs_actual, qs_actual = data[2]
            break
        end
    end
    (Ws_actual === nothing || Ts_actual === nothing || hs_actual === nothing || qs_actual === nothing) &&
        error("Context variable not found in data set: $context_variable")
    Ws_actual, Ts_actual, hs_actual, qs_actual = return_data_set(problem_instance, context_variable)
    primal_problem_cost(problem_instance, Ws_actual, Ts_actual, hs_actual, qs_actual, regularization_parameter, surrogate_decision)
end
=#

function loss(problem_instance, reg_param_surr, reg_param_prim, scenario_parameter, actual_scenario)
    nr_of_scenarios = size(scenario_parameter, 2)
    total_loss = 0.0
    
    for i in 1:nr_of_scenarios
        # Extract individual scenario vectors from the matrix
        scenario = scenario_parameter[:, i]
        actual = actual_scenario[:, i]
        
        # Generate surrogate solution using the existing function that expects vectors
        surrogate_decision = surrogate_solution(problem_instance, scenario, reg_param_surr)
        
        # Evaluate primal problem cost using the existing function that expects vectors
        scenario_loss = primal_problem_cost(problem_instance, actual, reg_param_prim, surrogate_decision)
        total_loss += scenario_loss
    end
    
    return total_loss / nr_of_scenarios  # Return average loss across scenarios
end

"""
---------------------------------------------------------------------------------------------
ChainRules.jl differentiation rules
---------------------------------------------------------------------------------------------
"""

"""
Provides the pullback for the s1_cost function, allowing for back-propagation through the neural network.
Uses the derivative computation from diff_s1_cost.
"""
function ChainRulesCore.rrule(::typeof(s1_cost), two_slp::TwoStageSLP, s1_decision, regularization_parameter, solver)
    cost_val = s1_cost(two_slp, s1_decision, regularization_parameter; solver=solver)
    
    function pullback(cost_hat)
        # Use the derivative from 2sp_differentials.jl
        cost_derivative = diff_s1_cost(two_slp, s1_decision, regularization_parameter; solver=solver)
        tangent = cost_derivative' * cost_hat
        return NoTangent(), NoTangent(), NoTangent(), tangent, NoTangent(), NoTangent()
    end
    
    return cost_val, pullback
end

"""
Provides the pullback for the primal LogBarCanLP solver that takes A, b, c as inputs.
This enables differentiation through the optimal solution computation with respect to the problem parameters.
"""
function ChainRulesCore.rrule(::typeof(LogBarCanLP_standard_solver_primal), constraint_matrix, constraint_vector, cost_vector, mu; solver_tolerance=1e-9, feasibility_margin=1e-8)
    # Create a temporary LogBarCanLP instance and solve for the optimal solution
    temp_lp = CanLP(constraint_matrix, constraint_vector, cost_vector)
    temp_instance = LogBarCanLP(temp_lp, mu)
    optimal_solution, optimal_dual, KKT_matrix = diff_cache_computation(temp_instance)
    
    function pullback(solution_hat)
        # Compute derivatives with respect to A, b, c using diff_opt        
        # Compute derivatives with respect to A, b, c
        D_A, D_b, D_c = diff_opt(temp_instance, optimal_solution, optimal_dual, KKT_matrix, solver=LogBarCanLP_standard_solver)
        
        # The pullback computes how changes in the optimal solution map back to changes in the problem parameters
        # solution_hat is a scalar (since we only return the primal solution)
        # We need to compute the contribution from each parameter
        
        # For A: D_A has shape (n, m, n) where n is number of variables, m is number of constraints
        # For b: D_b has shape (n, m) 
        # For c: D_c has shape (n, n)
        
        # The pullback for each parameter is the contraction with the solution tangent
        @einsum A_tangent[i,j] := v[k] * A[k,i,j]
        b_tangent = D_b' * solution_hat
        c_tangent = D_c' * solution_hat
        
        return NoTangent(), A_tangent, b_tangent, c_tangent, NoTangent(), NoTangent(), NoTangent()
    end
    
    return optimal_solution, pullback
end