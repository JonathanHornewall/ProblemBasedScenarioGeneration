# Define a multilayer perceptron model with input of size 3 and output of size 30
function scenario_neural_net(problem_instance, regularization_parameter)
# Make note to update input parameters, etc.
model = Chain(
    Dense(3, 3, relu),  
    Dense(3, 3, relu),   
    Dense(3, 9, relu),
    Dense(9, 18, relu),
    Dense(18, 30, relu)            
)

function surrogate_decision(x, W, T, h, q) 
    return surrogate_decision(problem_instance, x, W, T, h, q, regularization_parameter) 
end
function primal_problem_cost(context_parameter, first_stage_decision)
    return primal_problem_cost(problem_instance, context_parameter, regularization_parameter, first_stage_decision)
end
function ChainRulesCore.rrule(::typeof(surrogate_decision), x, W, T, h, q)
    y = surrogate_decision(x, W, T, h, q)
    
    function pullback(ȳ)
        A, b, c = first_stage_parameters(problem_instance)
        twoslp = TwoStageSLP(A, b, c, [W], [T], [h], [q])
        return NoTangent(), D_xiY(twoslp, regularization_parameter, return_scenario_type(problem_instance))
    end
    
    return y, pullback
end

function ChainRulesCore.rrule(::typeof(F), Gout)
    y = F(Gout)
    
    function F_pullback(ȳ)
        ∂G = derivative_F(Gout, ȳ)
        return NoTangent(), ∂G
    end
    
    return y, F_pullback
end
end