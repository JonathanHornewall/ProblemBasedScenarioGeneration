abstract type ProblemInstance end

function _field_value(problem_instance::ProblemInstance, names::Tuple)
    for name in names
        hasproperty(problem_instance, name) && return getproperty(problem_instance, name)
    end
    error("Problem instance $(typeof(problem_instance)) does not expose any of fields $(names).")
end

function A(problem_instance::ProblemInstance)
    return _field_value(problem_instance, (:A, :A1, :A_eq))
end

function b(problem_instance::ProblemInstance)
    return _field_value(problem_instance, (:b, :b1))
end

function c(problem_instance::ProblemInstance)
    return _field_value(problem_instance, (:c, :c1))
end

function W_base(problem_instance::ProblemInstance)
    return _field_value(problem_instance, (:W_base, :W, :W_eq))
end

function T_base(problem_instance::ProblemInstance)
    return _field_value(problem_instance, (:T_base, :T, :T_eq))
end

function h_base(problem_instance::ProblemInstance)
    return _field_value(problem_instance, (:h_base, :h))
end

function q_base(problem_instance::ProblemInstance)
    return _field_value(problem_instance, (:q_base, :q))
end

function context_sampler(problem_instance::ProblemInstance)
    return hasproperty(problem_instance, :context_sampler) ? getproperty(problem_instance, :context_sampler) : nothing
end

function scenario_sampler(problem_instance::ProblemInstance)
    return hasproperty(problem_instance, :scenario_sampler) ? getproperty(problem_instance, :scenario_sampler) : nothing
end

function scenario_parametrization(problem_instance::ProblemInstance)
    return hasproperty(problem_instance, :scenario_parametrization) ? getproperty(problem_instance, :scenario_parametrization) : identity
end
