abstract type ProblemInstance end

function A(problem_instance::ProblemInstance)
    return not_implemented(:A)
end

function b(problem_instance::ProblemInstance)
    return not_implemented(:b)
end

function c(problem_instance::ProblemInstance)
    return not_implemented(:c)
end

function W_base(problem_instance::ProblemInstance)
    return not_implemented(:W_base)
end

function T_base(problem_instance::ProblemInstance)
    return not_implemented(:T_base)
end

function h_base(problem_instance::ProblemInstance)
    return not_implemented(:h_base)
end

function q_base(problem_instance::ProblemInstance)
    return not_implemented(:q_base)
end

function context_sampler(problem_instance::ProblemInstance)
    return not_implemented(:context_sampler)
end

function scenario_sampler(problem_instance::ProblemInstance)
    return not_implemented(:scenario_sampler)
end

function scenario_parametrization(problem_instance::ProblemInstance)
    return not_implemented(:scenario_parametrization)
end
