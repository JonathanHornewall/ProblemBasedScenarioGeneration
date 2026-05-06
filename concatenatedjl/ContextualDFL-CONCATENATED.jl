# ContextualDFL Concatenated Source
# Generated from Julia files under src/ContextualDFL
# Generated at 2026-05-05T19:27:40+02:00
# File count: 96
# Excludes generated *concatenated*.jl files


# BEGIN FILE: src/ContextualDFL/ContextualDFL/src/ContextualDFL.jl
module ContextualDFL

import ChainRulesCore

export
    ContextSampler,
    ContextualDataPoint,
    ContextualDataSet,
    DflCLoss,
    DflScenLoss,
    GLPKSolver,
    GurobiSolver,
    HiGHSSolver,
    IpoptSolver,
    LP,
    LPSolver,
    LogBarSolver,
    LossFunction,
    MSEScenLoss,
    ParametricDecoder,
    ParametricScenario,
    ProjectedZLoss,
    ScenarioDecoder,
    ScenarioGenerator,
    ScenarioSampler,
    Solver,
    SPOPlusLoss,
    StochasticProgram,
    TransShipmentProblem,
    TransShipmentProblemData,
    TransShipmentScenarioDecoder,
    TransShipmentStochasticEntry,
    batch_size_schedule,
    constant_schedule,
    construct_lp,
    cost_function,
    cost_function_rrule,
    decode_scenario_collection,
    diff_solve,
    expected_cost,
    generate_context_set,
    generate_scenario_set,
    mu_schedule,
    rho_schedule,
    default_transshipment_data_dir,
    scenario_cost,
    sample_transshipment_parameters,
    solve,
    solve_rrule,
    step_size_schedule,
    transshipment_direct_mean_lp,
    transshipment_mean_lp,
    transshipment_mean_parameters,
    transshipment_mean_scenario_arrays,
    transshipment_scenario_arrays,
    train!,
    validate_transshipment_problem,
    VectorDecoder

include("linear_programming/LP.jl")
include("linear_programming/bound_form_lp.jl")
include("linear_programming/Solvers/solver_status.jl")
include("linear_programming/Solvers/solvers/LPSolvers/LPSolver.jl")
include("linear_programming/Solvers/solvers/LPSolvers/implemented_solvers/GLPKSolver.jl")
include("linear_programming/Solvers/solvers/LPSolvers/implemented_solvers/GurobiSolver.jl")
include("linear_programming/Solvers/solvers/LPSolvers/implemented_solvers/HiGHSSolver.jl")
include("linear_programming/Solvers/solvers/LogBarSolvers/LogBarSolver.jl")
include("linear_programming/Solvers/solvers/LogBarSolvers/implemented_solvers/IpoptSolver.jl")
include("linear_programming/Solvers/Solver.jl")
include("linear_programming/diff_lp.jl")

include("stochastic_programming/StochasticProgram.jl")
include("stochastic_programming/construct_lp.jl")
include("stochastic_programming/crash_recorder.jl")
include("stochastic_programming/solve.jl")
include("stochastic_programming/solve_rrule.jl")
include("stochastic_programming/cost_function.jl")
include("stochastic_programming/cost_function_rrule.jl")

include("scenario_decoders/ScenarioDecoder.jl")
include("learning/dataset.jl")
include("scenario_decoders/VectorDecoder.jl")
include("scenario_decoders/ParametricDecoder.jl")

include("data_generation/context_sampling/ContextSampler.jl")
include("data_generation/scenario_sampling/ScenarioSampler.jl")

include("ScenarioGenerator.jl")

include("learning/loss_functions/Loss.jl")
include("learning/loss_functions/DflScenLoss.jl")
include("learning/loss_functions/SPO_plus_loss.jl")
include("learning/loss_functions/dfl_c_loss.jl")
include("learning/loss_functions/MSE_scen_loss.jl")
include("learning/loss_functions/projected_z_loss.jl")
include("learning/train.jl")
include("learning/utils/test.jl")
include("learning/utils/hyper_parameter_helpers/schedules.jl")

include("implementations/TransShipmentProblem/TransShipmentProblem.jl")

end

# END FILE: src/ContextualDFL/ContextualDFL/src/ContextualDFL.jl

# BEGIN FILE: src/ContextualDFL/ContextualDFL/src/ScenarioGenerator.jl
struct ScenarioGenerator{TNeuralNet,TDecoder<:VectorDecoder}
    neural_net::TNeuralNet
    scenario_decoder::TDecoder
end

ScenarioGenerator(; neural_net, scenario_decoder) =
    ScenarioGenerator(neural_net, scenario_decoder)

(generator::ScenarioGenerator)(context) =
    generator.scenario_decoder(generator.neural_net(context))

# END FILE: src/ContextualDFL/ContextualDFL/src/ScenarioGenerator.jl

# BEGIN FILE: src/ContextualDFL/ContextualDFL/src/data_generation/context_sampling/ContextSampler.jl
abstract type ContextSampler end

(sampler::ContextSampler)(args...; kwargs...) =
    error("Context sampling is not defined for $(typeof(sampler)).")

generate_context_set(sampler::ContextSampler, nr_context::Integer; kwargs...) =
    error("Context-set generation has not been implemented yet.")

# END FILE: src/ContextualDFL/ContextualDFL/src/data_generation/context_sampling/ContextSampler.jl

# BEGIN FILE: src/ContextualDFL/ContextualDFL/src/data_generation/scenario_sampling/ScenarioSampler.jl
abstract type ScenarioSampler end

(sampler::ScenarioSampler)(context; kwargs...) =
    error("Scenario sampling is not defined for $(typeof(sampler)).")

generate_scenario_set(sampler::ScenarioSampler, context_data, scenarios_per_context; kwargs...) =
    error("Scenario-set generation has not been implemented yet.")

# END FILE: src/ContextualDFL/ContextualDFL/src/data_generation/scenario_sampling/ScenarioSampler.jl

# BEGIN FILE: src/ContextualDFL/ContextualDFL/src/implementations/TransShipmentProblem/TransShipmentProblem.jl
import Distributions
import LinearAlgebra
import Random

const TRANSSHIPMENT_SOURCE_REPOSITORY = "https://github.com/USC3DLAB/SD"
const TRANSSHIPMENT_SOURCE_DATA_URL =
    "https://github.com/USC3DLAB/SD/tree/master/spInput/transship"

struct TransShipmentStochasticEntry
    component::Symbol
    name::String
    index::Int
    mean::Float64
    std::Float64
end

struct TransShipmentProblemData
    source_repository::String
    source_data_url::String
    core_path::String
    time_path::String
    stochastic_path::String
    first_stage_variables::Vector{String}
    second_stage_variables::Vector{String}
    first_stage_rows::Vector{String}
    second_stage_rows::Vector{String}
    random_rhs_entries::Vector{TransShipmentStochasticEntry}
    random_objective_entries::Vector{TransShipmentStochasticEntry}
    direct_A_eq::Matrix{Float64}
    direct_b_eq::Vector{Float64}
    direct_c::Vector{Float64}
end

struct TransShipmentProblem
    data::TransShipmentProblemData
    stochastic_program::StochasticProgram
    base_scenario::NamedTuple
end

struct TransShipmentScenarioDecoder{TBase,TH,TQ,TVH,TVQ} <: VectorDecoder
    base_scenario::TBase
    h_selector::TH
    q_selector::TQ
    mean_rhs_values::TVH
    mean_objective_values::TVQ
end

struct _ParsedMPSCore
    name::String
    objective_row::String
    structural_rows::Vector{String}
    row_types::Dict{String,Char}
    variables::Vector{String}
    A::Matrix{Float64}
    rhs::Vector{Float64}
    c::Vector{Float64}
end

struct _StochasticSpec
    distribution_type::Symbol
    rhs_means::Dict{String,Float64}
    rhs_stds::Dict{String,Float64}
    objective_means::Dict{String,Float64}
    objective_stds::Dict{String,Float64}
end

default_transshipment_data_dir() = joinpath(@__DIR__, "data")

function TransShipmentProblem(; data_dir=default_transshipment_data_dir())
    core_path = joinpath(data_dir, "transship.cor")
    time_path = joinpath(data_dir, "transship.tim")
    stochastic_path = joinpath(data_dir, "transship.sto")

    core = _parse_mps_core(core_path)
    first_stage_variables, second_stage_variables, first_stage_rows, second_stage_rows =
        _parse_transshipment_stage_split(time_path, core)
    stochastic_spec = _parse_transshipment_stochastic_file(stochastic_path)

    all(==('E'), (core.row_types[row] for row in first_stage_rows)) ||
        throw(ArgumentError("Transshipment first-stage rows must all be equality rows."))
    all(==('E'), (core.row_types[row] for row in second_stage_rows)) ||
        throw(ArgumentError("Transshipment second-stage rows must all be equality rows."))

    first_var_indices = _indices(core.variables, first_stage_variables)
    second_var_indices = _indices(core.variables, second_stage_variables)
    first_row_indices = _indices(core.structural_rows, first_stage_rows)
    second_row_indices = _indices(core.structural_rows, second_stage_rows)

    A_eq = core.A[first_row_indices, first_var_indices]
    b_eq = core.rhs[first_row_indices]
    c = core.c[first_var_indices]
    W_eq = core.A[second_row_indices, second_var_indices]
    T_eq = core.A[second_row_indices, first_var_indices]
    h_eq = core.rhs[second_row_indices]
    q = core.c[second_var_indices]

    random_rhs_entries = TransShipmentStochasticEntry[]
    for row in sort(collect(keys(stochastic_spec.rhs_means)); by=_transshipment_index_key)
        row_index = _only_index(second_stage_rows, row)
        h_eq[row_index] = stochastic_spec.rhs_means[row]
        push!(
            random_rhs_entries,
            TransShipmentStochasticEntry(
                :h_eq,
                row,
                row_index,
                stochastic_spec.rhs_means[row],
                stochastic_spec.rhs_stds[row],
            ),
        )
    end

    random_objective_entries = TransShipmentStochasticEntry[]
    for variable in sort(collect(keys(stochastic_spec.objective_means)); by=_transshipment_index_key)
        variable_index = _only_index(second_stage_variables, variable)
        q[variable_index] = stochastic_spec.objective_means[variable]
        push!(
            random_objective_entries,
            TransShipmentStochasticEntry(
                :q,
                variable,
                variable_index,
                stochastic_spec.objective_means[variable],
                stochastic_spec.objective_stds[variable],
            ),
        )
    end

    nz = length(first_stage_variables)
    ny = length(second_stage_variables)
    program = StochasticProgram(
        A_eq=A_eq,
        A_ineq=-Matrix{Float64}(LinearAlgebra.I, nz, nz),
        b_eq=b_eq,
        b_ineq=zeros(Float64, nz),
        c=c,
    )

    base_scenario = (;
        W_eq=W_eq,
        W_ineq=-Matrix{Float64}(LinearAlgebra.I, ny, ny),
        T_eq=T_eq,
        T_ineq=zeros(Float64, ny, nz),
        h_eq=h_eq,
        h_ineq=zeros(Float64, ny),
        q=q,
    )

    direct_A_eq = hcat(T_eq, W_eq)
    direct_b_eq = copy(h_eq)
    direct_c = vcat(c, q)

    data = TransShipmentProblemData(
        TRANSSHIPMENT_SOURCE_REPOSITORY,
        TRANSSHIPMENT_SOURCE_DATA_URL,
        core_path,
        time_path,
        stochastic_path,
        first_stage_variables,
        second_stage_variables,
        first_stage_rows,
        second_stage_rows,
        random_rhs_entries,
        random_objective_entries,
        direct_A_eq,
        direct_b_eq,
        direct_c,
    )

    return TransShipmentProblem(data, program, base_scenario)
end

stochastic_program(problem::TransShipmentProblem) = problem.stochastic_program

base_scenario(problem::TransShipmentProblem) = problem.base_scenario

TransShipmentScenarioDecoder(problem::TransShipmentProblem) =
    TransShipmentScenarioDecoder(
        problem.base_scenario,
        _selector_matrix(length(problem.base_scenario.h_eq), problem.data.random_rhs_entries),
        _selector_matrix(length(problem.base_scenario.q), problem.data.random_objective_entries),
        [entry.mean for entry in problem.data.random_rhs_entries],
        [entry.mean for entry in problem.data.random_objective_entries],
    )

function transshipment_mean_parameters(problem::TransShipmentProblem)
    return (;
        rhs=[entry.mean for entry in problem.data.random_rhs_entries],
        q=[entry.mean for entry in problem.data.random_objective_entries],
    )
end

function sample_transshipment_parameters(
    problem::TransShipmentProblem;
    rng=Random.default_rng(),
    truncate_at_zero=true,
)
    rhs = _sample_transshipment_entries(problem.data.random_rhs_entries, rng; truncate_at_zero)
    q =
        _sample_transshipment_entries(problem.data.random_objective_entries, rng; truncate_at_zero)
    return (; rhs=rhs, q=q)
end

function transshipment_scenario_arrays(problem::TransShipmentProblem, scenario_collection)
    decoder = TransShipmentScenarioDecoder(problem)
    return decode_scenario_collection(decoder, scenario_collection)
end

function transshipment_mean_scenario_arrays(problem::TransShipmentProblem)
    scenario = transshipment_mean_parameters(problem)
    return transshipment_scenario_arrays(problem, [scenario])
end

function transshipment_mean_lp(problem::TransShipmentProblem)
    return construct_lp(problem.stochastic_program, transshipment_mean_scenario_arrays(problem)...)
end

function transshipment_direct_mean_lp(problem::TransShipmentProblem)
    n_variables = length(problem.data.direct_c)
    return LP(
        A_eq=problem.data.direct_A_eq,
        A_ineq=-Matrix{Float64}(LinearAlgebra.I, n_variables, n_variables),
        b_eq=problem.data.direct_b_eq,
        b_ineq=zeros(Float64, n_variables),
        c=problem.data.direct_c,
    )
end

function validate_transshipment_problem(
    problem::TransShipmentProblem=TransShipmentProblem();
    solver=Solver(IpoptSolver(), HiGHSSolver()),
    smoothing_cases=((; name=:unsmoothed, μ=0.0, ρ=0.0), (; name=:log_barrier, μ=1e-2, ρ=0.0), (; name=:quadratic, μ=0.0, ρ=1e-2), (; name=:log_barrier_quadratic, μ=1e-2, ρ=1e-2)),
    constraint_tolerance=1e-6,
)
    dimensions = (;
        n1=length(problem.data.first_stage_variables),
        n2=length(problem.data.second_stage_variables),
        m1=length(problem.data.first_stage_rows),
        m2=length(problem.data.second_stage_rows),
    )
    dimensions == (; n1=7, n2=77, m1=0, m2=35) ||
        throw(DimensionMismatch("Unexpected transshipment dimensions: $(dimensions)."))

    block_lp = transshipment_mean_lp(problem)
    direct_lp = transshipment_direct_mean_lp(problem)
    lp_match = _lp_difference_report(block_lp, direct_lp)
    lp_match.max_abs_A_eq <= 1e-12 ||
        throw(ArgumentError("Block and direct mean LP equality matrices differ."))
    lp_match.max_abs_b_eq <= 1e-12 ||
        throw(ArgumentError("Block and direct mean LP equality RHS vectors differ."))
    lp_match.max_abs_c <= 1e-12 ||
        throw(ArgumentError("Block and direct mean LP objective vectors differ."))

    perturbation_report = _transshipment_perturbation_report(problem)
    all(values(perturbation_report)) ||
        throw(ArgumentError("Transshipment perturbation validation failed."))

    solve_reports = map(smoothing_cases) do smoothing_case
        result = solve(
            solver,
            block_lp;
            μ=smoothing_case.μ,
            ρ=smoothing_case.ρ,
            constraint_tolerance=constraint_tolerance,
        )
        return _lp_solution_report(block_lp, result, smoothing_case.name)
    end

    return (;
        source_repository=problem.data.source_repository,
        source_data_url=problem.data.source_data_url,
        files=(;
            core=problem.data.core_path,
            time=problem.data.time_path,
            stochastic=problem.data.stochastic_path,
        ),
        dimensions=dimensions,
        random_rhs_entries=length(problem.data.random_rhs_entries),
        random_objective_entries=length(problem.data.random_objective_entries),
        stochastic_law=:INDEP_NORMAL,
        lp_match=lp_match,
        perturbation_report=perturbation_report,
        solve_reports=solve_reports,
    )
end

function (decoder::TransShipmentScenarioDecoder)(scenario_parameter::AbstractVector)
    return _decode_transshipment_scenario(decoder, scenario_parameter)
end

function (decoder::TransShipmentScenarioDecoder)(scenario_parameter)
    return _decode_transshipment_scenario(decoder, scenario_parameter)
end

function _decode_transshipment_scenario(decoder::TransShipmentScenarioDecoder, scenario_parameter)
    rhs_values, objective_values = _transshipment_parameter_values(decoder, scenario_parameter)

    h_eq =
        decoder.base_scenario.h_eq +
        decoder.h_selector * (rhs_values .- decoder.mean_rhs_values)
    q =
        decoder.base_scenario.q +
        decoder.q_selector * (objective_values .- decoder.mean_objective_values)

    return (
        decoder.base_scenario.W_eq,
        decoder.base_scenario.W_ineq,
        decoder.base_scenario.T_eq,
        decoder.base_scenario.T_ineq,
        h_eq,
        decoder.base_scenario.h_ineq,
        q,
    )
end

function _parse_mps_core(path::AbstractString)
    isfile(path) || throw(ArgumentError("MPS core file does not exist: $path"))

    name = ""
    current_section = :none
    row_names = String[]
    row_types = Dict{String,Char}()
    objective_row = ""
    variables = String[]
    variable_indices = Dict{String,Int}()
    coefficients = Dict{Tuple{String,String},Float64}()
    rhs_values = Dict{String,Float64}()

    for raw_line in eachline(path)
        line = strip(raw_line)
        (isempty(line) || startswith(line, "*")) && continue
        tokens = split(line)
        isempty(tokens) && continue

        marker = uppercase(tokens[1])
        if marker == "NAME"
            name = length(tokens) >= 2 ? tokens[2] : ""
            current_section = :name
            continue
        elseif marker == "ROWS"
            current_section = :rows
            continue
        elseif marker == "COLUMNS"
            current_section = :columns
            continue
        elseif marker == "RHS"
            current_section = :rhs
            continue
        elseif marker == "BOUNDS"
            current_section = :bounds
            continue
        elseif marker == "RANGES"
            current_section = :ranges
            continue
        elseif marker == "ENDATA"
            break
        end

        if current_section == :rows
            length(tokens) >= 2 || throw(ArgumentError("Malformed ROWS line in $path: $line"))
            row_type = only(tokens[1])
            row_name = tokens[2]
            push!(row_names, row_name)
            row_types[row_name] = row_type
            if row_type == 'N'
                objective_row = row_name
            end
        elseif current_section == :columns
            _parse_mps_column_line!(tokens, variables, variable_indices, coefficients)
        elseif current_section == :rhs
            _parse_mps_rhs_line!(tokens, row_types, rhs_values)
        elseif current_section in (:bounds, :ranges)
            throw(ArgumentError("Transshipment parser does not support $current_section sections."))
        end
    end

    objective_row != "" || throw(ArgumentError("MPS core has no objective row."))
    structural_rows = [row for row in row_names if row_types[row] != 'N']
    row_indices = Dict(row => index for (index, row) in enumerate(structural_rows))
    n_rows = length(structural_rows)
    n_variables = length(variables)
    A = zeros(Float64, n_rows, n_variables)
    c = zeros(Float64, n_variables)
    rhs = zeros(Float64, n_rows)

    for ((variable, row), value) in coefficients
        variable_index = variable_indices[variable]
        if row == objective_row
            c[variable_index] += value
        else
            row_index = row_indices[row]
            A[row_index, variable_index] += value
        end
    end

    for (row, value) in rhs_values
        haskey(row_indices, row) || continue
        rhs[row_indices[row]] = value
    end

    return _ParsedMPSCore(name, objective_row, structural_rows, row_types, variables, A, rhs, c)
end

function _parse_mps_column_line!(tokens, variables, variable_indices, coefficients)
    variable = tokens[1]
    if occursin("MARKER", join(tokens, " "))
        return nothing
    end
    if !haskey(variable_indices, variable)
        variable_indices[variable] = length(variables) + 1
        push!(variables, variable)
    end

    length(tokens) >= 3 || throw(ArgumentError("Malformed COLUMNS line: $(join(tokens, " "))"))
    pair_tokens = tokens[2:end]
    iseven(length(pair_tokens)) || throw(ArgumentError("Malformed COLUMNS row/value pairs."))
    for index in 1:2:length(pair_tokens)
        row = pair_tokens[index]
        value = parse(Float64, pair_tokens[index + 1])
        coefficients[(variable, row)] = get(coefficients, (variable, row), 0.0) + value
    end

    return nothing
end

function _parse_mps_rhs_line!(tokens, row_types, rhs_values)
    isempty(tokens) && return nothing
    start_index = haskey(row_types, tokens[1]) ? 1 : 2
    start_index <= length(tokens) || return nothing
    pair_tokens = tokens[start_index:end]
    iseven(length(pair_tokens)) || throw(ArgumentError("Malformed RHS row/value pairs."))
    for index in 1:2:length(pair_tokens)
        row = pair_tokens[index]
        value = parse(Float64, pair_tokens[index + 1])
        rhs_values[row] = value
    end

    return nothing
end

function _parse_transshipment_stage_split(path::AbstractString, core::_ParsedMPSCore)
    isfile(path) || throw(ArgumentError("SMPS time file does not exist: $path"))

    entries = NamedTuple[]
    in_periods = false
    for raw_line in eachline(path)
        line = strip(raw_line)
        isempty(line) && continue
        tokens = split(line)
        marker = uppercase(tokens[1])
        if marker == "PERIODS"
            in_periods = true
            continue
        elseif marker == "ENDATA"
            break
        elseif marker == "TIME"
            continue
        end

        in_periods || continue
        length(tokens) >= 3 || throw(ArgumentError("Malformed PERIODS line in $path: $line"))
        push!(
            entries,
            (;
                variable=String(tokens[1]),
                row=String(tokens[2]),
                stage=String(join(tokens[3:end], " ")),
            ),
        )
    end

    length(entries) == 2 ||
        throw(ArgumentError("Expected exactly two transshipment stages, found $(length(entries))."))
    second_stage_start_variable = entries[2].variable
    second_stage_start_row = entries[2].row
    second_variable_index = _only_index(core.variables, second_stage_start_variable)
    second_row_index = _only_index(core.structural_rows, second_stage_start_row)

    first_stage_variables = core.variables[1:(second_variable_index - 1)]
    second_stage_variables = core.variables[second_variable_index:end]
    first_stage_rows = core.structural_rows[1:(second_row_index - 1)]
    second_stage_rows = core.structural_rows[second_row_index:end]

    return first_stage_variables, second_stage_variables, first_stage_rows, second_stage_rows
end

function _parse_transshipment_stochastic_file(path::AbstractString)
    isfile(path) || throw(ArgumentError("SMPS stochastic file does not exist: $path"))

    distribution_type = :unknown
    rhs_means = Dict{String,Float64}()
    rhs_stds = Dict{String,Float64}()
    objective_means = Dict{String,Float64}()
    objective_stds = Dict{String,Float64}()

    for raw_line in eachline(path)
        line = strip(raw_line)
        isempty(line) && continue
        tokens = split(line)
        marker = uppercase(tokens[1])

        if marker == "STOCH"
            continue
        elseif marker == "INDEP"
            length(tokens) >= 2 || throw(ArgumentError("Malformed INDEP line in $path: $line"))
            distribution_type = Symbol("INDEP_" * uppercase(tokens[2]))
            continue
        elseif marker == "ENDATA"
            break
        elseif marker == "RHS"
            length(tokens) == 4 || throw(ArgumentError("Malformed stochastic RHS line: $line"))
            row = tokens[2]
            rhs_means[row] = parse(Float64, tokens[3])
            rhs_stds[row] = parse(Float64, tokens[4])
        else
            length(tokens) == 4 || throw(ArgumentError("Malformed stochastic objective line: $line"))
            variable = tokens[1]
            row = tokens[2]
            row == "obj" ||
                throw(ArgumentError("Expected stochastic objective row `obj`, found `$row`."))
            objective_means[variable] = parse(Float64, tokens[3])
            objective_stds[variable] = parse(Float64, tokens[4])
        end
    end

    distribution_type == :INDEP_NORMAL ||
        throw(ArgumentError("Expected INDEP NORMAL stochastic file, found $distribution_type."))
    return _StochasticSpec(
        distribution_type,
        rhs_means,
        rhs_stds,
        objective_means,
        objective_stds,
    )
end

function _transshipment_parameter_values(decoder::TransShipmentScenarioDecoder, scenario_parameter)
    n_rhs = length(decoder.mean_rhs_values)
    n_objective = length(decoder.mean_objective_values)

    if scenario_parameter isa AbstractVector
        values = vec(scenario_parameter)
        if length(values) == n_rhs + n_objective
            return values[1:n_rhs], values[(n_rhs + 1):end]
        elseif length(values) == n_rhs
            return values, decoder.mean_objective_values
        end
        throw(
            DimensionMismatch(
                "transshipment scenario vector has length $(length(values)); " *
                "expected $n_rhs or $(n_rhs + n_objective).",
            ),
        )
    end

    rhs_values = _scenario_property(
        scenario_parameter,
        (:rhs, :demand, :h, :h_eq, :h_eq_xi);
        default=decoder.mean_rhs_values,
    )
    objective_values = _scenario_property(
        scenario_parameter,
        (:q, :q_xi, :objective, :cost, :costs);
        default=decoder.mean_objective_values,
    )

    length(rhs_values) == n_rhs ||
        throw(DimensionMismatch("expected $n_rhs transshipment RHS values."))
    length(objective_values) == n_objective ||
        throw(DimensionMismatch("expected $n_objective transshipment objective values."))

    return vec(rhs_values), vec(objective_values)
end

function _scenario_property(scenario_parameter, names; default)
    for name in names
        if hasproperty(scenario_parameter, name)
            return vec(getproperty(scenario_parameter, name))
        end
    end

    return default
end

function _sample_transshipment_entries(entries, rng; truncate_at_zero)
    values = map(entries) do entry
        value = rand(rng, Distributions.Normal(entry.mean, entry.std))
        return truncate_at_zero ? max(0.0, value) : value
    end
    return Vector{Float64}(values)
end

function _selector_matrix(n_rows::Int, entries)
    selector = zeros(Float64, n_rows, length(entries))
    for (column, entry) in enumerate(entries)
        selector[entry.index, column] = 1.0
    end
    return selector
end

function _indices(names::AbstractVector{String}, selected::AbstractVector{String})
    index = Dict(name => position for (position, name) in enumerate(names))
    return [index[name] for name in selected]
end

function _only_index(names::AbstractVector{String}, selected::AbstractString)
    index = findfirst(==(selected), names)
    isnothing(index) && throw(ArgumentError("Could not find `$selected`."))
    return index
end

function _transshipment_index_key(name::String)
    matched = match(r"\((\d+)\)", name)
    isnothing(matched) && return typemax(Int)
    return parse(Int, matched.captures[1])
end

function _lp_difference_report(left::LP, right::LP)
    return (;
        max_abs_A_eq=_max_abs(left.A_eq - right.A_eq),
        max_abs_A_ineq=_max_abs(left.A_ineq - right.A_ineq),
        max_abs_b_eq=_max_abs(left.b_eq - right.b_eq),
        max_abs_b_ineq=_max_abs(left.b_ineq - right.b_ineq),
        max_abs_c=_max_abs(left.c - right.c),
    )
end

function _transshipment_perturbation_report(problem::TransShipmentProblem)
    decoder = TransShipmentScenarioDecoder(problem)
    mean_parameters = transshipment_mean_parameters(problem)
    base_arrays = transshipment_scenario_arrays(problem, [mean_parameters])

    rhs = copy(mean_parameters.rhs)
    rhs[1] += 1.0
    rhs_arrays = transshipment_scenario_arrays(problem, [(; rhs=rhs, q=mean_parameters.q)])

    q = copy(mean_parameters.q)
    q[1] += 1.0
    q_arrays = transshipment_scenario_arrays(problem, [(; rhs=mean_parameters.rhs, q=q)])

    rhs_h_delta = vec(rhs_arrays[5] - base_arrays[5])
    rhs_q_delta = vec(rhs_arrays[7] - base_arrays[7])
    q_h_delta = vec(q_arrays[5] - base_arrays[5])
    q_q_delta = vec(q_arrays[7] - base_arrays[7])

    return (;
        rhs_affects_h_only=
            count(!iszero, rhs_h_delta) == 1 &&
            rhs_h_delta[problem.data.random_rhs_entries[1].index] == 1.0 &&
            all(iszero, rhs_q_delta),
        objective_affects_q_only=
            count(!iszero, q_q_delta) == 1 &&
            q_q_delta[problem.data.random_objective_entries[1].index] == 1.0 &&
            all(iszero, q_h_delta),
        matrices_unchanged=
            base_arrays[1] == rhs_arrays[1] == q_arrays[1] &&
            base_arrays[2] == rhs_arrays[2] == q_arrays[2] &&
            base_arrays[3] == rhs_arrays[3] == q_arrays[3] &&
            base_arrays[4] == rhs_arrays[4] == q_arrays[4],
    )
end

function _lp_solution_report(lp::LP, result, name::Symbol)
    equality_residual = isempty(lp.b_eq) ? 0.0 : _max_abs(lp.A_eq * result.z - lp.b_eq)
    inequality_violation =
        isempty(lp.b_ineq) ? 0.0 : max(0.0, maximum(lp.A_ineq * result.z - lp.b_ineq))
    linear_objective = LinearAlgebra.dot(lp.c, result.z)

    return (;
        name=name,
        status=string(result.status),
        solver_objective=result.objective_value,
        linear_objective=linear_objective,
        objective_difference=result.objective_value - linear_objective,
        max_equality_residual=equality_residual,
        max_inequality_violation=inequality_violation,
        min_variable=minimum(result.z),
    )
end

_max_abs(values) = isempty(values) ? 0.0 : maximum(abs, values)

# END FILE: src/ContextualDFL/ContextualDFL/src/implementations/TransShipmentProblem/TransShipmentProblem.jl

# BEGIN FILE: src/ContextualDFL/ContextualDFL/src/learning/dataset.jl
struct ParametricScenario{W_EQ,W_INEQ,T_EQ,T_INEQ,H_EQ,H_INEQ,Q}
    W_eq_xi::W_EQ
    W_ineq_xi::W_INEQ
    T_eq_xi::T_EQ
    T_ineq_xi::T_INEQ
    h_eq_xi::H_EQ
    h_ineq_xi::H_INEQ
    q_xi::Q
end

function ParametricScenario(;
    W_eq_xi=0,
    W_ineq_xi=0,
    T_eq_xi=0,
    T_ineq_xi=0,
    h_eq_xi=0,
    h_ineq_xi=0,
    q_xi=0,
)
    return ParametricScenario(
        W_eq_xi,
        W_ineq_xi,
        T_eq_xi,
        T_ineq_xi,
        h_eq_xi,
        h_ineq_xi,
        q_xi,
    )
end

struct ContextualDataPoint{TContext<:AbstractVector,TScenarioParameter<:ParametricScenario}
    context::TContext
    scenario_parameters::Vector{TScenarioParameter}
end

const ContextualDataSet{TDataPoint<:ContextualDataPoint} = Vector{TDataPoint}

# END FILE: src/ContextualDFL/ContextualDFL/src/learning/dataset.jl

# BEGIN FILE: src/ContextualDFL/ContextualDFL/src/learning/loss_functions/DflScenLoss.jl
struct DflScenLoss{
    TInputScenarioDecoder<:VectorDecoder,
    TReferenceScenarioDecoder<:ScenarioDecoder,
    TSolver<:Solver,
    TProgram<:StochasticProgram,
} <: LossFunction
    input_scenario_decoder::TInputScenarioDecoder
    reference_scenario_decoder::TReferenceScenarioDecoder
    solver::TSolver
    program::TProgram
    nr_scenarios::Int
end

function DflScenLoss(
    input_scenario_decoder::VectorDecoder,
    reference_scenario_decoder::ScenarioDecoder,
    solver::Solver,
    program::StochasticProgram;
    nr_scenarios::Integer=1,
)
    nr_scenarios > 0 || throw(ArgumentError("nr_scenarios must be a positive integer."))
    return DflScenLoss{
        typeof(input_scenario_decoder),
        typeof(reference_scenario_decoder),
        typeof(solver),
        typeof(program),
    }(
        input_scenario_decoder,
        reference_scenario_decoder,
        solver,
        program,
        Int(nr_scenarios),
    )
end

function (loss::DflScenLoss)(
    input_scenario_parameter_collection,
    reference_scenario_parameter_collection,
    mu_in=0,
    mu_ref=mu_in;
    rho_in=0,
    rho_ref=rho_in,
    probabilities=nothing,
    nr_scenarios=loss.nr_scenarios,
    kwargs...,
)
    W_eq, W_ineq, T_eq, T_ineq, h_eq, h_ineq, q =
        decode_scenario_collection(
            loss.input_scenario_decoder,
            input_scenario_parameter_collection,
            nr_scenarios=nr_scenarios,
        )
    z, _, _, _, _, _ = solve(
        loss.solver,
        loss.program,
        W_eq,
        W_ineq,
        T_eq,
        T_ineq,
        h_eq,
        h_ineq,
        q;
        probabilities=probabilities,
        μ=mu_in,
        ρ=rho_in,
        kwargs...,
    )

    W_eq, W_ineq, T_eq, T_ineq, h_eq, h_ineq, q =
        decode_scenario_collection(
            loss.reference_scenario_decoder,
            reference_scenario_parameter_collection,
        )
    return cost_function(
        loss.program,
        loss.solver,
        z,
        W_eq,
        W_ineq,
        T_eq,
        T_ineq,
        h_eq,
        h_ineq,
        q;
        probabilities=probabilities,
        μ=mu_ref,
        ρ=rho_ref,
        kwargs...,
    )
end

# END FILE: src/ContextualDFL/ContextualDFL/src/learning/loss_functions/DflScenLoss.jl

# BEGIN FILE: src/ContextualDFL/ContextualDFL/src/learning/loss_functions/Loss.jl
abstract type LossFunction end

(loss::LossFunction)(
    input_scenario_parameter,
    reference_scenario_parameters,
    mu;
    kwargs...,
) =
    error("Loss evaluation is not defined for $(typeof(loss)).")

# END FILE: src/ContextualDFL/ContextualDFL/src/learning/loss_functions/Loss.jl

# BEGIN FILE: src/ContextualDFL/ContextualDFL/src/learning/loss_functions/MSE_scen_loss.jl
struct MSEScenLoss <: LossFunction end

(loss::MSEScenLoss)(
    input_scenario_parameter,
    reference_scenario_parameters,
    mu;
    kwargs...,
) =
    error("MSE scenario loss has not been implemented yet.")

# END FILE: src/ContextualDFL/ContextualDFL/src/learning/loss_functions/MSE_scen_loss.jl

# BEGIN FILE: src/ContextualDFL/ContextualDFL/src/learning/loss_functions/SPO_plus_loss.jl
struct SPOPlusLoss{
    TInputScenarioDecoder<:VectorDecoder,
    TReferenceScenarioDecoder<:ScenarioDecoder,
    TSolver<:Solver,
    TProgram<:StochasticProgram,
} <: LossFunction
    input_scenario_decoder::TInputScenarioDecoder
    reference_scenario_decoder::TReferenceScenarioDecoder
    solver::TSolver
    program::TProgram
    nr_scenarios::Int
end

function SPOPlusLoss(
    input_scenario_decoder::VectorDecoder,
    reference_scenario_decoder::ScenarioDecoder,
    solver::Solver,
    program::StochasticProgram;
    nr_scenarios::Integer=1,
)
    nr_scenarios > 0 || throw(ArgumentError("nr_scenarios must be a positive integer."))
    return SPOPlusLoss{
        typeof(input_scenario_decoder),
        typeof(reference_scenario_decoder),
        typeof(solver),
        typeof(program),
    }(
        input_scenario_decoder,
        reference_scenario_decoder,
        solver,
        program,
        Int(nr_scenarios),
    )
end

function (loss::SPOPlusLoss)(
    input_scenario_parameter_collection,
    reference_scenario_parameter_collection,
    mu_in=0,
    mu_ref=mu_in;
    rho_in=0,
    rho_ref=rho_in,
    probabilities=nothing,
    nr_scenarios=loss.nr_scenarios,
    validate_fixed_feasible_set=true,
    fixed_feasible_set_atol=0,
    fixed_feasible_set_rtol=0,
    kwargs...,
)
    _check_spo_plus_mu(mu_in, mu_ref)

    input_arrays = decode_scenario_collection(
        loss.input_scenario_decoder,
        input_scenario_parameter_collection;
        nr_scenarios=nr_scenarios,
    )
    reference_arrays = decode_scenario_collection(
        loss.reference_scenario_decoder,
        reference_scenario_parameter_collection,
    )

    return _spo_plus_loss_value(
        loss.program,
        loss.solver,
        input_arrays...,
        reference_arrays...;
        rho_in=rho_in,
        rho_ref=rho_ref,
        probabilities=probabilities,
        validate_fixed_feasible_set=validate_fixed_feasible_set,
        fixed_feasible_set_atol=fixed_feasible_set_atol,
        fixed_feasible_set_rtol=fixed_feasible_set_rtol,
        kwargs...,
    )
end

function _spo_plus_loss_value(
    program::StochasticProgram,
    solver::Solver,
    input_W_eq_array,
    input_W_ineq_array,
    input_T_eq_array,
    input_T_ineq_array,
    input_h_eq_array,
    input_h_ineq_array,
    input_q_array,
    reference_W_eq_array,
    reference_W_ineq_array,
    reference_T_eq_array,
    reference_T_ineq_array,
    reference_h_eq_array,
    reference_h_ineq_array,
    reference_q_array;
    rho_in=0,
    rho_ref=rho_in,
    probabilities=nothing,
    validate_fixed_feasible_set=true,
    fixed_feasible_set_atol=0,
    fixed_feasible_set_rtol=0,
    kwargs...,
)
    value, _, _, _ = _spo_plus_oracle(
        program,
        solver,
        input_W_eq_array,
        input_W_ineq_array,
        input_T_eq_array,
        input_T_ineq_array,
        input_h_eq_array,
        input_h_ineq_array,
        input_q_array,
        reference_W_eq_array,
        reference_W_ineq_array,
        reference_T_eq_array,
        reference_T_ineq_array,
        reference_h_eq_array,
        reference_h_ineq_array,
        reference_q_array;
        rho_in=rho_in,
        rho_ref=rho_ref,
        probabilities=probabilities,
        validate_fixed_feasible_set=validate_fixed_feasible_set,
        fixed_feasible_set_atol=fixed_feasible_set_atol,
        fixed_feasible_set_rtol=fixed_feasible_set_rtol,
        kwargs...,
    )
    return value
end

function ChainRulesCore.rrule(
    ::typeof(_spo_plus_loss_value),
    program::StochasticProgram,
    solver::Solver,
    input_W_eq_array,
    input_W_ineq_array,
    input_T_eq_array,
    input_T_ineq_array,
    input_h_eq_array,
    input_h_ineq_array,
    input_q_array,
    reference_W_eq_array,
    reference_W_ineq_array,
    reference_T_eq_array,
    reference_T_ineq_array,
    reference_h_eq_array,
    reference_h_ineq_array,
    reference_q_array;
    rho_in=0,
    rho_ref=rho_in,
    probabilities=nothing,
    validate_fixed_feasible_set=true,
    fixed_feasible_set_atol=0,
    fixed_feasible_set_rtol=0,
    kwargs...,
)
    value, reference_y, perturbed_y, p_vector = _spo_plus_oracle(
        program,
        solver,
        input_W_eq_array,
        input_W_ineq_array,
        input_T_eq_array,
        input_T_ineq_array,
        input_h_eq_array,
        input_h_ineq_array,
        input_q_array,
        reference_W_eq_array,
        reference_W_ineq_array,
        reference_T_eq_array,
        reference_T_ineq_array,
        reference_h_eq_array,
        reference_h_ineq_array,
        reference_q_array;
        rho_in=rho_in,
        rho_ref=rho_ref,
        probabilities=probabilities,
        validate_fixed_feasible_set=validate_fixed_feasible_set,
        fixed_feasible_set_atol=fixed_feasible_set_atol,
        fixed_feasible_set_rtol=fixed_feasible_set_rtol,
        kwargs...,
    )

    function spo_plus_loss_pullback(value_tangent)
        tangent = _spo_plus_scalar_tangent(value_tangent)
        dq = similar(
            input_q_array,
            promote_type(eltype(input_q_array), eltype(reference_y), eltype(perturbed_y), eltype(p_vector), typeof(tangent)),
            size(input_q_array),
        )
        for k in axes(input_q_array, 2)
            dq[:, k] .= tangent .* (2 .* p_vector[k]) .* (reference_y[:, k] .- perturbed_y[:, k])
        end

        return (
            ChainRulesCore.NoTangent(),
            ChainRulesCore.NoTangent(),
            ChainRulesCore.NoTangent(),
            ChainRulesCore.NoTangent(),
            ChainRulesCore.NoTangent(),
            ChainRulesCore.NoTangent(),
            ChainRulesCore.NoTangent(),
            ChainRulesCore.NoTangent(),
            ChainRulesCore.NoTangent(),
            dq,
            ChainRulesCore.NoTangent(),
            ChainRulesCore.NoTangent(),
            ChainRulesCore.NoTangent(),
            ChainRulesCore.NoTangent(),
            ChainRulesCore.NoTangent(),
            ChainRulesCore.NoTangent(),
            ChainRulesCore.NoTangent(),
        )
    end

    return value, spo_plus_loss_pullback
end

function _spo_plus_oracle(
    program::StochasticProgram,
    solver::Solver,
    input_W_eq_array,
    input_W_ineq_array,
    input_T_eq_array,
    input_T_ineq_array,
    input_h_eq_array,
    input_h_ineq_array,
    input_q_array,
    reference_W_eq_array,
    reference_W_ineq_array,
    reference_T_eq_array,
    reference_T_ineq_array,
    reference_h_eq_array,
    reference_h_ineq_array,
    reference_q_array;
    rho_in=0,
    rho_ref=rho_in,
    probabilities=nothing,
    validate_fixed_feasible_set=true,
    fixed_feasible_set_atol=0,
    fixed_feasible_set_rtol=0,
    kwargs...,
)
    _sp_n_scenarios(
        input_W_eq_array,
        input_W_ineq_array,
        input_T_eq_array,
        input_T_ineq_array,
        input_h_eq_array,
        input_h_ineq_array,
        input_q_array,
    )
    K = _sp_n_scenarios(
        reference_W_eq_array,
        reference_W_ineq_array,
        reference_T_eq_array,
        reference_T_ineq_array,
        reference_h_eq_array,
        reference_h_ineq_array,
        reference_q_array,
    )
    size(input_q_array) == size(reference_q_array) ||
        throw(DimensionMismatch("input and reference q arrays must have the same size."))

    if validate_fixed_feasible_set
        _check_spo_plus_fixed_feasible_set(
            input_W_eq_array,
            input_W_ineq_array,
            input_T_eq_array,
            input_T_ineq_array,
            input_h_eq_array,
            input_h_ineq_array,
            reference_W_eq_array,
            reference_W_ineq_array,
            reference_T_eq_array,
            reference_T_ineq_array,
            reference_h_eq_array,
            reference_h_ineq_array;
            atol=fixed_feasible_set_atol,
            rtol=fixed_feasible_set_rtol,
        )
    end

    p_vector = _spo_plus_probability_vector(reference_q_array, probabilities)
    perturbed_q_array = 2 .* input_q_array .- reference_q_array

    reference_z, reference_y, _, _, _, _ = solve(
        solver,
        program,
        reference_W_eq_array,
        reference_W_ineq_array,
        reference_T_eq_array,
        reference_T_ineq_array,
        reference_h_eq_array,
        reference_h_ineq_array,
        reference_q_array;
        probabilities=probabilities,
        μ=0,
        ρ=rho_ref,
        kwargs...,
    )
    perturbed_z, perturbed_y, _, _, _, _ = solve(
        solver,
        program,
        reference_W_eq_array,
        reference_W_ineq_array,
        reference_T_eq_array,
        reference_T_ineq_array,
        reference_h_eq_array,
        reference_h_ineq_array,
        perturbed_q_array;
        probabilities=probabilities,
        μ=0,
        ρ=rho_in,
        kwargs...,
    )

    value =
        _spo_plus_objective(program, reference_z, reference_y, perturbed_q_array; probabilities=probabilities, ρ=rho_ref) -
        _spo_plus_objective(program, perturbed_z, perturbed_y, perturbed_q_array; probabilities=probabilities, ρ=rho_in)

    return value, reference_y, perturbed_y, p_vector
end

function _spo_plus_objective(
    program::StochasticProgram,
    z,
    y,
    q_array;
    probabilities=nothing,
    ρ=0,
    rho=ρ,
)
    K = size(q_array, 2)
    p_vector = _spo_plus_probability_vector(q_array, probabilities)
    first_stage_ρ = _first_stage_quadratic_parameter(program.first_stage_lp, q_array, rho)
    value =
        sum(program.first_stage_lp.c .* z) +
        _first_stage_quadratic_value(z, first_stage_ρ)
    for k in 1:K
        scenario_ρ = _scenario_quadratic_parameter(
            size(q_array, 1),
            K,
            rho,
            k,
            length(program.first_stage_lp.c),
            p_vector[k],
        )
        scenario_ρ_vector = _quadratic_parameter_vector(size(q_array, 1), scenario_ρ)
        y_k = view(y, :, k)
        value += p_vector[k] * (
            sum(view(q_array, :, k) .* y_k) +
            0.5 * sum(scenario_ρ_vector .* (y_k .^ 2))
        )
    end
    return value
end

function _spo_plus_linear_objective(
    program::StochasticProgram,
    z,
    y,
    q_array;
    probabilities=nothing,
)
    K = size(q_array, 2)
    p_vector = _spo_plus_probability_vector(q_array, probabilities)
    value = sum(program.first_stage_lp.c .* z)
    for k in 1:K
        value += p_vector[k] * sum(view(q_array, :, k) .* view(y, :, k))
    end
    return value
end

function _spo_plus_probability_vector(q_array, probabilities)
    K = size(q_array, 2)
    T = eltype(q_array)
    if isnothing(probabilities)
        return fill(one(T) / K, K)
    end

    length(probabilities) == K ||
        throw(DimensionMismatch("probabilities must have one entry per scenario."))
    return probabilities
end

function _check_spo_plus_mu(mu_in, mu_ref)
    _is_zero_barrier_parameter(mu_in) && _is_zero_barrier_parameter(mu_ref) && return nothing
    throw(ArgumentError(
        "SPOPlusLoss does not support log-barrier smoothing; pass mu_in=0 and mu_ref=0.",
    ))
end

function _check_spo_plus_fixed_feasible_set(
    input_W_eq_array,
    input_W_ineq_array,
    input_T_eq_array,
    input_T_ineq_array,
    input_h_eq_array,
    input_h_ineq_array,
    reference_W_eq_array,
    reference_W_ineq_array,
    reference_T_eq_array,
    reference_T_ineq_array,
    reference_h_eq_array,
    reference_h_ineq_array;
    atol,
    rtol,
)
    _check_spo_plus_same_array(:W_eq, input_W_eq_array, reference_W_eq_array; atol=atol, rtol=rtol)
    _check_spo_plus_same_array(:W_ineq, input_W_ineq_array, reference_W_ineq_array; atol=atol, rtol=rtol)
    _check_spo_plus_same_array(:T_eq, input_T_eq_array, reference_T_eq_array; atol=atol, rtol=rtol)
    _check_spo_plus_same_array(:T_ineq, input_T_ineq_array, reference_T_ineq_array; atol=atol, rtol=rtol)
    _check_spo_plus_same_array(:h_eq, input_h_eq_array, reference_h_eq_array; atol=atol, rtol=rtol)
    _check_spo_plus_same_array(:h_ineq, input_h_ineq_array, reference_h_ineq_array; atol=atol, rtol=rtol)
    return nothing
end

function _check_spo_plus_same_array(name, input_array, reference_array; atol, rtol)
    size(input_array) == size(reference_array) ||
        throw(DimensionMismatch("SPOPlusLoss requires matching $(name) arrays."))
    isapprox(input_array, reference_array; atol=atol, rtol=rtol) ||
        throw(ArgumentError(
            "SPOPlusLoss supports objective-vector predictions only; predicted $(name) must match the reference $(name).",
        ))
    return nothing
end

function _spo_plus_scalar_tangent(value_tangent)
    value_tangent = ChainRulesCore.unthunk(value_tangent)
    _is_zero_cotangent(value_tangent) && return 0
    value_tangent isa Number && return value_tangent
    throw(ArgumentError(
        "Expected scalar cotangent for scalar SPOPlusLoss output; got $(typeof(value_tangent)).",
    ))
end

# END FILE: src/ContextualDFL/ContextualDFL/src/learning/loss_functions/SPO_plus_loss.jl

# BEGIN FILE: src/ContextualDFL/ContextualDFL/src/learning/loss_functions/dfl_c_loss.jl
struct DflCLoss{
    TSolver<:Solver,
    TProgram<:StochasticProgram,
    TMu,
} <: LossFunction
    solver::TSolver
    program::TProgram
    mu::TMu
end

(loss::DflCLoss)(
    input_scenario_parameter,
    reference_scenario_parameters,
    mu;
    kwargs...,
) =
    error("DFL cost loss has not been implemented yet.")

# END FILE: src/ContextualDFL/ContextualDFL/src/learning/loss_functions/dfl_c_loss.jl

# BEGIN FILE: src/ContextualDFL/ContextualDFL/src/learning/loss_functions/projected_z_loss.jl
struct ProjectedZLoss{
    TSolver<:Solver,
    TProgram<:StochasticProgram,
} <: LossFunction
    solver::TSolver
    program::TProgram
end

(loss::ProjectedZLoss)(
    input_scenario_parameter,
    reference_scenario_parameters,
    mu;
    kwargs...,
) =
    error("Projected-z loss has not been implemented yet.")

# END FILE: src/ContextualDFL/ContextualDFL/src/learning/loss_functions/projected_z_loss.jl

# BEGIN FILE: src/ContextualDFL/ContextualDFL/src/learning/train.jl
import Flux
import Plots
import Random
import Serialization
import Statistics

# %%% Core training loop

"""
    train!(
        neural_net,
        loss,
        relative_loss,
        mu_in_schedule,
        mu_ref_schedule,
        data_set;
        kwargs...,
    )

Flux training loop for a contextual scenario dataset.

Each data point must have a context vector and a scenario-parameter collection.
At epoch `k`, `loss` is called as:

    loss(neural_net(context), scenario_parameters, mu_in_schedule[k], mu_ref_schedule[k])
"""
function train!(
    neural_net,
    loss,
    relative_loss,
    mu_in_schedule::AbstractVector,
    mu_ref_schedule::AbstractVector,
    data_set;
    opt=nothing,
    optimizer_type=Flux.Adam,
    learning_rate=1e-3,
    epochs::Integer=length(mu_in_schedule),
    batchsize::Integer=1,
    display_iterations::Bool=false,
    verbose::Bool=display_iterations,
    display_plot::Bool=display_iterations,
    save_model::Bool=false,
    model_save_path::AbstractString="trained_model.jls",
    shuffle::Bool=false,
    rng::Random.AbstractRNG=Random.default_rng(),
    opt_state=nothing,
    reset_optimizer_each_epoch::Bool=false,
    on_epoch_end=nothing,
    nr_scenarios=nothing,
    rho_in_schedule=nothing,
    rho_ref_schedule=nothing,
    display_smooth::Bool=false,
    display_real=nothing,
    display_reference_input=nothing,
)
    epochs >= 0 || throw(ArgumentError("epochs must be non-negative."))
    batchsize > 0 || throw(ArgumentError("batchsize must be positive."))
    length(mu_in_schedule) == epochs ||
        throw(ArgumentError("mu_in_schedule must have one value per epoch."))
    length(mu_ref_schedule) == epochs ||
        throw(ArgumentError("mu_ref_schedule must have one value per epoch."))
    rho_in_values = _optional_schedule_values(rho_in_schedule, epochs, :rho_in_schedule)
    rho_ref_values = _optional_schedule_values(
        isnothing(rho_ref_schedule) ? rho_in_schedule : rho_ref_schedule,
        epochs,
        :rho_ref_schedule,
    )
    pass_rho = any(!iszero, rho_in_values) || any(!iszero, rho_ref_values)
    isempty(data_set) && throw(ArgumentError("training data must not be empty."))
    _validate_nr_scenarios(nr_scenarios)
    _validate_display_options(display_smooth, display_real, display_reference_input)

    optimizer = isnothing(opt) ? _make_optimizer(optimizer_type, learning_rate) : opt
    state = isnothing(opt_state) ? Flux.setup(optimizer, neural_net) : opt_state
    show_progress = display_iterations || verbose
    loss_kwargs = _training_loss_kwargs(nr_scenarios)
    display_reference_cache = Dict{Any,Vector{Float64}}()

    if display_smooth
        for (mu_ref, rho_ref) in unique(collect(zip(mu_ref_schedule, rho_ref_values)))
            _display_reference_values!(
                display_reference_cache,
                loss,
                data_set,
                display_reference_input,
                mu_ref;
                rho_ref=rho_ref,
                pass_rho=pass_rho,
                loss_kwargs=loss_kwargs,
            )
        end
    end

    real_reference_values = if isnothing(display_real)
        nothing
    else
        _display_reference_values!(
            display_reference_cache,
            loss,
            data_set,
            display_reference_input,
            0.0;
            rho_ref=0.0,
            pass_rho=pass_rho,
            loss_kwargs=loss_kwargs,
        )
    end

    history = NamedTuple[]
    displayed_epoch_losses = Float64[]

    for epoch_number in 1:epochs
        epoch_started = time()
        mu_in = mu_in_schedule[epoch_number]
        mu_ref = mu_ref_schedule[epoch_number]
        rho_in = rho_in_values[epoch_number]
        rho_ref = rho_ref_values[epoch_number]
        epoch_loss_kwargs = _with_rho_loss_kwargs(loss_kwargs, pass_rho, rho_in, rho_ref)

        if reset_optimizer_each_epoch
            state = Flux.setup(optimizer, neural_net)
        end

        show_progress && print("Epoch ", epoch_number)
        epoch_losses = Float64[]
        epoch_display_losses = Float64[]

        indices = shuffle ? Random.randperm(rng, length(data_set)) : collect(eachindex(data_set))
        for idxs_iter in Iterators.partition(indices, batchsize)
            idxs = collect(idxs_iter)

            loss_value, gradients = Flux.withgradient(neural_net) do trainable_neural_net
                Statistics.mean(
                    loss(
                        trainable_neural_net(_context(data_set[index])),
                        _scenario_parameters(data_set[index]),
                        mu_in,
                        mu_ref;
                        epoch_loss_kwargs...,
                    )
                    for index in idxs
                )
            end
            iteration_number = length(epoch_losses) + 1
            loss_float = _checked_loss_float(
                loss_value,
                "training loss";
                epoch=epoch_number,
                iteration=iteration_number,
                mu_in=mu_in,
                mu_ref=mu_ref,
            )
            Flux.update!(state, neural_net, gradients[1])
            push!(epoch_losses, loss_float)

            if display_smooth
                reference_key = _display_reference_cache_key(mu_ref, rho_ref, pass_rho)
                reference_mean = Statistics.mean(
                    display_reference_cache[reference_key][index] for index in idxs
                )
                display_loss = _relative_display_loss(loss_float, reference_mean)
                push!(
                    epoch_display_losses,
                    _checked_loss_float(
                        display_loss,
                        "smooth display loss";
                        epoch=epoch_number,
                        iteration=iteration_number,
                        mu_in=mu_in,
                        mu_ref=mu_ref,
                    ),
                )
            elseif show_progress || !isnothing(relative_loss)
                display_loss_function = isnothing(relative_loss) ? loss : relative_loss
                display_loss = Statistics.mean(
                    display_loss_function(
                        neural_net(_context(data_set[index])),
                        _scenario_parameters(data_set[index]),
                        mu_in,
                        mu_ref;
                        epoch_loss_kwargs...,
                    )
                    for index in idxs
                )
                push!(
                    epoch_display_losses,
                    _checked_loss_float(
                        display_loss,
                        "display loss";
                        epoch=epoch_number,
                        iteration=iteration_number,
                        mu_in=mu_in,
                        mu_ref=mu_ref,
                    ),
                )
            end
        end

        average_loss = Statistics.mean(epoch_losses)
        average_display_loss = isempty(epoch_display_losses) ?
            average_loss :
            Statistics.mean(epoch_display_losses)
        real_display_loss = if isnothing(display_real) ||
                               epoch_number % Int(display_real) != 0
            nothing
        else
            real_loss = Statistics.mean(
                loss(
                    neural_net(_context(data_point)),
                    _scenario_parameters(data_point),
                    mu_in,
                    0.0;
                    _with_rho_loss_kwargs(loss_kwargs, pass_rho, rho_in, 0.0)...,
                )
                for data_point in data_set
            )
            real_loss_float = _checked_loss_float(
                real_loss,
                "real display loss";
                epoch=epoch_number,
                iteration=length(epoch_losses),
                mu_in=mu_in,
                mu_ref=0.0,
            )
            real_reference_mean = Statistics.mean(real_reference_values)
            _checked_loss_float(
                _relative_display_loss(real_loss_float, real_reference_mean),
                "relative real display loss";
                epoch=epoch_number,
                iteration=length(epoch_losses),
                mu_in=mu_in,
                mu_ref=0.0,
            )
        end
        epoch_seconds = time() - epoch_started
        epoch_metadata = (;
            epoch=Int(epoch_number),
            mu=mu_in,
            mu_in=mu_in,
            mu_ref=mu_ref,
            rho_in=rho_in,
            rho_ref=rho_ref,
            iterations=length(epoch_losses),
            epoch_seconds=epoch_seconds,
            real_display_loss=real_display_loss,
        )

        if show_progress
            print(" with avg loss ", average_display_loss)
            isnothing(real_display_loss) || print(" real loss ", real_display_loss)
            println(" (", length(epoch_display_losses), " iterations)")
            push!(displayed_epoch_losses, average_display_loss)
        end

        if !isnothing(on_epoch_end)
            _call_epoch_callback(
                on_epoch_end,
                Int(epoch_number),
                average_loss,
                average_display_loss,
                epoch_metadata,
            )
        end

        push!(
            history,
            (;
                epoch=Int(epoch_number),
                mu=mu_in,
                mu_in=mu_in,
                mu_ref=mu_ref,
                rho_in=rho_in,
                rho_ref=rho_ref,
                loss=average_loss,
                display_loss=average_display_loss,
                real_display_loss=real_display_loss,
                iterations=length(epoch_losses),
                epoch_seconds=epoch_seconds,
            ),
        )
    end

    # %%% Optional model storage
    if save_model
        Serialization.serialize(model_save_path, neural_net)
        println("Model saved to: $model_save_path")
    end

    # %%% Optional training-loss plot
    if display_plot && show_progress && !isempty(displayed_epoch_losses)
        plt = Plots.plot(
            1:length(displayed_epoch_losses),
            displayed_epoch_losses;
            xlabel="Epoch",
            ylabel="Loss",
            title="Training Loss",
        )
        display(plt)
    end

    return (; model=neural_net, history=history, opt_state=state)
end

train!(
    neural_net,
    loss,
    relative_loss,
    mu_in_schedule::AbstractVector,
    data_set;
    mu_ref_schedule=nothing,
    kwargs...,
) =
    train!(
        neural_net,
        loss,
        relative_loss,
        mu_in_schedule,
        _default_mu_ref_schedule(mu_in_schedule, mu_ref_schedule),
        data_set;
        kwargs...,
    )

train!(
    neural_net,
    loss,
    mu_schedule::AbstractVector,
    data_set;
    kwargs...,
) =
    train!(
        neural_net,
        loss,
        nothing,
        mu_schedule,
        data_set;
        kwargs...,
    )

train!(
    neural_net,
    loss,
    mu_in_schedule::AbstractVector,
    mu_ref_schedule::AbstractVector,
    data_set;
    kwargs...,
) =
    train!(
        neural_net,
        loss,
        nothing,
        mu_in_schedule,
        mu_ref_schedule,
        data_set;
        kwargs...,
    )

# %%% Small core helpers

_context(data_point::ContextualDataPoint) = data_point.context
_context(data_point::Tuple) = data_point[1]

_scenario_parameters(data_point::ContextualDataPoint) = data_point.scenario_parameters
_scenario_parameters(data_point::Tuple) = data_point[2]

function _validate_nr_scenarios(nr_scenarios)
    isnothing(nr_scenarios) && return nothing
    nr_scenarios isa Integer && nr_scenarios > 0 ||
        throw(ArgumentError("nr_scenarios must be a positive integer."))
    return nothing
end

_training_loss_kwargs(nr_scenarios) =
    isnothing(nr_scenarios) ? NamedTuple() : (; nr_scenarios=Int(nr_scenarios))

function _optional_schedule_values(schedule, epochs, name::Symbol)
    if isnothing(schedule)
        return zeros(Float64, epochs)
    end

    length(schedule) == epochs ||
        throw(ArgumentError("$(name) must have one value per epoch."))
    return collect(schedule)
end

function _with_rho_loss_kwargs(loss_kwargs, pass_rho, rho_in, rho_ref)
    pass_rho || return loss_kwargs
    return merge(loss_kwargs, (; rho_in=rho_in, rho_ref=rho_ref))
end

_default_mu_ref_schedule(mu_in_schedule, mu_ref_schedule) =
    isnothing(mu_ref_schedule) ? mu_in_schedule : mu_ref_schedule

function _validate_display_options(display_smooth, display_real, display_reference_input)
    if !isnothing(display_real)
        display_real isa Integer && !(display_real isa Bool) && display_real > 0 ||
            throw(ArgumentError("display_real must be nothing or a positive integer."))
    end

    if (display_smooth || !isnothing(display_real)) && isnothing(display_reference_input)
        throw(
            ArgumentError(
                "display_smooth/display_real require display_reference_input, " *
                "a function from data point to exact-scenario decoder input.",
            ),
        )
    end

    return nothing
end

function _display_reference_values!(
    cache,
    loss,
    data_set,
    display_reference_input,
    mu_ref;
    rho_ref=0,
    pass_rho=false,
    loss_kwargs,
)
    cache_key = _display_reference_cache_key(mu_ref, rho_ref, pass_rho)
    haskey(cache, cache_key) && return cache[cache_key]
    reference_loss_kwargs = _with_rho_loss_kwargs(loss_kwargs, pass_rho, rho_ref, rho_ref)

    # Reference inputs represent the exact scenario, so this is the cached baseline.
    values = [
        _checked_loss_float(
            loss(
                display_reference_input(data_point),
                _scenario_parameters(data_point),
                mu_ref,
                mu_ref;
                reference_loss_kwargs...,
            ),
            "display reference loss";
            epoch=0,
            iteration=index,
            mu_in=mu_ref,
            mu_ref=mu_ref,
        ) for (index, data_point) in enumerate(data_set)
    ]
    cache[cache_key] = values
    return values
end

_display_reference_cache_key(mu_ref, rho_ref, pass_rho) =
    pass_rho ? (mu_ref, rho_ref) : mu_ref

_relative_display_loss(loss_value, reference_value) =
    (Float64(loss_value) - Float64(reference_value)) / abs(Float64(reference_value))

_float(value::Number) = Float64(value)
_float(value::AbstractArray) = Float64(only(value))

function _checked_loss_float(value, label; epoch, iteration, mu=nothing, mu_in=mu, mu_ref=0)
    float_value = _float(value)
    isfinite(float_value) || throw(
        DomainError(
            float_value,
            "$label became non-finite at epoch=$(epoch) iteration=$(iteration) mu_in=$(mu_in) mu_ref=$(mu_ref)",
        ),
    )
    return float_value
end

function _call_epoch_callback(callback, epoch, loss_value, display_loss, metadata)
    if applicable(callback, epoch, loss_value, display_loss, metadata)
        return callback(epoch, loss_value, display_loss, metadata)
    end
    return callback(epoch, loss_value, display_loss)
end

# %%% Optimizer helpers

function _make_optimizer(optimizer_type::Symbol, learning_rate)
    if optimizer_type === :adam
        return Flux.Adam(learning_rate)
    elseif optimizer_type in (:descent, :sgd)
        return Flux.Descent(learning_rate)
    elseif optimizer_type === :rmsprop
        return Flux.RMSProp(learning_rate)
    end

    throw(ArgumentError("unsupported optimizer_type `$optimizer_type`."))
end

_make_optimizer(optimizer_type, learning_rate) = optimizer_type(learning_rate)

# END FILE: src/ContextualDFL/ContextualDFL/src/learning/train.jl

# BEGIN FILE: src/ContextualDFL/ContextualDFL/src/learning/utils/hyper_parameter_helpers/schedules.jl
constant_schedule(value) =
    error("constant_schedule has not been implemented yet.")

mu_schedule(args...; kwargs...) =
    error("mu_schedule has not been implemented yet.")

rho_schedule(args...; kwargs...) =
    error("rho_schedule has not been implemented yet.")

batch_size_schedule(args...; kwargs...) =
    error("batch_size_schedule has not been implemented yet.")

step_size_schedule(args...; kwargs...) =
    error("step_size_schedule has not been implemented yet.")

# END FILE: src/ContextualDFL/ContextualDFL/src/learning/utils/hyper_parameter_helpers/schedules.jl

# BEGIN FILE: src/ContextualDFL/ContextualDFL/src/learning/utils/test.jl
test(z_policy, data_set::ContextualDataSet; kwargs...) =
    error("Policy testing has not been implemented yet.")

# END FILE: src/ContextualDFL/ContextualDFL/src/learning/utils/test.jl

# BEGIN FILE: src/ContextualDFL/ContextualDFL/src/linear_programming/LP.jl
const _MaybeMatrix = Union{Nothing,AbstractMatrix}
const _MaybeVector = Union{Nothing,AbstractVector}

struct LP{
    TAeq<:AbstractMatrix,
    TAineq<:AbstractMatrix,
    Tbeq<:AbstractVector,
    Tbineq<:AbstractVector,
    Tc<:AbstractVector,
}
    A_eq::TAeq
    A_ineq::TAineq
    b_eq::Tbeq
    b_ineq::Tbineq
    c::Tc

    function LP(
        A_eq::TAeq,
        A_ineq::TAineq,
        b_eq::Tbeq,
        b_ineq::Tbineq,
        c::Tc,
    ) where {
        TAeq<:AbstractMatrix,
        TAineq<:AbstractMatrix,
        Tbeq<:AbstractVector,
        Tbineq<:AbstractVector,
        Tc<:AbstractVector,
    }
        _validate_lp_dimensions(A_eq, A_ineq, b_eq, b_ineq, c)
        return new{TAeq,TAineq,Tbeq,Tbineq,Tc}(A_eq, A_ineq, b_eq, b_ineq, c)
    end
end

function LP(
    A_eq::_MaybeMatrix,
    A_ineq::_MaybeMatrix,
    b_eq::_MaybeVector,
    b_ineq::_MaybeVector,
    c::_MaybeVector,
)
    return LP(_canonical_lp_data(A_eq, A_ineq, b_eq, b_ineq, c)...)
end

LP(; A_eq=nothing, A_ineq=nothing, b_eq=nothing, b_ineq=nothing, c=nothing) =
    LP(A_eq, A_ineq, b_eq, b_ineq, c)

function _canonical_lp_data(A_eq, A_ineq, b_eq, b_ineq, c)
    n_variables = _infer_variable_count(A_eq, A_ineq, c)
    T = _infer_lp_eltype(A_eq, A_ineq, b_eq, b_ineq, c)

    A_eq, b_eq = _canonical_constraint_pair(:A_eq, :b_eq, A_eq, b_eq, n_variables, T)
    A_ineq, b_ineq =
        _canonical_constraint_pair(:A_ineq, :b_ineq, A_ineq, b_ineq, n_variables, T)
    c = isnothing(c) ? zeros(T, n_variables) : c

    return A_eq, A_ineq, b_eq, b_ineq, c
end

function _infer_variable_count(A_eq, A_ineq, c)
    counts = Int[]
    isnothing(A_eq) || push!(counts, size(A_eq, 2))
    isnothing(A_ineq) || push!(counts, size(A_ineq, 2))
    isnothing(c) || push!(counts, length(c))

    isempty(counts) && return 0

    n_variables = first(counts)
    all(==(n_variables), counts) ||
        throw(DimensionMismatch("LP inputs disagree on the number of variables."))

    return n_variables
end

function _infer_lp_eltype(values...)
    types = Type[]

    for value in values
        if !isnothing(value) && (!isempty(value) || eltype(value) !== Any)
            push!(types, eltype(value))
        end
    end

    return isempty(types) ? Float64 : promote_type(types...)
end

function _canonical_constraint_pair(A_name, b_name, A, b, n_variables, T)
    if isnothing(A)
        if isnothing(b) || isempty(b)
            return Matrix{T}(undef, 0, n_variables), Vector{T}(undef, 0)
        end

        throw(ArgumentError("$(A_name) must be provided when $(b_name) has entries."))
    end

    if isnothing(b)
        return A, zeros(T, size(A, 1))
    end

    return A, b
end

function _validate_lp_dimensions(A_eq, A_ineq, b_eq, b_ineq, c)
    n_variables = length(c)

    size(A_eq, 2) == n_variables ||
        throw(DimensionMismatch("A_eq must have length(c) columns."))
    size(A_ineq, 2) == n_variables ||
        throw(DimensionMismatch("A_ineq must have length(c) columns."))
    size(A_eq, 1) == length(b_eq) ||
        throw(DimensionMismatch("A_eq and b_eq must have matching row counts."))
    size(A_ineq, 1) == length(b_ineq) ||
        throw(DimensionMismatch("A_ineq and b_ineq must have matching row counts."))

    return nothing
end

# END FILE: src/ContextualDFL/ContextualDFL/src/linear_programming/LP.jl

# BEGIN FILE: src/ContextualDFL/ContextualDFL/src/linear_programming/Solvers/Solver.jl
struct Solver{TLogBarSolver<:LogBarSolver,TLPSolver<:LPSolver}
    log_bar_solver::TLogBarSolver
    lp_solver::TLPSolver
end

function solve(solver::Solver, lp::LP; μ=0, ρ=0, rho=ρ, kwargs...)
    μ_vector = _barrier_parameter_vector(lp, μ)
    ρ_vector = _quadratic_parameter_vector(lp, rho)

    if _is_zero_barrier_parameter(μ_vector) && _is_zero_quadratic_parameter(ρ_vector)
        return solve(solver.lp_solver, lp; kwargs...)
    end

    return solve(solver.log_bar_solver, lp; μ=μ_vector, ρ=ρ_vector, kwargs...)
end

# END FILE: src/ContextualDFL/ContextualDFL/src/linear_programming/Solvers/Solver.jl

# BEGIN FILE: src/ContextualDFL/ContextualDFL/src/linear_programming/Solvers/solver_status.jl
import JuMP

function _assert_successful_solve(
    model,
    solver;
    accepted_statuses,
    accepted_primal_statuses=("FEASIBLE_POINT", "NEARLY_FEASIBLE_POINT"),
)
    status = JuMP.termination_status(model)
    primal_status = JuMP.primal_status(model)

    status_name = string(status)
    primal_status_name = string(primal_status)
    accepted_status_names = Set(string.(accepted_statuses))
    accepted_primal_status_names = Set(string.(accepted_primal_statuses))

    if !(status_name in accepted_status_names) ||
       !(primal_status_name in accepted_primal_status_names)
        throw(
            ErrorException(
                string(
                    typeof(solver),
                    " failed to solve the optimization problem: ",
                    "termination_status=",
                    status,
                    ", primal_status=",
                    primal_status,
                    ", dual_status=",
                    JuMP.dual_status(model),
                    ", raw_status=",
                    JuMP.raw_status(model),
                    ".",
                ),
            ),
        )
    end

    return status
end

function _assert_lp_solution_feasible(lp::LP, z; atol=1e-6)
    all(isfinite, z) ||
        throw(DomainError(z, "The solver returned non-finite primal values."))

    if !isempty(lp.b_eq)
        equality_residual = lp.A_eq * z - lp.b_eq
        maximum(abs, equality_residual) <= atol ||
            throw(
                DomainError(
                    equality_residual,
                    "The solver returned a solution that violates equality constraints.",
                ),
            )
    end

    if !isempty(lp.b_ineq)
        inequality_violation = lp.A_ineq * z - lp.b_ineq
        maximum(inequality_violation) <= atol ||
            throw(
                DomainError(
                    inequality_violation,
                    "The solver returned a solution that violates inequality constraints.",
                ),
            )
    end

    return nothing
end

# END FILE: src/ContextualDFL/ContextualDFL/src/linear_programming/Solvers/solver_status.jl

# BEGIN FILE: src/ContextualDFL/ContextualDFL/src/linear_programming/Solvers/solvers/LPSolvers/LPSolver.jl
abstract type LPSolver end

solve(solver::LPSolver, lp::LP; kwargs...) =
    error("LP solving is not defined for $(typeof(solver)).")

# END FILE: src/ContextualDFL/ContextualDFL/src/linear_programming/Solvers/solvers/LPSolvers/LPSolver.jl

# BEGIN FILE: src/ContextualDFL/ContextualDFL/src/linear_programming/Solvers/solvers/LPSolvers/implemented_solvers/GLPKSolver.jl
struct GLPKSolver <: LPSolver end

# END FILE: src/ContextualDFL/ContextualDFL/src/linear_programming/Solvers/solvers/LPSolvers/implemented_solvers/GLPKSolver.jl

# BEGIN FILE: src/ContextualDFL/ContextualDFL/src/linear_programming/Solvers/solvers/LPSolvers/implemented_solvers/GurobiSolver.jl
struct GurobiSolver <: LPSolver end

# END FILE: src/ContextualDFL/ContextualDFL/src/linear_programming/Solvers/solvers/LPSolvers/implemented_solvers/GurobiSolver.jl

# BEGIN FILE: src/ContextualDFL/ContextualDFL/src/linear_programming/Solvers/solvers/LPSolvers/implemented_solvers/HiGHSSolver.jl
import HiGHS
import JuMP

struct HiGHSSolver <: LPSolver end

function solve(solver::HiGHSSolver, lp::LP; constraint_tolerance=1e-6, kwargs...)
    bound_lp, bound_map = _extract_variable_bounds_for_solver(solver, lp)

    model = JuMP.Model(HiGHS.Optimizer)
    JuMP.set_silent(model)
    _set_optimizer_attributes(model, kwargs)

    n_variables = length(bound_lp.c)
    JuMP.@variable(model, z[1:n_variables])
    _set_variable_bounds!(z, bound_lp.lower_bounds, bound_lp.upper_bounds)

    eq_constraints = JuMP.@constraint(model, bound_lp.A_eq * z .== bound_lp.b_eq)
    ineq_constraints = JuMP.@constraint(model, bound_lp.A_ineq * z .<= bound_lp.b_ineq)

    JuMP.@objective(model, Min, sum(bound_lp.c[j] * z[j] for j in 1:n_variables))
    JuMP.optimize!(model)

    status = _assert_successful_solve(model, solver; accepted_statuses=("OPTIMAL",))
    z_value = JuMP.value.(z)
    _assert_lp_solution_feasible(lp, z_value; atol=constraint_tolerance)

    lower_bound_dual, upper_bound_dual =
        _normalized_variable_bound_duals(z, bound_lp.lower_bounds, bound_lp.upper_bounds)
    raw_result = BoundFormSolveResult(
        z_value,
        bound_lp.b_ineq - bound_lp.A_ineq * z_value,
        -JuMP.dual.(ineq_constraints),
        JuMP.dual.(eq_constraints),
        lower_bound_dual,
        upper_bound_dual,
        JuMP.objective_value(model),
        status,
        (;
            primal_status=JuMP.primal_status(model),
            dual_status=JuMP.dual_status(model),
            raw_status=JuMP.raw_status(model),
            solver=solver,
        ),
    )

    return _reconstruct_original_lp_result(lp, bound_map, raw_result)
end

function _set_optimizer_attributes(model, kwargs)
    for (attribute, value) in kwargs
        JuMP.set_optimizer_attribute(model, String(attribute), value)
    end

    return nothing
end

function _set_variable_bounds!(z, lower_bounds, upper_bounds)
    @inbounds for j in eachindex(z)
        if isfinite(lower_bounds[j])
            JuMP.set_lower_bound(z[j], lower_bounds[j])
        end
        if isfinite(upper_bounds[j])
            JuMP.set_upper_bound(z[j], upper_bounds[j])
        end
    end

    return nothing
end

function _normalized_variable_bound_duals(z, lower_bounds, upper_bounds)
    T = promote_type(Float64, eltype(lower_bounds), eltype(upper_bounds))
    lower_bound_dual = zeros(T, length(z))
    upper_bound_dual = zeros(T, length(z))

    @inbounds for j in eachindex(z)
        if isfinite(lower_bounds[j])
            lower_bound_dual[j] = T(JuMP.dual(JuMP.LowerBoundRef(z[j])))
        end
        if isfinite(upper_bounds[j])
            upper_bound_dual[j] = -T(JuMP.dual(JuMP.UpperBoundRef(z[j])))
        end
    end

    return lower_bound_dual, upper_bound_dual
end

# END FILE: src/ContextualDFL/ContextualDFL/src/linear_programming/Solvers/solvers/LPSolvers/implemented_solvers/HiGHSSolver.jl

# BEGIN FILE: src/ContextualDFL/ContextualDFL/src/linear_programming/Solvers/solvers/LogBarSolvers/LogBarSolver.jl
abstract type LogBarSolver end

solve(solver::LogBarSolver, lp::LP; μ=nothing, ρ=0, kwargs...) =
    error("Log-barrier LP solving is not defined for $(typeof(solver)).")

function _barrier_parameter_vector(n_inequalities::Integer, μ)
    isnothing(μ) &&
        throw(ArgumentError("A log-barrier parameter μ must be provided."))
    n_inequalities >= 0 || throw(ArgumentError("n_inequalities must be non-negative."))

    if μ isa Number
        μ >= zero(μ) || throw(ArgumentError("μ must be non-negative."))
        return fill(μ, n_inequalities)
    end

    μ isa AbstractVector ||
        throw(ArgumentError("μ must be a scalar or a vector with one entry per inequality."))
    length(μ) == n_inequalities ||
        throw(DimensionMismatch("μ must have one entry per inequality."))
    any(value -> value < zero(value), μ) &&
        throw(ArgumentError("μ entries must be non-negative."))

    return collect(μ)
end

_barrier_parameter_vector(lp::LP, μ) =
    _barrier_parameter_vector(length(lp.b_ineq), μ)

function _quadratic_parameter_vector(n_variables::Integer, ρ)
    n_variables >= 0 || throw(ArgumentError("n_variables must be non-negative."))

    if ρ isa Number
        ρ >= zero(ρ) || throw(ArgumentError("ρ must be non-negative."))
        return fill(ρ, n_variables)
    end

    ρ isa AbstractVector ||
        throw(ArgumentError("ρ must be a scalar or a vector with one entry per variable."))
    length(ρ) == n_variables ||
        throw(DimensionMismatch("ρ must have one entry per variable."))
    any(value -> value < zero(value), ρ) &&
        throw(ArgumentError("ρ entries must be non-negative."))

    return collect(ρ)
end

_quadratic_parameter_vector(lp::LP, ρ) =
    _quadratic_parameter_vector(length(lp.c), ρ)

_is_zero_barrier_parameter(μ) =
    μ isa Number ? iszero(μ) : all(iszero, μ)

_is_zero_quadratic_parameter(ρ) =
    ρ isa Number ? iszero(ρ) : all(iszero, ρ)

# END FILE: src/ContextualDFL/ContextualDFL/src/linear_programming/Solvers/solvers/LogBarSolvers/LogBarSolver.jl

# BEGIN FILE: src/ContextualDFL/ContextualDFL/src/linear_programming/Solvers/solvers/LogBarSolvers/implemented_solvers/IpoptSolver.jl
import Ipopt
import JuMP

struct IpoptSolver <: LogBarSolver end

function solve(
    solver::IpoptSolver,
    lp::LP;
    μ=nothing,
    ρ=0,
    rho=ρ,
    slack_lower_bound=1e-9,
    constraint_tolerance=1e-6,
    kwargs...,
)
    μ_value = isnothing(μ) ? zeros(eltype(lp.c), length(lp.b_ineq)) : μ
    μ_vector = _barrier_parameter_vector(lp, μ_value)
    ρ_vector = _quadratic_parameter_vector(lp, rho)
    positive_barrier_indices = findall(!iszero, μ_vector)
    positive_quadratic_indices = findall(!iszero, ρ_vector)
    isempty(positive_barrier_indices) && isempty(positive_quadratic_indices) &&
        throw(ArgumentError("IpoptSolver requires at least one positive smoothing weight."))
    slack_lower_bound > zero(slack_lower_bound) ||
        throw(ArgumentError("slack_lower_bound must be positive."))
    bound_lp, bound_map = _extract_variable_bounds_for_solver(
        solver,
        lp;
        μ_vector=μ_vector,
        slack_lower_bound=slack_lower_bound,
    )

    model = JuMP.Model(Ipopt.Optimizer)
    JuMP.set_optimizer_attribute(model, "print_level", 0)
    JuMP.set_optimizer_attribute(model, "sb", "yes")
    JuMP.set_optimizer_attribute(model, "mu_strategy", "monotone")
    JuMP.set_optimizer_attribute(model, "nlp_scaling_method", "none")
    _set_optimizer_attributes(model, kwargs)

    n_variables = length(bound_lp.c)
    n_general_inequalities = length(bound_lp.b_ineq)

    JuMP.@variable(model, z[1:n_variables])
    _set_variable_bounds!(z, bound_lp.lower_bounds, bound_lp.upper_bounds)

    s = Vector{JuMP.VariableRef}(undef, n_general_inequalities)
    slack_constraints = Vector{JuMP.ConstraintRef}(undef, n_general_inequalities)
    if n_general_inequalities > 0
        JuMP.@variable(model, general_slack[1:n_general_inequalities] >= 0)
        s = general_slack
        for k in 1:n_general_inequalities
            if !iszero(μ_vector[bound_map.general_rows[k]])
                JuMP.set_lower_bound(s[k], slack_lower_bound)
            end
        end
        slack_constraints =
            JuMP.@constraint(model, bound_lp.A_ineq * z .+ s .== bound_lp.b_ineq)
    end

    eq_constraints = JuMP.@constraint(model, bound_lp.A_eq * z .== bound_lp.b_eq)

    positive_general_positions = [
        k for k in eachindex(bound_map.general_rows) if !iszero(μ_vector[bound_map.general_rows[k]])
    ]
    positive_bound_rows = [
        row for row in bound_map.bound_rows if !iszero(μ_vector[row.original_row])
    ]
    general_original_rows = bound_map.general_rows
    bound_original_rows = [row.original_row for row in positive_bound_rows]
    bound_variables = [row.variable for row in positive_bound_rows]
    bound_coefficients = [row.coefficient for row in positive_bound_rows]
    bound_rhs = [row.rhs for row in positive_bound_rows]

    JuMP.@NLobjective(
        model,
        Min,
        sum(bound_lp.c[j] * z[j] for j in 1:n_variables) +
        0.5 * sum(ρ_vector[j] * z[j]^2 for j in positive_quadratic_indices) -
        sum(
            μ_vector[general_original_rows[k]] * log(s[k]) for
            k in positive_general_positions
        ) -
        sum(
            μ_vector[bound_original_rows[k]] *
            log(bound_rhs[k] - bound_coefficients[k] * z[bound_variables[k]]) for
            k in eachindex(bound_original_rows)
        ),
    )
    JuMP.optimize!(model)

    status =
        _assert_successful_solve(model, solver; accepted_statuses=("OPTIMAL", "LOCALLY_SOLVED"))
    z_value = JuMP.value.(z)
    _assert_lp_solution_feasible(lp, z_value; atol=constraint_tolerance)

    lower_bound_dual, upper_bound_dual =
        _normalized_variable_bound_duals(z, bound_lp.lower_bounds, bound_lp.upper_bounds)
    raw_result = BoundFormSolveResult(
        z_value,
        n_general_inequalities == 0 ? similar(z_value, 0) : JuMP.value.(s),
        n_general_inequalities == 0 ? similar(z_value, 0) : -JuMP.dual.(slack_constraints),
        JuMP.dual.(eq_constraints),
        lower_bound_dual,
        upper_bound_dual,
        JuMP.objective_value(model),
        status,
        (;
            primal_status=JuMP.primal_status(model),
            dual_status=JuMP.dual_status(model),
            raw_status=JuMP.raw_status(model),
            solver=solver,
            ρ_vector=ρ_vector,
        ),
    )

    return _reconstruct_original_lp_result(
        lp,
        bound_map,
        raw_result;
        μ_vector=μ_vector,
        include_slack=true,
    )
end

# END FILE: src/ContextualDFL/ContextualDFL/src/linear_programming/Solvers/solvers/LogBarSolvers/implemented_solvers/IpoptSolver.jl

# BEGIN FILE: src/ContextualDFL/ContextualDFL/src/linear_programming/bound_form_lp.jl
import SparseArrays: findnz, issparse

struct ExtractedBoundRow{T}
    original_row::Int
    variable::Int
    coefficient::T
    rhs::T
    is_lower::Bool
end

struct InequalityBoundMap{T}
    bound_rows::Vector{ExtractedBoundRow{T}}
    general_rows::Vector{Int}
    lower_bounds::Vector{T}
    upper_bounds::Vector{T}
    lower_owner::Vector{Int}
    upper_owner::Vector{Int}
end

struct BoundFormLP{
    TAeq,
    TAineq,
    Tbeq,
    Tbineq,
    Tc,
    Tbounds,
    Tmap,
}
    A_eq::TAeq
    A_ineq::TAineq
    b_eq::Tbeq
    b_ineq::Tbineq
    c::Tc
    lower_bounds::Tbounds
    upper_bounds::Tbounds
    bound_map::Tmap
end

struct BoundFormSolveResult{
    Tz,
    TgeneralSlack,
    TgeneralDualIneq,
    TdualEq,
    TlowerBoundDual,
    TupperBoundDual,
    TobjectiveValue,
    Tstatus,
    Tmetadata,
}
    z::Tz
    general_slack::TgeneralSlack
    general_dual_ineq::TgeneralDualIneq
    dual_eq::TdualEq
    lower_bound_dual::TlowerBoundDual
    upper_bound_dual::TupperBoundDual
    objective_value::TobjectiveValue
    status::Tstatus
    metadata::Tmetadata
end

function _extract_variable_bounds(
    lp::LP;
    μ_vector=nothing,
    slack_lower_bound::Real=1e-9,
    coefficient_atol::Real=0.0,
    convert_zero_barrier_rows::Bool=true,
)
    A = lp.A_ineq
    b = lp.b_ineq
    m, n = size(A)

    if !isnothing(μ_vector) && length(μ_vector) != m
        throw(DimensionMismatch("μ_vector must have one entry per inequality."))
    end

    T = promote_type(
        Float64,
        eltype(A),
        eltype(b),
        eltype(lp.c),
        typeof(slack_lower_bound),
    )

    row_counts = zeros(Int, m)
    row_variables = zeros(Int, m)
    row_coefficients = zeros(T, m)
    coefficient_threshold = T(coefficient_atol)

    if issparse(A)
        I, J, V = findnz(A)
        @inbounds for k in eachindex(V)
            a = T(V[k])
            abs(a) <= coefficient_threshold && continue

            i = I[k]
            row_counts[i] += 1
            if row_counts[i] == 1
                row_variables[i] = J[k]
                row_coefficients[i] = a
            end
        end
    else
        @inbounds for j in 1:n
            for i in 1:m
                a = T(A[i, j])
                abs(a) <= coefficient_threshold && continue

                row_counts[i] += 1
                if row_counts[i] == 1
                    row_variables[i] = j
                    row_coefficients[i] = a
                end
            end
        end
    end

    bound_rows = ExtractedBoundRow{T}[]
    general_rows = Int[]
    lower_bounds = fill(-T(Inf), n)
    upper_bounds = fill(T(Inf), n)
    lower_owner = zeros(Int, n)
    upper_owner = zeros(Int, n)

    @inbounds for i in 1:m
        μ_i = isnothing(μ_vector) ? zero(T) : T(μ_vector[i])
        eligible = row_counts[i] == 1 && (convert_zero_barrier_rows || !iszero(μ_i))

        if !eligible
            push!(general_rows, i)
            continue
        end

        j = row_variables[i]
        a = row_coefficients[i]
        rhs = T(b[i])

        if iszero(a)
            push!(general_rows, i)
            continue
        end

        is_lower = a < zero(T)
        push!(bound_rows, ExtractedBoundRow{T}(i, j, a, rhs, is_lower))

        raw_bound = rhs / a
        effective_bound = if iszero(μ_i)
            raw_bound
        elseif is_lower
            raw_bound + T(slack_lower_bound) / (-a)
        else
            raw_bound - T(slack_lower_bound) / a
        end

        if is_lower
            if effective_bound > lower_bounds[j]
                lower_bounds[j] = effective_bound
                lower_owner[j] = i
            end
        else
            if effective_bound < upper_bounds[j]
                upper_bounds[j] = effective_bound
                upper_owner[j] = i
            end
        end
    end

    @inbounds for j in 1:n
        if lower_bounds[j] > upper_bounds[j]
            throw(
                ArgumentError(
                    "Extracted inconsistent bounds for variable $j: lower bound " *
                    "$(lower_bounds[j]) exceeds upper bound $(upper_bounds[j]).",
                ),
            )
        end
    end

    A_general = A[general_rows, :]
    b_general = b[general_rows]
    bound_map = InequalityBoundMap(
        bound_rows,
        general_rows,
        lower_bounds,
        upper_bounds,
        lower_owner,
        upper_owner,
    )
    bound_lp = BoundFormLP(
        lp.A_eq,
        A_general,
        lp.b_eq,
        b_general,
        lp.c,
        lower_bounds,
        upper_bounds,
        bound_map,
    )

    return bound_lp, bound_map
end

function _extract_variable_bounds_for_solver(solver, lp::LP; kwargs...)
    try
        return _extract_variable_bounds(lp; kwargs...)
    catch error
        if error isa ArgumentError &&
           occursin("Extracted inconsistent bounds", sprint(showerror, error))
            throw(
                ErrorException(
                    string(
                        typeof(solver),
                        " failed to solve the optimization problem: ",
                        "extracted inconsistent variable bounds.",
                    ),
                ),
            )
        end
        rethrow()
    end
end

function _reconstruct_original_inequality_info(
    lp::LP,
    bound_map::InequalityBoundMap,
    raw::BoundFormSolveResult;
    μ_vector=nothing,
)
    n_inequalities = length(lp.b_ineq)
    T = promote_type(
        Float64,
        eltype(lp.A_ineq),
        eltype(lp.b_ineq),
        eltype(raw.z),
        eltype(raw.general_slack),
        eltype(raw.general_dual_ineq),
        eltype(raw.lower_bound_dual),
        eltype(raw.upper_bound_dual),
    )

    slack = Vector{T}(undef, n_inequalities)
    dual_ineq = zeros(T, n_inequalities)

    @inbounds for k in eachindex(bound_map.general_rows)
        original_row = bound_map.general_rows[k]
        slack[original_row] = T(raw.general_slack[k])

        μ_i = isnothing(μ_vector) ? zero(T) : T(μ_vector[original_row])
        dual_ineq[original_row] =
            iszero(μ_i) ? T(raw.general_dual_ineq[k]) : μ_i / slack[original_row]
    end

    @inbounds for rowinfo in bound_map.bound_rows
        i = rowinfo.original_row
        j = rowinfo.variable
        a = T(rowinfo.coefficient)
        rhs = T(rowinfo.rhs)

        s_i = rhs - a * T(raw.z[j])
        slack[i] = s_i

        μ_i = isnothing(μ_vector) ? zero(T) : T(μ_vector[i])
        if !iszero(μ_i)
            dual_ineq[i] = μ_i / s_i
        elseif rowinfo.is_lower
            if bound_map.lower_owner[j] == i
                dual_ineq[i] = T(raw.lower_bound_dual[j]) / (-a)
            end
        elseif bound_map.upper_owner[j] == i
            dual_ineq[i] = T(raw.upper_bound_dual[j]) / a
        end
    end

    return slack, dual_ineq
end

function _reconstruct_original_lp_result(
    lp::LP,
    bound_map::InequalityBoundMap,
    raw::BoundFormSolveResult;
    μ_vector=nothing,
    include_slack::Bool=false,
)
    slack, dual_ineq =
        _reconstruct_original_inequality_info(lp, bound_map, raw; μ_vector=μ_vector)

    if include_slack
        return (;
            z=raw.z,
            slack=slack,
            dual_eq=raw.dual_eq,
            dual_ineq=dual_ineq,
            objective_value=raw.objective_value,
            status=raw.status,
            metadata=raw.metadata,
        )
    end

    return (;
        z=raw.z,
        dual_eq=raw.dual_eq,
        dual_ineq=dual_ineq,
        objective_value=raw.objective_value,
        status=raw.status,
        metadata=raw.metadata,
    )
end

# END FILE: src/ContextualDFL/ContextualDFL/src/linear_programming/bound_form_lp.jl

# BEGIN FILE: src/ContextualDFL/ContextualDFL/src/linear_programming/diff_lp.jl
import LinearAlgebra: Diagonal, I, Symmetric, bunchkaufman, factorize, rank
import SparseArrays: issparse, sparse, spzeros

function diff_solve(
    solver,
    lp::LP,
    μ;
    ρ=0,
    rho=ρ,
    pre_computed=nothing,
    dc=zeros(eltype(lp.c), length(lp.c)),
    db_eq=zeros(eltype(lp.b_eq), length(lp.b_eq)),
    db_ineq=zeros(eltype(lp.b_ineq), length(lp.b_ineq)),
    tight_tol=1e-7,
    kwargs...,
)
    cache = _diff_precompute(solver, lp, μ, rho, pre_computed, tight_tol; kwargs...)
    n = length(lp.c)
    m_eq = length(lp.b_eq)
    m_ineq = length(lp.b_ineq)

    length(dc) == n || throw(DimensionMismatch("dc must have length $n."))
    length(db_eq) == m_eq || throw(DimensionMismatch("db_eq must have length $m_eq."))
    length(db_ineq) == m_ineq || throw(DimensionMismatch("db_ineq must have length $m_ineq."))

    μ_vector = cache.μ
    ρ_vector = cache.ρ
    T = promote_type(eltype(cache.z), eltype(lp.c), eltype(μ_vector), eltype(ρ_vector))
    rhs_x = zeros(T, n)

    if _is_zero_barrier_parameter(μ_vector)
        if _is_zero_quadratic_parameter(ρ_vector)
            all(iszero, db_ineq) ||
            (rhs_x .+= transpose(lp.A_ineq[cache.loose, :]) * (cache.d .* db_ineq[cache.loose]))
        else
            all(iszero, dc) || (rhs_x .-= dc)
        end
        rhs = vcat(rhs_x, db_eq, db_ineq[cache.tight])
    else
        rhs_eq = zeros(T, m_eq)
        all(iszero, dc) || (rhs_x .-= dc)
        all(iszero, db_eq) || (rhs_eq .= db_eq)
        all(iszero, db_ineq) ||
            (rhs_x .+= transpose(lp.A_ineq) * ((μ_vector .* cache.d) .* db_ineq))
        rhs = vcat(rhs_x, rhs_eq)
    end

    solution = cache.K_factorization \ rhs
    return solution[1:n]
end

function _diff_precompute(solver, lp::LP, μ, pre_computed, tight_tol; kwargs...)
    return _diff_precompute(solver, lp, μ, 0, pre_computed, tight_tol; kwargs...)
end

function _diff_precompute(solver, lp::LP, μ, ρ, pre_computed, tight_tol; kwargs...)
    μ_vector = _barrier_parameter_vector(lp, μ)
    ρ_vector = _quadratic_parameter_vector(lp, ρ)

    if !isnothing(pre_computed) && hasproperty(pre_computed, :K_factorization)
        pre_computed.μ == μ_vector ||
            throw(ArgumentError("pre_computed was built with a different μ."))
        hasproperty(pre_computed, :ρ) && pre_computed.ρ == ρ_vector ||
            throw(ArgumentError("pre_computed was built with a different ρ."))
        return pre_computed
    end

    solve_result = if isnothing(pre_computed)
        if _is_zero_barrier_parameter(μ_vector) && _is_zero_quadratic_parameter(ρ_vector)
            solve(solver, lp; kwargs...)
        else
            solve(solver, lp; μ=μ_vector, ρ=ρ_vector, kwargs...)
        end
    else
        pre_computed
    end
    z = solve_result isa AbstractVector ? solve_result : solve_result.z
    length(z) == length(lp.c) ||
        throw(DimensionMismatch("The solution must have length $(length(lp.c))."))

    n = length(lp.c)
    m_eq = length(lp.b_eq)
    slack = lp.b_ineq - lp.A_ineq * z
    any(<(-tight_tol), slack) &&
        throw(DomainError(slack, "The solution violates inequality constraints."))

    if _is_zero_barrier_parameter(μ_vector)
        tight = findall(abs.(slack) .<= tight_tol)
        loose = findall(slack .> tight_tol)
        rank(Matrix(lp.A_eq)) == size(lp.A_eq, 1) ||
            throw(ArgumentError("Differentiation requires A_eq to have full row rank."))

        F = Matrix(lp.A_eq)
        selected_tight = Int[]
        current_rank = rank(F)
        for index in tight
            candidate = [F; lp.A_ineq[index:index, :]]
            candidate_rank = rank(candidate)
            if candidate_rank > current_rank
                push!(selected_tight, index)
                F = candidate
                current_rank = candidate_rank
            end
        end

        tight = selected_tight
        d = _is_zero_quadratic_parameter(ρ_vector) ?
            one(eltype(slack)) ./ (slack[loose] .^ 2) :
            zeros(promote_type(eltype(slack), eltype(ρ_vector)), length(loose))

        H = if _is_zero_quadratic_parameter(ρ_vector)
            A_loose = lp.A_ineq[loose, :]
            transpose(A_loose) * (Diagonal(d) * A_loose)
        else
            Diagonal(ρ_vector)
        end
        T = promote_type(eltype(H), eltype(F), eltype(μ_vector), eltype(ρ_vector))
        K = if issparse(H) || issparse(F)
            [
                sparse(H) sparse(transpose(F))
                sparse(F) spzeros(T, size(F, 1), size(F, 1))
            ]
        else
            [
                H transpose(F)
                F zeros(T, size(F, 1), size(F, 1))
            ]
        end

        K_factorization = issparse(K) ? factorize(K) : bunchkaufman(Symmetric(K))
        return (; z=z, d=d, K_factorization=K_factorization, μ=μ_vector, ρ=ρ_vector, tight=tight, loose=loose)
    end

    all(>(zero(eltype(slack))), slack) ||
        throw(DomainError(slack, "The log-barrier solution must have positive inequality slack."))
    rank(Matrix(lp.A_eq)) == size(lp.A_eq, 1) ||
        throw(ArgumentError("Log-barrier differentiation requires A_eq to have full row rank."))

    d = one(eltype(slack)) ./ (slack .^ 2)
    H = transpose(lp.A_ineq) * (Diagonal(μ_vector .* d) * lp.A_ineq)
    H = _is_zero_quadratic_parameter(ρ_vector) ? H : H + Diagonal(ρ_vector)
    T = promote_type(eltype(H), eltype(lp.A_eq), eltype(μ_vector), eltype(ρ_vector))
    K = if issparse(H) || issparse(lp.A_eq)
        [
            sparse(H) sparse(transpose(lp.A_eq))
            sparse(lp.A_eq) spzeros(T, m_eq, m_eq)
        ]
    else
        [
            H transpose(lp.A_eq)
            lp.A_eq zeros(T, m_eq, m_eq)
        ]
    end

    K_factorization = issparse(K) ? factorize(K) : bunchkaufman(Symmetric(K))
    return (; z=z, d=d, K_factorization=K_factorization, μ=μ_vector, ρ=ρ_vector, tight=Int[], loose=collect(1:length(lp.b_ineq)))
end

# END FILE: src/ContextualDFL/ContextualDFL/src/linear_programming/diff_lp.jl

# BEGIN FILE: src/ContextualDFL/ContextualDFL/src/scenario_decoders/ParametricDecoder.jl
const SCENARIO_COMPONENTS = (:W_eq, :W_ineq, :T_eq, :T_ineq, :h_eq, :h_ineq, :q)

struct ParametricDecoder{IC,W_EQ,W_INEQ,T_EQ,T_INEQ,H_EQ,H_INEQ,Q} <: ScenarioDecoder
    input_components::IC
    base_W_eq::W_EQ
    base_W_ineq::W_INEQ
    base_T_eq::T_EQ
    base_T_ineq::T_INEQ
    base_h_eq::H_EQ
    base_h_ineq::H_INEQ
    base_q::Q
end

function ParametricDecoder(
    input_components=SCENARIO_COMPONENTS;
    base_W_eq=nothing,
    base_W_ineq=nothing,
    base_T_eq=nothing,
    base_T_ineq=nothing,
    base_h_eq=nothing,
    base_h_ineq=nothing,
    base_q=nothing,
)
    issubset(input_components, SCENARIO_COMPONENTS) ||
        throw(ArgumentError("input_components must be a subset of $SCENARIO_COMPONENTS."))

    return ParametricDecoder(
        input_components,
        base_W_eq,
        base_W_ineq,
        base_T_eq,
        base_T_ineq,
        base_h_eq,
        base_h_ineq,
        base_q,
    )
end

function (decoder::ParametricDecoder)(scenario_parameters::ParametricScenario)
    W_eq = :W_eq in decoder.input_components ? scenario_parameters.W_eq_xi : decoder.base_W_eq
    W_ineq = :W_ineq in decoder.input_components ? scenario_parameters.W_ineq_xi : decoder.base_W_ineq
    T_eq = :T_eq in decoder.input_components ? scenario_parameters.T_eq_xi : decoder.base_T_eq
    T_ineq = :T_ineq in decoder.input_components ? scenario_parameters.T_ineq_xi : decoder.base_T_ineq
    h_eq = :h_eq in decoder.input_components ? scenario_parameters.h_eq_xi : decoder.base_h_eq
    h_ineq = :h_ineq in decoder.input_components ? scenario_parameters.h_ineq_xi : decoder.base_h_ineq
    q = :q in decoder.input_components ? scenario_parameters.q_xi : decoder.base_q

    any(isnothing, (W_eq, W_ineq, T_eq, T_ineq, h_eq, h_ineq, q)) &&
        throw(ArgumentError("All scenario components must be provided."))

    return W_eq, W_ineq, T_eq, T_ineq, h_eq, h_ineq, q
end

function _tuple_cotangent_component(output_tangent, index)
    output_tangent = ChainRulesCore.unthunk(output_tangent)

    if output_tangent isa ChainRulesCore.AbstractZero
        return ChainRulesCore.ZeroTangent()
    elseif output_tangent isa Tuple
        index > length(output_tangent) && return ChainRulesCore.ZeroTangent()
        return ChainRulesCore.unthunk(output_tangent[index])
    elseif output_tangent isa ChainRulesCore.Tangent
        index > length(output_tangent) && return ChainRulesCore.ZeroTangent()
        return ChainRulesCore.unthunk(output_tangent[index])
    end

    throw(
        ArgumentError(
            "Expected tuple-like cotangent for decode_scenario_collection output; got $(typeof(output_tangent)).",
        ),
    )
end

function _maybe_array_component(component, template; name)
    component = ChainRulesCore.unthunk(component)

    if _is_zero_cotangent(component)
        return ChainRulesCore.NoTangent()
    end

    component isa AbstractArray ||
        throw(ArgumentError("Expected array cotangent for $name; got $(typeof(component))."))

    size(component) == size(template) || throw(
        DimensionMismatch(
            "Cotangent for $name has size $(size(component)); expected $(size(template)).",
        ),
    )

    return component
end

function ChainRulesCore.rrule(
    ::typeof(decode_scenario_collection),
    decoder::ParametricDecoder,
    scenario_parameter_collection::AbstractVector{<:ParametricScenario},
)
    output = decode_scenario_collection(decoder, scenario_parameter_collection)
    W_eq_array,
    W_ineq_array,
    T_eq_array,
    T_ineq_array,
    h_eq_array,
    h_ineq_array,
    q_array = output

    function decode_scenario_collection_pullback(output_tangent)
        dW_eq_array = _maybe_array_component(
            _tuple_cotangent_component(output_tangent, 1),
            W_eq_array;
            name=:W_eq,
        )
        dW_ineq_array = _maybe_array_component(
            _tuple_cotangent_component(output_tangent, 2),
            W_ineq_array;
            name=:W_ineq,
        )
        dT_eq_array = _maybe_array_component(
            _tuple_cotangent_component(output_tangent, 3),
            T_eq_array;
            name=:T_eq,
        )
        dT_ineq_array = _maybe_array_component(
            _tuple_cotangent_component(output_tangent, 4),
            T_ineq_array;
            name=:T_ineq,
        )
        dh_eq_array = _maybe_array_component(
            _tuple_cotangent_component(output_tangent, 5),
            h_eq_array;
            name=:h_eq,
        )
        dh_ineq_array = _maybe_array_component(
            _tuple_cotangent_component(output_tangent, 6),
            h_ineq_array;
            name=:h_ineq,
        )
        dq_array = _maybe_array_component(
            _tuple_cotangent_component(output_tangent, 7),
            q_array;
            name=:q,
        )

        if all(
            tangent -> !(tangent isa AbstractArray),
            (dW_eq_array, dW_ineq_array, dT_eq_array, dT_ineq_array, dh_eq_array, dh_ineq_array, dq_array),
        )
            return (
                ChainRulesCore.NoTangent(),
                ChainRulesCore.NoTangent(),
                ChainRulesCore.NoTangent(),
            )
        end

        scenario_parameter_tangents = map(enumerate(scenario_parameter_collection)) do (k, scenario_parameters)
            return ChainRulesCore.Tangent{typeof(scenario_parameters)}(
                W_eq_xi=
                if :W_eq in decoder.input_components && dW_eq_array isa AbstractArray
                    ChainRulesCore.ProjectTo(scenario_parameters.W_eq_xi)(
                        view(dW_eq_array, :, :, k),
                    )
                else
                    ChainRulesCore.NoTangent()
                end,
                W_ineq_xi=
                if :W_ineq in decoder.input_components && dW_ineq_array isa AbstractArray
                    ChainRulesCore.ProjectTo(scenario_parameters.W_ineq_xi)(
                        view(dW_ineq_array, :, :, k),
                    )
                else
                    ChainRulesCore.NoTangent()
                end,
                T_eq_xi=
                if :T_eq in decoder.input_components && dT_eq_array isa AbstractArray
                    ChainRulesCore.ProjectTo(scenario_parameters.T_eq_xi)(
                        view(dT_eq_array, :, :, k),
                    )
                else
                    ChainRulesCore.NoTangent()
                end,
                T_ineq_xi=
                if :T_ineq in decoder.input_components && dT_ineq_array isa AbstractArray
                    ChainRulesCore.ProjectTo(scenario_parameters.T_ineq_xi)(
                        view(dT_ineq_array, :, :, k),
                    )
                else
                    ChainRulesCore.NoTangent()
                end,
                h_eq_xi=
                if :h_eq in decoder.input_components && dh_eq_array isa AbstractArray
                    ChainRulesCore.ProjectTo(scenario_parameters.h_eq_xi)(
                        view(dh_eq_array, :, k),
                    )
                else
                    ChainRulesCore.NoTangent()
                end,
                h_ineq_xi=
                if :h_ineq in decoder.input_components && dh_ineq_array isa AbstractArray
                    ChainRulesCore.ProjectTo(scenario_parameters.h_ineq_xi)(
                        view(dh_ineq_array, :, k),
                    )
                else
                    ChainRulesCore.NoTangent()
                end,
                q_xi=
                if :q in decoder.input_components && dq_array isa AbstractArray
                    ChainRulesCore.ProjectTo(scenario_parameters.q_xi)(
                        view(dq_array, :, k),
                    )
                else
                    ChainRulesCore.NoTangent()
                end,
            )
        end

        return (
            ChainRulesCore.NoTangent(),
            ChainRulesCore.NoTangent(),
            scenario_parameter_tangents,
        )
    end

    return output, decode_scenario_collection_pullback
end

# END FILE: src/ContextualDFL/ContextualDFL/src/scenario_decoders/ParametricDecoder.jl

# BEGIN FILE: src/ContextualDFL/ContextualDFL/src/scenario_decoders/ScenarioDecoder.jl
abstract type ScenarioDecoder end

(decoder::ScenarioDecoder)(scenario_parameter) =
    error("Scenario decoding is not defined for $(typeof(decoder)).")

function decode_scenario_collection(
    decoder::ScenarioDecoder,
    scenario_parameter_collection::AbstractVector,
)
    return _decode_scenario_collection(decoder, scenario_parameter_collection)
end

function _decode_scenario_collection(
    decoder::ScenarioDecoder,
    scenario_parameter_collection::AbstractVector,
)
    K = length(scenario_parameter_collection)
    K > 0 || throw(ArgumentError("scenario_parameter_collection must not be empty."))

    decoded_scenarios = map(decoder, scenario_parameter_collection)
    W_eq, W_ineq, T_eq, T_ineq, h_eq, h_ineq, q = first(decoded_scenarios)

    W_eq isa AbstractMatrix || throw(ArgumentError("W_eq must be a matrix."))
    W_ineq isa AbstractMatrix || throw(ArgumentError("W_ineq must be a matrix."))
    T_eq isa AbstractMatrix || throw(ArgumentError("T_eq must be a matrix."))
    T_ineq isa AbstractMatrix || throw(ArgumentError("T_ineq must be a matrix."))
    h_eq isa AbstractVector || throw(ArgumentError("h_eq must be a vector."))
    h_ineq isa AbstractVector || throw(ArgumentError("h_ineq must be a vector."))
    q isa AbstractVector || throw(ArgumentError("q must be a vector."))

    W_eq_array = _stack_scenario_matrices(map(scenario -> scenario[1], decoded_scenarios))
    W_ineq_array = _stack_scenario_matrices(map(scenario -> scenario[2], decoded_scenarios))
    T_eq_array = _stack_scenario_matrices(map(scenario -> scenario[3], decoded_scenarios))
    T_ineq_array = _stack_scenario_matrices(map(scenario -> scenario[4], decoded_scenarios))
    h_eq_array = _stack_scenario_vectors(map(scenario -> scenario[5], decoded_scenarios))
    h_ineq_array = _stack_scenario_vectors(map(scenario -> scenario[6], decoded_scenarios))
    q_array = _stack_scenario_vectors(map(scenario -> scenario[7], decoded_scenarios))

    return W_eq_array,
        W_ineq_array,
        T_eq_array,
        T_ineq_array,
        h_eq_array,
        h_ineq_array,
        q_array
end

function _stack_scenario_matrices(matrices)
    matrix = first(matrices)
    length(matrices) == 1 &&
        return reshape(matrix, size(matrix, 1), size(matrix, 2), 1)
    return cat(matrices...; dims=3)
end

function _stack_scenario_vectors(vectors)
    vector = first(vectors)
    length(vectors) == 1 &&
        return reshape(vector, length(vector), 1)
    return reduce(hcat, vectors)
end

# END FILE: src/ContextualDFL/ContextualDFL/src/scenario_decoders/ScenarioDecoder.jl

# BEGIN FILE: src/ContextualDFL/ContextualDFL/src/scenario_decoders/VectorDecoder.jl
"""
    VectorDecoder

Abstract decoder for vector-valued model outputs, especially neural net outputs.
Concrete subtypes decode one scenario from an `AbstractVector`; the collection
method below splits a longer vector into `nr_scenarios` equally sized scenario
vectors before assembling the decoded scenario collection.
"""
abstract type VectorDecoder <: ScenarioDecoder end

(decoder::VectorDecoder)(vector::AbstractVector) =
    error("Vector decoding is not defined for $(typeof(decoder)).")

function decode_scenario_collection(
    decoder::VectorDecoder,
    vector_or_collection::AbstractVector;
    nr_scenarios=nothing,
)
    if isnothing(nr_scenarios)
        return _decode_scenario_collection(decoder, vector_or_collection)
    end

    nr_scenarios isa Integer && nr_scenarios > 0 ||
        throw(ArgumentError("nr_scenarios must be a positive integer."))

    L = length(vector_or_collection)
    L % nr_scenarios == 0 ||
        throw(ArgumentError("vector length $L is not divisible by nr_scenarios=$nr_scenarios."))

    scenario_width = L ÷ nr_scenarios
    scenario_matrix = reshape(vector_or_collection, scenario_width, nr_scenarios)
    scenario_parameter_collection = [
        view(scenario_matrix, :, scenario_index)
        for scenario_index in 1:nr_scenarios
    ]

    return decode_scenario_collection(decoder, scenario_parameter_collection)
end

# END FILE: src/ContextualDFL/ContextualDFL/src/scenario_decoders/VectorDecoder.jl

# BEGIN FILE: src/ContextualDFL/ContextualDFL/src/stochastic_programming/StochasticProgram.jl
struct StochasticProgram{TLP<:LP}
    first_stage_lp::TLP
end

StochasticProgram(args...; kwargs...) = StochasticProgram(LP(args...; kwargs...))

function Base.getproperty(sp::StochasticProgram, name::Symbol)
    name === :A_eq && return getfield(sp, :first_stage_lp).A_eq
    name === :A_ineq && return getfield(sp, :first_stage_lp).A_ineq
    name === :b_eq && return getfield(sp, :first_stage_lp).b_eq
    name === :b_ineq && return getfield(sp, :first_stage_lp).b_ineq
    name === :c && return getfield(sp, :first_stage_lp).c
    return getfield(sp, name)
end

function Base.propertynames(sp::StochasticProgram, private::Bool=false)
    names = (:first_stage_lp, :A_eq, :A_ineq, :b_eq, :b_ineq, :c)
    return private ? (names..., fieldnames(typeof(sp))...) : names
end

# END FILE: src/ContextualDFL/ContextualDFL/src/stochastic_programming/StochasticProgram.jl

# BEGIN FILE: src/ContextualDFL/ContextualDFL/src/stochastic_programming/construct_lp.jl
function construct_lp(
    sp::StochasticProgram,
    W_eq_array,
    W_ineq_array,
    T_eq_array,
    T_ineq_array,
    h_eq_array,
    h_ineq_array,
    q_array;
    probabilities=nothing,
)
    K = _sp_n_scenarios(W_eq_array, W_ineq_array, T_eq_array, T_ineq_array, h_eq_array, h_ineq_array, q_array)
    first_stage_lp = sp.first_stage_lp
    T = _sp_eltype(
        first_stage_lp.A_eq,
        first_stage_lp.A_ineq,
        first_stage_lp.b_eq,
        first_stage_lp.b_ineq,
        first_stage_lp.c,
        W_eq_array,
        W_ineq_array,
        T_eq_array,
        T_ineq_array,
        h_eq_array,
        h_ineq_array,
        q_array,
    )

    p_vector = if isnothing(probabilities)
        fill(one(T) / K, K)
    else
        length(probabilities) == K ||
            throw(DimensionMismatch("probabilities must have one entry per scenario."))
        probabilities
    end
    T = promote_type(T, eltype(p_vector))

    # The extensive-form variable is v = [z; y_1; ...; y_K].
    nz = size(first_stage_lp.A_eq, 2)
    ny = size(q_array, 1)
    nvars = nz + K * ny

    m1_eq = length(first_stage_lp.b_eq)
    m1_ineq = length(first_stage_lp.b_ineq)
    m2_eq = size(W_eq_array, 1)
    m2_ineq = size(W_ineq_array, 1)

    A_eq = spzeros(T, m1_eq + K * m2_eq, nvars)
    A_ineq = spzeros(T, m1_ineq + K * m2_ineq, nvars)
    b_eq = zeros(T, m1_eq + K * m2_eq)
    b_ineq = zeros(T, m1_ineq + K * m2_ineq)
    c = zeros(T, nvars)

    # First-stage rows: A_eq z = b_eq and A_ineq z <= b_ineq.
    z_cols = 1:nz
    A_eq[1:m1_eq, z_cols] = first_stage_lp.A_eq
    A_ineq[1:m1_ineq, z_cols] = first_stage_lp.A_ineq
    b_eq[1:m1_eq] = first_stage_lp.b_eq
    b_ineq[1:m1_ineq] = first_stage_lp.b_ineq
    c[z_cols] = first_stage_lp.c

    for k in 1:K
        y_cols = (nz + (k - 1) * ny + 1):(nz + k * ny)

        # Scenario-k equality rows: T_eq_array[k] z + W_eq_array[k] y_k = h_eq_array[k].
        eq_rows = (m1_eq + (k - 1) * m2_eq + 1):(m1_eq + k * m2_eq)
        A_eq[eq_rows, z_cols] = view(T_eq_array, :, :, k)
        A_eq[eq_rows, y_cols] = view(W_eq_array, :, :, k)
        b_eq[eq_rows] = view(h_eq_array, :, k)

        # Scenario-k inequality rows: T_ineq_array[k] z + W_ineq_array[k] y_k <= h_ineq_array[k].
        ineq_rows = (m1_ineq + (k - 1) * m2_ineq + 1):(m1_ineq + k * m2_ineq)
        A_ineq[ineq_rows, z_cols] = view(T_ineq_array, :, :, k)
        A_ineq[ineq_rows, y_cols] = view(W_ineq_array, :, :, k)
        b_ineq[ineq_rows] = view(h_ineq_array, :, k)

        # Expected second-stage objective: sum_k p_k q_k' y_k.
        c[y_cols] = p_vector[k] .* view(q_array, :, k)
    end

    return LP(A_eq, A_ineq, b_eq, b_ineq, c)
end

function _sp_n_scenarios(
    W_eq_array,
    W_ineq_array,
    T_eq_array,
    T_ineq_array,
    h_eq_array,
    h_ineq_array,
    q_array,
)
    # Scenario data always uses the last array index for scenarios.
    # Matrices are 3D arrays; vectors are stored as columns of 2D matrices.
    matrix_arrays = (W_eq_array, W_ineq_array, T_eq_array, T_ineq_array)
    vector_arrays = (h_eq_array, h_ineq_array, q_array)
    all(component -> ndims(component) == 3, matrix_arrays) ||
        throw(ArgumentError("W_eq_array, W_ineq_array, T_eq_array, and T_ineq_array must be 3D arrays."))
    all(component -> ndims(component) == 2, vector_arrays) ||
        throw(ArgumentError("h_eq_array, h_ineq_array, and q_array must be matrices with one scenario per column."))

    K = size(W_eq_array, 3)
    all(component -> size(component, 3) == K, matrix_arrays) &&
        all(component -> size(component, 2) == K, vector_arrays) ||
        throw(DimensionMismatch("Scenario components disagree on the number of scenarios."))
    return K
end

function _sp_eltype(values...)
    types = [eltype(value) for value in values]
    return promote_type(types...)
end

# END FILE: src/ContextualDFL/ContextualDFL/src/stochastic_programming/construct_lp.jl

# BEGIN FILE: src/ContextualDFL/ContextualDFL/src/stochastic_programming/cost_function.jl
function cost_function(
    program::StochasticProgram,
    solver::Solver,
    z,
    W_eq_array,
    W_ineq_array,
    T_eq_array,
    T_ineq_array,
    h_eq_array,
    h_ineq_array,
    q_array;
    μ=0,
    ρ=0,
    rho=ρ,
    probabilities=nothing,
    return_dual=false,
    kwargs...,
)
    return G(
        program,
        solver,
        z,
        W_eq_array,
        W_ineq_array,
        T_eq_array,
        T_ineq_array,
        h_eq_array,
        h_ineq_array,
        q_array,
        ;
        μ=μ,
        ρ=rho,
        probabilities=probabilities,
        return_dual=return_dual,
        kwargs...,
    )
end

function G(
    program::StochasticProgram,
    solver::Solver,
    z,
    W_eq_array,
    W_ineq_array,
    T_eq_array,
    T_ineq_array,
    h_eq_array,
    h_ineq_array,
    q_array;
    μ=0,
    ρ=0,
    rho=ρ,
    probabilities=nothing,
    return_dual=false,
    kwargs...,
)
    K = _sp_n_scenarios(W_eq_array, W_ineq_array, T_eq_array, T_ineq_array, h_eq_array, h_ineq_array, q_array)
    first_stage_lp = program.first_stage_lp
    T = _sp_eltype(
        first_stage_lp.c,
        W_eq_array,
        W_ineq_array,
        T_eq_array,
        T_ineq_array,
        h_eq_array,
        h_ineq_array,
        q_array,
    )

    p_vector = if isnothing(probabilities)
        fill(one(T) / K, K)
    else
        length(probabilities) == K ||
            throw(DimensionMismatch("probabilities must have one entry per scenario."))
        probabilities
    end
    first_stage_μ = _first_stage_barrier_parameter(first_stage_lp, W_ineq_array, μ)
    first_stage_ρ = _first_stage_quadratic_parameter(first_stage_lp, q_array, rho)
    T = promote_type(T, eltype(p_vector), eltype(first_stage_μ), eltype(first_stage_ρ))

    second_stage_value = zero(T)
    λ_h_eq_array = zeros(T, size(h_eq_array))
    λ_h_ineq_array = zeros(T, size(h_ineq_array))

    for k in 1:K
        scenario_μ = _scenario_barrier_parameter(
            size(W_ineq_array, 1),
            K,
            μ,
            k,
            length(first_stage_lp.b_ineq),
            p_vector[k],
        )
        scenario_ρ = _scenario_quadratic_parameter(
            size(q_array, 1),
            K,
            rho,
            k,
            length(first_stage_lp.c),
            p_vector[k],
        )
        scenario_value_or_dual = try
            G_hat(
                solver,
                z,
                view(W_eq_array, :, :, k),
                view(W_ineq_array, :, :, k),
                view(T_eq_array, :, :, k),
                view(T_ineq_array, :, :, k),
                view(h_eq_array, :, k),
                view(h_ineq_array, :, k),
                view(q_array, :, k),
                ;
                μ=scenario_μ,
                ρ=scenario_ρ,
                return_dual=return_dual,
                kwargs...,
            )
        catch error
            _throw_stochastic_program_failure(
                error,
                :second_stage_cost,
                solver,
                program,
                W_eq_array,
                W_ineq_array,
                T_eq_array,
                T_ineq_array,
                h_eq_array,
                h_ineq_array,
                q_array;
                μ=μ,
                ρ=rho,
                probabilities=probabilities,
                kwargs=(; kwargs...),
                z=z,
                scenario_index=k,
                scenario_μ=scenario_μ,
                scenario_ρ=scenario_ρ,
            )
        end

        if return_dual
            y, λ_h_eq, λ_h_ineq = scenario_value_or_dual
            scenario_value = sum(view(q_array, :, k) .* y)
            scenario_ρ_vector = _quadratic_parameter_vector(length(y), scenario_ρ)
            scenario_value += 0.5 * sum(scenario_ρ_vector .* (y .^ 2))
            scenario_μ_vector = _barrier_parameter_vector(size(W_ineq_array, 1), scenario_μ)
            positive_barrier_indices = findall(!iszero, scenario_μ_vector)
            if !isempty(positive_barrier_indices)
                slack =
                    view(h_ineq_array, :, k) - view(T_ineq_array, :, :, k) * z -
                    view(W_ineq_array, :, :, k) * y
                scenario_value -= sum(
                    scenario_μ_vector[i] * log(slack[i])
                    for i in positive_barrier_indices
                )
            end

            second_stage_value += p_vector[k] * scenario_value
            λ_h_eq_array[:, k] = λ_h_eq
            λ_h_ineq_array[:, k] = λ_h_ineq
        else
            second_stage_value += p_vector[k] * scenario_value_or_dual
        end
    end

    value =
        sum(first_stage_lp.c .* z) +
        _first_stage_quadratic_value(z, first_stage_ρ) -
        _first_stage_barrier_value(first_stage_lp, z, first_stage_μ) +
        second_stage_value

    return_dual && return value, λ_h_eq_array, λ_h_ineq_array
    return value
end

function G_hat(
    solver::Solver,
    z,
    W_eq,
    W_ineq,
    T_eq,
    T_ineq,
    h_eq,
    h_ineq,
    q;
    μ=0,
    ρ=0,
    rho=ρ,
    return_dual=false,
    kwargs...,
)
    # Fix z in the second-stage recourse problem:
    # W_eq y = h_eq - T_eq z and W_ineq y <= h_ineq - T_ineq z.
    second_stage_lp = LP(
        W_eq,
        W_ineq,
        h_eq - T_eq * z,
        h_ineq - T_ineq * z,
        q,
    )

    result = solve(solver, second_stage_lp; μ=μ, ρ=rho, kwargs...)

    if return_dual
        # The dual variables are returned for differentiability purposes.
        λ_h_eq = result.dual_eq
        λ_h_ineq = result.dual_ineq
        return result.z, λ_h_eq, λ_h_ineq
    end

    return result.objective_value
end

function _first_stage_barrier_parameter(first_stage_lp, W_ineq_array, μ)
    n_first_stage_inequalities = length(first_stage_lp.b_ineq)
    μ isa Number && return _barrier_parameter_vector(n_first_stage_inequalities, μ)
    n_first_stage_inequalities == 0 && return view(μ, 1:0)

    n_extensive_inequalities =
        n_first_stage_inequalities + size(W_ineq_array, 1) * size(W_ineq_array, 3)
    length(μ) == n_extensive_inequalities ||
        throw(DimensionMismatch(
            "μ must be a scalar or have one entry per extensive-form inequality when first-stage inequalities are present.",
        ))

    return view(μ, 1:n_first_stage_inequalities)
end

function _first_stage_quadratic_parameter(first_stage_lp, q_array, ρ)
    n_first_stage_variables = length(first_stage_lp.c)
    ρ isa Number && return _quadratic_parameter_vector(n_first_stage_variables, ρ)

    n_extensive_variables = n_first_stage_variables + size(q_array, 1) * size(q_array, 2)
    ρ_vector = _quadratic_parameter_vector(n_extensive_variables, ρ)
    return view(ρ_vector, 1:n_first_stage_variables)
end

function _first_stage_quadratic_value(z, ρ_vector)
    positive_quadratic_indices = findall(!iszero, ρ_vector)
    isempty(positive_quadratic_indices) && return zero(_sp_eltype(z, ρ_vector))

    return 0.5 * sum(ρ_vector[i] * z[i]^2 for i in positive_quadratic_indices)
end

function _add_first_stage_quadratic_gradient!(dz, z, ρ_vector)
    _is_zero_quadratic_parameter(ρ_vector) && return dz
    dz .+= ρ_vector .* z
    return dz
end

function _first_stage_barrier_value(first_stage_lp, z, μ_vector)
    positive_barrier_indices = findall(!iszero, μ_vector)
    isempty(positive_barrier_indices) && return zero(_sp_eltype(first_stage_lp.b_ineq, z, μ_vector))

    slack = first_stage_lp.b_ineq - first_stage_lp.A_ineq * z
    all(i -> slack[i] > zero(slack[i]), positive_barrier_indices) ||
        throw(DomainError(slack, "The first-stage log-barrier cost requires positive inequality slack."))

    return sum(μ_vector[i] * log(slack[i]) for i in positive_barrier_indices)
end

function _add_first_stage_barrier_gradient!(dz, first_stage_lp, z, μ_vector)
    positive_barrier_indices = findall(!iszero, μ_vector)
    isempty(positive_barrier_indices) && return dz

    slack = first_stage_lp.b_ineq - first_stage_lp.A_ineq * z
    all(i -> slack[i] > zero(slack[i]), positive_barrier_indices) ||
        throw(DomainError(slack, "The first-stage log-barrier cost requires positive inequality slack."))

    weights = zeros(promote_type(eltype(slack), eltype(μ_vector)), length(μ_vector))
    for i in positive_barrier_indices
        weights[i] = μ_vector[i] / slack[i]
    end

    dz .+= transpose(first_stage_lp.A_ineq) * weights
    return dz
end

function _scenario_barrier_parameter(
    n_inequalities,
    n_scenarios,
    μ,
    scenario_index,
    n_first_stage_inequalities=0,
    probability=nothing,
)
    μ isa Number && return μ

    if n_first_stage_inequalities > 0
        n_extensive_inequalities =
            n_first_stage_inequalities + n_inequalities * n_scenarios
        if length(μ) == n_extensive_inequalities
            rows = (
                n_first_stage_inequalities + (scenario_index - 1) * n_inequalities + 1
            ):(n_first_stage_inequalities + scenario_index * n_inequalities)
            scenario_μ = view(μ, rows)
            isnothing(probability) && return scenario_μ
            iszero(probability) &&
                throw(ArgumentError("probabilities must be nonzero when μ is an extensive-form vector."))
            return scenario_μ ./ probability
        end
    end

    length(μ) == n_inequalities && return μ
    length(μ) == n_inequalities * n_scenarios ||
        throw(DimensionMismatch("μ must have one entry per scenario inequality or per stacked scenario inequality."))

    rows = ((scenario_index - 1) * n_inequalities + 1):(scenario_index * n_inequalities)
    return view(μ, rows)
end

function _scenario_quadratic_parameter(
    n_variables,
    n_scenarios,
    ρ,
    scenario_index,
    n_first_stage_variables=0,
    probability=nothing,
)
    ρ isa Number && return ρ

    if n_first_stage_variables > 0
        n_extensive_variables = n_first_stage_variables + n_variables * n_scenarios
        if length(ρ) == n_extensive_variables
            cols = (
                n_first_stage_variables + (scenario_index - 1) * n_variables + 1
            ):(n_first_stage_variables + scenario_index * n_variables)
            scenario_ρ = view(ρ, cols)
            isnothing(probability) && return scenario_ρ
            iszero(probability) &&
                throw(ArgumentError("probabilities must be nonzero when ρ is an extensive-form vector."))
            return scenario_ρ ./ probability
        end
    end

    if length(ρ) == n_variables * n_scenarios
        cols = ((scenario_index - 1) * n_variables + 1):(scenario_index * n_variables)
        scenario_ρ = view(ρ, cols)
        isnothing(probability) && return scenario_ρ
        iszero(probability) &&
            throw(ArgumentError("probabilities must be nonzero when ρ is a stacked extensive-form vector."))
        return scenario_ρ ./ probability
    end

    length(ρ) == n_variables ||
        throw(DimensionMismatch("ρ must have one entry per scenario variable or per extensive-form variable."))

    return ρ
end

# END FILE: src/ContextualDFL/ContextualDFL/src/stochastic_programming/cost_function.jl

# BEGIN FILE: src/ContextualDFL/ContextualDFL/src/stochastic_programming/cost_function_rrule.jl
# Differentiates cost_function with respect to z only. Scenario arrays
# W_*, T_*, h_*, and q_array are treated as constants by this rrule.
function ChainRulesCore.rrule(
    ::typeof(cost_function),
    program::StochasticProgram,
    solver::Solver,
    z,
    W_eq_array,
    W_ineq_array,
    T_eq_array,
    T_ineq_array,
    h_eq_array,
    h_ineq_array,
    q_array;
    μ=0,
    ρ=0,
    rho=ρ,
    probabilities=nothing,
    return_dual=false,
    kwargs...,
)
    return_dual &&
        throw(ArgumentError("The cost_function rrule is defined for scalar cost output."))

    value, dz = _cost_and_z_gradient(
        program,
        solver,
        z,
        W_eq_array,
        W_ineq_array,
        T_eq_array,
        T_ineq_array,
        h_eq_array,
        h_ineq_array,
        q_array,
        ;
        μ=μ,
        ρ=rho,
        probabilities=probabilities,
        kwargs...,
    )
    T = eltype(dz)

    function cost_function_pullback(value_tangent)
        value_tangent = ChainRulesCore.unthunk(value_tangent)
        tangent = if _is_zero_cotangent(value_tangent)
            zero(T)
        elseif value_tangent isa Number
            value_tangent
        else
            throw(ArgumentError(
                "Expected scalar cotangent for scalar cost_function output; got $(typeof(value_tangent)).",
            ))
        end

        return (
            ChainRulesCore.NoTangent(),
            ChainRulesCore.NoTangent(),
            ChainRulesCore.NoTangent(),
            tangent .* dz,
            ChainRulesCore.NoTangent(),
            ChainRulesCore.NoTangent(),
            ChainRulesCore.NoTangent(),
            ChainRulesCore.NoTangent(),
            ChainRulesCore.NoTangent(),
            ChainRulesCore.NoTangent(),
            ChainRulesCore.NoTangent(),
        )
    end

    return value, cost_function_pullback
end

function _cost_and_z_gradient(
    program::StochasticProgram,
    solver::Solver,
    z,
    W_eq_array,
    W_ineq_array,
    T_eq_array,
    T_ineq_array,
    h_eq_array,
    h_ineq_array,
    q_array;
    μ=0,
    ρ=0,
    rho=ρ,
    probabilities=nothing,
    kwargs...,
)
    K = _sp_n_scenarios(
        W_eq_array,
        W_ineq_array,
        T_eq_array,
        T_ineq_array,
        h_eq_array,
        h_ineq_array,
        q_array,
    )
    first_stage_lp = program.first_stage_lp
    T = _sp_eltype(
        first_stage_lp.c,
        z,
        W_eq_array,
        W_ineq_array,
        T_eq_array,
        T_ineq_array,
        h_eq_array,
        h_ineq_array,
        q_array,
    )

    p_vector = if isnothing(probabilities)
        fill(one(T) / K, K)
    else
        length(probabilities) == K ||
            throw(DimensionMismatch("probabilities must have one entry per scenario."))
        probabilities
    end
    first_stage_μ = _first_stage_barrier_parameter(first_stage_lp, W_ineq_array, μ)
    first_stage_ρ = _first_stage_quadratic_parameter(first_stage_lp, q_array, rho)
    T = promote_type(T, eltype(p_vector), eltype(first_stage_μ), eltype(first_stage_ρ))

    value =
        sum(first_stage_lp.c .* z) +
        _first_stage_quadratic_value(z, first_stage_ρ) -
        _first_stage_barrier_value(first_stage_lp, z, first_stage_μ)
    dz = T.(first_stage_lp.c)
    _add_first_stage_quadratic_gradient!(dz, z, first_stage_ρ)
    _add_first_stage_barrier_gradient!(dz, first_stage_lp, z, first_stage_μ)

    for k in 1:K
        scenario_μ = _scenario_barrier_parameter(
            size(W_ineq_array, 1),
            K,
            μ,
            k,
            length(first_stage_lp.b_ineq),
            p_vector[k],
        )
        scenario_ρ = _scenario_quadratic_parameter(
            size(q_array, 1),
            K,
            rho,
            k,
            length(first_stage_lp.c),
            p_vector[k],
        )
        result = try
            second_stage_lp = LP(
                view(W_eq_array, :, :, k),
                view(W_ineq_array, :, :, k),
                view(h_eq_array, :, k) - view(T_eq_array, :, :, k) * z,
                view(h_ineq_array, :, k) - view(T_ineq_array, :, :, k) * z,
                view(q_array, :, k),
            )

            solve(solver, second_stage_lp; μ=scenario_μ, ρ=scenario_ρ, kwargs...)
        catch error
            _throw_stochastic_program_failure(
                error,
                :second_stage_cost,
                solver,
                program,
                W_eq_array,
                W_ineq_array,
                T_eq_array,
                T_ineq_array,
                h_eq_array,
                h_ineq_array,
                q_array;
                μ=μ,
                ρ=rho,
                probabilities=probabilities,
                kwargs=(; kwargs...),
                z=z,
                scenario_index=k,
                scenario_μ=scenario_μ,
                scenario_ρ=scenario_ρ,
            )
        end

        value += p_vector[k] * result.objective_value
        dz .+= p_vector[k] .* (
            -transpose(view(T_eq_array, :, :, k)) * result.dual_eq +
            transpose(view(T_ineq_array, :, :, k)) * result.dual_ineq
        )
    end

    return value, dz
end

# END FILE: src/ContextualDFL/ContextualDFL/src/stochastic_programming/cost_function_rrule.jl

# BEGIN FILE: src/ContextualDFL/ContextualDFL/src/stochastic_programming/crash_recorder.jl
import Dates
import Serialization

const _STOCHASTIC_CRASH_DEFAULT_ROOT = "/tmp/contextual-dfl"
const _STOCHASTIC_CRASH_ROOT = Ref{String}(_STOCHASTIC_CRASH_DEFAULT_ROOT)

struct StochasticProgramFailure <: Exception
    location::Symbol
    crash_file::String
    scenario_index::Union{Nothing,Int}
end

function Base.showerror(io::IO, error::StochasticProgramFailure)
    if error.location === :single_scenario_solve
        print(io, "single-scenario problem failed.")
    elseif error.location === :stochastic_program_solve
        print(io, "stochastic program solve failed.")
    elseif error.location === :second_stage_cost
        print(io, "second-stage problem failed in scenario $(error.scenario_index).")
    else
        print(io, "stochastic program failure.")
    end

    print(io, " Crash data serialized at ", error.crash_file)
end

function _set_stochastic_crash_root!(root::AbstractString)
    previous = _STOCHASTIC_CRASH_ROOT[]
    _STOCHASTIC_CRASH_ROOT[] = String(root)
    return previous
end

function _reset_stochastic_crash_root!()
    return _set_stochastic_crash_root!(_STOCHASTIC_CRASH_DEFAULT_ROOT)
end

_stochastic_crash_root() = _STOCHASTIC_CRASH_ROOT[]

_crash_copy(value) = value
_crash_copy(value::AbstractArray) = copy(value)

function _stochastic_failure_location(W_eq_array)
    return size(W_eq_array, 3) == 1 ? :single_scenario_solve : :stochastic_program_solve
end

function _stochastic_crash_file()
    root = _stochastic_crash_root()
    mkpath(root)

    timestamp = Dates.format(Dates.now(), "yyyymmddTHHMMSSsss")
    crash_dir = mktempdir(root; prefix="crashed_$(timestamp)_")
    return joinpath(crash_dir, "stochastic_program_failure.jls")
end

function _first_stage_crash_payload(sp::StochasticProgram)
    first_stage_lp = sp.first_stage_lp
    return (;
        A_eq=_crash_copy(first_stage_lp.A_eq),
        A_ineq=_crash_copy(first_stage_lp.A_ineq),
        b_eq=_crash_copy(first_stage_lp.b_eq),
        b_ineq=_crash_copy(first_stage_lp.b_ineq),
        c=_crash_copy(first_stage_lp.c),
    )
end

function _scenario_crash_payload(
    W_eq_array,
    W_ineq_array,
    T_eq_array,
    T_ineq_array,
    h_eq_array,
    h_ineq_array,
    q_array,
)
    return (;
        W_eq_array=_crash_copy(W_eq_array),
        W_ineq_array=_crash_copy(W_ineq_array),
        T_eq_array=_crash_copy(T_eq_array),
        T_ineq_array=_crash_copy(T_ineq_array),
        h_eq_array=_crash_copy(h_eq_array),
        h_ineq_array=_crash_copy(h_ineq_array),
        q_array=_crash_copy(q_array),
    )
end

function _record_stochastic_program_failure(
    error,
    location::Symbol,
    solver::Solver,
    sp::StochasticProgram,
    W_eq_array,
    W_ineq_array,
    T_eq_array,
    T_ineq_array,
    h_eq_array,
    h_ineq_array,
    q_array;
    μ,
    ρ=0,
    effective_μ=nothing,
    effective_ρ=nothing,
    probabilities=nothing,
    kwargs=(;),
    z=nothing,
    scenario_index=nothing,
    scenario_μ=nothing,
    scenario_ρ=nothing,
)
    crash_file = _stochastic_crash_file()
    payload = (;
        location=location,
        timestamp=Dates.now(),
        first_stage=_first_stage_crash_payload(sp),
        scenario_data=_scenario_crash_payload(
            W_eq_array,
            W_ineq_array,
            T_eq_array,
            T_ineq_array,
            h_eq_array,
            h_ineq_array,
            q_array,
        ),
        μ=_crash_copy(μ),
        ρ=_crash_copy(ρ),
        effective_μ=_crash_copy(effective_μ),
        effective_ρ=_crash_copy(effective_ρ),
        scenario_μ=_crash_copy(scenario_μ),
        scenario_ρ=_crash_copy(scenario_ρ),
        probabilities=_crash_copy(probabilities),
        solver_type=string(typeof(solver)),
        kwargs=kwargs,
        z=_crash_copy(z),
        scenario_index=scenario_index,
        original_error_type=string(typeof(error)),
        original_error_text=sprint(showerror, error),
    )

    Serialization.serialize(crash_file, payload)
    return crash_file
end

function _throw_stochastic_program_failure(
    error,
    location::Symbol,
    solver::Solver,
    sp::StochasticProgram,
    W_eq_array,
    W_ineq_array,
    T_eq_array,
    T_ineq_array,
    h_eq_array,
    h_ineq_array,
    q_array;
    μ,
    ρ=0,
    effective_μ=nothing,
    effective_ρ=nothing,
    probabilities=nothing,
    kwargs=(;),
    z=nothing,
    scenario_index=nothing,
    scenario_μ=nothing,
    scenario_ρ=nothing,
)
    crash_file = _record_stochastic_program_failure(
        error,
        location,
        solver,
        sp,
        W_eq_array,
        W_ineq_array,
        T_eq_array,
        T_ineq_array,
        h_eq_array,
        h_ineq_array,
        q_array;
        μ=μ,
        ρ=ρ,
        effective_μ=effective_μ,
        effective_ρ=effective_ρ,
        probabilities=probabilities,
        kwargs=kwargs,
        z=z,
        scenario_index=scenario_index,
        scenario_μ=scenario_μ,
        scenario_ρ=scenario_ρ,
    )
    throw(StochasticProgramFailure(location, crash_file, scenario_index))
end

# END FILE: src/ContextualDFL/ContextualDFL/src/stochastic_programming/crash_recorder.jl

# BEGIN FILE: src/ContextualDFL/ContextualDFL/src/stochastic_programming/solve.jl
function solve(
    solver::Solver,
    sp::StochasticProgram,
    W_eq_array,
    W_ineq_array,
    T_eq_array,
    T_ineq_array,
    h_eq_array,
    h_ineq_array,
    q_array;
    probabilities=nothing,
    μ=0,
    ρ=0,
    rho=ρ,
    kwargs...,
)
    _, _, _, result = _solve_stochastic_extensive(
        solver,
        sp,
        W_eq_array,
        W_ineq_array,
        T_eq_array,
        T_ineq_array,
        h_eq_array,
        h_ineq_array,
        q_array;
        probabilities=probabilities,
        μ=μ,
        ρ=rho,
        kwargs...,
    )
    return _split_stochastic_solution(sp, result, W_eq_array, W_ineq_array, q_array)
end

function _solve_stochastic_extensive(
    solver::Solver,
    sp::StochasticProgram,
    W_eq_array,
    W_ineq_array,
    T_eq_array,
    T_ineq_array,
    h_eq_array,
    h_ineq_array,
    q_array;
    probabilities=nothing,
    μ=0,
    ρ=0,
    rho=ρ,
    kwargs...,
)
    lp = construct_lp(
        sp,
        W_eq_array,
        W_ineq_array,
        T_eq_array,
        T_ineq_array,
        h_eq_array,
        h_ineq_array,
        q_array;
        probabilities=probabilities,
    )
    μ_vector =
        _stochastic_barrier_parameter_vector(lp, sp, W_ineq_array, μ; probabilities=probabilities)
    ρ_vector =
        _stochastic_quadratic_parameter_vector(lp, sp, q_array, rho; probabilities=probabilities)

    result = try
        solve(solver, lp; μ=μ_vector, ρ=ρ_vector, kwargs...)
    catch error
        _throw_stochastic_program_failure(
            error,
            _stochastic_failure_location(W_eq_array),
            solver,
            sp,
            W_eq_array,
            W_ineq_array,
            T_eq_array,
            T_ineq_array,
            h_eq_array,
            h_ineq_array,
            q_array;
            μ=μ,
            ρ=rho,
            effective_μ=μ_vector,
            effective_ρ=ρ_vector,
            probabilities=probabilities,
            kwargs=(; kwargs...),
        )
    end

    return lp, μ_vector, ρ_vector, result
end

function _stochastic_barrier_parameter_vector(
    lp::LP,
    sp::StochasticProgram,
    W_ineq_array,
    μ;
    probabilities=nothing,
)
    μ isa AbstractVector && return _barrier_parameter_vector(lp, μ)

    K = size(W_ineq_array, 3)
    first_stage_inequalities = length(sp.first_stage_lp.b_ineq)
    second_stage_inequalities = size(W_ineq_array, 1)

    T = promote_type(
        typeof(μ),
        isnothing(probabilities) ? Float64 : eltype(probabilities),
    )
    μ_vector = zeros(T, length(lp.b_ineq))

    μ >= zero(μ) || throw(ArgumentError("μ must be non-negative."))
    μ_vector[1:first_stage_inequalities] .= μ

    probability_vector = if isnothing(probabilities)
        fill(one(T) / K, K)
    else
        length(probabilities) == K ||
            throw(DimensionMismatch("probabilities must have one entry per scenario."))
        probabilities
    end

    for k in 1:K
        first_row = first_stage_inequalities + (k - 1) * second_stage_inequalities + 1
        last_row = first_stage_inequalities + k * second_stage_inequalities
        rows = first_row:last_row
        μ_vector[rows] .= μ .* probability_vector[k]
    end

    return μ_vector
end

function _stochastic_quadratic_parameter_vector(
    lp::LP,
    sp::StochasticProgram,
    q_array,
    ρ;
    probabilities=nothing,
)
    ρ isa AbstractVector && return _quadratic_parameter_vector(lp, ρ)

    K = size(q_array, 2)
    first_stage_variables = length(sp.first_stage_lp.c)
    second_stage_variables = size(q_array, 1)

    T = promote_type(
        typeof(ρ),
        isnothing(probabilities) ? Float64 : eltype(probabilities),
    )
    ρ_vector = zeros(T, length(lp.c))

    ρ >= zero(ρ) || throw(ArgumentError("ρ must be non-negative."))
    ρ_vector[1:first_stage_variables] .= ρ

    probability_vector = if isnothing(probabilities)
        fill(one(T) / K, K)
    else
        length(probabilities) == K ||
            throw(DimensionMismatch("probabilities must have one entry per scenario."))
        probabilities
    end

    for k in 1:K
        first_col = first_stage_variables + (k - 1) * second_stage_variables + 1
        last_col = first_stage_variables + k * second_stage_variables
        cols = first_col:last_col
        ρ_vector[cols] .= ρ .* probability_vector[k]
    end

    return ρ_vector
end

function _split_stochastic_solution(sp::StochasticProgram, result, W_eq_array, W_ineq_array, q_array)
    first_stage_lp = sp.first_stage_lp
    K = size(W_eq_array, 3)
    nz = length(first_stage_lp.c)
    ny = size(q_array, 1)

    m1_eq = length(first_stage_lp.b_eq)
    m1_ineq = length(first_stage_lp.b_ineq)
    m2_eq = size(W_eq_array, 1)
    m2_ineq = size(W_ineq_array, 1)

    z = result.z[1:nz]
    y = reshape(result.z[(nz + 1):(nz + K * ny)], ny, K)

    λ_b_eq = result.dual_eq[1:m1_eq]
    λ_b_ineq = result.dual_ineq[1:m1_ineq]
    λ_h_eq_array = reshape(
        result.dual_eq[(m1_eq + 1):(m1_eq + K * m2_eq)],
        m2_eq,
        K,
    )
    λ_h_ineq_array = reshape(
        result.dual_ineq[(m1_ineq + 1):(m1_ineq + K * m2_ineq)],
        m2_ineq,
        K,
    )

    return (
        z,
        y,
        λ_b_eq,
        λ_b_ineq,
        λ_h_eq_array,
        λ_h_ineq_array,
    )
end

# END FILE: src/ContextualDFL/ContextualDFL/src/stochastic_programming/solve.jl

# BEGIN FILE: src/ContextualDFL/ContextualDFL/src/stochastic_programming/solve_rrule.jl
import LinearAlgebra: lu

# Differentiates solve with respect to h_eq_array, h_ineq_array, and q_array.
# W_eq_array, W_ineq_array, T_eq_array, and T_ineq_array are treated as constants.
function ChainRulesCore.rrule(
    ::typeof(solve),
    solver::Solver,
    sp::StochasticProgram,
    W_eq_array,
    W_ineq_array,
    T_eq_array,
    T_ineq_array,
    h_eq_array,
    h_ineq_array,
    q_array;
    probabilities=nothing,
    μ=0,
    ρ=0,
    rho=ρ,
    kwargs...,
)
    lp, μ_vector, ρ_vector, result = _solve_stochastic_extensive(
        solver,
        sp,
        W_eq_array,
        W_ineq_array,
        T_eq_array,
        T_ineq_array,
        h_eq_array,
        h_ineq_array,
        q_array;
        probabilities=probabilities,
        μ=μ,
        ρ=rho,
        kwargs...,
    )
    output = _split_stochastic_solution(sp, result, W_eq_array, W_ineq_array, q_array)

    p_vector = if isnothing(probabilities)
        fill(one(eltype(lp.c)) / size(W_eq_array, 3), size(W_eq_array, 3))
    else
        probabilities
    end

    function stochastic_solve_pullback(output_tangent)
        z, y, _, _, _, _ = output
        dz = _maybe_array_cotangent(output_tangent, 1; name=:z)
        dy = _maybe_array_cotangent(output_tangent, 2; name=:y)

        _assert_zero_cotangent_component(output_tangent, 3; name=:λ_b_eq)
        _assert_zero_cotangent_component(output_tangent, 4; name=:λ_b_ineq)
        _assert_zero_cotangent_component(output_tangent, 5; name=:λ_h_eq_array)
        _assert_zero_cotangent_component(output_tangent, 6; name=:λ_h_ineq_array)

        T = promote_type(
            eltype(lp.c),
            eltype(z),
            eltype(y),
            isnothing(dz) ? eltype(z) : eltype(dz),
            isnothing(dy) ? eltype(y) : eltype(dy),
        )
        primal_tangent = zeros(T, length(lp.c))
        nz = length(z)
        if !isnothing(dz)
            length(dz) == nz || throw(DimensionMismatch("z cotangent must have length $(nz)."))
            primal_tangent[1:nz] .= dz
        end
        if !isnothing(dy)
            length(dy) == length(y) ||
                throw(DimensionMismatch("y cotangent must have length $(length(y))."))
            offset = nz
            @inbounds for value in dy
                offset += 1
                primal_tangent[offset] = value
            end
        end

        dc, db_eq, db_ineq = _lp_reverse_from_primal_tangent(
            solver,
            lp,
            μ_vector,
            ρ_vector,
            result,
            primal_tangent;
            kwargs...,
        )

        first_stage_lp = sp.first_stage_lp
        K = size(W_eq_array, 3)
        nz = length(first_stage_lp.c)
        ny = size(q_array, 1)
        m1_eq = length(first_stage_lp.b_eq)
        m1_ineq = length(first_stage_lp.b_ineq)
        m2_eq = size(W_eq_array, 1)
        m2_ineq = size(W_ineq_array, 1)

        T = promote_type(eltype(dc), eltype(db_eq), eltype(db_ineq))
        dh_eq_array = zeros(T, size(h_eq_array))
        dh_ineq_array = zeros(T, size(h_ineq_array))
        dq_array = zeros(T, size(q_array))

        for k in 1:K
            y_cols = (nz + (k - 1) * ny + 1):(nz + k * ny)
            eq_rows = (m1_eq + (k - 1) * m2_eq + 1):(m1_eq + k * m2_eq)
            ineq_rows = (m1_ineq + (k - 1) * m2_ineq + 1):(m1_ineq + k * m2_ineq)

            dh_eq_array[:, k] = view(db_eq, eq_rows)
            dh_ineq_array[:, k] = view(db_ineq, ineq_rows)
            dq_array[:, k] = p_vector[k] .* view(dc, y_cols)
        end

        return (
            ChainRulesCore.NoTangent(),
            ChainRulesCore.NoTangent(),
            ChainRulesCore.NoTangent(),
            ChainRulesCore.NoTangent(),
            ChainRulesCore.NoTangent(),
            ChainRulesCore.NoTangent(),
            ChainRulesCore.NoTangent(),
            dh_eq_array,
            dh_ineq_array,
            dq_array,
        )
    end

    return output, stochastic_solve_pullback
end

function _lp_reverse_from_primal_tangent(
    solver,
    lp::LP,
    μ,
    ρ,
    result,
    primal_tangent;
    kwargs...,
)
    tight_tol = haskey(kwargs, :tight_tol) ? kwargs[:tight_tol] : 1e-7
    cache = _lp_reverse_precompute(lp, μ, ρ, result, tight_tol)

    n = length(lp.c)
    m_eq = length(lp.b_eq)
    m_ineq = length(lp.b_ineq)
    T = promote_type(
        eltype(lp.c),
        eltype(lp.b_eq),
        eltype(lp.b_ineq),
        eltype(cache.μ),
        eltype(cache.ρ),
        eltype(primal_tangent),
    )

    μ_vector = cache.μ
    ρ_vector = cache.ρ
    kkt_size = _is_zero_barrier_parameter(μ_vector) ? n + m_eq + length(cache.tight) : n + m_eq
    rhs = zeros(T, kkt_size)
    rhs[1:n] = primal_tangent
    adjoint_solution = cache.K_factorization \ rhs
    adjoint_primal = adjoint_solution[1:n]

    dc = zeros(T, n)
    db_eq = zeros(T, m_eq)
    db_ineq = zeros(T, m_ineq)

    if _is_zero_barrier_parameter(μ_vector)
        adjoint_constraints = adjoint_solution[(n + 1):end]
        db_eq .= view(adjoint_constraints, 1:m_eq)
        db_ineq[cache.tight] .= view(adjoint_constraints, (m_eq + 1):length(adjoint_constraints))

        if _is_zero_quadratic_parameter(ρ_vector) && !isempty(cache.loose)
            db_ineq[cache.loose] .=
                cache.d .* (view(lp.A_ineq, cache.loose, :) * adjoint_primal)
        end
        if !_is_zero_quadratic_parameter(ρ_vector)
            dc .= .-adjoint_primal
        end

        return dc, db_eq, db_ineq
    end

    dc .= .-adjoint_primal
    db_eq .= view(adjoint_solution, (n + 1):(n + m_eq))
    if m_ineq > 0
        db_ineq .= μ_vector .* cache.d .* (lp.A_ineq * adjoint_primal)
    end

    return dc, db_eq, db_ineq
end

function _lp_reverse_precompute(lp::LP, μ, result, tight_tol)
    return _lp_reverse_precompute(lp, μ, 0, result, tight_tol)
end

function _lp_reverse_precompute(lp::LP, μ, ρ, result, tight_tol)
    μ_vector = _barrier_parameter_vector(lp, μ)
    ρ_vector = _quadratic_parameter_vector(lp, ρ)

    z = result isa AbstractVector ? result : result.z
    n = length(lp.c)
    m_eq = length(lp.b_eq)
    length(z) == n || throw(DimensionMismatch("The solution must have length $(n)."))

    slack = hasproperty(result, :slack) ? result.slack : lp.b_ineq - lp.A_ineq * z
    any(<(-tight_tol), slack) &&
        throw(DomainError(slack, "The solution violates inequality constraints."))

    if _is_zero_barrier_parameter(μ_vector)
        tight = findall(abs.(slack) .<= tight_tol)
        loose = findall(slack .> tight_tol)

        F = Matrix(lp.A_eq)
        selected_tight = Int[]
        current_rank = rank(F)
        for index in tight
            candidate = [F; lp.A_ineq[index:index, :]]
            candidate_rank = rank(candidate)
            if candidate_rank > current_rank
                push!(selected_tight, index)
                F = candidate
                current_rank = candidate_rank
            end
        end

        tight = selected_tight
        d = _is_zero_quadratic_parameter(ρ_vector) ?
            one(eltype(slack)) ./ (slack[loose] .^ 2) :
            zeros(promote_type(eltype(slack), eltype(ρ_vector)), length(loose))

        H = if _is_zero_quadratic_parameter(ρ_vector)
            A_loose = lp.A_ineq[loose, :]
            transpose(A_loose) * (Diagonal(d) * A_loose)
        else
            Diagonal(ρ_vector)
        end
        T = promote_type(eltype(H), eltype(F), eltype(μ_vector), eltype(ρ_vector))
        K = if issparse(H) || issparse(F)
            [
                sparse(H) sparse(transpose(F))
                sparse(F) spzeros(T, size(F, 1), size(F, 1))
            ]
        else
            [
                H transpose(F)
                F zeros(T, size(F, 1), size(F, 1))
            ]
        end

        K_factorization = issparse(K) ? lu(K) : bunchkaufman(Symmetric(K))
        return (; z=z, d=d, K_factorization=K_factorization, μ=μ_vector, ρ=ρ_vector, tight=tight, loose=loose)
    end

    all(>(zero(eltype(slack))), slack) ||
        throw(DomainError(slack, "The log-barrier solution must have positive inequality slack."))

    d = one(eltype(slack)) ./ (slack .^ 2)
    H = transpose(lp.A_ineq) * (Diagonal(μ_vector .* d) * lp.A_ineq)
    H = _is_zero_quadratic_parameter(ρ_vector) ? H : H + Diagonal(ρ_vector)
    T = promote_type(eltype(H), eltype(lp.A_eq), eltype(μ_vector), eltype(ρ_vector))
    K = if issparse(H) || issparse(lp.A_eq)
        [
            sparse(H) sparse(transpose(lp.A_eq))
            sparse(lp.A_eq) spzeros(T, m_eq, m_eq)
        ]
    else
        [
            H transpose(lp.A_eq)
            lp.A_eq zeros(T, m_eq, m_eq)
        ]
    end

    K_factorization = issparse(K) ? lu(K) : bunchkaufman(Symmetric(K))
    return (; z=z, d=d, K_factorization=K_factorization, μ=μ_vector, ρ=ρ_vector, tight=Int[], loose=collect(1:length(lp.b_ineq)))
end

function _array_cotangent(output_tangent, index, template; name)
    component = _cotangent_component(output_tangent, index)
    if component isa AbstractArray
        return component
    end
    _is_zero_cotangent(component) && return zeros(eltype(template), size(template))

    throw(ArgumentError("Expected array or zero cotangent for $(name), got $(typeof(component))."))
end

function _maybe_array_cotangent(output_tangent, index; name)
    component = _cotangent_component(output_tangent, index)
    _is_zero_cotangent(component) && return nothing
    component isa AbstractArray && return component

    throw(ArgumentError("Expected array or zero cotangent for $(name), got $(typeof(component))."))
end

function _assert_zero_cotangent_component(output_tangent, index; name)
    component = _cotangent_component(output_tangent, index)
    _is_zero_cotangent(component) && return nothing

    throw(ArgumentError("The solve rrule does not support nonzero cotangents for $(name)."))
end

function _cotangent_component(output_tangent, index)
    tangent = ChainRulesCore.unthunk(output_tangent)
    if tangent isa ChainRulesCore.AbstractZero
        return ChainRulesCore.ZeroTangent()
    elseif tangent isa Tuple
        index > length(tangent) && return ChainRulesCore.ZeroTangent()
        return ChainRulesCore.unthunk(tangent[index])
    elseif tangent isa ChainRulesCore.Tangent
        index in propertynames(tangent) || return ChainRulesCore.ZeroTangent()
        return ChainRulesCore.unthunk(getproperty(tangent, index))
    end

    throw(ArgumentError("Expected tuple-like cotangent for solve output, got $(typeof(tangent))."))
end

_is_zero_cotangent(component::AbstractArray) = all(iszero, component)
_is_zero_cotangent(component::ChainRulesCore.AbstractZero) = true
_is_zero_cotangent(component::Number) = iszero(component)
_is_zero_cotangent(component) = false

# END FILE: src/ContextualDFL/ContextualDFL/src/stochastic_programming/solve_rrule.jl

# BEGIN FILE: src/ContextualDFL/ContextualDFL/test/implementations/transshipment/runtests.jl
using ContextualDFL
using LinearAlgebra
using Random
using Test

@testset "transshipment implementation" begin
    problem = TransShipmentProblem()

    @test problem.data.source_repository == "https://github.com/USC3DLAB/SD"
    @test isfile(problem.data.core_path)
    @test isfile(problem.data.time_path)
    @test isfile(problem.data.stochastic_path)
    @test length(problem.data.first_stage_variables) == 7
    @test length(problem.data.second_stage_variables) == 77
    @test isempty(problem.data.first_stage_rows)
    @test length(problem.data.second_stage_rows) == 35
    @test length(problem.data.random_rhs_entries) == 7
    @test length(problem.data.random_objective_entries) == 7
    @test first(problem.data.first_stage_variables) == "orderUp(0)"
    @test first(problem.data.second_stage_variables) == "begEnd(0)"
    @test first(problem.data.second_stage_rows) == "initInv(0)"

    @test size(problem.stochastic_program.A_eq) == (0, 7)
    @test size(problem.stochastic_program.A_ineq) == (7, 7)
    @test problem.stochastic_program.A_ineq == -Matrix{Float64}(I, 7, 7)
    @test size(problem.base_scenario.W_eq) == (35, 77)
    @test size(problem.base_scenario.T_eq) == (35, 7)
    @test size(problem.base_scenario.W_ineq) == (77, 77)
    @test problem.base_scenario.W_ineq == -Matrix{Float64}(I, 77, 77)

    mean_parameters = transshipment_mean_parameters(problem)
    @test mean_parameters.rhs == [100.0, 200.0, 150.0, 170.0, 180.0, 170.0, 170.0]
    @test mean_parameters.q == [4.0, 4.2, 4.4, 4.6, 4.8, 5.0, 5.2]

    arrays = transshipment_scenario_arrays(problem, [mean_parameters])
    @test size(arrays[1]) == (35, 77, 1)
    @test size(arrays[3]) == (35, 7, 1)
    @test size(arrays[5]) == (35, 1)
    @test size(arrays[7]) == (77, 1)

    block_lp = transshipment_mean_lp(problem)
    direct_lp = transshipment_direct_mean_lp(problem)
    @test block_lp.A_eq == direct_lp.A_eq
    @test block_lp.A_ineq == direct_lp.A_ineq
    @test block_lp.b_eq == direct_lp.b_eq
    @test block_lp.b_ineq == direct_lp.b_ineq
    @test block_lp.c == direct_lp.c

    decoder = TransShipmentScenarioDecoder(problem)
    perturbed_rhs = copy(mean_parameters.rhs)
    perturbed_rhs[4] += 1.0
    rhs_arrays = transshipment_scenario_arrays(problem, [(; rhs=perturbed_rhs, q=mean_parameters.q)])
    @test count(!iszero, vec(rhs_arrays[5] - arrays[5])) == 1
    @test all(iszero, vec(rhs_arrays[7] - arrays[7]))

    perturbed_q = copy(mean_parameters.q)
    perturbed_q[4] += 1.0
    q_arrays = transshipment_scenario_arrays(problem, [(; rhs=mean_parameters.rhs, q=perturbed_q)])
    @test count(!iszero, vec(q_arrays[7] - arrays[7])) == 1
    @test all(iszero, vec(q_arrays[5] - arrays[5]))

    compact_scenario = ContextualDFL.ParametricScenario(;
        h_eq_xi=mean_parameters.rhs .+ 1.0,
        q_xi=mean_parameters.q .+ 1.0,
    )
    compact_arrays = decode_scenario_collection(decoder, [compact_scenario])
    @test count(!iszero, vec(compact_arrays[5] - arrays[5])) == 7
    @test count(!iszero, vec(compact_arrays[7] - arrays[7])) == 7

    vector_arrays = decode_scenario_collection(
        decoder,
        vcat(mean_parameters.rhs, mean_parameters.q);
        nr_scenarios=1,
    )
    @test vector_arrays[5] == arrays[5]
    @test vector_arrays[7] == arrays[7]

    sampled = sample_transshipment_parameters(
        problem;
        rng=Random.MersenneTwister(11),
        truncate_at_zero=true,
    )
    @test length(sampled.rhs) == 7
    @test length(sampled.q) == 7
    @test all(>=(0.0), sampled.rhs)
    @test all(>=(0.0), sampled.q)

    report = validate_transshipment_problem(problem)
    @test report.dimensions == (; n1=7, n2=77, m1=0, m2=35)
    @test report.random_rhs_entries == 7
    @test report.random_objective_entries == 7
    @test all(values(report.perturbation_report))
    @test length(report.solve_reports) == 4
    @test all(item -> item.status in ("OPTIMAL", "LOCALLY_SOLVED"), report.solve_reports)
    @test all(item -> item.max_equality_residual <= 1e-6, report.solve_reports)
    @test all(item -> item.max_inequality_violation <= 1e-6, report.solve_reports)

    W_eq, W_ineq, T_eq, T_ineq, h_eq, h_ineq, q = arrays
    solver = Solver(IpoptSolver(), HiGHSSolver())
    z, y = solve(
        solver,
        problem.stochastic_program,
        W_eq,
        W_ineq,
        T_eq,
        T_ineq,
        h_eq,
        h_ineq,
        q;
        μ=0.0,
        ρ=0.0,
    )[1:2]
    @test length(z) == 7
    @test size(y) == (77, 1)
    @test minimum(z) >= -1e-7
    @test minimum(y) >= -1e-7
end

# END FILE: src/ContextualDFL/ContextualDFL/test/implementations/transshipment/runtests.jl

# BEGIN FILE: src/ContextualDFL/ContextualDFL/test/learning/runtests.jl
import Flux
import ChainRulesCore

function supervised_dataset(inputs, targets)
    points = [
        ContextualDataPoint(Float32[input], [ParametricScenario(h_eq_xi=Float32[target])])
        for (input, target) in zip(inputs, targets)
    ]
    return ContextualDataSet{eltype(points)}(points)
end

target_vector(scenario_parameters) = only(scenario_parameters).h_eq_xi

@testset "learning" begin
    @testset "train! calls epoch callback" begin
        model = Flux.Chain(Flux.Dense(1 => 1))
        data = supervised_dataset(1:4, 2:2:8)
        loss(prediction, scenario_parameters, mu_in, mu_ref; kwargs...) =
            sum(abs2, prediction .- target_vector(scenario_parameters))
        callbacks = NamedTuple[]

        result = train!(
            model,
            loss,
            nothing,
            fill(0.0, 2),
            data;
            optimizer_type=Flux.Descent,
            learning_rate=1e-4,
            epochs=2,
            batchsize=2,
            on_epoch_end=(epoch, loss_value, display_loss, metadata) -> push!(
                callbacks,
                (;
                    epoch=epoch,
                    loss=loss_value,
                    display_loss=display_loss,
                    metadata=metadata,
                ),
            ),
        )

        @test length(result.history) == 2
        @test length(callbacks) == 2
        @test [callback.epoch for callback in callbacks] == [1, 2]
        @test all(callback -> callback.loss isa Float64, callbacks)
        @test all(callback -> callback.display_loss isa Float64, callbacks)
        @test [callback.metadata.epoch for callback in callbacks] == [1, 2]
        @test [callback.metadata.iterations for callback in callbacks] == [2, 2]
        @test all(callback -> callback.metadata.epoch_seconds >= 0, callbacks)
    end

    @testset "train! defaults mu_ref_schedule to mu_in_schedule" begin
        model = Flux.Chain(Flux.Dense(1 => 1))
        data = supervised_dataset(1:4, 2:2:8)
        mu_schedule = [0.1, 0.2]
        loss(prediction, scenario_parameters, mu_in, mu_ref; kwargs...) =
            sum(abs2, prediction .- target_vector(scenario_parameters))

        result = train!(
            model,
            loss,
            nothing,
            mu_schedule,
            data;
            optimizer_type=Flux.Descent,
            learning_rate=1e-4,
            epochs=length(mu_schedule),
            batchsize=2,
        )

        @test [row.mu_in for row in result.history] == mu_schedule
        @test [row.mu_ref for row in result.history] == mu_schedule
        @test [row.mu for row in result.history] == mu_schedule
    end

    @testset "train! threads rho schedules only when requested" begin
        model = Flux.Chain(Flux.Dense(1 => 1))
        data = supervised_dataset(1:4, 2:2:8)
        mu_schedule = [0.0, 0.0]
        rho_in_schedule = [0.0, 0.2]
        rho_ref_schedule = [0.1, 0.3]
        rho_calls = Tuple{Float64,Float64}[]

        function rho_loss(prediction, scenario_parameters, mu_in, mu_ref; rho_in, rho_ref, kwargs...)
            ChainRulesCore.ignore_derivatives() do
                push!(rho_calls, (Float64(rho_in), Float64(rho_ref)))
            end
            return sum(abs2, prediction .- target_vector(scenario_parameters))
        end

        result = train!(
            model,
            rho_loss,
            nothing,
            mu_schedule,
            data;
            optimizer_type=Flux.Descent,
            learning_rate=1e-4,
            epochs=length(mu_schedule),
            batchsize=2,
            rho_in_schedule=rho_in_schedule,
            rho_ref_schedule=rho_ref_schedule,
        )

        @test [row.rho_in for row in result.history] == rho_in_schedule
        @test [row.rho_ref for row in result.history] == rho_ref_schedule
        @test Set(rho_calls) == Set(zip(rho_in_schedule, rho_ref_schedule))

        strict_loss(prediction, scenario_parameters, mu_in, mu_ref) =
            sum(abs2, prediction .- target_vector(scenario_parameters))
        strict_result = train!(
            model,
            strict_loss,
            nothing,
            fill(0.0, 1),
            data;
            optimizer_type=Flux.Descent,
            learning_rate=1e-4,
            epochs=1,
            batchsize=2,
            rho_in_schedule=[0.0],
            rho_ref_schedule=[0.0],
        )
        @test only(strict_result.history).rho_in == 0.0
        @test only(strict_result.history).rho_ref == 0.0
    end

    @testset "train! rejects non-finite training loss" begin
        model = Flux.Chain(Flux.Dense(1 => 1))
        data = supervised_dataset(1:1, 2:2)
        nan_loss(prediction, scenario_parameters, mu_in, mu_ref; kwargs...) =
            sum(prediction) * Float32(NaN)
        callback_count = Ref(0)

        @test_throws DomainError train!(
            model,
            nan_loss,
            nothing,
            fill(0.0, 1),
            data;
            optimizer_type=Flux.Descent,
            learning_rate=1.0,
            epochs=1,
            batchsize=1,
            on_epoch_end=(args...) -> (callback_count[] += 1),
        )

        @test callback_count[] == 0
        @test all(parameter -> all(isfinite, parameter), Flux.trainables(model))
    end

    @testset "train! smooth display uses cached references" begin
        model = Flux.Chain(Flux.Dense(1 => 1))
        data = supervised_dataset(1:4, 2:2:8)
        mu_in_schedule = [0.1, 0.2]
        mu_ref_schedule = [0.1, 0.2]
        reference_calls = Ref(0)
        relative_calls = Ref(0)

        function loss(input, scenario_parameters, mu_in, mu_ref; kwargs...)
            input === target_vector(scenario_parameters) && (reference_calls[] += 1)
            return sum(abs2, input) + sum(target_vector(scenario_parameters)) + mu_in + mu_ref
        end
        relative_loss(args...; kwargs...) = (relative_calls[] += 1)
        display_reference_input(point) = target_vector(point.scenario_parameters)

        result = train!(
            model,
            loss,
            relative_loss,
            mu_in_schedule,
            mu_ref_schedule,
            data;
            optimizer_type=Flux.Descent,
            learning_rate=1e-4,
            epochs=length(mu_in_schedule),
            batchsize=2,
            display_smooth=true,
            display_reference_input=display_reference_input,
        )

        @test reference_calls[] == length(unique(mu_ref_schedule)) * length(data)
        @test relative_calls[] == 0
        @test all(row -> row.display_loss isa Float64, result.history)
        @test all(row -> row.real_display_loss === nothing, result.history)
    end

    @testset "train! display modes validate reference input" begin
        model = Flux.Chain(Flux.Dense(1 => 1))
        data = supervised_dataset(1:1, 2:2)
        loss(input, scenario_parameters, mu_in, mu_ref; kwargs...) =
            sum(abs2, input) + sum(target_vector(scenario_parameters))
        display_reference_input(point) = target_vector(point.scenario_parameters)

        @test_throws ArgumentError train!(
            model,
            loss,
            nothing,
            fill(0.0, 1),
            data;
            optimizer_type=Flux.Descent,
            learning_rate=1e-4,
            epochs=1,
            batchsize=1,
            display_smooth=true,
        )
        @test_throws ArgumentError train!(
            model,
            loss,
            nothing,
            fill(0.0, 1),
            data;
            optimizer_type=Flux.Descent,
            learning_rate=1e-4,
            epochs=1,
            batchsize=1,
            display_real=1,
        )
        @test_throws ArgumentError train!(
            model,
            loss,
            nothing,
            fill(0.0, 1),
            data;
            optimizer_type=Flux.Descent,
            learning_rate=1e-4,
            epochs=1,
            batchsize=1,
            display_real=0,
            display_reference_input=display_reference_input,
        )
    end

    @testset "train! real display runs on requested epochs" begin
        model = Flux.Chain(Flux.Dense(1 => 1))
        data = supervised_dataset(1:2, 2:2:4)
        mu_schedule = [0.3, 0.2, 0.1]
        reference_calls = Ref(0)
        real_calls = Ref(0)

        function loss(input, scenario_parameters, mu_in, mu_ref; kwargs...)
            if input === target_vector(scenario_parameters)
                reference_calls[] += 1
            elseif mu_ref == 0.0
                real_calls[] += 1
            end
            return sum(abs2, input) + sum(target_vector(scenario_parameters)) + mu_in + mu_ref
        end
        display_reference_input(point) = target_vector(point.scenario_parameters)

        result = train!(
            model,
            loss,
            nothing,
            mu_schedule,
            data;
            optimizer_type=Flux.Descent,
            learning_rate=1e-4,
            epochs=length(mu_schedule),
            batchsize=1,
            display_real=2,
            display_reference_input=display_reference_input,
        )

        @test reference_calls[] == length(data)
        @test real_calls[] == length(data)
        @test result.history[1].real_display_loss === nothing
        @test result.history[2].real_display_loss isa Float64
        @test result.history[3].real_display_loss === nothing
    end

end

# END FILE: src/ContextualDFL/ContextualDFL/test/learning/runtests.jl

# BEGIN FILE: src/ContextualDFL/ContextualDFL/test/linear_programming/base_suite/runtests.jl
struct TestLPSolver <: LPSolver end
struct TestLogBarSolver <: LogBarSolver end

ContextualDFL.solve(::TestLPSolver, lp::LP; kwargs...) = (; method=:lp, lp, kwargs)
ContextualDFL.solve(::TestLogBarSolver, lp::LP; μ=nothing, ρ=nothing, kwargs...) =
    (; method=:log_barrier, lp, μ, ρ, kwargs)

@testset "base LP suite" begin
    @testset "LP construction" begin
        lp = LP(c=[1.0, 2.0])

        @test size(lp.A_eq) == (0, 2)
        @test size(lp.A_ineq) == (0, 2)
        @test isempty(lp.b_eq)
        @test isempty(lp.b_ineq)
        @test lp.c == [1.0, 2.0]
        @test eltype(lp.A_eq) == Float64
        @test eltype(lp.b_eq) == Float64

        zero_variable_lp = LP()

        @test size(zero_variable_lp.A_eq) == (0, 0)
        @test size(zero_variable_lp.A_ineq) == (0, 0)
        @test isempty(zero_variable_lp.b_eq)
        @test isempty(zero_variable_lp.b_ineq)
        @test isempty(zero_variable_lp.c)

        matrix_only_lp = LP(A_eq=[1.0 2.0; 3.0 4.0])

        @test matrix_only_lp.A_eq == [1.0 2.0; 3.0 4.0]
        @test matrix_only_lp.b_eq == zeros(2)
        @test matrix_only_lp.c == zeros(2)
        @test size(matrix_only_lp.A_ineq) == (0, 2)

        @test_throws ArgumentError LP(b_eq=[1.0], c=[1.0, 2.0])
        @test_throws DimensionMismatch LP(A_eq=[1.0 2.0], c=[1.0])
        @test_throws DimensionMismatch LP(A_eq=[1.0 2.0], b_eq=[1.0, 2.0], c=[1.0, 2.0])
    end

    @testset "solver dispatch" begin
        lp = LP(c=[1.0, 2.0])
        log_barrier_lp = LP(A_ineq=[1.0 0.0], b_ineq=[10.0], c=[1.0, 2.0])
        solver = Solver(TestLogBarSolver(), TestLPSolver())

        lp_solution = solve(solver, lp; warm_start=:basis)
        log_barrier_solution = solve(solver, log_barrier_lp; μ=0.5, max_iter=100)

        @test lp_solution.method == :lp
        @test lp_solution.lp === lp
        @test lp_solution.kwargs[:warm_start] == :basis

        @test log_barrier_solution.method == :log_barrier
        @test log_barrier_solution.lp === log_barrier_lp
        @test log_barrier_solution.μ == [0.5]
        @test log_barrier_solution.kwargs[:max_iter] == 100

        vector_barrier_solution = solve(solver, log_barrier_lp; μ=[0.25])
        zero_vector_solution = solve(solver, log_barrier_lp; μ=zeros(1))
        quadratic_solution = solve(solver, lp; ρ=0.5)
        vector_quadratic_solution = solve(solver, lp; rho=[0.25, 0.75])
        zero_quadratic_solution = solve(solver, lp; ρ=zeros(2))
        combined_solution = solve(solver, log_barrier_lp; μ=0.5, ρ=0.25)

        @test vector_barrier_solution.method == :log_barrier
        @test vector_barrier_solution.μ == [0.25]
        @test zero_vector_solution.method == :lp
        @test quadratic_solution.method == :log_barrier
        @test quadratic_solution.ρ == [0.5, 0.5]
        @test vector_quadratic_solution.ρ == [0.25, 0.75]
        @test zero_quadratic_solution.method == :lp
        @test combined_solution.method == :log_barrier
        @test combined_solution.μ == [0.5]
        @test combined_solution.ρ == [0.25, 0.25]
        @test_throws DimensionMismatch solve(solver, log_barrier_lp; μ=[0.25, 0.5])
        @test_throws DimensionMismatch solve(solver, lp; ρ=[0.25])
        @test_throws ArgumentError solve(solver, lp; ρ=-0.25)
        @test_throws ArgumentError solve(solver, lp; ρ=[0.25, -0.75])
    end

    @testset "infeasible solves throw" begin
        infeasible_lp = LP(
            A_ineq=reshape([1.0, -1.0], 2, 1),
            b_ineq=[0.0, -1.0],
            c=[0.0],
        )

        @test_throws ErrorException solve(TEST_HIGHS_SOLVER, infeasible_lp)
        @test_throws ErrorException solve(TEST_SOLVER, infeasible_lp)
        @test_throws ErrorException solve(TEST_SOLVER, infeasible_lp; μ=1.0, max_iter=50)
    end

    @testset "geometric LP cases" begin
        square_A, square_b = square_2d()
        case_2_A = [square_A; 1.0 1.0]
        case_2_b = [square_b; 0.5]
        case_3_A = [
            case_2_A
            1.0 -1.0
            -1.0 0.4
            -0.3 -1.0
        ]
        case_3_b = [case_2_b; 1.0; 1.2; 1.4]

        simplex_2_A, simplex_2_b = nonnegative_orthant(2)
        simplex_5_A, simplex_5_b = nonnegative_orthant(5)

        tube_A = [square_A zeros(4)]
        tube_eq = [0.0 0.0 1.0]

        tilted_A = [case_3_A zeros(size(case_3_A, 1))]
        tilted_eq = [-0.2 0.1 1.0]
        tilted_expected = [-0.5, 1.0, 0.8]
        tilted_extra_A = [
            tilted_A
            0.2 1.0 0.0
            -0.4 -0.6 0.0
        ]
        tilted_extra_b = [case_3_b; 1.5; 1.3]

        cases = [
            (;
                name="case 1: square in 2D",
                lp=LP(A_ineq=square_A, b_ineq=square_b, c=[-1.0, -2.0]),
                expected_status="OPTIMAL",
                expected_z=[1.0, 1.0],
            ),
            (;
                name="case 2: square in 2D with one cut",
                lp=LP(A_ineq=case_2_A, b_ineq=case_2_b, c=[-1.0, -2.0]),
                expected_status="OPTIMAL",
                expected_z=[-0.5, 1.0],
            ),
            (;
                name="case 3: square in 2D with several cuts",
                lp=LP(A_ineq=case_3_A, b_ineq=case_3_b, c=[-1.0, -2.0]),
                expected_status="OPTIMAL",
                expected_z=[-0.5, 1.0],
            ),
            (;
                name="case 4: simplex in dimension 2",
                lp=LP(
                    A_eq=[1.0 1.0],
                    b_eq=[1.0],
                    A_ineq=simplex_2_A,
                    b_ineq=simplex_2_b,
                    c=[1.0, 2.0],
                ),
                expected_status="OPTIMAL",
                expected_z=[1.0, 0.0],
            ),
            (;
                name="case 5: simplex in dimension 5",
                lp=LP(
                    A_eq=ones(1, 5),
                    b_eq=[1.0],
                    A_ineq=simplex_5_A,
                    b_ineq=simplex_5_b,
                    c=collect(1.0:5.0),
                ),
                expected_status="OPTIMAL",
                expected_z=[1.0, 0.0, 0.0, 0.0, 0.0],
            ),
            (;
                name="case 6: square tube",
                lp=LP(
                    A_eq=tube_eq,
                    b_eq=[1.0],
                    A_ineq=tube_A,
                    b_ineq=square_b,
                    c=[-1.0, -2.0, 0.0],
                ),
                expected_status="OPTIMAL",
                expected_z=[1.0, 1.0, 1.0],
            ),
            (;
                name="case 7: square tube with tilted base floor",
                lp=LP(
                    A_eq=tilted_eq,
                    b_eq=[1.0],
                    A_ineq=tilted_A,
                    b_ineq=case_3_b,
                    c=[-1.0, -2.0, 0.0],
                ),
                expected_status="OPTIMAL",
                expected_z=tilted_expected,
            ),
            (;
                name="case 8: tilted square tube with extra cuts",
                lp=LP(
                    A_eq=tilted_eq,
                    b_eq=[1.0],
                    A_ineq=tilted_extra_A,
                    b_ineq=tilted_extra_b,
                    c=[-1.0, -2.0, 0.0],
                ),
                expected_status="OPTIMAL",
                expected_z=tilted_expected,
            ),
        ]

        for case in cases
            run_smooth_case(case)
        end

        for solver in (TEST_SOLVER, TEST_HIGHS_SOLVER)
            case_2_solution = solve(solver, cases[2].lp).z
            case_3_solution = solve(solver, cases[3].lp).z
            @test case_3_solution ≈ case_2_solution atol = 1e-8
        end
    end
end

# END FILE: src/ContextualDFL/ContextualDFL/test/linear_programming/base_suite/runtests.jl

# BEGIN FILE: src/ContextualDFL/ContextualDFL/test/linear_programming/extended_suite/runtests.jl
@testset "extended LP suite" begin
    @testset "equality-only sanity cases" begin
        singleton_case = (;
            name="E1 singleton equality system",
            lp=LP(
                A_eq=[1.0 1.0; 1.0 -1.0],
                b_eq=[1.0, 0.0],
                c=[3.0, -2.0],
            ),
            expected_status="OPTIMAL",
            expected_z=[0.5, 0.5],
        )
        assert_lp_case_with_highs(singleton_case)
        singleton_barrier = solve(TEST_SOLVER, singleton_case.lp; μ=1.0)
        @test is_optimal_status(singleton_barrier.status)
        @test singleton_barrier.z ≈ singleton_case.expected_z atol = 1e-8

        redundant_case = (;
            name="E2 redundant equality system",
            lp=LP(
                A_eq=[1.0 1.0; 1.0 -1.0; 2.0 2.0],
                b_eq=[1.0, 0.0, 2.0],
                c=[3.0, -2.0],
            ),
            expected_status="OPTIMAL",
            expected_z=[0.5, 0.5],
        )
        assert_lp_case_with_highs(redundant_case)

        @testset "E3 equality-only unbounded LP" begin
            unbounded_lp = LP(A_eq=[1.0 1.0], b_eq=[1.0], c=[1.0, 0.0])

            @test_throws ErrorException solve(TEST_SOLVER, unbounded_lp)
            @test_throws ErrorException solve(TEST_HIGHS_SOLVER, unbounded_lp)
        end

        degenerate_case = (;
            name="E4 equality-only degenerate optimal face",
            lp=LP(A_eq=[1.0 1.0], b_eq=[1.0], c=[1.0, 1.0]),
            expected_status="OPTIMAL",
        )
        assert_lp_case_with_highs(degenerate_case)
    end

    @testset "equality and inequality barrier cases" begin
        @testset "quadratic smoothing cases" begin
            one_dimensional_qp = LP(
                A_ineq=reshape([-1.0, 1.0], 2, 1),
                b_ineq=[0.0, 10.0],
                c=[-4.0],
            )
            quadratic_result = solve(TEST_SOLVER, one_dimensional_qp; ρ=2.0, tol=1e-10)

            @test is_optimal_status(quadratic_result.status)
            @test quadratic_result.z ≈ [2.0] atol = 1e-7
            @test quadratic_result.objective_value ≈ -4.0 atol = 1e-7
            @test diff_solve(
                TEST_SOLVER,
                one_dimensional_qp,
                0.0;
                ρ=2.0,
                pre_computed=quadratic_result,
                dc=[1.0],
            ) ≈ [-0.5] atol = 1e-8

            combined_lp = LP(
                A_ineq=reshape([-1.0, 1.0], 2, 1),
                b_ineq=[0.0, 10.0],
                c=[-1.0],
            )
            μ = 0.1
            ρ = 0.5
            combined_result = solve(TEST_SOLVER, combined_lp; μ=μ, ρ=ρ, tol=1e-10)
            slack = combined_lp.b_ineq - combined_lp.A_ineq * combined_result.z
            stationarity =
                combined_lp.c .+
                ρ .* combined_result.z .+
                transpose(combined_lp.A_ineq) * (fill(μ, length(slack)) ./ slack)

            @test is_optimal_status(combined_result.status)
            @test minimum(slack) > 1e-8
            @test norm(stationarity, Inf) ≤ 5e-5
        end

        @testset "vector barrier parameters" begin
            vector_barrier_lp = LP(
                A_eq=ones(1, 2),
                b_eq=[1.0],
                A_ineq=[1.0 0.0; -1.0 0.0; 0.0 1.0; 0.0 -1.0],
                b_ineq=[0.9, -0.1, 0.9, -0.1],
                c=[1.0, 2.0],
            )

            scalar_result = solve(TEST_SOLVER, vector_barrier_lp; μ=0.2, tol=1e-10)
            vector_result = solve(
                TEST_SOLVER,
                vector_barrier_lp;
                μ=fill(0.2, length(vector_barrier_lp.b_ineq)),
                tol=1e-10,
            )
            custom_vector_result = solve(
                TEST_SOLVER,
                vector_barrier_lp;
                μ=[0.1, 0.2, 0.3, 0.4],
                tol=1e-10,
            )

            @test scalar_result.z ≈ vector_result.z atol = 1e-8
            @test is_optimal_status(custom_vector_result.status)
            @test_throws DimensionMismatch solve(TEST_SOLVER, vector_barrier_lp; μ=[0.1])
        end

        slice_A, slice_b = box_constraints(fill(0.1, 3), fill(0.8, 3))
        tilted_A, tilted_b = box_constraints(fill(-1.0, 3), fill(1.0, 3))
        simplex_3_A, simplex_3_b = nonnegative_orthant(3)

        cases = [
            (;
                name="E6: affine slice of a box",
                lp=LP(
                    A_eq=ones(1, 3),
                    b_eq=[1.0],
                    A_ineq=slice_A,
                    b_ineq=slice_b,
                    c=[1.0, 2.0, 3.0],
                ),
                expected_status="OPTIMAL",
                expected_z=[0.8, 0.1, 0.1],
            ),
            (;
                name="E7: tilted affine slice of a box",
                lp=LP(
                    A_eq=[1.0 2.0 3.0],
                    b_eq=[1.0],
                    A_ineq=tilted_A,
                    b_ineq=tilted_b,
                    c=[-1.0, 0.5, 2.0],
                ),
                expected_status="OPTIMAL",
            ),
            (;
                name="E8: simplex with one oblique cut",
                lp=LP(
                    A_eq=ones(1, 3),
                    b_eq=[1.0],
                    A_ineq=[simplex_3_A; 1.0 2.0 0.0],
                    b_ineq=[simplex_3_b; 0.8],
                    c=[2.0, -2.0, 1.0],
                ),
                expected_status="OPTIMAL",
                expected_z=[0.0, 0.4, 0.6],
            ),
        ]

        for case in cases
            run_smooth_case(case)
        end
    end

    @testset "realistic equality-structured toy LPs" begin
        transport_A = [
            1.0 1.0 1.0 0.0 0.0 0.0
            0.0 0.0 0.0 1.0 1.0 1.0
            1.0 0.0 0.0 1.0 0.0 0.0
            0.0 1.0 0.0 0.0 1.0 0.0
        ]
        transport_A_redundant = [
            transport_A
            0.0 0.0 1.0 0.0 0.0 1.0
        ]
        transport_b = [1.0, 2.0, 0.5, 1.0]
        transport_b_redundant = [transport_b; 1.5]
        transport_ineq_A, transport_ineq_b = nonnegative_orthant(6)

        network_A = [
            1.0 1.0 0.0 0.0 0.0
            -1.0 0.0 1.0 1.0 0.0
            0.0 -1.0 -1.0 0.0 1.0
        ]
        network_ineq_A, network_ineq_b = box_constraints(zeros(5), ones(5))

        inventory_A_eq = [
            -1.0 0.0 0.0 1.0 0.0 0.0
            0.0 -1.0 0.0 -1.0 1.0 0.0
            0.0 0.0 -1.0 0.0 -1.0 1.0
        ]
        inventory_b_eq = [-0.3, -0.4, -0.2]
        production_upper = [Matrix{Float64}(I, 3, 3) zeros(3, 3)]
        production_nonnegative = [-Matrix{Float64}(I, 3, 3) zeros(3, 3)]
        inventory_nonnegative = [zeros(3, 3) -Matrix{Float64}(I, 3, 3)]
        inventory_A_ineq = [
            production_upper
            production_nonnegative
            inventory_nonnegative
        ]
        inventory_b_ineq = [ones(3); zeros(3); zeros(3)]

        cases = [
            (;
                name="E9: transportation polytope, full-row-rank equalities",
                lp=LP(
                    A_eq=transport_A,
                    b_eq=transport_b,
                    A_ineq=transport_ineq_A,
                    b_ineq=transport_ineq_b,
                    c=[1.0, 4.0, 2.0, 3.0, 1.0, 5.0],
                ),
                expected_status="OPTIMAL",
            ),
            (;
                name="E10: network flow with capacities",
                lp=LP(
                    A_eq=network_A,
                    b_eq=[1.0, 0.0, 0.0],
                    A_ineq=network_ineq_A,
                    b_ineq=network_ineq_b,
                    c=[1.0, 2.0, 0.5, 3.0, 1.0],
                ),
                expected_status="OPTIMAL",
            ),
            (;
                name="E11: inventory balance model",
                lp=LP(
                    A_eq=inventory_A_eq,
                    b_eq=inventory_b_eq,
                    A_ineq=inventory_A_ineq,
                    b_ineq=inventory_b_ineq,
                    c=[1.0, 1.2, 1.1, 0.1, 0.1, 0.1],
                ),
                expected_status="OPTIMAL",
            ),
        ]

        for case in cases
            run_smooth_case(case)
        end

        rank_deficient_transport = LP(
            A_eq=transport_A_redundant,
            b_eq=transport_b_redundant,
            A_ineq=transport_ineq_A,
            b_ineq=transport_ineq_b,
            c=[1.0, 4.0, 2.0, 3.0, 1.0, 5.0],
        )
        assert_lp_case_with_highs((;
            name="E9 redundant transportation polytope",
            lp=rank_deficient_transport,
            expected_status="OPTIMAL",
        ))

        for μ in TEST_BARRIER_MUS
            barrier_result =
                solve(TEST_SOLVER, rank_deficient_transport; μ=μ, tol=1e-10, max_iter=1_000)
            @test is_optimal_status(barrier_result.status)
            @test_throws ArgumentError construct_jacobian(
                TEST_SOLVER,
                rank_deficient_transport,
                μ;
                pre_computed=barrier_result.z,
            )
        end
    end

    @testset "bound-aware value preservation" begin
        @testset "extraction helper keeps provenance" begin
            lp = LP(
                A_ineq=[
                    1.0 0.0
                    -2.0 0.0
                    0.0 1.0
                    1.0 1.0
                ],
                b_ineq=[3.0, -2.0, 5.0, 7.0],
                c=[1.0, 2.0],
            )

            bound_lp, bound_map = ContextualDFL._extract_variable_bounds(lp)

            @test [row.original_row for row in bound_map.bound_rows] == [1, 2, 3]
            @test bound_map.general_rows == [4]
            @test bound_lp.lower_bounds == [1.0, -Inf]
            @test bound_lp.upper_bounds == [3.0, 5.0]
            @test bound_map.lower_owner == [2, 0]
            @test bound_map.upper_owner == [1, 3]
            @test bound_lp.A_ineq == reshape([1.0, 1.0], 1, 2)
            @test bound_lp.b_ineq == [7.0]

            sparse_lp = LP(
                A_ineq=SparseArrays.sparse(lp.A_ineq),
                b_ineq=lp.b_ineq,
                c=lp.c,
            )
            sparse_bound_lp, sparse_bound_map =
                ContextualDFL._extract_variable_bounds(sparse_lp)

            @test SparseArrays.issparse(sparse_bound_lp.A_ineq)
            @test sparse_bound_map.general_rows == [4]
            @test sparse_bound_lp.lower_bounds == bound_lp.lower_bounds
            @test sparse_bound_lp.upper_bounds == bound_lp.upper_bounds

            inconsistent_lp = LP(
                A_ineq=reshape([1.0, -1.0], 2, 1),
                b_ineq=[0.0, -1.0],
                c=[0.0],
            )
            @test_throws ArgumentError ContextualDFL._extract_variable_bounds(inconsistent_lp)
        end

        @testset "all singleton bounds" begin
            lp = LP(
                A_eq=ones(1, 2),
                b_eq=[1.0],
                A_ineq=[
                    1.0 0.0
                    -1.0 0.0
                    0.0 1.0
                    0.0 -1.0
                ],
                b_ineq=[0.8, 0.0, 0.9, 0.0],
                c=[1.0, 2.0],
            )

            assert_bound_aware_value_preserving(lp)
        end

        @testset "mixed bounds and residual rows" begin
            lp = LP(
                A_ineq=[
                    -1.0 0.0
                    0.0 -1.0
                    1.0 0.0
                    0.0 1.0
                    -1.0 -1.0
                ],
                b_ineq=[0.0, 0.0, 2.0, 2.0, -1.0],
                c=[1.0, 2.0],
            )

            assert_bound_aware_value_preserving(lp)
        end

        @testset "duplicate singleton rows" begin
            lp = LP(
                A_eq=ones(1, 2),
                b_eq=[1.0],
                A_ineq=[
                    -1.0 0.0
                    -2.0 0.0
                    0.0 -1.0
                    0.0 1.0
                    1.0 0.0
                ],
                b_ineq=[0.0, 0.0, 0.0, 1.0, 1.0],
                c=[2.0, 1.0],
            )

            assert_bound_aware_value_preserving(lp)
        end

        @testset "sparse extensive-form LP" begin
            program = StochasticProgram(
                A_eq=zeros(0, 1),
                A_ineq=reshape([-1.0, 1.0], 2, 1),
                b_eq=Float64[],
                b_ineq=[0.0, 2.0],
                c=[0.5],
            )
            W_eq_array = zeros(0, 1, 1)
            W_ineq_array = reshape([-1.0, 1.0], 2, 1, 1)
            T_eq_array = zeros(0, 1, 1)
            T_ineq_array = zeros(2, 1, 1)
            h_eq_array = zeros(0, 1)
            h_ineq_array = reshape([0.0, 2.0], 2, 1)
            q_array = reshape([1.0], 1, 1)

            lp = construct_lp(
                program,
                W_eq_array,
                W_ineq_array,
                T_eq_array,
                T_ineq_array,
                h_eq_array,
                h_ineq_array,
                q_array,
            )

            @test SparseArrays.issparse(lp.A_ineq)
            assert_bound_aware_value_preserving(lp)
        end
    end
end

# END FILE: src/ContextualDFL/ContextualDFL/test/linear_programming/extended_suite/runtests.jl

# BEGIN FILE: src/ContextualDFL/ContextualDFL/test/linear_programming/runtests.jl
@testset "linear_programming" begin
    include("test_helpers.jl")
    include("base_suite/runtests.jl")
    include("extended_suite/runtests.jl")
end

# END FILE: src/ContextualDFL/ContextualDFL/test/linear_programming/runtests.jl

# BEGIN FILE: src/ContextualDFL/ContextualDFL/test/linear_programming/test_helpers.jl
import HiGHS
import Ipopt
import JuMP
import LinearAlgebra: Diagonal, I, dot, norm
import SparseArrays

const TEST_SOLVER = Solver(IpoptSolver(), HiGHSSolver())
const TEST_HIGHS_SOLVER = HiGHSSolver()
const TEST_BARRIER_MUS = (1.0, 0.1)
# μ=0 derivative finite-difference checks are intentionally skipped for now:
# the LP solution map is nonsmooth at active-set changes, so these failures do
# not tell us anything useful about the smooth log-barrier derivative path.
const TEST_DERIVATIVE_MUS = (1.0, 0.1)

status_name(status) = string(status)
is_optimal_status(status) = status_name(status) in ("OPTIMAL", "LOCALLY_SOLVED")

function assert_feasible(lp, z; atol=1e-6)
    if !isempty(lp.A_eq)
        @test norm(lp.A_eq * z - lp.b_eq, Inf) ≤ atol
    end

    if !isempty(lp.A_ineq)
        @test maximum(lp.A_ineq * z - lp.b_ineq) ≤ atol
    end
end

function assert_barrier_stationarity(lp, z, μ; atol=5e-4)
    slack = lp.b_ineq - lp.A_ineq * z
    μ_vector = ContextualDFL._barrier_parameter_vector(length(slack), μ)
    stationarity = lp.c + transpose(lp.A_ineq) * (μ_vector ./ slack)

    if !isempty(lp.A_eq)
        λ = -(transpose(lp.A_eq) \ stationarity)
        stationarity += transpose(lp.A_eq) * λ
    end

    @test norm(stationarity, Inf) ≤ atol
end

function assert_lp_kkt(lp, result; atol=1e-6)
    dual_ineq = result.dual_ineq

    if !isempty(lp.A_ineq)
        @test minimum(dual_ineq) ≥ -atol
        @test norm(dual_ineq .* (lp.A_ineq * result.z - lp.b_ineq), Inf) ≤ atol
    end

    stationarity = copy(lp.c)
    isempty(lp.A_eq) || (stationarity .-= transpose(lp.A_eq) * result.dual_eq)
    isempty(lp.A_ineq) || (stationarity .+= transpose(lp.A_ineq) * dual_ineq)

    @test norm(stationarity, Inf) ≤ atol
end

function solve_row_explicit_highs(lp; constraint_tolerance=1e-6, kwargs...)
    model = JuMP.Model(HiGHS.Optimizer)
    JuMP.set_silent(model)
    for (attribute, value) in kwargs
        JuMP.set_optimizer_attribute(model, String(attribute), value)
    end

    n_variables = length(lp.c)
    JuMP.@variable(model, z[1:n_variables])
    eq_constraints = JuMP.@constraint(model, lp.A_eq * z .== lp.b_eq)
    ineq_constraints = JuMP.@constraint(model, lp.A_ineq * z .<= lp.b_ineq)
    JuMP.@objective(model, Min, sum(lp.c[j] * z[j] for j in 1:n_variables))
    JuMP.optimize!(model)

    status = ContextualDFL._assert_successful_solve(
        model,
        TEST_HIGHS_SOLVER;
        accepted_statuses=("OPTIMAL",),
    )
    z_value = JuMP.value.(z)
    ContextualDFL._assert_lp_solution_feasible(lp, z_value; atol=constraint_tolerance)

    return (;
        z=z_value,
        dual_eq=JuMP.dual.(eq_constraints),
        dual_ineq=-JuMP.dual.(ineq_constraints),
        objective_value=JuMP.objective_value(model),
        status=status,
    )
end

function solve_row_explicit_ipopt(
    lp,
    μ;
    slack_lower_bound=1e-9,
    constraint_tolerance=1e-6,
    kwargs...,
)
    μ_vector = ContextualDFL._barrier_parameter_vector(lp, μ)
    positive_barrier_indices = findall(!iszero, μ_vector)

    model = JuMP.Model(Ipopt.Optimizer)
    JuMP.set_optimizer_attribute(model, "print_level", 0)
    JuMP.set_optimizer_attribute(model, "sb", "yes")
    JuMP.set_optimizer_attribute(model, "mu_strategy", "monotone")
    JuMP.set_optimizer_attribute(model, "nlp_scaling_method", "none")
    for (attribute, value) in kwargs
        JuMP.set_optimizer_attribute(model, String(attribute), value)
    end

    n_variables = length(lp.c)
    n_inequalities = length(lp.b_ineq)
    JuMP.@variable(model, z[1:n_variables])
    JuMP.@variable(model, s[1:n_inequalities] >= 0)
    for i in positive_barrier_indices
        JuMP.set_lower_bound(s[i], slack_lower_bound)
    end

    eq_constraints = JuMP.@constraint(model, lp.A_eq * z .== lp.b_eq)
    slack_constraints = JuMP.@constraint(model, lp.A_ineq * z .+ s .== lp.b_ineq)
    JuMP.@NLobjective(
        model,
        Min,
        sum(lp.c[j] * z[j] for j in 1:n_variables) -
        sum(μ_vector[i] * log(s[i]) for i in positive_barrier_indices),
    )
    JuMP.optimize!(model)

    status = ContextualDFL._assert_successful_solve(
        model,
        IpoptSolver();
        accepted_statuses=("OPTIMAL", "LOCALLY_SOLVED"),
    )
    z_value = JuMP.value.(z)
    ContextualDFL._assert_lp_solution_feasible(lp, z_value; atol=constraint_tolerance)

    return (;
        z=z_value,
        slack=JuMP.value.(s),
        dual_eq=JuMP.dual.(eq_constraints),
        dual_ineq=-JuMP.dual.(slack_constraints),
        objective_value=JuMP.objective_value(model),
        status=status,
    )
end

function log_barrier_objective(lp, z, μ)
    μ_vector = ContextualDFL._barrier_parameter_vector(lp, μ)
    slack = lp.b_ineq - lp.A_ineq * z
    return dot(lp.c, z) - dot(μ_vector, log.(slack))
end

function assert_bound_aware_value_preserving(lp; μ=0.2, atol=5e-5)
    row_explicit_lp = solve_row_explicit_highs(lp)
    bound_aware_lp = solve(TEST_HIGHS_SOLVER, lp)

    @test bound_aware_lp.z ≈ row_explicit_lp.z atol = atol rtol = atol
    @test dot(lp.c, bound_aware_lp.z) ≈ dot(lp.c, row_explicit_lp.z) atol = atol rtol = atol
    @test bound_aware_lp.objective_value ≈ row_explicit_lp.objective_value atol = atol rtol = atol
    @test lp.b_ineq - lp.A_ineq * bound_aware_lp.z ≈
          lp.b_ineq - lp.A_ineq * row_explicit_lp.z atol = atol rtol = atol

    row_explicit_barrier =
        solve_row_explicit_ipopt(lp, μ; tol=1e-10, max_iter=1_000)
    bound_aware_barrier =
        solve(TEST_SOLVER, lp; μ=μ, tol=1e-10, max_iter=1_000)

    @test bound_aware_barrier.z ≈ row_explicit_barrier.z atol = atol rtol = atol
    @test bound_aware_barrier.slack ≈
          lp.b_ineq - lp.A_ineq * bound_aware_barrier.z atol = atol rtol = atol
    @test bound_aware_barrier.slack ≈ row_explicit_barrier.slack atol = atol rtol = atol
    @test log_barrier_objective(lp, bound_aware_barrier.z, μ) ≈
          log_barrier_objective(lp, row_explicit_barrier.z, μ) atol = atol rtol = atol
    @test bound_aware_barrier.objective_value ≈
          row_explicit_barrier.objective_value atol = atol rtol = atol
    @test bound_aware_barrier.dual_ineq ≈
          ContextualDFL._barrier_parameter_vector(lp, μ) ./ bound_aware_barrier.slack atol =
          5e-4 rtol = 5e-4
end

function assert_lp_case(case, solver=TEST_SOLVER)
    result = solve(solver, case.lp)

    @test status_name(result.status) == case.expected_status

    if case.expected_status == "OPTIMAL"
        assert_feasible(case.lp, result.z)
        @test result.objective_value ≈ dot(case.lp.c, result.z) atol = 1e-7
        assert_lp_kkt(case.lp, result)

        if haskey(case, :expected_z)
            @test result.z ≈ case.expected_z atol = 1e-7
        end
    end
end

function assert_lp_case_with_highs(case)
    @testset "LP solver strategy" begin
        assert_lp_case(case, TEST_SOLVER)
    end

    @testset "HiGHS direct" begin
        assert_lp_case(case, TEST_HIGHS_SOLVER)
    end
end

function solve_reference_z(lp, μ)
    μ_vector = ContextualDFL._barrier_parameter_vector(lp, μ)
    result = ContextualDFL._is_zero_barrier_parameter(μ_vector) ?
        solve(TEST_SOLVER, lp) :
        solve(TEST_SOLVER, lp; μ=μ_vector, tol=1e-10, max_iter=1_000)
    @test is_optimal_status(result.status)
    return result.z
end

function assert_log_barrier_case(case, μ)
    result = solve(TEST_SOLVER, case.lp; μ=μ, tol=1e-10, max_iter=1_000)

    @test is_optimal_status(result.status)
    assert_feasible(case.lp, result.z; atol=1e-6)
    @test minimum(case.lp.b_ineq - case.lp.A_ineq * result.z) > 1e-7
    assert_barrier_stationarity(case.lp, result.z, μ)

    return result
end

function deterministic_direction(length_value, scale, phase)
    return [scale * sin(i + phase) for i in 1:length_value]
end

function finite_difference_action(lp, μ, dc, db_eq, db_ineq)
    ε = 1e-3
    lp_plus = LP(
        A_eq=lp.A_eq,
        A_ineq=lp.A_ineq,
        b_eq=lp.b_eq + ε .* db_eq,
        b_ineq=lp.b_ineq + ε .* db_ineq,
        c=lp.c + ε .* dc,
    )
    lp_minus = LP(
        A_eq=lp.A_eq,
        A_ineq=lp.A_ineq,
        b_eq=lp.b_eq - ε .* db_eq,
        b_ineq=lp.b_ineq - ε .* db_ineq,
        c=lp.c - ε .* dc,
    )

    return (solve_reference_z(lp_plus, μ) - solve_reference_z(lp_minus, μ)) ./ (2ε)
end

function finite_difference_jacobian(lp, μ, component)
    n = length(lp.c)

    if component === :c
        J = zeros(n, n)
        for j in 1:n
            dc = zeros(n)
            dc[j] = 1.0
            J[:, j] = finite_difference_action(
                lp,
                μ,
                dc,
                zeros(length(lp.b_eq)),
                zeros(length(lp.b_ineq)),
            )
        end
        return J
    elseif component === :b_eq
        J = zeros(n, length(lp.b_eq))
        for j in 1:length(lp.b_eq)
            db_eq = zeros(length(lp.b_eq))
            db_eq[j] = 1.0
            J[:, j] = finite_difference_action(
                lp,
                μ,
                zeros(n),
                db_eq,
                zeros(length(lp.b_ineq)),
            )
        end
        return J
    elseif component === :b_ineq
        J = zeros(n, length(lp.b_ineq))
        for j in 1:length(lp.b_ineq)
            db_ineq = zeros(length(lp.b_ineq))
            db_ineq[j] = 1.0
            J[:, j] = finite_difference_action(
                lp,
                μ,
                zeros(n),
                zeros(length(lp.b_eq)),
                db_ineq,
            )
        end
        return J
    end

    throw(ArgumentError("Unknown derivative component: $component"))
end

function construct_jacobian(
    solver,
    lp::LP,
    μ;
    pre_computed=nothing,
    compute_J_c=true,
    compute_J_b_eq=true,
    compute_J_b_ineq=true,
    tight_tol=1e-7,
    kwargs...,
)
    cache = ContextualDFL._diff_precompute(solver, lp, μ, pre_computed, tight_tol; kwargs...)
    n = length(lp.c)
    m_eq = length(lp.b_eq)
    m_ineq = length(lp.b_ineq)
    μ_vector = cache.μ
    T = promote_type(eltype(cache.z), eltype(lp.c), eltype(μ_vector))

    J_c = nothing
    J_b_eq = nothing
    J_b_ineq = nothing

    if ContextualDFL._is_zero_barrier_parameter(μ_vector)
        if compute_J_c
            J_c = zeros(T, n, n)
        end

        if compute_J_b_eq
            rhs_b_eq = vcat(
                zeros(T, n, m_eq),
                Matrix{T}(I, m_eq, m_eq),
                zeros(T, length(cache.tight), m_eq),
            )
            J_b_eq = (cache.K_factorization \ rhs_b_eq)[1:n, :]
        end

        if compute_J_b_ineq
            top = zeros(T, n, m_ineq)
            if !isempty(cache.loose)
                top[:, cache.loose] = transpose(lp.A_ineq[cache.loose, :]) * Diagonal(cache.d)
            end

            bottom = zeros(T, length(cache.tight), m_ineq)
            for (row, index) in enumerate(cache.tight)
                bottom[row, index] = one(T)
            end

            rhs_b_ineq = vcat(top, zeros(T, m_eq, m_ineq), bottom)
            J_b_ineq = (cache.K_factorization \ rhs_b_ineq)[1:n, :]
        end
    else
        if compute_J_c
            rhs_c = vcat(Matrix{T}(I, n, n), zeros(T, m_eq, n))
            J_c = -(cache.K_factorization \ rhs_c)[1:n, :]
        end

        if compute_J_b_eq
            rhs_b_eq = vcat(zeros(T, n, m_eq), Matrix{T}(I, m_eq, m_eq))
            J_b_eq = (cache.K_factorization \ rhs_b_eq)[1:n, :]
        end

        if compute_J_b_ineq
            C = transpose(lp.A_ineq) * Diagonal(μ_vector .* cache.d)
            rhs_b_ineq = vcat(Matrix{T}(C), zeros(T, m_eq, m_ineq))
            J_b_ineq = (cache.K_factorization \ rhs_b_ineq)[1:n, :]
        end
    end

    return (;
        J_c=J_c,
        J_b_eq=J_b_eq,
        J_b_ineq=J_b_ineq,
        pre_computed=cache,
    )
end

function assert_diff_solve_column(lp, μ, jac, component, column)
    n = length(lp.c)
    dc = zeros(n)
    db_eq = zeros(length(lp.b_eq))
    db_ineq = zeros(length(lp.b_ineq))

    if component === :c
        dc[column] = 1.0
        expected = jac.J_c[:, column]
    elseif component === :b_eq
        db_eq[column] = 1.0
        expected = jac.J_b_eq[:, column]
    elseif component === :b_ineq
        db_ineq[column] = 1.0
        expected = jac.J_b_ineq[:, column]
    else
        throw(ArgumentError("Unknown derivative component: $component"))
    end

    actual = diff_solve(
        TEST_SOLVER,
        lp,
        μ;
        pre_computed=jac.pre_computed,
        dc=dc,
        db_eq=db_eq,
        db_ineq=db_ineq,
    )

    @test actual ≈ expected atol = 1e-8 rtol = 1e-8
end

function assert_diff_case(case, μ)
    lp = case.lp
    z = solve_reference_z(lp, μ)
    jac = construct_jacobian(TEST_SOLVER, lp, μ; pre_computed=z)
    fd_J_c = finite_difference_jacobian(lp, μ, :c)
    fd_J_b_eq = finite_difference_jacobian(lp, μ, :b_eq)
    fd_J_b_ineq = finite_difference_jacobian(lp, μ, :b_ineq)

    @test jac.J_c ≈ fd_J_c atol = 5e-4 rtol = 5e-3
    @test jac.J_b_eq ≈ fd_J_b_eq atol = 5e-4 rtol = 5e-3
    @test jac.J_b_ineq ≈ fd_J_b_ineq atol = 5e-4 rtol = 5e-3

    for j in 1:length(lp.c)
        assert_diff_solve_column(lp, μ, jac, :c, j)
    end

    for j in 1:length(lp.b_eq)
        assert_diff_solve_column(lp, μ, jac, :b_eq, j)
    end

    for j in 1:length(lp.b_ineq)
        assert_diff_solve_column(lp, μ, jac, :b_ineq, j)
    end
end

function run_smooth_case(case)
    @testset "$(case.name)" begin
        assert_lp_case_with_highs(case)

        for μ in TEST_BARRIER_MUS
            @testset "log-barrier μ=$(μ)" begin
                assert_log_barrier_case(case, μ)
            end
        end

        for μ in TEST_DERIVATIVE_MUS
            @testset "derivative μ=$(μ)" begin
                assert_diff_case(case, μ)
            end
        end
    end
end

function square_2d()
    return [1.0 0.0; -1.0 0.0; 0.0 1.0; 0.0 -1.0], ones(4)
end

function nonnegative_orthant(n)
    return -Matrix{Float64}(I, n, n), zeros(n)
end

function box_constraints(lower, upper)
    n = length(lower)
    return [Matrix{Float64}(I, n, n); -Matrix{Float64}(I, n, n)], [upper; -lower]
end

# END FILE: src/ContextualDFL/ContextualDFL/test/linear_programming/test_helpers.jl

# BEGIN FILE: src/ContextualDFL/ContextualDFL/test/loss_functions/runtests.jl
import Flux
import LinearAlgebra: dot

struct DflTestVectorDecoder <: VectorDecoder end

function (::DflTestVectorDecoder)(vector::AbstractVector)
    return (
        reshape(view(vector, 1:1), 1, 1),
        zeros(eltype(vector), 0, 1),
        reshape(view(vector, 2:2), 1, 1),
        zeros(eltype(vector), 0, 1),
        view(vector, 3:3),
        zeros(eltype(vector), 0),
        view(vector, 4:4),
    )
end

struct SPOPlusTestQDecoder <: VectorDecoder end

function (::SPOPlusTestQDecoder)(q::AbstractVector)
    return (
        zeros(eltype(q), 0, 1),
        reshape([-one(eltype(q)), one(eltype(q))], 2, 1),
        zeros(eltype(q), 0, 0),
        zeros(eltype(q), 2, 0),
        zeros(eltype(q), 0),
        [zero(eltype(q)), one(eltype(q))],
        q,
    )
end

function spo_plus_test_scenario(q)
    return ParametricScenario(;
        W_eq_xi=zeros(0, 1),
        W_ineq_xi=reshape([-1.0, 1.0], 2, 1),
        T_eq_xi=zeros(0, 0),
        T_ineq_xi=zeros(2, 0),
        h_eq_xi=Float64[],
        h_ineq_xi=[0.0, 1.0],
        q_xi=[q],
    )
end

@testset "loss_functions" begin
    input_decoder = DflTestVectorDecoder()
    reference_decoder = ParametricDecoder(
        (:h_eq,);
        base_W_eq=:base_W_eq,
        base_W_ineq=:base_W_ineq,
        base_T_eq=:base_T_eq,
        base_T_ineq=:base_T_ineq,
        base_h_ineq=:base_h_ineq,
        base_q=:base_q,
    )
    solver = Solver(IpoptSolver(), HiGHSSolver())
    program = StochasticProgram(c=[1.0])

    loss = DflScenLoss(input_decoder, reference_decoder, solver, program; nr_scenarios=2)

    @test loss.input_scenario_decoder === input_decoder
    @test loss.reference_scenario_decoder === reference_decoder
    @test loss.solver === solver
    @test loss.program === program
    @test loss.nr_scenarios == 2

    passthrough_decoder = ParametricDecoder()
    bounded_program = StochasticProgram(
        A_eq=zeros(0, 1),
        A_ineq=reshape([-1.0, 1.0], 2, 1),
        b_eq=Float64[],
        b_ineq=[0.0, 10.0],
        c=[0.0],
    )
    dfl_loss = DflScenLoss(input_decoder, passthrough_decoder, solver, bounded_program)

    input_scenario_parameter_collection = [1.0, 1.0, 5.0, 1.0]
    reference_scenario_parameter_collection = [
        ParametricScenario(;
            W_eq_xi=reshape([1.0], 1, 1),
            W_ineq_xi=reshape([1.0], 1, 1),
            T_eq_xi=reshape([1.0], 1, 1),
            T_ineq_xi=reshape([0.0], 1, 1),
            h_eq_xi=[20.0],
            h_ineq_xi=[30.0],
            q_xi=[2.0],
        ),
    ]

    @test dfl_loss(
        input_scenario_parameter_collection,
        reference_scenario_parameter_collection,
        0.0,
    ) ≈ 20.0

    positive_mu = 0.1
    default_reference_mu_loss = dfl_loss(
        input_scenario_parameter_collection,
        reference_scenario_parameter_collection,
        positive_mu,
    )
    explicit_reference_mu_loss = dfl_loss(
        input_scenario_parameter_collection,
        reference_scenario_parameter_collection,
        positive_mu,
        positive_mu,
    )
    zero_reference_mu_loss = dfl_loss(
        input_scenario_parameter_collection,
        reference_scenario_parameter_collection,
        positive_mu,
        0.0,
    )

    @test default_reference_mu_loss ≈ explicit_reference_mu_loss atol = 1e-7 rtol = 1e-7
    @test !isapprox(default_reference_mu_loss, zero_reference_mu_loss; atol=1e-4, rtol=1e-4)

    rho = 0.2
    default_reference_rho_loss = dfl_loss(
        input_scenario_parameter_collection,
        reference_scenario_parameter_collection,
        0.0,
        0.0;
        rho_in=rho,
        tol=1e-10,
    )
    explicit_reference_rho_loss = dfl_loss(
        input_scenario_parameter_collection,
        reference_scenario_parameter_collection,
        0.0,
        0.0;
        rho_in=rho,
        rho_ref=rho,
        tol=1e-10,
    )
    zero_reference_rho_loss = dfl_loss(
        input_scenario_parameter_collection,
        reference_scenario_parameter_collection,
        0.0,
        0.0;
        rho_in=rho,
        rho_ref=0.0,
    )

    @test default_reference_rho_loss ≈ explicit_reference_rho_loss atol = 1e-7 rtol = 1e-7
    @test !isapprox(default_reference_rho_loss, zero_reference_rho_loss; atol=1e-4, rtol=1e-4)

    rho_direction = [0.0, 0.0, 0.1, -0.05]
    rho_objective(v) = dfl_loss(
        v,
        reference_scenario_parameter_collection,
        0.0,
        0.0;
        rho_in=rho,
        rho_ref=rho,
        tol=1e-10,
    )
    rho_gradient = only(Flux.gradient(rho_objective, input_scenario_parameter_collection))
    ϵ = 1e-5
    rho_finite_difference = (
        rho_objective(input_scenario_parameter_collection .+ ϵ .* rho_direction) -
        rho_objective(input_scenario_parameter_collection .- ϵ .* rho_direction)
    ) / (2ϵ)

    @test dot(rho_gradient, rho_direction) ≈ rho_finite_difference atol = 1e-4 rtol = 1e-3

    @testset "SPOPlusLoss objective-vector surrogate" begin
        spo_program = StochasticProgram(
            A_eq=zeros(0, 0),
            A_ineq=zeros(0, 0),
            b_eq=Float64[],
            b_ineq=Float64[],
            c=Float64[],
        )
        q_decoder = SPOPlusTestQDecoder()
        spo_loss = SPOPlusLoss(q_decoder, ParametricDecoder(), solver, spo_program)

        @test spo_loss.input_scenario_decoder === q_decoder
        @test spo_loss.reference_scenario_decoder isa ParametricDecoder
        @test spo_loss.solver === solver
        @test spo_loss.program === spo_program
        @test spo_loss.nr_scenarios == 1
        @test_throws ArgumentError SPOPlusLoss(q_decoder, ParametricDecoder(), solver, spo_program; nr_scenarios=0)

        reference = [spo_plus_test_scenario(2.0)]
        prediction = [0.25]
        @test spo_loss(prediction, reference, 0.0) ≈ 1.5 atol = 1e-8
        @test spo_loss([2.0], reference, 0.0) ≈ 0.0 atol = 1e-8
        @test only(Flux.gradient(q -> spo_loss(q, reference, 0.0), prediction)) ≈ [-2.0] atol = 1e-8

        ϵ = 1e-6
        finite_difference_gradient =
            (
                spo_loss(prediction .+ ϵ, reference, 0.0) -
                spo_loss(prediction .- ϵ, reference, 0.0)
            ) / (2ϵ)
        @test finite_difference_gradient ≈ -2.0 atol = 1e-5
        @test_throws ArgumentError spo_loss(prediction, reference, 0.1)

        rho_spo = 4.0
        rho_spo_value = spo_loss(prediction, reference, 0.0; rho_in=rho_spo, tol=1e-10)
        explicit_rho_spo_value = spo_loss(
            prediction,
            reference,
            0.0;
            rho_in=rho_spo,
            rho_ref=rho_spo,
            tol=1e-10,
        )
        rho_spo_gradient = only(
            Flux.gradient(
                q -> spo_loss(q, reference, 0.0; rho_in=rho_spo, rho_ref=rho_spo, tol=1e-10),
                prediction,
            ),
        )
        rho_spo_finite_difference =
            (
                spo_loss(prediction .+ ϵ, reference, 0.0; rho_in=rho_spo, rho_ref=rho_spo, tol=1e-10) -
                spo_loss(prediction .- ϵ, reference, 0.0; rho_in=rho_spo, rho_ref=rho_spo, tol=1e-10)
            ) / (2ϵ)

        @test rho_spo_value ≈ explicit_rho_spo_value atol = 1e-8
        @test rho_spo_gradient[1] ≈ rho_spo_finite_difference atol = 1e-5 rtol = 1e-5

        two_scenario_loss =
            SPOPlusLoss(q_decoder, ParametricDecoder(), solver, spo_program; nr_scenarios=2)
        two_reference = [spo_plus_test_scenario(2.0), spo_plus_test_scenario(-3.0)]
        two_prediction = [0.25, -0.5]
        probabilities = [0.25, 0.75]

        @test two_scenario_loss(
            two_prediction,
            two_reference,
            0.0;
            probabilities=probabilities,
        ) ≈ 1.875 atol = 1e-8
        @test only(
            Flux.gradient(
                q -> two_scenario_loss(q, two_reference, 0.0; probabilities=probabilities),
                two_prediction,
            ),
        ) ≈ [-0.5, 1.5] atol = 1e-8

        mismatched_feasible_loss =
            SPOPlusLoss(input_decoder, passthrough_decoder, solver, bounded_program)
        @test_throws DimensionMismatch mismatched_feasible_loss(
            input_scenario_parameter_collection,
            reference_scenario_parameter_collection,
            0.0,
        )
    end
end

# END FILE: src/ContextualDFL/ContextualDFL/test/loss_functions/runtests.jl

# BEGIN FILE: src/ContextualDFL/ContextualDFL/test/resource_allocation_training/resource_allocation_instance.jl
import ChainRulesCore
import Distributions
import Flux
import LinearAlgebra
import Random
import Statistics

module ResourceAllocationLegacyParameters
include(
    normpath(
        joinpath(
            @__DIR__,
            "..",
            "..",
            "..",
            "..",
            "ProblemBasedScenarioGeneration",
            "src",
            "problem_instances",
            "resource_allocation",
            "parameters.jl",
        ),
    ),
)
end

struct ResourceAllocationProblemData
    service_rate_parameters::Matrix{Float64}
    first_stage_costs::Vector{Float64}
    second_stage_costs::Vector{Float64}
    yield_parameters::Vector{Float64}
end

struct ResourceAllocationTestInstance
    problem_data::ResourceAllocationProblemData
    legacy_first_stage::NamedTuple
    stochastic_program::StochasticProgram
    base_scenario::NamedTuple
end

struct ResourceAllocationDemandDecoder{TB} <: ScenarioDecoder
    base_scenario::TB
    resource_count::Int
    demand_count::Int
end

function imported_resource_allocation_data()
    data = ResourceAllocationProblemData(
        Matrix{Float64}(ResourceAllocationLegacyParameters.μᵢⱼ),
        vec(Float64.(ResourceAllocationLegacyParameters.cz)),
        vec(Float64.(ResourceAllocationLegacyParameters.qw)),
        vec(Float64.(ResourceAllocationLegacyParameters.ρᵢ)),
    )

    resource_count, demand_count = size(data.service_rate_parameters)
    length(data.first_stage_costs) == resource_count ||
        throw(DimensionMismatch("first-stage costs must match resource count."))
    length(data.second_stage_costs) == demand_count ||
        throw(DimensionMismatch("second-stage costs must match demand count."))
    length(data.yield_parameters) == resource_count ||
        throw(DimensionMismatch("yield parameters must match resource count."))

    return data
end

function resource_allocation_instance(; resource_indices=:, demand_indices=:)
    imported = imported_resource_allocation_data()
    service_rates = Matrix{Float64}(imported.service_rate_parameters[resource_indices, demand_indices])
    first_costs = vec(Float64.(imported.first_stage_costs[resource_indices]))
    second_costs = vec(Float64.(imported.second_stage_costs[demand_indices]))
    yields = vec(Float64.(imported.yield_parameters[resource_indices]))
    data = ResourceAllocationProblemData(service_rates, first_costs, second_costs, yields)

    resource_count, demand_count = size(service_rates)
    recourse_variables = demand_count + resource_count * demand_count + resource_count + demand_count
    recourse_rows = resource_count + demand_count

    W_eq = zeros(Float64, recourse_rows, recourse_variables)
    for resource_index in 1:resource_count
        for demand_index in 1:demand_count
            allocation_index = demand_count + demand_count * (resource_index - 1) + demand_index
            W_eq[resource_index, allocation_index] = 1.0
        end
        W_eq[resource_index, demand_count + resource_count * demand_count + resource_index] = 1.0
    end

    for demand_index in 1:demand_count
        row = resource_count + demand_index
        W_eq[row, demand_index] = 1.0
        for resource_index in 1:resource_count
            allocation_index = demand_count + demand_count * (resource_index - 1) + demand_index
            W_eq[row, allocation_index] = service_rates[resource_index, demand_index]
        end
        slack_index = demand_count + resource_count * demand_count + resource_count + demand_index
        W_eq[row, slack_index] = -1.0
    end

    T_eq = zeros(Float64, recourse_rows, resource_count)
    for resource_index in 1:resource_count
        T_eq[resource_index, resource_index] = -yields[resource_index]
    end

    q = zeros(Float64, recourse_variables)
    q[1:demand_count] .= second_costs

    first_stage_nonnegativity = -Matrix{Float64}(LinearAlgebra.I, resource_count, resource_count)
    recourse_nonnegativity = -Matrix{Float64}(LinearAlgebra.I, recourse_variables, recourse_variables)

    program = StochasticProgram(
        A_eq=zeros(Float64, 0, resource_count),
        A_ineq=first_stage_nonnegativity,
        b_eq=Float64[],
        b_ineq=zeros(Float64, resource_count),
        c=first_costs,
    )

    base_scenario = (;
        W_eq=W_eq,
        W_ineq=recourse_nonnegativity,
        T_eq=T_eq,
        T_ineq=zeros(Float64, recourse_variables, resource_count),
        h_ineq=zeros(Float64, recourse_variables),
        q=q,
    )

    legacy_first_stage = (;
        A=zeros(Float64, 1, resource_count),
        b=[0.0],
        c=first_costs,
    )

    return ResourceAllocationTestInstance(data, legacy_first_stage, program, base_scenario)
end

ResourceAllocationDemandDecoder(instance::ResourceAllocationTestInstance) =
    ResourceAllocationDemandDecoder(
        instance.base_scenario,
        size(instance.problem_data.service_rate_parameters, 1),
        size(instance.problem_data.service_rate_parameters, 2),
    )

function (decoder::ResourceAllocationDemandDecoder)(scenario_parameter)
    raw = _resource_allocation_demand_or_rhs(decoder, scenario_parameter)
    h_eq = if length(raw) == decoder.demand_count
        vcat(zeros(eltype(raw), decoder.resource_count), raw)
    elseif length(raw) == decoder.resource_count + decoder.demand_count
        raw
    else
        throw(
            DimensionMismatch(
                "resource allocation scenario parameter has length $(length(raw)); " *
                "expected $(decoder.demand_count) or $(decoder.resource_count + decoder.demand_count).",
            ),
        )
    end

    return (
        decoder.base_scenario.W_eq,
        decoder.base_scenario.W_ineq,
        decoder.base_scenario.T_eq,
        decoder.base_scenario.T_ineq,
        h_eq,
        decoder.base_scenario.h_ineq,
        decoder.base_scenario.q,
    )
end

function _resource_allocation_demand_or_rhs(decoder, scenario_parameter)
    value = if scenario_parameter isa AbstractVector
        scenario_parameter
    elseif hasproperty(scenario_parameter, :h_eq)
        getproperty(scenario_parameter, :h_eq)
    elseif hasproperty(scenario_parameter, :h)
        getproperty(scenario_parameter, :h)
    else
        throw(ArgumentError("scenario parameter must be a demand vector or have field `h_eq`/`h`."))
    end

    return vec(value)
end

function demand_parameter_collection(demand_matrix::AbstractMatrix)
    return [(; h_eq=view(demand_matrix, :, k)) for k in axes(demand_matrix, 2)]
end

function demand_matrix(scenario_collection)
    return hcat((_resource_allocation_demand_or_rhs(nothing, scenario) for scenario in scenario_collection)...)
end

function decoded_resource_allocation_arrays(
    decoder::ResourceAllocationDemandDecoder,
    scenario_collection,
)
    return decode_scenario_collection(decoder, scenario_collection)
end

function ChainRulesCore.rrule(
    ::typeof(decode_scenario_collection),
    decoder::ResourceAllocationDemandDecoder,
    scenario_parameter_collection::AbstractVector,
)
    output = decode_scenario_collection(decoder, scenario_parameter_collection)

    function resource_allocation_decode_pullback(output_tangent)
        dh_eq_array = ContextualDFL._array_cotangent(
            output_tangent,
            5,
            output[5];
            name=:h_eq_array,
        )

        scenario_parameter_tangents = map(enumerate(scenario_parameter_collection)) do (k, scenario_parameter)
            raw = _resource_allocation_demand_or_rhs(decoder, scenario_parameter)
            h_tangent = if length(raw) == decoder.demand_count
                view(dh_eq_array, (decoder.resource_count + 1):(decoder.resource_count + decoder.demand_count), k)
            else
                view(dh_eq_array, :, k)
            end

            if scenario_parameter isa AbstractVector
                return ChainRulesCore.ProjectTo(scenario_parameter)(h_tangent)
            end

            names = propertynames(scenario_parameter)
            values = map(names) do name
                name in (:h_eq, :h) ? h_tangent : ChainRulesCore.NoTangent()
            end
            return NamedTuple{names}(values)
        end

        return (
            ChainRulesCore.NoTangent(),
            ChainRulesCore.NoTangent(),
            scenario_parameter_tangents,
        )
    end

    return output, resource_allocation_decode_pullback
end

function generate_random_correlation_matrix(rng::Random.AbstractRNG, dimension::Int)
    beta_parameter = 2.0
    partial_correlation = zeros(Float64, dimension, dimension)
    correlation = Matrix{Float64}(LinearAlgebra.I, dimension, dimension)

    for k in 1:(dimension - 1)
        for i in (k + 1):dimension
            partial_correlation[k, i] = (rand(rng, Distributions.Beta(beta_parameter, beta_parameter)) - 0.5) * 2.0
            rho = partial_correlation[k, i]
            for j in (k - 1):-1:1
                rho =
                    rho *
                    sqrt((1 - partial_correlation[j, i]^2) * (1 - partial_correlation[j, k]^2)) +
                    partial_correlation[j, i] * partial_correlation[j, k]
            end
            correlation[k, i] = rho
            correlation[i, k] = rho
        end
    end

    permutation = Random.randperm(rng, dimension)
    return correlation[permutation, permutation]
end

function sample_resource_allocation_demand_parameters(
    rng::Random.AbstractRNG,
    demand_count::Int,
)
    intercept = 50 .+ 5 .* rand(rng, Distributions.Normal(0, 1), demand_count)
    B1 = 10 .+ rand(rng, Distributions.Uniform(-4, 4), demand_count)
    B2 = 5 .+ rand(rng, Distributions.Uniform(-4, 4), demand_count)
    B3 = 2 .+ rand(rng, Distributions.Uniform(-4, 4), demand_count)
    return intercept, hcat(B1, B2, B3)
end

function generate_resource_allocation_context_scenarios(
    instance::ResourceAllocationTestInstance;
    n_contexts::Int,
    n_scenarios::Int,
    sigma::Real,
    p::Real,
    L::Int,
    rng::Random.AbstractRNG=Random.default_rng(),
)
    L <= 3 || throw(ArgumentError("The legacy resource allocation generator has three context terms."))

    demand_count = size(instance.problem_data.service_rate_parameters, 2)
    correlation = generate_random_correlation_matrix(rng, 3)
    distribution = Distributions.MvNormal(zeros(3), LinearAlgebra.Symmetric(correlation + 1e-8LinearAlgebra.I))
    x_array = abs.(rand(rng, distribution, n_contexts))
    intercept, slopes = sample_resource_allocation_demand_parameters(rng, demand_count)

    scenario_collections = Vector{Vector{NamedTuple}}(undef, n_contexts)
    for context_index in 1:n_contexts
        collection = NamedTuple[]
        context = view(x_array, :, context_index)
        for _ in 1:n_scenarios
            demand = zeros(Float64, demand_count)
            for demand_index in 1:demand_count
                signal = intercept[demand_index]
                for term in 1:L
                    signal += slopes[demand_index, term] * context[term]^p
                end
                demand[demand_index] = signal + rand(rng, Distributions.Normal(0, sigma))
            end
            push!(collection, (; h_eq=demand))
        end
        scenario_collections[context_index] = collection
    end

    data = [
        (copy(view(x_array, :, context_index)), scenario_collections[context_index])
        for context_index in 1:n_contexts
    ]

    return (;
        x_array=x_array,
        scenario_collections=scenario_collections,
        data=data,
        demand_intercepts=intercept,
        demand_slopes=slopes,
        correlation_matrix=correlation,
    )
end

function construct_resource_allocation_neural_net(
    instance::ResourceAllocationTestInstance;
    n_scenarios::Int=1,
)
    demand_count = size(instance.problem_data.service_rate_parameters, 2)
    output_dim = demand_count * n_scenarios
    return Flux.Chain(
        Flux.Dense(3, 128, Flux.relu),
        Flux.Dense(128, 128, Flux.relu),
        Flux.Dense(128, 128, Flux.relu),
        Flux.Dense(128, output_dim, Flux.relu),
        x -> reshape(x, demand_count, n_scenarios),
    ) |> Flux.f64
end

function resource_allocation_training_loss(
    predicted_demands,
    reference_collection,
    mu_in=0.0,
    mu_ref=0.0;
    kwargs...,
)
    target = ChainRulesCore.ignore_derivatives() do
        demand_matrix(reference_collection)
    end
    size(predicted_demands) == size(target) ||
        throw(DimensionMismatch("predicted demand matrix and target matrix have different sizes."))
    return Statistics.mean(abs2, predicted_demands .- target)
end

function relative_resource_allocation_training_loss(
    predicted_demands,
    reference_collection,
    mu_in=0.0,
    mu_ref=0.0;
    kwargs...,
)
    target = ChainRulesCore.ignore_derivatives() do
        demand_matrix(reference_collection)
    end
    denominator = max(Statistics.mean(abs2, target), eps(Float64))
    return resource_allocation_training_loss(predicted_demands, reference_collection) / denominator
end

function mean_resource_allocation_training_loss(model, data)
    return Statistics.mean(resource_allocation_training_loss(model(x), scenarios) for (x, scenarios) in data)
end

function resource_allocation_scenario_arrays(instance::ResourceAllocationTestInstance, scenario_collection)
    decoder = ResourceAllocationDemandDecoder(instance)
    return decoded_resource_allocation_arrays(decoder, scenario_collection)
end

function status_is_optimal(status)
    return string(status) in ("OPTIMAL", "LOCALLY_SOLVED")
end

function assert_resource_allocation_feasible(lp::LP, z; atol=1e-6)
    isempty(lp.b_eq) || @test LinearAlgebra.norm(lp.A_eq * z - lp.b_eq, Inf) <= atol
    isempty(lp.b_ineq) || @test maximum(lp.A_ineq * z - lp.b_ineq) <= atol
end

function deterministic_resource_allocation_direction(shape; scale=1.0, phase=0.0)
    values = [scale * sin(index + phase) for index in 1:prod(shape)]
    return reshape(values, shape)
end

# END FILE: src/ContextualDFL/ContextualDFL/test/resource_allocation_training/resource_allocation_instance.jl

# BEGIN FILE: src/ContextualDFL/ContextualDFL/test/resource_allocation_training/runtests.jl
using ChainRulesCore
using ContextualDFL
using Flux
using LinearAlgebra
using Random
using Statistics
using Test

include("resource_allocation_instance.jl")

@testset "resource_allocation_training" begin
    @testset "instance import, decoder, and data generation" begin
        imported = imported_resource_allocation_data()

        @test size(imported.service_rate_parameters) == (20, 30)
        @test length(imported.first_stage_costs) == 20
        @test length(imported.second_stage_costs) == 30
        @test length(imported.yield_parameters) == 20
        @test all(>=(0.0), imported.service_rate_parameters)
        @test all(>(0.0), imported.first_stage_costs)
        @test all(>(0.0), imported.second_stage_costs)

        instance = resource_allocation_instance()
        resource_count, demand_count = size(instance.problem_data.service_rate_parameters)
        recourse_variables = demand_count + resource_count * demand_count + resource_count + demand_count
        recourse_rows = resource_count + demand_count

        @test instance.legacy_first_stage.A == zeros(1, resource_count)
        @test instance.legacy_first_stage.b == [0.0]
        @test instance.legacy_first_stage.c == instance.problem_data.first_stage_costs
        @test size(instance.stochastic_program.A_ineq) == (resource_count, resource_count)
        @test instance.stochastic_program.A_ineq == -Matrix{Float64}(I, resource_count, resource_count)
        @test instance.stochastic_program.c == instance.problem_data.first_stage_costs

        @test size(instance.base_scenario.W_eq) == (recourse_rows, recourse_variables)
        @test size(instance.base_scenario.T_eq) == (recourse_rows, resource_count)
        @test size(instance.base_scenario.W_ineq) == (recourse_variables, recourse_variables)
        @test instance.base_scenario.W_ineq == -Matrix{Float64}(I, recourse_variables, recourse_variables)
        @test instance.base_scenario.q[1:demand_count] == instance.problem_data.second_stage_costs
        @test all(iszero, instance.base_scenario.q[(demand_count + 1):end])

        decoder = ResourceAllocationDemandDecoder(instance)
        demand = collect(1.0:demand_count)
        W_eq, W_ineq, T_eq, T_ineq, h_eq, h_ineq, q = decoder((; h_eq=demand))

        @test W_eq === instance.base_scenario.W_eq
        @test W_ineq === instance.base_scenario.W_ineq
        @test T_eq === instance.base_scenario.T_eq
        @test T_ineq === instance.base_scenario.T_ineq
        @test h_eq[1:resource_count] == zeros(resource_count)
        @test h_eq[(resource_count + 1):end] == demand
        @test h_ineq === instance.base_scenario.h_ineq
        @test q === instance.base_scenario.q

        generated = generate_resource_allocation_context_scenarios(
            instance;
            n_contexts=5,
            n_scenarios=3,
            sigma=1.0,
            p=1.0,
            L=3,
            rng=Random.MersenneTwister(7),
        )

        @test size(generated.x_array) == (3, 5)
        @test length(generated.scenario_collections) == 5
        @test length(generated.data) == 5
        for scenario_collection in generated.scenario_collections
            @test length(scenario_collection) == 3
            for scenario in scenario_collection
                @test propertynames(scenario) == (:h_eq,)
                @test length(scenario.h_eq) == demand_count
                @test all(isfinite, scenario.h_eq)
            end
        end

        arrays = decoded_resource_allocation_arrays(decoder, generated.scenario_collections[1])
        @test size(arrays[1]) == (recourse_rows, recourse_variables, 3)
        @test size(arrays[3]) == (recourse_rows, resource_count, 3)
        @test size(arrays[5]) == (recourse_rows, 3)
        @test arrays[5][1:resource_count, :] == zeros(resource_count, 3)
        @test arrays[5][(resource_count + 1):end, 1] == generated.scenario_collections[1][1].h_eq

        _, pullback = ChainRulesCore.rrule(
            decode_scenario_collection,
            decoder,
            generated.scenario_collections[1],
        )
        dh_eq = ones(recourse_rows, 3)
        tangents = pullback((
            zeros(size(arrays[1])),
            zeros(size(arrays[2])),
            zeros(size(arrays[3])),
            zeros(size(arrays[4])),
            dh_eq,
            zeros(size(arrays[6])),
            zeros(size(arrays[7])),
        ))
        @test tangents[3][1].h_eq == ones(demand_count)
        @test tangents[3][2].h_eq == ones(demand_count)

        vector_collection = [collect(scenario.h_eq) for scenario in generated.scenario_collections[1]]
        vector_arrays = decoded_resource_allocation_arrays(decoder, vector_collection)
        _, vector_pullback = ChainRulesCore.rrule(
            decode_scenario_collection,
            decoder,
            vector_collection,
        )
        vector_dh_eq = reshape(collect(1.0:(recourse_rows * 3)), recourse_rows, 3)
        vector_tangents = vector_pullback(ntuple(
            index -> index == 5 ? vector_dh_eq : zeros(size(vector_arrays[index])),
            length(vector_arrays),
        ))
        @test vector_tangents[3][1] == vector_dh_eq[(resource_count + 1):end, 1]
        @test vector_tangents[3][3] == vector_dh_eq[(resource_count + 1):end, 3]

        full_rhs_collection = [
            vcat(fill(-Float64(k), resource_count), collect(scenario.h_eq))
            for (k, scenario) in enumerate(generated.scenario_collections[1])
        ]
        full_rhs_arrays = decoded_resource_allocation_arrays(decoder, full_rhs_collection)
        _, full_rhs_pullback = ChainRulesCore.rrule(
            decode_scenario_collection,
            decoder,
            full_rhs_collection,
        )
        full_rhs_dh_eq = reshape(collect(101.0:(100.0 + recourse_rows * 3)), recourse_rows, 3)
        full_rhs_tangents = full_rhs_pullback(ntuple(
            index -> index == 5 ? full_rhs_dh_eq : zeros(size(full_rhs_arrays[index])),
            length(full_rhs_arrays),
        ))
        @test full_rhs_tangents[3][1] == full_rhs_dh_eq[:, 1]
        @test full_rhs_tangents[3][2] == full_rhs_dh_eq[:, 2]

        malformed_cotangent = ntuple(
            index -> index == 5 ? "not an array cotangent" : zeros(size(arrays[index])),
            length(arrays),
        )
        @test_throws ArgumentError pullback(malformed_cotangent)
    end

    @testset "resource allocation LP, stochastic solve, and cost" begin
        instance = resource_allocation_instance(resource_indices=1:4, demand_indices=1:5)
        solver = Solver(IpoptSolver(), HiGHSSolver())
        generated = generate_resource_allocation_context_scenarios(
            instance;
            n_contexts=2,
            n_scenarios=2,
            sigma=0.25,
            p=1.0,
            L=3,
            rng=Random.MersenneTwister(11),
        )
        arrays = resource_allocation_scenario_arrays(instance, generated.scenario_collections[1])
        W_eq, W_ineq, T_eq, T_ineq, h_eq, h_ineq, q = arrays
        probabilities = [0.4, 0.6]

        lp = construct_lp(
            instance.stochastic_program,
            W_eq,
            W_ineq,
            T_eq,
            T_ineq,
            h_eq,
            h_ineq,
            q;
            probabilities=probabilities,
        )
        lp_result = solve(HiGHSSolver(), lp)

        @test status_is_optimal(lp_result.status)
        assert_resource_allocation_feasible(lp, lp_result.z)
        @test lp_result.objective_value ≈ dot(lp.c, lp_result.z) atol = 1e-7
        @test minimum(lp_result.z) >= -1e-7

        z, y, λ_b_eq, λ_b_ineq, λ_h_eq, λ_h_ineq = solve(
            solver,
            instance.stochastic_program,
            W_eq,
            W_ineq,
            T_eq,
            T_ineq,
            h_eq,
            h_ineq,
            q;
            probabilities=probabilities,
        )

        @test length(z) == size(instance.problem_data.service_rate_parameters, 1)
        @test size(y) == (size(q, 1), size(q, 2))
        @test isempty(λ_b_eq)
        @test length(λ_b_ineq) == length(instance.stochastic_program.b_ineq)
        @test size(λ_h_eq) == size(h_eq)
        @test size(λ_h_ineq) == size(h_ineq)
        @test minimum(z) >= -1e-7
        @test minimum(y) >= -1e-7

        value = cost_function(
            instance.stochastic_program,
            solver,
            z,
            W_eq,
            W_ineq,
            T_eq,
            T_ineq,
            h_eq,
            h_ineq,
            q;
            probabilities=probabilities,
        )
        @test value ≈ lp_result.objective_value atol = 1e-6
    end

    @testset "resource allocation differentiation and rrules" begin
        instance = resource_allocation_instance(resource_indices=1:3, demand_indices=1:4)
        solver = Solver(IpoptSolver(), HiGHSSolver())
        generated = generate_resource_allocation_context_scenarios(
            instance;
            n_contexts=1,
            n_scenarios=1,
            sigma=0.1,
            p=1.0,
            L=3,
            rng=Random.MersenneTwister(13),
        )
        W_eq, W_ineq, T_eq, T_ineq, h_eq, h_ineq, q =
            resource_allocation_scenario_arrays(instance, generated.scenario_collections[1])
        program = instance.stochastic_program
        μ = 0.25
        ρ = 0.1

        lp = construct_lp(program, W_eq, W_ineq, T_eq, T_ineq, h_eq, h_ineq, q)
        base_result = solve(solver, lp; μ=μ, ρ=ρ, tol=1e-9)
        @test status_is_optimal(base_result.status)
        assert_resource_allocation_feasible(lp, base_result.z; atol=1e-5)

        dc = vec(deterministic_resource_allocation_direction((length(lp.c),); scale=0.02, phase=0.1))
        db_eq = vec(deterministic_resource_allocation_direction((length(lp.b_eq),); scale=0.02, phase=0.3))
        db_ineq = vec(deterministic_resource_allocation_direction((length(lp.b_ineq),); scale=0.02, phase=0.5))
        dz = diff_solve(
            solver,
            lp,
            μ;
            ρ=ρ,
            pre_computed=base_result,
            dc=dc,
            db_eq=db_eq,
            db_ineq=db_ineq,
            tol=1e-9,
        )

        ϵ = 1e-4
        lp_plus = LP(
            A_eq=lp.A_eq,
            A_ineq=lp.A_ineq,
            b_eq=lp.b_eq + ϵ .* db_eq,
            b_ineq=lp.b_ineq + ϵ .* db_ineq,
            c=lp.c + ϵ .* dc,
        )
        lp_minus = LP(
            A_eq=lp.A_eq,
            A_ineq=lp.A_ineq,
            b_eq=lp.b_eq - ϵ .* db_eq,
            b_ineq=lp.b_ineq - ϵ .* db_ineq,
            c=lp.c - ϵ .* dc,
        )
        finite_difference_dz =
            (solve(solver, lp_plus; μ=μ, ρ=ρ, tol=1e-9).z - solve(solver, lp_minus; μ=μ, ρ=ρ, tol=1e-9).z) ./ (2ϵ)
        @test dz ≈ finite_difference_dz atol = 2e-3 rtol = 2e-2

        z_for_cost = fill(15.0, length(program.c))
        value, cost_pullback = ChainRulesCore.rrule(
            cost_function,
            program,
            solver,
            z_for_cost,
            W_eq,
            W_ineq,
            T_eq,
            T_ineq,
            h_eq,
            h_ineq,
            q;
            μ=μ,
            ρ=ρ,
            tol=1e-9,
        )
        dz_cost = cost_pullback(1.0)[4]
        direction = vec(deterministic_resource_allocation_direction(size(z_for_cost); scale=0.1, phase=0.7))
        finite_difference_cost = (
            cost_function(program, solver, z_for_cost + ϵ .* direction, W_eq, W_ineq, T_eq, T_ineq, h_eq, h_ineq, q; μ=μ, ρ=ρ, tol=1e-9) -
            cost_function(program, solver, z_for_cost - ϵ .* direction, W_eq, W_ineq, T_eq, T_ineq, h_eq, h_ineq, q; μ=μ, ρ=ρ, tol=1e-9)
        ) / (2ϵ)

        @test value isa Number
        @test dot(dz_cost, direction) ≈ finite_difference_cost atol = 2e-3 rtol = 2e-2

        output, solve_pullback = ChainRulesCore.rrule(
            solve,
            solver,
            program,
            W_eq,
            W_ineq,
            T_eq,
            T_ineq,
            h_eq,
            h_ineq,
            q;
            μ=μ,
            ρ=ρ,
            tol=1e-9,
        )
        dy_tangent = deterministic_resource_allocation_direction(size(output[2]); scale=0.03, phase=0.2)
        solve_tangents = solve_pullback((
            zeros(size(output[1])),
            dy_tangent,
            zeros(size(output[3])),
            zeros(size(output[4])),
            zeros(size(output[5])),
            zeros(size(output[6])),
        ))
        dh_eq_tangent = solve_tangents[8]
        h_direction = zeros(size(h_eq))
        resource_count = size(instance.problem_data.service_rate_parameters, 1)
        h_direction[(resource_count + 1):end, :] .=
            deterministic_resource_allocation_direction(
                size(view(h_direction, (resource_count + 1):size(h_direction, 1), :));
                scale=0.1,
                phase=1.3,
            )

        function solve_scalar(h_eq_candidate)
            candidate_output = solve(
                solver,
                program,
                W_eq,
                W_ineq,
                T_eq,
                T_ineq,
                h_eq_candidate,
                h_ineq,
                q;
                μ=μ,
                ρ=ρ,
                tol=1e-9,
            )
            return sum(candidate_output[2] .* dy_tangent)
        end

        finite_difference_solve =
            (solve_scalar(h_eq + ϵ .* h_direction) - solve_scalar(h_eq - ϵ .* h_direction)) / (2ϵ)
        ad_dh_eq = only(Flux.gradient(solve_scalar, h_eq))
        @test abs(finite_difference_solve) > 1e-8
        @test size(ad_dh_eq) == size(h_eq)
        @test sum(ad_dh_eq .* h_direction) ≈ finite_difference_solve atol = 3e-3 rtol = 3e-2
        @test sum(dh_eq_tangent .* h_direction) ≈ finite_difference_solve atol = 3e-3 rtol = 3e-2
    end

    @testset "train! loop learns resource allocation demand scenarios" begin
        Random.seed!(23)
        instance = resource_allocation_instance()
        n_scenarios = 2
        generated = generate_resource_allocation_context_scenarios(
            instance;
            n_contexts=24,
            n_scenarios=n_scenarios,
            sigma=0.5,
            p=1.0,
            L=3,
            rng=Random.MersenneTwister(23),
        )
        model = construct_resource_allocation_neural_net(instance; n_scenarios=n_scenarios)
        initial_loss = mean_resource_allocation_training_loss(model, generated.data)

        result = train!(
            model,
            resource_allocation_training_loss,
            relative_resource_allocation_training_loss,
            fill(0.0, 12),
            generated.data;
            opt=Flux.Adam(1e-3),
            epochs=12,
            batchsize=4,
            display_iterations=true,
            display_plot=false,
            shuffle=true,
            rng=Random.MersenneTwister(29),
        )

        final_loss = mean_resource_allocation_training_loss(model, generated.data)
        history_losses = [row.loss for row in result.history]
        history_display_losses = [row.display_loss for row in result.history]

        @test length(result.history) == 12
        @test all(isfinite, history_losses)
        @test all(isfinite, history_display_losses)
        @test final_loss < initial_loss
        @test last(history_display_losses) < first(history_display_losses)
        @test minimum(history_losses) <= first(history_losses)
    end
end

# END FILE: src/ContextualDFL/ContextualDFL/test/resource_allocation_training/runtests.jl

# BEGIN FILE: src/ContextualDFL/ContextualDFL/test/runtests.jl
using ContextualDFL
using Test

@testset "ContextualDFL" begin
    include("linear_programming/runtests.jl")
    include("learning/runtests.jl")
    include("loss_functions/runtests.jl")
    include("scenario_decoders/runtests.jl")
    include("stochastic_programming/runtests.jl")
    include("resource_allocation_training/runtests.jl")
    include("implementations/transshipment/runtests.jl")
end

# END FILE: src/ContextualDFL/ContextualDFL/test/runtests.jl

# BEGIN FILE: src/ContextualDFL/ContextualDFL/test/scenario_decoders/runtests.jl
import ChainRulesCore
import Flux

struct TestScenarioDecoder <: ScenarioDecoder end

struct TestVectorDecoder <: VectorDecoder end

function (::TestVectorDecoder)(vector::AbstractVector)
    return (
        reshape(view(vector, 1:1), 1, 1),
        zeros(eltype(vector), 0, 1),
        reshape(view(vector, 2:2), 1, 1),
        zeros(eltype(vector), 0, 1),
        view(vector, 3:3),
        zeros(eltype(vector), 0),
        view(vector, 4:4),
    )
end

@testset "scenario_decoders" begin
    @test_throws ErrorException TestScenarioDecoder()(:ξ)
    @test_throws ErrorException TestVectorDecoder()(:bad)

    scenario_parameters = ParametricScenario(;
        W_eq_xi=:parameter_W_eq,
        W_ineq_xi=:parameter_W_ineq,
        T_eq_xi=:parameter_T_eq,
        T_ineq_xi=:parameter_T_ineq,
        h_eq_xi=:parameter_h_eq,
        h_ineq_xi=:parameter_h_ineq,
        q_xi=:parameter_q,
    )

    decoder = ParametricDecoder(
        (:W_eq, :h_eq);
        base_W_eq=:base_W_eq,
        base_W_ineq=:base_W_ineq,
        base_T_eq=:base_T_eq,
        base_T_ineq=:base_T_ineq,
        base_h_eq=:base_h_eq,
        base_h_ineq=:base_h_ineq,
        base_q=:base_q,
    )
    W_eq, W_ineq, T_eq, T_ineq, h_eq, h_ineq, q = decoder(scenario_parameters)

    @test W_eq === :parameter_W_eq
    @test h_eq === :parameter_h_eq
    @test W_ineq === :base_W_ineq
    @test T_eq === :base_T_eq
    @test T_ineq === :base_T_ineq
    @test h_ineq === :base_h_ineq
    @test q === :base_q

    @test ParametricDecoder(
        (:T_eq,);
        base_W_eq=:base_W_eq,
        base_W_ineq=:base_W_ineq,
        base_T_ineq=:base_T_ineq,
        base_h_eq=:base_h_eq,
        base_h_ineq=:base_h_ineq,
        base_q=:base_q,
    )(scenario_parameters)[3] === :parameter_T_eq
    @test_throws ArgumentError ParametricDecoder((:bad_component,))
    @test_throws ArgumentError ParametricDecoder((:q,))(scenario_parameters)

    scenario_collection = [
        ParametricScenario(;
            W_eq_xi=[1.0 2.0; 3.0 4.0],
            W_ineq_xi=[5.0 6.0],
            T_eq_xi=reshape([7.0, 8.0], 2, 1),
            T_ineq_xi=reshape([9.0], 1, 1),
            h_eq_xi=[10.0, 11.0],
            h_ineq_xi=[12.0],
            q_xi=[13.0, 14.0],
        ),
        ParametricScenario(;
            W_eq_xi=[15.0 16.0; 17.0 18.0],
            W_ineq_xi=[19.0 20.0],
            T_eq_xi=reshape([21.0, 22.0], 2, 1),
            T_ineq_xi=reshape([23.0], 1, 1),
            h_eq_xi=[24.0, 25.0],
            h_ineq_xi=[26.0],
            q_xi=[27.0, 28.0],
        ),
    ]
    collection_decoder =
        ParametricDecoder((:W_eq, :W_ineq, :T_eq, :T_ineq, :h_eq, :h_ineq, :q))

    W_eq_array,
    W_ineq_array,
    T_eq_array,
    T_ineq_array,
    h_eq_array,
    h_ineq_array,
    q_array = decode_scenario_collection(collection_decoder, scenario_collection)

    @test W_eq_array[:, :, 1] == scenario_collection[1].W_eq_xi
    @test W_eq_array[:, :, 2] == scenario_collection[2].W_eq_xi
    @test W_ineq_array[:, :, 1] == scenario_collection[1].W_ineq_xi
    @test W_ineq_array[:, :, 2] == scenario_collection[2].W_ineq_xi
    @test T_eq_array[:, :, 1] == scenario_collection[1].T_eq_xi
    @test T_eq_array[:, :, 2] == scenario_collection[2].T_eq_xi
    @test T_ineq_array[:, :, 1] == scenario_collection[1].T_ineq_xi
    @test T_ineq_array[:, :, 2] == scenario_collection[2].T_ineq_xi
    @test h_eq_array[:, 1] == scenario_collection[1].h_eq_xi
    @test h_eq_array[:, 2] == scenario_collection[2].h_eq_xi
    @test h_ineq_array[:, 1] == scenario_collection[1].h_ineq_xi
    @test h_ineq_array[:, 2] == scenario_collection[2].h_ineq_xi
    @test q_array[:, 1] == scenario_collection[1].q_xi
    @test q_array[:, 2] == scenario_collection[2].q_xi

    _, pullback =
        ChainRulesCore.rrule(decode_scenario_collection, collection_decoder, scenario_collection)
    collection_tangent = pullback((
        fill(1.0, size(W_eq_array)),
        fill(2.0, size(W_ineq_array)),
        fill(3.0, size(T_eq_array)),
        fill(4.0, size(T_ineq_array)),
        fill(5.0, size(h_eq_array)),
        fill(6.0, size(h_ineq_array)),
        fill(7.0, size(q_array)),
    ))[3]

    @test collection_tangent[1].W_eq_xi == fill(1.0, size(scenario_collection[1].W_eq_xi))
    @test collection_tangent[2].W_ineq_xi == fill(2.0, size(scenario_collection[2].W_ineq_xi))
    @test collection_tangent[1].T_eq_xi == fill(3.0, size(scenario_collection[1].T_eq_xi))
    @test collection_tangent[2].T_ineq_xi == fill(4.0, size(scenario_collection[2].T_ineq_xi))
    @test collection_tangent[1].h_eq_xi == fill(5.0, size(scenario_collection[1].h_eq_xi))
    @test collection_tangent[2].h_ineq_xi == fill(6.0, size(scenario_collection[2].h_ineq_xi))
    @test collection_tangent[1].q_xi == fill(7.0, size(scenario_collection[1].q_xi))

    zero_component_tangent = pullback((
        ChainRulesCore.ZeroTangent(),
        ChainRulesCore.ZeroTangent(),
        ChainRulesCore.ZeroTangent(),
        ChainRulesCore.ZeroTangent(),
        ChainRulesCore.ZeroTangent(),
        ChainRulesCore.ZeroTangent(),
        fill(7.0, size(q_array)),
    ))[3]

    @test zero_component_tangent[1].W_eq_xi isa ChainRulesCore.NoTangent
    @test zero_component_tangent[1].q_xi == fill(7.0, size(scenario_collection[1].q_xi))

    @test_throws ArgumentError pullback((1.0,))
    @test_throws DimensionMismatch pullback((fill(1.0, size(W_eq_array, 1), size(W_eq_array, 2)),))

    float32_collection = [
        ParametricScenario(;
            W_eq_xi=Float32[1.0 2.0; 3.0 4.0],
            W_ineq_xi=reshape(Float32[5.0, 6.0], 1, 2),
            T_eq_xi=reshape(Float32[7.0, 8.0], 2, 1),
            T_ineq_xi=reshape(Float32[9.0], 1, 1),
            h_eq_xi=Float32[10.0, 11.0],
            h_ineq_xi=Float32[12.0],
            q_xi=Float32[13.0, 14.0],
        ),
    ]
    float32_output, float32_pullback =
        ChainRulesCore.rrule(decode_scenario_collection, collection_decoder, float32_collection)
    float32_tangent = float32_pullback((
        fill(1.0, size(float32_output[1])),
        fill(2.0, size(float32_output[2])),
        fill(3.0, size(float32_output[3])),
        fill(4.0, size(float32_output[4])),
        fill(5.0, size(float32_output[5])),
        fill(6.0, size(float32_output[6])),
        fill(7.0, size(float32_output[7])),
    ))[3][1]

    @test eltype(float32_tangent.W_eq_xi) == Float32
    @test eltype(float32_tangent.h_eq_xi) == Float32
    @test float32_tangent.W_eq_xi == fill(Float32(1.0), size(float32_collection[1].W_eq_xi))
    @test float32_tangent.h_eq_xi == fill(Float32(5.0), size(float32_collection[1].h_eq_xi))

    function zygote_collection_sum(x)
        W_eq_array,
        W_ineq_array,
        T_eq_array,
        T_ineq_array,
        h_eq_array,
        h_ineq_array,
        q_array = decode_scenario_collection(
            collection_decoder,
            [
                ParametricScenario(;
                    W_eq_xi=reshape([x], 1, 1),
                    W_ineq_xi=reshape([2x], 1, 1),
                    T_eq_xi=reshape([3x], 1, 1),
                    T_ineq_xi=reshape([4x], 1, 1),
                    h_eq_xi=[5x],
                    h_ineq_xi=[6x],
                    q_xi=[7x],
                ),
            ],
        )

        return only(W_eq_array) +
               only(W_ineq_array) +
               only(T_eq_array) +
               only(T_ineq_array) +
               only(h_eq_array) +
               only(h_ineq_array) +
               only(q_array)
    end

    @test only(Flux.gradient(zygote_collection_sum, 2.0)) == 28.0
    @test_throws ArgumentError decode_scenario_collection(collection_decoder, ParametricScenario[])

    vector_arrays = decode_scenario_collection(TestVectorDecoder(), collect(1.0:8.0); nr_scenarios=2)
    @test size(vector_arrays[1]) == (1, 1, 2)
    @test vector_arrays[1][:, :, 1] == reshape([1.0], 1, 1)
    @test vector_arrays[1][:, :, 2] == reshape([5.0], 1, 1)
    @test vector_arrays[5] == reshape([3.0, 7.0], 1, 2)
    @test_throws ArgumentError decode_scenario_collection(TestVectorDecoder(), collect(1.0:5.0); nr_scenarios=2)
end

# END FILE: src/ContextualDFL/ContextualDFL/test/scenario_decoders/runtests.jl

# BEGIN FILE: src/ContextualDFL/ContextualDFL/test/stochastic_programming/runtests.jl
import ChainRulesCore
import Serialization
import SparseArrays

function _captured_failure(f)
    try
        f()
    catch error
        return error
    end

    error("Expected stochastic program failure.")
end

function _crash_payload_from_failure(error)
    message = sprint(showerror, error)
    matched = match(r"Crash data serialized at (.+stochastic_program_failure\.jls)$", message)
    @test matched !== nothing
    crash_file = matched.captures[1]
    @test isfile(crash_file)
    return Serialization.deserialize(crash_file), crash_file, message
end

function _with_stochastic_crash_root(f)
    crash_root = mktempdir()
    previous_root = ContextualDFL._set_stochastic_crash_root!(crash_root)
    try
        return f(crash_root)
    finally
        ContextualDFL._set_stochastic_crash_root!(previous_root)
        rm(crash_root; recursive=true, force=true)
    end
end

@testset "stochastic_programming" begin
    solver = Solver(IpoptSolver(), HiGHSSolver())

    @testset "first-stage wrapper" begin
        first_stage_lp = LP(
            A_eq=zeros(0, 1),
            A_ineq=zeros(0, 1),
            b_eq=Float64[],
            b_ineq=Float64[],
            c=[2.0],
        )
        program = StochasticProgram(first_stage_lp)

        @test program.first_stage_lp === first_stage_lp
        @test program.A_eq == first_stage_lp.A_eq
        @test program.A_ineq == first_stage_lp.A_ineq
        @test program.b_eq == first_stage_lp.b_eq
        @test program.b_ineq == first_stage_lp.b_ineq
        @test program.c == first_stage_lp.c
    end

    @testset "single-scenario solve crash serialization" begin
        _with_stochastic_crash_root() do crash_root
            program = StochasticProgram(
                A_eq=zeros(0, 0),
                A_ineq=zeros(0, 0),
                b_eq=Float64[],
                b_ineq=Float64[],
                c=Float64[],
            )
            W_eq_array = zeros(0, 1, 1)
            W_ineq_array = reshape([1.0, -1.0], 2, 1, 1)
            T_eq_array = zeros(0, 0, 1)
            T_ineq_array = zeros(2, 0, 1)
            h_eq_array = zeros(0, 1)
            h_ineq_array = reshape([0.0, -1.0], 2, 1)
            q_array = reshape([0.0], 1, 1)
            probabilities = [1.0]

            failure = _captured_failure() do
                solve(
                    solver,
                    program,
                    W_eq_array,
                    W_ineq_array,
                    T_eq_array,
                    T_ineq_array,
                    h_eq_array,
                    h_ineq_array,
                    q_array;
                    probabilities=probabilities,
                    μ=0,
                    constraint_tolerance=1e-7,
                )
            end

            payload, crash_file, message = _crash_payload_from_failure(failure)
            @test failure isa ContextualDFL.StochasticProgramFailure
            @test startswith(crash_file, crash_root)
            @test occursin("single-scenario problem failed.", message)
            @test payload.location === :single_scenario_solve
            @test payload.first_stage.A_eq == program.A_eq
            @test payload.first_stage.A_ineq == program.A_ineq
            @test payload.first_stage.b_eq == program.b_eq
            @test payload.first_stage.b_ineq == program.b_ineq
            @test payload.first_stage.c == program.c
            @test payload.scenario_data.W_eq_array == W_eq_array
            @test payload.scenario_data.W_ineq_array == W_ineq_array
            @test payload.scenario_data.T_eq_array == T_eq_array
            @test payload.scenario_data.T_ineq_array == T_ineq_array
            @test payload.scenario_data.h_eq_array == h_eq_array
            @test payload.scenario_data.h_ineq_array == h_ineq_array
            @test payload.scenario_data.q_array == q_array
            @test isempty(payload.scenario_data.W_eq_array)
            @test isempty(payload.scenario_data.h_eq_array)
            @test payload.μ == 0
            @test payload.effective_μ == zeros(2)
            @test payload.ρ == 0
            @test payload.effective_ρ == zeros(1)
            @test payload.probabilities == probabilities
            @test payload.kwargs.constraint_tolerance == 1e-7
            @test payload.original_error_text != ""
        end
    end

    @testset "second-stage cost crash serialization" begin
        _with_stochastic_crash_root() do crash_root
            program = StochasticProgram(
                A_eq=zeros(0, 1),
                A_ineq=zeros(0, 1),
                b_eq=Float64[],
                b_ineq=Float64[],
                c=[0.0],
            )
            z = [0.0]
            W_eq_array = zeros(0, 1, 1)
            W_ineq_array = zeros(0, 1, 1)
            T_eq_array = zeros(0, 1, 1)
            T_ineq_array = zeros(0, 1, 1)
            h_eq_array = zeros(0, 1)
            h_ineq_array = zeros(0, 1)
            q_array = reshape([-1.0], 1, 1)

            failure = _captured_failure() do
                cost_function(
                    program,
                    solver,
                    z,
                    W_eq_array,
                    W_ineq_array,
                    T_eq_array,
                    T_ineq_array,
                    h_eq_array,
                    h_ineq_array,
                    q_array;
                    μ=0,
                    constraint_tolerance=1e-8,
                )
            end

            payload, crash_file, message = _crash_payload_from_failure(failure)
            @test failure isa ContextualDFL.StochasticProgramFailure
            @test startswith(crash_file, crash_root)
            @test occursin("second-stage problem failed in scenario 1.", message)
            @test payload.location === :second_stage_cost
            @test payload.scenario_index == 1
            @test payload.z == z
            @test payload.first_stage.c == program.c
            @test payload.scenario_data.W_eq_array == W_eq_array
            @test payload.scenario_data.W_ineq_array == W_ineq_array
            @test payload.scenario_data.T_eq_array == T_eq_array
            @test payload.scenario_data.T_ineq_array == T_ineq_array
            @test payload.scenario_data.h_eq_array == h_eq_array
            @test payload.scenario_data.h_ineq_array == h_ineq_array
            @test payload.scenario_data.q_array == q_array
            @test isempty(payload.scenario_data.W_ineq_array)
            @test isempty(payload.scenario_data.h_ineq_array)
            @test payload.μ == 0
            @test payload.scenario_μ == 0
            @test payload.ρ == 0
            @test payload.scenario_ρ == 0
            @test payload.kwargs.constraint_tolerance == 1e-8
            @test payload.original_error_text != ""
        end
    end

    @testset "two-scenario equality recourse" begin
        program = StochasticProgram(
            A_eq=zeros(0, 1),
            A_ineq=zeros(0, 1),
            b_eq=Float64[],
            b_ineq=Float64[],
            c=[2.0],
        )

        W_eq_array = reshape([1.0, 1.0], 1, 1, 2)
        W_ineq_array = zeros(0, 1, 2)
        T_eq_array = reshape([1.0, 2.0], 1, 1, 2)
        T_ineq_array = zeros(0, 1, 2)
        h_eq_array = reshape([5.0, 8.0], 1, 2)
        h_ineq_array = zeros(0, 2)
        q_array = reshape([3.0, 4.0], 1, 2)
        probabilities = [0.25, 0.75]
        z = [1.0]

        extensive_lp = construct_lp(
            program,
            W_eq_array,
            W_ineq_array,
            T_eq_array,
            T_ineq_array,
            h_eq_array,
            h_ineq_array,
            q_array;
            probabilities=probabilities,
        )

        @test extensive_lp.c == [2.0, 0.75, 3.0]
        @test SparseArrays.issparse(extensive_lp.A_eq)
        @test SparseArrays.issparse(extensive_lp.A_ineq)
        @test extensive_lp.A_eq == [1.0 1.0 0.0; 2.0 0.0 1.0]
        @test size(extensive_lp.A_ineq) == (0, 3)
        @test extensive_lp.b_eq == [5.0, 8.0]
        @test isempty(extensive_lp.b_ineq)

        @test cost_function(
            program,
            solver,
            z,
            W_eq_array,
            W_ineq_array,
            T_eq_array,
            T_ineq_array,
            h_eq_array,
            h_ineq_array,
            q_array;
            probabilities=probabilities,
        ) ≈ 23.0 atol = 1e-8
        @test cost_function(
            program,
            solver,
            z,
            W_eq_array,
            W_ineq_array,
            T_eq_array,
            T_ineq_array,
            h_eq_array,
            h_ineq_array,
            q_array;
            probabilities=probabilities,
            ρ=0.5,
            tol=1e-10,
        ) ≈ 31.0 atol = 1e-7

        primal, dual_eq, dual_ineq = ContextualDFL.G_hat(
            solver,
            z,
            view(W_eq_array, :, :, 1),
            view(W_ineq_array, :, :, 1),
            view(T_eq_array, :, :, 1),
            view(T_ineq_array, :, :, 1),
            view(h_eq_array, :, 1),
            view(h_ineq_array, :, 1),
            view(q_array, :, 1);
            return_dual=true,
        )

        @test primal ≈ [4.0] atol = 1e-8
        @test dual_eq ≈ [3.0] atol = 1e-8
        @test isempty(dual_ineq)

        value, pullback = ChainRulesCore.rrule(
            cost_function,
            program,
            solver,
            z,
            W_eq_array,
            W_ineq_array,
            T_eq_array,
            T_ineq_array,
            h_eq_array,
            h_ineq_array,
            q_array;
            μ=0,
            probabilities=probabilities,
        )
        tangents = pullback(1.0)
        zero_tangents = pullback(ChainRulesCore.ZeroTangent())

        @test value ≈ 23.0 atol = 1e-8
        @test length(tangents) == 11
        @test tangents[4] ≈ [-4.75] atol = 1e-8
        @test zero_tangents[4] == zeros(size(z))
        @test_throws ArgumentError pullback([1.0])

        ϵ = 1e-5
        finite_difference_gradient = (
            cost_function(
                program,
                solver,
                z .+ ϵ,
                W_eq_array,
                W_ineq_array,
                T_eq_array,
                T_ineq_array,
                h_eq_array,
                h_ineq_array,
                q_array;
                probabilities=probabilities,
            ) -
            cost_function(
                program,
                solver,
                z .- ϵ,
                W_eq_array,
                W_ineq_array,
                T_eq_array,
                T_ineq_array,
                h_eq_array,
                h_ineq_array,
                q_array;
                probabilities=probabilities,
            )
        ) / (2ϵ)

        @test tangents[4][1] ≈ finite_difference_gradient atol = 1e-5

        rho_value = 0.5
        value_rho, pullback_rho = ChainRulesCore.rrule(
            cost_function,
            program,
            solver,
            z,
            W_eq_array,
            W_ineq_array,
            T_eq_array,
            T_ineq_array,
            h_eq_array,
            h_ineq_array,
            q_array;
            μ=0,
            ρ=rho_value,
            probabilities=probabilities,
            tol=1e-10,
        )
        tangents_rho = pullback_rho(1.0)
        finite_difference_gradient_rho = (
            cost_function(
                program,
                solver,
                z .+ ϵ,
                W_eq_array,
                W_ineq_array,
                T_eq_array,
                T_ineq_array,
                h_eq_array,
                h_ineq_array,
                q_array;
                ρ=rho_value,
                probabilities=probabilities,
                tol=1e-10,
            ) -
            cost_function(
                program,
                solver,
                z .- ϵ,
                W_eq_array,
                W_ineq_array,
                T_eq_array,
                T_ineq_array,
                h_eq_array,
                h_ineq_array,
                q_array;
                ρ=rho_value,
                probabilities=probabilities,
                tol=1e-10,
            )
        ) / (2ϵ)

        @test value_rho ≈ 31.0 atol = 1e-7
        @test tangents_rho[4][1] ≈ finite_difference_gradient_rho atol = 1e-5
    end

    @testset "probability-scaled log barrier" begin
        program = StochasticProgram(
            A_eq=zeros(0, 1),
            A_ineq=reshape([-1.0, 1.0], 2, 1),
            b_eq=Float64[],
            b_ineq=[0.0, 10.0],
            c=[1.0],
        )

        W_eq_array = zeros(0, 1, 2)
        W_ineq_array = zeros(2, 1, 2)
        W_ineq_array[:, :, 1] = reshape([-1.0, 1.0], 2, 1)
        W_ineq_array[:, :, 2] = reshape([-1.0, 1.0], 2, 1)
        T_eq_array = zeros(0, 1, 2)
        T_ineq_array = zeros(2, 1, 2)
        h_eq_array = zeros(0, 2)
        h_ineq_array = [0.0 0.0; 5.0 7.0]
        q_array = reshape([2.0, 3.0], 1, 2)
        probabilities = [0.25, 0.75]
        μ = 0.4

        stochastic_result = solve(
            solver,
            program,
            W_eq_array,
            W_ineq_array,
            T_eq_array,
            T_ineq_array,
            h_eq_array,
            h_ineq_array,
            q_array;
            probabilities=probabilities,
            μ=μ,
            tol=1e-10,
        )

        extensive_lp = construct_lp(
            program,
            W_eq_array,
            W_ineq_array,
            T_eq_array,
            T_ineq_array,
            h_eq_array,
            h_ineq_array,
            q_array;
            probabilities=probabilities,
        )
        manual_barrier = [
            μ,
            μ,
            μ * probabilities[1],
            μ * probabilities[1],
            μ * probabilities[2],
            μ * probabilities[2],
        ]
        manual_result = solve(solver, extensive_lp; μ=manual_barrier, tol=1e-10)

        @test vcat(stochastic_result[1], vec(stochastic_result[2])) ≈ manual_result.z atol = 1e-8
    end

    @testset "probability-scaled quadratic smoothing" begin
        program = StochasticProgram(
            A_eq=zeros(0, 1),
            A_ineq=reshape([-1.0, 1.0], 2, 1),
            b_eq=Float64[],
            b_ineq=[0.0, 10.0],
            c=[1.0],
        )

        W_eq_array = zeros(0, 1, 2)
        W_ineq_array = zeros(2, 1, 2)
        W_ineq_array[:, :, 1] = reshape([-1.0, 1.0], 2, 1)
        W_ineq_array[:, :, 2] = reshape([-1.0, 1.0], 2, 1)
        T_eq_array = zeros(0, 1, 2)
        T_ineq_array = zeros(2, 1, 2)
        h_eq_array = zeros(0, 2)
        h_ineq_array = [0.0 0.0; 5.0 7.0]
        q_array = reshape([2.0, 3.0], 1, 2)
        probabilities = [0.25, 0.75]
        ρ = 0.7

        stochastic_result = solve(
            solver,
            program,
            W_eq_array,
            W_ineq_array,
            T_eq_array,
            T_ineq_array,
            h_eq_array,
            h_ineq_array,
            q_array;
            probabilities=probabilities,
            ρ=ρ,
            tol=1e-10,
        )

        extensive_lp = construct_lp(
            program,
            W_eq_array,
            W_ineq_array,
            T_eq_array,
            T_ineq_array,
            h_eq_array,
            h_ineq_array,
            q_array;
            probabilities=probabilities,
        )
        manual_ρ = [ρ, ρ * probabilities[1], ρ * probabilities[2]]
        manual_result = solve(solver, extensive_lp; ρ=manual_ρ, tol=1e-10)

        @test vcat(stochastic_result[1], vec(stochastic_result[2])) ≈ manual_result.z atol = 1e-8
    end

    @testset "cost_function includes first-stage log barrier" begin
        program = StochasticProgram(
            A_eq=zeros(0, 1),
            A_ineq=reshape([-1.0, 1.0], 2, 1),
            b_eq=Float64[],
            b_ineq=[0.0, 10.0],
            c=[1.0],
        )

        W_eq_array = zeros(0, 1, 1)
        W_ineq_array = reshape([-1.0, 1.0], 2, 1, 1)
        T_eq_array = zeros(0, 1, 1)
        T_ineq_array = zeros(2, 1, 1)
        h_eq_array = zeros(0, 1)
        h_ineq_array = reshape([0.0, 1.0], 2, 1)
        q_array = reshape([0.0], 1, 1)
        μ = 0.2

        extensive_lp = construct_lp(
            program,
            W_eq_array,
            W_ineq_array,
            T_eq_array,
            T_ineq_array,
            h_eq_array,
            h_ineq_array,
            q_array,
        )
        extensive_result = solve(solver, extensive_lp; μ=fill(μ, length(extensive_lp.b_ineq)), tol=1e-10)
        z = extensive_result.z[1:1]

        @test cost_function(
            program,
            solver,
            z,
            W_eq_array,
            W_ineq_array,
            T_eq_array,
            T_ineq_array,
            h_eq_array,
            h_ineq_array,
            q_array;
            μ=μ,
            tol=1e-10,
        ) ≈ extensive_result.objective_value atol = 2e-8

        z_fixed = [2.0]
        _, pullback = ChainRulesCore.rrule(
            cost_function,
            program,
            solver,
            z_fixed,
            W_eq_array,
            W_ineq_array,
            T_eq_array,
            T_ineq_array,
            h_eq_array,
            h_ineq_array,
            q_array;
            μ=μ,
            tol=1e-10,
        )
        first_stage_slack = program.b_ineq - program.A_ineq * z_fixed
        expected_dz =
            program.c + transpose(program.A_ineq) * (fill(μ, length(program.b_ineq)) ./ first_stage_slack)

        @test pullback(1.0)[4] ≈ expected_dz atol = 2e-8
    end

    @testset "log-barrier inequality dual convention" begin
        μ = 0.1
        lp = LP(
            A_eq=zeros(0, 1),
            A_ineq=reshape([-1.0, -1.0], 2, 1),
            b_eq=Float64[],
            b_ineq=[-8.0, 0.0],
            c=[1.0],
        )

        result = solve(solver, lp; μ=μ, tol=1e-10)
        slack = lp.b_ineq - lp.A_ineq * result.z

        @test minimum(result.dual_ineq) >= -1e-8
        @test result.dual_ineq ≈ fill(μ, length(slack)) ./ slack atol = 2e-4 rtol = 2e-4
    end

    @testset "log-barrier cost rrule inequality sign" begin
        program = StochasticProgram(
            A_eq=zeros(0, 1),
            A_ineq=zeros(0, 1),
            b_eq=Float64[],
            b_ineq=Float64[],
            c=[0.0],
        )

        W_eq_array = zeros(0, 1, 1)
        W_ineq_array = reshape([-1.0, -1.0], 2, 1, 1)
        T_eq_array = zeros(0, 1, 1)
        T_ineq_array = reshape([-1.0, 0.0], 2, 1, 1)
        h_eq_array = zeros(0, 1)
        h_ineq_array = reshape([-10.0, 0.0], 2, 1)
        q_array = reshape([1.0], 1, 1)
        z = [2.0]
        μ = 0.1

        value, pullback = ChainRulesCore.rrule(
            cost_function,
            program,
            solver,
            z,
            W_eq_array,
            W_ineq_array,
            T_eq_array,
            T_ineq_array,
            h_eq_array,
            h_ineq_array,
            q_array;
            μ=μ,
            tol=1e-10,
        )
        dz = pullback(1.0)[4]

        direction = [0.2]
        ϵ = 1e-5
        finite_difference_gradient = (
            cost_function(
                program,
                solver,
                z .+ ϵ .* direction,
                W_eq_array,
                W_ineq_array,
                T_eq_array,
                T_ineq_array,
                h_eq_array,
                h_ineq_array,
                q_array;
                μ=μ,
                tol=1e-10,
            ) -
            cost_function(
                program,
                solver,
                z .- ϵ .* direction,
                W_eq_array,
                W_ineq_array,
                T_eq_array,
                T_ineq_array,
                h_eq_array,
                h_ineq_array,
                q_array;
                μ=μ,
                tol=1e-10,
            )
        ) / (2ϵ)

        @test value isa Number
        @test sum(dz .* direction) ≈ finite_difference_gradient atol = 2e-4 rtol = 2e-4
    end

    @testset "single-scenario equality and inequality recourse" begin
        z = [1.0]
        W_eq = [1.0 0.0]
        W_ineq = [0.0 1.0]
        T_eq = reshape([1.0], 1, 1)
        T_ineq = reshape([-1.0], 1, 1)
        h_eq = [4.0]
        h_ineq = [3.0]
        q = [1.0, -2.0]

        @test ContextualDFL.G_hat(
            solver,
            z,
            W_eq,
            W_ineq,
            T_eq,
            T_ineq,
            h_eq,
            h_ineq,
            q,
        ) ≈ -5.0 atol = 1e-8

        primal, dual_eq, dual_ineq = ContextualDFL.G_hat(
            solver,
            z,
            W_eq,
            W_ineq,
            T_eq,
            T_ineq,
            h_eq,
            h_ineq,
            q;
            return_dual=true,
        )

        @test primal ≈ [3.0, 4.0] atol = 1e-8
        @test dual_eq ≈ [1.0] atol = 1e-8
        @test dual_ineq ≈ [2.0] atol = 1e-8
    end

    @testset "solve packages first and second stage solutions" begin
        program = StochasticProgram(
            A_eq=reshape([1.0], 1, 1),
            A_ineq=zeros(0, 1),
            b_eq=[1.0],
            b_ineq=Float64[],
            c=[0.0],
        )

        W_eq_array = reshape([1.0, 1.0], 1, 1, 2)
        W_ineq_array = zeros(0, 1, 2)
        T_eq_array = reshape([1.0, 2.0], 1, 1, 2)
        T_ineq_array = zeros(0, 1, 2)
        h_eq_array = reshape([5.0, 8.0], 1, 2)
        h_ineq_array = zeros(0, 2)
        q_array = reshape([3.0, 4.0], 1, 2)

        z, y, λ_b_eq, λ_b_ineq, λ_h_eq_array, λ_h_ineq_array = solve(
            solver,
            program,
            W_eq_array,
            W_ineq_array,
            T_eq_array,
            T_ineq_array,
            h_eq_array,
            h_ineq_array,
            q_array;
        )

        @test z ≈ [1.0] atol = 1e-8
        @test y ≈ reshape([4.0, 6.0], 1, 2) atol = 1e-8
        @test length(λ_b_eq) == 1
        @test isempty(λ_b_ineq)
        @test size(λ_h_eq_array) == (1, 2)
        @test size(λ_h_ineq_array) == (0, 2)
    end

    @testset "solve optimizes first-stage recourse tradeoff" begin
        program = StochasticProgram(
            A_eq=zeros(0, 1),
            A_ineq=reshape([1.0, -1.0], 2, 1),
            b_eq=Float64[],
            b_ineq=[3.0, 0.0],
            c=[2.0],
        )

        W_eq_array = reshape([1.0, 1.0], 1, 1, 2)
        W_ineq_array = zeros(0, 1, 2)
        T_eq_array = reshape([1.0, 1.0], 1, 1, 2)
        T_ineq_array = zeros(0, 1, 2)
        h_eq_array = reshape([4.0, 6.0], 1, 2)
        h_ineq_array = zeros(0, 2)
        q_array = reshape([3.0, 4.0], 1, 2)
        probabilities = [0.5, 0.5]

        z, y, _, _, _, _ = solve(
            solver,
            program,
            W_eq_array,
            W_ineq_array,
            T_eq_array,
            T_ineq_array,
            h_eq_array,
            h_ineq_array,
            q_array;
            probabilities=probabilities,
        )

        @test z ≈ [3.0] atol = 1e-8
        @test y ≈ reshape([1.0, 3.0], 1, 2) atol = 1e-8
        @test cost_function(
            program,
            solver,
            z,
            W_eq_array,
            W_ineq_array,
            T_eq_array,
            T_ineq_array,
            h_eq_array,
            h_ineq_array,
            q_array;
            probabilities=probabilities,
        ) ≈ 13.5 atol = 1e-8
    end

    @testset "solve rrule rejects malformed nonzero cotangent" begin
        program = StochasticProgram(
            A_eq=reshape([1.0], 1, 1),
            A_ineq=reshape([1.0], 1, 1),
            b_eq=[1.0],
            b_ineq=[2.0],
            c=[0.0],
        )

        W_eq_array = reshape([1.0], 1, 1, 1)
        W_ineq_array = reshape([1.0], 1, 1, 1)
        T_eq_array = reshape([1.0], 1, 1, 1)
        T_ineq_array = reshape([0.0], 1, 1, 1)
        h_eq_array = reshape([2.0], 1, 1)
        h_ineq_array = reshape([3.0], 1, 1)
        q_array = reshape([1.0], 1, 1)

        _, pullback = ChainRulesCore.rrule(
            solve,
            solver,
            program,
            W_eq_array,
            W_ineq_array,
            T_eq_array,
            T_ineq_array,
            h_eq_array,
            h_ineq_array,
            q_array;
            μ=0.25,
            tol=1e-9,
        )

        @test_throws ArgumentError pullback("not a valid cotangent")
    end

    @testset "LP reverse precompute uses solver slack" begin
        lp = LP(
            A_eq=zeros(0, 1),
            A_ineq=reshape([1.0], 1, 1),
            b_eq=Float64[],
            b_ineq=[0.0],
            c=[1.0],
        )
        result = (; z=[1.0], slack=[0.5])

        cache = ContextualDFL._lp_reverse_precompute(lp, 0.1, result, 1e-7)

        @test cache.z === result.z
        @test cache.d ≈ [4.0] atol = 1e-12
    end

    @testset "log-barrier solve rrule q sensitivity" begin
        program = StochasticProgram(
            A_eq=zeros(0, 0),
            A_ineq=zeros(0, 0),
            b_eq=Float64[],
            b_ineq=Float64[],
            c=Float64[],
        )

        W_eq_array = zeros(0, 1, 1)
        W_ineq_array = reshape([1.0, -1.0], 2, 1, 1)
        T_eq_array = zeros(0, 0, 1)
        T_ineq_array = zeros(2, 0, 1)
        h_eq_array = zeros(0, 1)
        h_ineq_array = reshape([1.0, 1.0], 2, 1)
        q_array = reshape([0.0], 1, 1)

        output, pullback = ChainRulesCore.rrule(
            solve,
            solver,
            program,
            W_eq_array,
            W_ineq_array,
            T_eq_array,
            T_ineq_array,
            h_eq_array,
            h_ineq_array,
            q_array;
            μ=0.5,
        )
        output_tangent = (
            Float64[],
            ones(1, 1),
            Float64[],
            Float64[],
            zeros(0, 1),
            zeros(2, 1),
        )
        tangents = pullback(output_tangent)

        @test output[2] ≈ reshape([0.0], 1, 1) atol = 1e-5
        @test tangents[10] ≈ reshape([-1.0], 1, 1) atol = 1e-4

        ϵ = 1e-2
        q_fd = (
            solve(
                solver,
                program,
                W_eq_array,
                W_ineq_array,
                T_eq_array,
                T_ineq_array,
                h_eq_array,
                h_ineq_array,
                q_array .+ ϵ;
                μ=0.5,
                tol=1e-10,
            )[2][1, 1] -
            solve(
                solver,
                program,
                W_eq_array,
                W_ineq_array,
                T_eq_array,
                T_ineq_array,
                h_eq_array,
                h_ineq_array,
                q_array .- ϵ;
                μ=0.5,
                tol=1e-10,
            )[2][1, 1]
        ) / (2ϵ)

        @test tangents[10][1, 1] ≈ q_fd atol = 1e-3
    end

    @testset "quadratic solve rrule scenario RHS sensitivity" begin
        program = StochasticProgram(
            A_eq=zeros(0, 1),
            A_ineq=zeros(0, 1),
            b_eq=Float64[],
            b_ineq=Float64[],
            c=[0.0],
        )

        W_eq_array = reshape([1.0], 1, 1, 1)
        W_ineq_array = zeros(0, 1, 1)
        T_eq_array = reshape([1.0], 1, 1, 1)
        T_ineq_array = zeros(0, 1, 1)
        h_eq_array = reshape([2.0], 1, 1)
        h_ineq_array = zeros(0, 1)
        q_array = reshape([1.0], 1, 1)
        ρ = 1.0

        output, pullback = ChainRulesCore.rrule(
            solve,
            solver,
            program,
            W_eq_array,
            W_ineq_array,
            T_eq_array,
            T_ineq_array,
            h_eq_array,
            h_ineq_array,
            q_array;
            ρ=ρ,
            tol=1e-10,
        )
        output_tangent = (
            ones(1),
            zeros(1, 1),
            Float64[],
            Float64[],
            zeros(1, 1),
            zeros(0, 1),
        )
        tangents = pullback(output_tangent)

        f(h_candidate) = solve(
            solver,
            program,
            W_eq_array,
            W_ineq_array,
            T_eq_array,
            T_ineq_array,
            h_candidate,
            h_ineq_array,
            q_array;
            ρ=ρ,
            tol=1e-10,
        )[1][1]

        ϵ = 1e-5
        h_fd = (f(h_eq_array .+ ϵ) - f(h_eq_array .- ϵ)) / (2ϵ)

        @test output[1] ≈ [1.5] atol = 1e-7
        @test output[2] ≈ reshape([0.5], 1, 1) atol = 1e-7
        @test tangents[8][1, 1] ≈ h_fd atol = 1e-5
    end
end

# END FILE: src/ContextualDFL/ContextualDFL/test/stochastic_programming/runtests.jl

# BEGIN FILE: src/ContextualDFL/ContextualDFLExperiments/experiments/resource_allocation_annealing/annealing.jl
import Pkg

Pkg.activate(joinpath(@__DIR__, "..", ".."))
Pkg.instantiate()

using ContextualDFL
using ContextualDFLExperiments

import Random
import Serialization
import SparseArrays
import Statistics

const Flux = ContextualDFL.Flux

rng = Random.default_rng()

problem = ResourceAllocationProblem(default_resource_allocation_problem_data())

# Match the prototype training and SAA testing data settings.
Ntraining_samples = 100
Ntesting_samples = 30
sigma = 5
p = 2
L = 3
N_xi_per_x = 100

context_generator = ResourceAllocationContextDataGenerator(rng=rng)
scenario_generator = ResourceAllocationScenarioDataGenerator(
    problem;
    sigma=sigma,
    p=p,
    L=L,
    rng=rng,
)

contexts = [Vector{Float64}(context_generator()) for _ in 1:Ntraining_samples]
scenarios = [scenario_generator(context) for context in contexts]
data_set_training = generate_contextual_data_set(contexts, scenarios)

testing_splits = 30
testing_contexts = [Vector{Float64}(context_generator()) for _ in 1:Ntesting_samples]
testing_scenarios = [
    [scenario_generator(context) for _ in 1:(testing_splits * N_xi_per_x)]
    for context in testing_contexts
]
data_set_testing = generate_contextual_data_set(testing_contexts, testing_scenarios)

nr_scenarios = 1
demand_count = size(problem.problem_data.service_rate_parameters, 2)

# Same hidden architecture as the old resource-allocation prototype.
# The new VectorDecoder expects a flat demand vector, so we leave out the old final reshape.
model = Flux.Chain(
    Flux.Dense(3 => 128, Flux.relu),
    Flux.Dense(128 => 128, Flux.relu),
    Flux.Dense(128 => 128, Flux.relu),
    Flux.Dense(128 => demand_count * nr_scenarios, Flux.relu),
) |> Flux.f64

solver = ContextualDFL.Solver(ContextualDFL.IpoptSolver(), ContextualDFL.HiGHSSolver())
loss = ContextualDFL.DflScenLoss(
    ResourceAllocationDemandVectorDecoder(problem),
    ResourceAllocationDemandParametricDecoder(problem),
    solver,
    stochastic_program(problem);
    nr_scenarios=nr_scenarios,
)

display_reference_input(point) =
    reduce(vcat, (scenario.h_eq_xi for scenario in point.scenario_parameters))

function write_testing_csv(path, rows)
    open(path, "w") do io
        println(io, "sample_index,policy_value,optimal_value,regret,relative_regret,ucb_percent")
        for row in rows
            println(
                io,
                join(
                    (
                        row.sample_index,
                        row.policy_value,
                        row.optimal_value,
                        row.regret,
                        row.relative_regret,
                        row.ucb_percent,
                    ),
                    ",",
                ),
            )
        end
    end
end

function normalized_gap_ucb_percent(policy_split_values, optimal_split_values)
    gaps = Float64.(policy_split_values) .- Float64.(optimal_split_values)
    optimal_mean = Statistics.mean(Float64.(optimal_split_values))
    gap_variance = length(gaps) > 1 ? Statistics.var(gaps) : 0.0
    ucb_gap = Statistics.mean(gaps) + 2.462 * sqrt(gap_variance / length(gaps))
    return 100 * ucb_gap / max(abs(optimal_mean), eps(Float64))
end

function testing_split_ranges(data_point, split_count)
    scenario_count = length(data_point.scenario_parameters)
    scenario_count % split_count == 0 ||
        throw(ArgumentError(
            "scenario count $scenario_count is not divisible by splits=$split_count.",
        ))

    split_size = scenario_count ÷ split_count
    return [
        ((split_index - 1) * split_size + 1):(split_index * split_size) for
        split_index in 1:split_count
    ]
end

function write_testing_partial(
    path,
    rows,
    testing_count,
    testing_splits,
    csv_path;
    active_context_index=nothing,
    optimal_split_values=Float64[],
    policy_split_values=Float64[],
)
    Serialization.serialize(
        path,
        (;
            ucb_rows=rows,
            completed_contexts=length(rows),
            testing_context_limit=testing_count,
            testing_splits=testing_splits,
            csv_path=csv_path,
            active_context_index=active_context_index,
            optimal_split_values=Float64.(collect(optimal_split_values)),
            policy_split_values=Float64.(collect(policy_split_values)),
        ),
    )
end

function resource_allocation_h_eq_matrix(problem, scenario_parameters)
    scenario = base_scenario(problem)
    resource_count = size(scenario.T_eq, 2)
    demand_count = length(scenario.h_eq) - resource_count
    h_eq_array = zeros(Float64, length(scenario.h_eq), length(scenario_parameters))

    for (index, scenario_parameter) in enumerate(scenario_parameters)
        length(scenario_parameter.h_eq_xi) == demand_count ||
            throw(DimensionMismatch("scenario demand vector must have length $demand_count."))
        h_eq_array[(resource_count + 1):end, index] = scenario_parameter.h_eq_xi
    end

    return h_eq_array
end

function resource_allocation_h_eq_vector(problem, scenario_parameter)
    h_eq_array = resource_allocation_h_eq_matrix(problem, (scenario_parameter,))
    return vec(h_eq_array)
end

function resource_allocation_extensive_lp(problem, scenario_parameters)
    program = stochastic_program(problem)
    first_stage_lp = program.first_stage_lp
    scenario = base_scenario(problem)
    h_eq_array = resource_allocation_h_eq_matrix(problem, scenario_parameters)

    split_count = size(h_eq_array, 2)
    z_count = length(first_stage_lp.c)
    y_count = length(scenario.q)
    variable_count = z_count + split_count * y_count
    first_ineq_count = length(first_stage_lp.b_ineq)
    recourse_eq_count = size(scenario.W_eq, 1)
    recourse_ineq_count = size(scenario.W_ineq, 1)

    A_eq = SparseArrays.spzeros(Float64, split_count * recourse_eq_count, variable_count)
    A_ineq = SparseArrays.spzeros(
        Float64,
        first_ineq_count + split_count * recourse_ineq_count,
        variable_count,
    )
    b_eq = zeros(Float64, split_count * recourse_eq_count)
    b_ineq = zeros(Float64, first_ineq_count + split_count * recourse_ineq_count)
    c = zeros(Float64, variable_count)

    z_cols = 1:z_count
    c[z_cols] = first_stage_lp.c
    if first_ineq_count > 0
        A_ineq[1:first_ineq_count, z_cols] = SparseArrays.sparse(first_stage_lp.A_ineq)
        b_ineq[1:first_ineq_count] = first_stage_lp.b_ineq
    end

    W_eq = SparseArrays.sparse(scenario.W_eq)
    W_ineq = SparseArrays.sparse(scenario.W_ineq)
    T_eq = SparseArrays.sparse(scenario.T_eq)
    T_ineq = SparseArrays.sparse(scenario.T_ineq)
    probability = 1.0 / split_count

    for split_index in 1:split_count
        y_cols = (z_count + (split_index - 1) * y_count + 1):(z_count + split_index * y_count)
        eq_rows = ((split_index - 1) * recourse_eq_count + 1):(split_index * recourse_eq_count)
        ineq_rows = (
            first_ineq_count + (split_index - 1) * recourse_ineq_count + 1
        ):(first_ineq_count + split_index * recourse_ineq_count)

        A_eq[eq_rows, z_cols] = T_eq
        A_eq[eq_rows, y_cols] = W_eq
        b_eq[eq_rows] = view(h_eq_array, :, split_index)

        A_ineq[ineq_rows, z_cols] = T_ineq
        A_ineq[ineq_rows, y_cols] = W_ineq
        b_ineq[ineq_rows] = scenario.h_ineq

        c[y_cols] = probability .* scenario.q
    end

    return ContextualDFL.LP(A_eq, A_ineq, b_eq, b_ineq, c)
end

function resource_allocation_extensive_rho(problem, scenario_parameters, rho)
    rho isa Number ||
        throw(ArgumentError("resource-allocation SAA testing expects scalar rho."))
    rho >= 0 || throw(ArgumentError("rho must be non-negative."))

    program = stochastic_program(problem)
    scenario = base_scenario(problem)
    split_count = length(scenario_parameters)
    z_count = length(program.first_stage_lp.c)
    y_count = length(scenario.q)
    probability = 1.0 / split_count

    rho_vector = zeros(Float64, z_count + split_count * y_count)
    rho_vector[1:z_count] .= rho
    for split_index in 1:split_count
        y_cols = (z_count + (split_index - 1) * y_count + 1):(z_count + split_index * y_count)
        rho_vector[y_cols] .= probability * rho
    end
    return rho_vector
end

function resource_allocation_optimal_split_value(problem, solver, scenario_parameters; mu, rho)
    iszero(mu) || throw(ArgumentError("resource-allocation SAA testing expects mu=0."))

    lp = resource_allocation_extensive_lp(problem, scenario_parameters)
    result = ContextualDFL.solve(
        solver,
        lp;
        μ=mu,
        ρ=resource_allocation_extensive_rho(problem, scenario_parameters, rho),
    )
    return result.objective_value
end

function resource_allocation_policy_split_value(problem, solver, z, scenario_parameters; mu, rho)
    iszero(mu) || throw(ArgumentError("resource-allocation SAA testing expects mu=0."))
    rho isa Number ||
        throw(ArgumentError("resource-allocation SAA testing expects scalar rho."))
    rho >= 0 || throw(ArgumentError("rho must be non-negative."))

    program = stochastic_program(problem)
    scenario = base_scenario(problem)
    probability = 1.0 / length(scenario_parameters)
    value = sum(program.first_stage_lp.c .* z) + 0.5 * rho * sum(abs2, z)

    for scenario_parameter in scenario_parameters
        h_eq = resource_allocation_h_eq_vector(problem, scenario_parameter)
        value += probability * ContextualDFL.G_hat(
            solver,
            z,
            scenario.W_eq,
            scenario.W_ineq,
            scenario.T_eq,
            scenario.T_ineq,
            h_eq,
            scenario.h_ineq,
            scenario.q;
            μ=mu,
            ρ=rho,
        )
    end

    return value
end

function solve_data_point_to_optimality_with_progress(
    data_point,
    problem,
    solver;
    mu,
    rho=0,
    splits,
    context_index,
    testing_count,
    partial_path,
    rows,
    csv_path,
    resume_values=Float64[],
)
    split_ranges = testing_split_ranges(data_point, splits)
    objective_values = Float64.(collect(resume_values))

    for split_index in (length(objective_values) + 1):splits
        println("Testing context $(context_index)/$(testing_count): optimal split $(split_index)/$(splits)...")
        scenario_range = split_ranges[split_index]
        objective_value = resource_allocation_optimal_split_value(
            problem,
            solver,
            view(data_point.scenario_parameters, scenario_range);
            mu=mu,
            rho=rho,
        )

        push!(objective_values, objective_value)
        write_testing_partial(
            partial_path,
            rows,
            testing_count,
            splits,
            csv_path;
            active_context_index=context_index,
            optimal_split_values=objective_values,
        )
    end

    return objective_values
end

function evaluate_policy_on_data_point_with_progress(
    policy,
    data_point,
    problem,
    solver;
    mu,
    rho=0,
    splits,
    context_index,
    testing_count,
    partial_path,
    rows,
    csv_path,
    optimal_split_values,
    resume_values=Float64[],
)
    split_ranges = testing_split_ranges(data_point, splits)
    policy_split_values = Float64.(collect(resume_values))
    decision_set = generate_decision_set(policy, [data_point])
    z = view(decision_set, :, 1)

    for split_index in (length(policy_split_values) + 1):splits
        println("Testing context $(context_index)/$(testing_count): policy split $(split_index)/$(splits)...")
        scenario_range = split_ranges[split_index]
        policy_value = resource_allocation_policy_split_value(
            problem,
            solver,
            z,
            view(data_point.scenario_parameters, scenario_range);
            mu=mu,
            rho=rho,
        )

        push!(policy_split_values, policy_value)
        write_testing_partial(
            partial_path,
            rows,
            testing_count,
            splits,
            csv_path;
            active_context_index=context_index,
            optimal_split_values=optimal_split_values,
            policy_split_values=policy_split_values,
        )
    end

    return policy_split_values
end

function testing_sample_row(index, policy_split_values, optimal_split_values)
    policy_value = Statistics.mean(Float64.(policy_split_values))
    optimal_value = Statistics.mean(Float64.(optimal_split_values))
    regret = policy_value - optimal_value
    relative_regret = regret / max(abs(optimal_value), eps(Float64))

    return (;
        sample_index=index,
        policy_value=policy_value,
        optimal_value=optimal_value,
        regret=regret,
        relative_regret=relative_regret,
        policy_split_values=Float64.(collect(policy_split_values)),
        optimal_split_values=Float64.(collect(optimal_split_values)),
        ucb_percent=normalized_gap_ucb_percent(policy_split_values, optimal_split_values),
    )
end

function run_saa_testing(
    model,
    problem,
    solver,
    data_set_testing;
    output_dir,
    reg_param_surr,
    reg_param_ref,
    rho_surr=0.0,
    rho_ref=0.0,
    testing_splits,
    testing_context_limit=length(data_set_testing),
)
    program = stochastic_program(problem)
    policy = ScenarioGenerationPolicy(
        ContextualDFL.ScenarioGenerator(;
            neural_net=model,
            scenario_decoder=ResourceAllocationDemandVectorDecoder(problem),
        ),
        solver,
        program;
        mu=reg_param_surr,
        rho=rho_surr,
    )

    csv_path = joinpath(output_dir, "testing_saa_results.csv")
    partial_path = joinpath(output_dir, "testing_saa_partial.jls")
    testing_count = min(Int(testing_context_limit), length(data_set_testing))
    rows = NamedTuple[]
    active_context_index = nothing
    active_optimal_split_values = Float64[]
    active_policy_split_values = Float64[]

    if isfile(partial_path)
        partial = Serialization.deserialize(partial_path)
        if hasproperty(partial, :ucb_rows)
            rows = collect(partial.ucb_rows)
            println("Resuming testing SAA from $(length(rows)) completed contexts.")
        end
        if hasproperty(partial, :active_context_index)
            active_context_index = partial.active_context_index
            active_optimal_split_values = hasproperty(partial, :optimal_split_values) ?
                Float64.(collect(partial.optimal_split_values)) : Float64[]
            active_policy_split_values = hasproperty(partial, :policy_split_values) ?
                Float64.(collect(partial.policy_split_values)) : Float64[]
        end
    end

    length(rows) > testing_count && resize!(rows, testing_count)
    write_testing_csv(csv_path, rows)

    for index in (length(rows) + 1):testing_count
        resume_current_context = active_context_index == index
        optimal_resume_values =
            resume_current_context ? active_optimal_split_values : Float64[]
        policy_resume_values =
            resume_current_context ? active_policy_split_values : Float64[]

        println("Testing context $(index)/$(testing_count): solving SAA optima...")
        data_point = data_set_testing[index]
        optimal_split_values = solve_data_point_to_optimality_with_progress(
            data_point,
            problem,
            solver;
            mu=reg_param_ref,
            rho=rho_ref,
            splits=testing_splits,
            context_index=index,
            testing_count=testing_count,
            partial_path=partial_path,
            rows=rows,
            csv_path=csv_path,
            resume_values=optimal_resume_values,
        )

        println("Testing context $(index)/$(testing_count): evaluating policy...")
        policy_split_values = evaluate_policy_on_data_point_with_progress(
            policy,
            data_point,
            problem,
            solver;
            mu=reg_param_ref,
            rho=rho_ref,
            splits=testing_splits,
            context_index=index,
            testing_count=testing_count,
            partial_path=partial_path,
            rows=rows,
            csv_path=csv_path,
            optimal_split_values=optimal_split_values,
            resume_values=policy_resume_values,
        )

        row = testing_sample_row(index, policy_split_values, optimal_split_values)
        push!(rows, row)
        write_testing_csv(csv_path, rows)
        write_testing_partial(
            partial_path,
            rows,
            testing_count,
            testing_splits,
            csv_path,
        )
        active_context_index = nothing
        empty!(active_optimal_split_values)
        empty!(active_policy_split_values)
        println(
            "Testing context $(index)/$(testing_count): UCB = $(row.ucb_percent), " *
            "relative regret = $(row.relative_regret)",
        )
    end

    clean_ucbs = [row.ucb_percent for row in rows if isfinite(row.ucb_percent)]
    mean_ucb = isempty(clean_ucbs) ? NaN : Statistics.mean(clean_ucbs)
    println("mean UCB: ", mean_ucb)

    println("Testing SAA results saved to: $(csv_path)")

    return (;
        ucb_rows=rows,
        mean_ucb=mean_ucb,
        tested_contexts=length(rows),
        testing_splits=testing_splits,
        csv_path=csv_path,
        partial_path=partial_path,
    )
end

reg_param_ref = 0.0
rho_ref = 0.0
batchsize = 1
default_epochs = 10
step_size = 1e-3
save_model_training = true

param_list = [1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.08, 0.06, 0.04, 0.02, 0.01]
epoch_list = fill(default_epochs, length(param_list) + 1)
epoch_list[1] = 20
@assert length(epoch_list) == length(param_list) + 1

output_dir = joinpath(@__DIR__, "results")
mkpath(output_dir)
model_save_path = joinpath(output_dir, "trained_model_annealing.jls")
state_save_path = joinpath(output_dir, "experiment_state_annealing.jls")
skip_training = get(ENV, "CDFL_ANNEALING_SKIP_TRAINING", "0") == "1"
resume_training = get(ENV, "CDFL_ANNEALING_RESUME_TRAINING", "0") == "1"
skip_testing = get(ENV, "CDFL_ANNEALING_SKIP_TESTING", "0") == "1"
testing_context_limit = parse(
    Int,
    get(ENV, "CDFL_ANNEALING_TEST_CONTEXTS", string(Ntesting_samples)),
)

experiment_parameters = (;
    Ntraining_samples=Ntraining_samples,
    Ntesting_samples=Ntesting_samples,
    sigma=sigma,
    p=p,
    L=L,
    N_xi_per_x=N_xi_per_x,
    testing_splits=testing_splits,
    reg_param_ref=reg_param_ref,
    rho_ref=rho_ref,
    batchsize=batchsize,
    default_epochs=default_epochs,
    step_size=step_size,
    param_list=param_list,
    epoch_list=epoch_list,
    nr_scenarios=nr_scenarios,
)

final_reg_param_surr = param_list[end]
final_stage_epochs = epoch_list[end]
final_reg_param_prim = 0.0
final_rho_surr = 0.0
final_stage_index = length(param_list) + 1
stage_histories = NamedTuple[]
completed_stage_numbers = Set{Int}()

if skip_training
    isfile(state_save_path) ||
        error("CDFL_ANNEALING_SKIP_TRAINING=1 requires saved state at $(state_save_path).")
    saved_state = Serialization.deserialize(state_save_path)
    model = saved_state.model
    data_set_training = saved_state.data_set_training
    data_set_testing = saved_state.data_set_testing
    stage_histories = collect(saved_state.stage_histories)
    completed_stage_numbers = Set(Int(stage.stage) for stage in stage_histories)
    println("Skipping training and resuming from: $(state_save_path)")
else
    if resume_training && isfile(state_save_path)
        saved_state = Serialization.deserialize(state_save_path)
        model = saved_state.model
        data_set_training = saved_state.data_set_training
        data_set_testing = saved_state.data_set_testing
        stage_histories = collect(saved_state.stage_histories)
        completed_stage_numbers = Set(Int(stage.stage) for stage in stage_histories)
        println(
            "Resuming training from $(state_save_path) with " *
            "$(length(completed_stage_numbers)) completed stages.",
        )
    else
        println("Starting training with annealing...")
    end

    for (idx, reg_param_surr) in enumerate(param_list)
        if idx in completed_stage_numbers
            println("Skipping completed annealing stage $(idx).")
            continue
        end

        stage_epochs = epoch_list[idx]
        reg_param_prim = reg_param_surr
        println(
            "Starting annealing stage $(idx) with reg_param_surr = $(reg_param_surr), " *
            "reg_param_prim = $(reg_param_prim), epochs = $(stage_epochs)",
        )

        result = ContextualDFL.train!(
            model,
            loss,
            fill(reg_param_surr, stage_epochs),
            fill(reg_param_prim, stage_epochs),
            data_set_training;
            opt=Flux.Adam(step_size),
            epochs=stage_epochs,
            batchsize=batchsize,
            display_iterations=true,
            display_plot=false,
            save_model=save_model_training,
            model_save_path=model_save_path,
            reset_optimizer_each_epoch=true,
            nr_scenarios=nr_scenarios,
            display_smooth=true,
            display_reference_input=display_reference_input,
        )

        push!(
            stage_histories,
            (;
                stage=idx,
                reg_param_surr=reg_param_surr,
                reg_param_prim=reg_param_prim,
                epochs=stage_epochs,
                history=result.history,
            ),
        )
        push!(completed_stage_numbers, idx)

        Serialization.serialize(
            state_save_path,
            (;
                model=model,
                data_set_training=data_set_training,
                data_set_testing=data_set_testing,
                problem=problem,
                stage_histories=stage_histories,
                parameters=experiment_parameters,
            ),
        )
    end

    if final_stage_index in completed_stage_numbers
        println("Skipping completed final annealing stage.")
    else
        println(
            "Starting final annealing stage with reg_param_surr = $(final_reg_param_surr), " *
            "reg_param_prim = $(final_reg_param_prim), epochs = $(final_stage_epochs)",
        )

        final_result = ContextualDFL.train!(
            model,
            loss,
            fill(final_reg_param_surr, final_stage_epochs),
            fill(final_reg_param_prim, final_stage_epochs),
            data_set_training;
            opt=Flux.Adam(step_size),
            epochs=final_stage_epochs,
            batchsize=batchsize,
            display_iterations=true,
            display_plot=false,
            save_model=save_model_training,
            model_save_path=model_save_path,
            reset_optimizer_each_epoch=true,
            nr_scenarios=nr_scenarios,
            display_smooth=true,
            display_reference_input=display_reference_input,
        )

        push!(
            stage_histories,
            (;
                stage=final_stage_index,
                reg_param_surr=final_reg_param_surr,
                reg_param_prim=final_reg_param_prim,
                epochs=final_stage_epochs,
                history=final_result.history,
            ),
        )
        push!(completed_stage_numbers, final_stage_index)

        Serialization.serialize(
            state_save_path,
            (;
                model=model,
                data_set_training=data_set_training,
                data_set_testing=data_set_testing,
                problem=problem,
                stage_histories=stage_histories,
                parameters=experiment_parameters,
            ),
        )
    end

    println("Training completed!")
end

if skip_testing
    println("Skipping testing because CDFL_ANNEALING_SKIP_TESTING=1.")
else
    println("Testing the trained model...")
    testing_result = run_saa_testing(
        model,
        problem,
        solver,
        data_set_testing;
        output_dir=output_dir,
        reg_param_surr=final_reg_param_surr,
        reg_param_ref=reg_param_ref,
        rho_surr=final_rho_surr,
        rho_ref=rho_ref,
        testing_splits=testing_splits,
        testing_context_limit=testing_context_limit,
    )

    Serialization.serialize(
        state_save_path,
        (;
            model=model,
            data_set_training=data_set_training,
            data_set_testing=data_set_testing,
            problem=problem,
            stage_histories=stage_histories,
            testing_result=testing_result,
            parameters=experiment_parameters,
        ),
    )
end

println("Model saved to: $(model_save_path)")
println("Experiment state saved to: $(state_save_path)")

# END FILE: src/ContextualDFL/ContextualDFLExperiments/experiments/resource_allocation_annealing/annealing.jl

# BEGIN FILE: src/ContextualDFL/ContextualDFLExperiments/experiments/resource_allocation_annealing/benchmark_old_vs_new.jl
import Printf

const REPO_ROOT = normpath(joinpath(@__DIR__, "..", "..", "..", "..", ".."))
const NEW_PROJECT = normpath(joinpath(@__DIR__, "..", ".."))
const REPEATS = parse(Int, get(ENV, "CDFL_BENCH_REPEATS", "3"))
const WARMUPS = parse(Int, get(ENV, "CDFL_BENCH_WARMUPS", "2"))
const RUN_PROFILES = get(ENV, "CDFL_BENCH_PROFILE", "1") == "1"

function write_temp_script(source)
    path = tempname() * ".jl"
    write(path, source)
    return path
end

function benchmark_command(script, project, repo_root)
    return `$(Base.julia_cmd()) --project=$(project) $(script) $(repo_root) $(REPEATS) $(WARMUPS) $(Int(RUN_PROFILES))`
end

function run_benchmark(label, command)
    println("Running $(label) benchmark...")
    output = read(command, String)
    print(output)
    return output
end

function parse_results(output)
    rows = NamedTuple[]
    for line in split(output, '\n')
        startswith(line, "RESULT\t") || continue
        fields = split(line, '\t')
        length(fields) == 8 || continue
        push!(
            rows,
            (;
                implementation=fields[2],
                measurement=fields[3],
                min_seconds=parse(Float64, fields[4]),
                mean_seconds=parse(Float64, fields[5]),
                median_seconds=parse(Float64, fields[6]),
                mean_alloc_mib=parse(Float64, fields[7]),
                value=fields[8],
            ),
        )
    end
    return rows
end

function print_table(rows)
    isempty(rows) && return
    println()
    println("Timing summary")
    Printf.@printf(
        "%-14s  %-24s  %10s  %10s  %10s  %12s  %s\n",
        "impl",
        "measurement",
        "min_s",
        "mean_s",
        "median_s",
        "alloc_MiB",
        "value",
    )
    println("-"^105)
    for row in rows
        Printf.@printf(
            "%-14s  %-24s  %10.4f  %10.4f  %10.4f  %12.2f  %s\n",
            row.implementation,
            row.measurement,
            row.min_seconds,
            row.mean_seconds,
            row.median_seconds,
            row.mean_alloc_mib,
            row.value,
        )
    end
end

const OLD_SCRIPT = raw"""
repo_root = ARGS[1]
repeats = parse(Int, ARGS[2])
warmups = parse(Int, ARGS[3])
run_profiles = parse(Int, ARGS[4]) == 1

import Pkg
try
    redirect_stdout(devnull) do
        redirect_stderr(devnull) do
            Pkg.develop(path=joinpath(repo_root, "src", "ProblemBasedScenarioGeneration"))
            Pkg.add(["Flux", "ChainRulesCore"])
            Pkg.instantiate()
        end
    end
catch error
    @error "temporary old-package environment setup failed" exception=(error, catch_backtrace())
    rethrow()
end

using ChainRulesCore
using Flux
using LinearAlgebra
using Printf
using ProblemBasedScenarioGeneration
using Profile
using Statistics

using ProblemBasedScenarioGeneration: ResourceAllocationProblemData,
    ResourceAllocationProblem,
    LogBarCanLP,
    LogBarCanLP_standard_solver,
    TwoStageSLP,
    diff_opt_b,
    diff_s1_cost,
    s1_cost,
    scenario_collection_realization

import ProblemBasedScenarioGeneration: loss, surrogate_solution

include(joinpath(repo_root, "scripts", "resource_allocation_prototype", "custom_code", "neural_net.jl"))
include(joinpath(repo_root, "src", "ContextualDFL", "ContextualDFLExperiments", "src", "implementations", "resource_allocation_problem", "problem_data", "parameters.jl"))

problem_data = ResourceAllocationProblemData(
    RESOURCE_ALLOCATION_SERVICE_RATE_PARAMETERS,
    vec(Float64.(RESOURCE_ALLOCATION_FIRST_STAGE_COSTS)),
    vec(Float64.(RESOURCE_ALLOCATION_SECOND_STAGE_COSTS)),
    vec(Float64.(RESOURCE_ALLOCATION_YIELD_PARAMETERS)),
)
problem = ResourceAllocationProblem(problem_data)

mu_in = 1.0
mu_ref = 1.0
demand_count = size(problem.problem_data.service_rate_parameters, 2)
predicted_demand = 50.0 .+ 0.1 .* collect(1:demand_count)
actual_demand = 55.0 .+ 0.2 .* collect(1:demand_count)
z_for_cost = surrogate_solution(problem, mu_in, predicted_demand)

function summary_value(value)
    if value isa Number
        return Printf.@sprintf("%.6g", Float64(value))
    elseif value isa AbstractArray
        return Printf.@sprintf("array(len=%d,norm=%.6g)", length(value), LinearAlgebra.norm(value))
    else
        return string(typeof(value))
    end
end

function emit_result(implementation, measurement, samples)
    times = [sample.time for sample in samples]
    bytes = [sample.bytes for sample in samples]
    Printf.@printf(
        "RESULT\t%s\t%s\t%.9f\t%.9f\t%.9f\t%.6f\t%s\n",
        implementation,
        measurement,
        minimum(times),
        Statistics.mean(times),
        Statistics.median(times),
        Statistics.mean(bytes) / 1024^2,
        summary_value(samples[end].value),
    )
end

function measure(implementation, measurement, f)
    for _ in 1:warmups
        f()
    end
    samples = NamedTuple[]
    for _ in 1:repeats
        GC.gc()
        push!(samples, @timed f())
    end
    emit_result(implementation, measurement, samples)
    return samples[end].value
end

function profile_once(implementation, measurement, f)
    run_profiles || return
    for _ in 1:warmups
        f()
    end
    Profile.init(delay=0.001)
    Profile.clear()
    println("PROFILE_START\t$(implementation)\t$(measurement)")
    @profile f()
    Profile.print(format=:flat, sortedby=:count, mincount=5, maxdepth=18)
    println("PROFILE_END\t$(implementation)\t$(measurement)")
end

old_loss(demand) = loss(problem, mu_in, mu_ref, demand, actual_demand)
old_relative_display(demand) = begin
    evaluated = loss(problem, mu_in, mu_ref, demand, actual_demand)
    reference = loss(problem, mu_ref, mu_ref, actual_demand, actual_demand)
    (evaluated - reference) / abs(reference)
end

measure("old", "forward_loss", () -> old_loss(predicted_demand))
measure("old", "gradient_demand", () -> Flux.gradient(d -> old_loss(d), predicted_demand)[1])
measure("old", "surrogate_solve", () -> surrogate_solution(problem, mu_in, predicted_demand))
measure("old", "recourse_cost", () -> primal_problem_cost(problem, mu_ref, actual_demand, z_for_cost))
measure("old", "recourse_gradient_z", () -> derivative_primal_problem_cost(problem, mu_ref, actual_demand, z_for_cost))
measure("old", "relative_display_loss", () -> old_relative_display(predicted_demand))

profile_once("old", "gradient_demand", () -> Flux.gradient(d -> old_loss(d), predicted_demand)[1])
"""

const NEW_SCRIPT = raw"""
repo_root = ARGS[1]
repeats = parse(Int, ARGS[2])
warmups = parse(Int, ARGS[3])
run_profiles = parse(Int, ARGS[4]) == 1

using ContextualDFL
using ContextualDFLExperiments
using LinearAlgebra
using Printf
using Profile
using Statistics

const Flux = ContextualDFL.Flux

mu_kw(value) = NamedTuple{(Symbol(Char(0x03bc)),)}((value,))

problem = ResourceAllocationProblem(default_resource_allocation_problem_data())
program = stochastic_program(problem)
solver = ContextualDFL.Solver(ContextualDFL.IpoptSolver(), ContextualDFL.HiGHSSolver())
vector_decoder = ResourceAllocationDemandVectorDecoder(problem)
parametric_decoder = ResourceAllocationDemandParametricDecoder(problem)
loss_object = ContextualDFL.DflScenLoss(
    vector_decoder,
    parametric_decoder,
    solver,
    program;
    nr_scenarios=1,
)

mu_in = 1.0
mu_ref = 1.0
demand_count = size(problem.problem_data.service_rate_parameters, 2)
predicted_demand = 50.0 .+ 0.1 .* collect(1:demand_count)
actual_demand = 55.0 .+ 0.2 .* collect(1:demand_count)
actual_scenario = ContextualDFL.ParametricScenario(;
    W_eq_xi=Float64[],
    W_ineq_xi=Float64[],
    T_eq_xi=Float64[],
    T_ineq_xi=Float64[],
    h_eq_xi=actual_demand,
    h_ineq_xi=Float64[],
    q_xi=Float64[],
)
actual_collection = [actual_scenario]
decoded_actual = ContextualDFL.decode_scenario_collection(parametric_decoder, actual_collection)
decoded_predicted = ContextualDFL.decode_scenario_collection(
    vector_decoder,
    predicted_demand;
    nr_scenarios=1,
)
z_for_cost = ContextualDFL.solve(solver, program, decoded_predicted...; mu_kw(mu_in)...)[1]

function summary_value(value)
    if value isa Number
        return Printf.@sprintf("%.6g", Float64(value))
    elseif value isa AbstractArray
        return Printf.@sprintf("array(len=%d,norm=%.6g)", length(value), LinearAlgebra.norm(value))
    else
        return string(typeof(value))
    end
end

function emit_result(implementation, measurement, samples)
    times = [sample.time for sample in samples]
    bytes = [sample.bytes for sample in samples]
    Printf.@printf(
        "RESULT\t%s\t%s\t%.9f\t%.9f\t%.9f\t%.6f\t%s\n",
        implementation,
        measurement,
        minimum(times),
        Statistics.mean(times),
        Statistics.median(times),
        Statistics.mean(bytes) / 1024^2,
        summary_value(samples[end].value),
    )
end

function measure(implementation, measurement, f)
    for _ in 1:warmups
        f()
    end
    samples = NamedTuple[]
    for _ in 1:repeats
        GC.gc()
        push!(samples, @timed f())
    end
    emit_result(implementation, measurement, samples)
    return samples[end].value
end

function profile_once(implementation, measurement, f)
    run_profiles || return
    for _ in 1:warmups
        f()
    end
    Profile.init(delay=0.001)
    Profile.clear()
    println("PROFILE_START\t$(implementation)\t$(measurement)")
    @profile f()
    Profile.print(format=:flat, sortedby=:count, mincount=5, maxdepth=18)
    println("PROFILE_END\t$(implementation)\t$(measurement)")
end

function new_loss(demand, mu_in_value, mu_ref_value; kwargs...)
    return loss_object(
        demand,
        actual_collection,
        mu_in_value,
        mu_ref_value;
        nr_scenarios=1,
        kwargs...,
    )
end

const reference_input = reduce(vcat, (scenario.h_eq_xi for scenario in actual_collection))
const reference_cache = Dict{Any,Float64}()

function reference_cache_key(mu_ref_value; kwargs...)
    return (mu_ref_value, Tuple((key, value) for (key, value) in pairs(kwargs)))
end

function cached_reference_value(mu_ref_value; kwargs...)
    key = reference_cache_key(mu_ref_value; kwargs...)
    return get!(reference_cache, key) do
        Float64(
            loss_object(
                reference_input,
                actual_collection,
                mu_ref_value,
                mu_ref_value;
                nr_scenarios=1,
                kwargs...,
            ),
        )
    end
end

function new_relative_display(demand, mu_in_value, mu_ref_value; kwargs...)
    evaluated = loss_object(
        demand,
        actual_collection,
        mu_in_value,
        mu_ref_value;
        nr_scenarios=1,
        kwargs...,
    )
    reference = cached_reference_value(mu_ref_value; kwargs...)
    return (evaluated - reference) / abs(reference)
end

function surrogate_solve(demand, mu_value; kwargs...)
    decoded = ContextualDFL.decode_scenario_collection(vector_decoder, demand; nr_scenarios=1)
    return ContextualDFL.solve(solver, program, decoded...; mu_kw(mu_value)..., kwargs...)[1]
end

function recourse_cost(z, mu_value; kwargs...)
    return ContextualDFL.cost_function(
        program,
        solver,
        z,
        decoded_actual...;
        mu_kw(mu_value)...,
        kwargs...,
    )
end

function recourse_gradient_z(z, mu_value; kwargs...)
    return Flux.gradient(z_value -> recourse_cost(z_value, mu_value; kwargs...), z)[1]
end

function measure_suite(
    implementation;
    mu_in_value=mu_in,
    mu_ref_value=mu_ref,
    loss_kwargs=(;),
    solve_kwargs=loss_kwargs,
)
    cached_reference_value(mu_ref_value; loss_kwargs...)
    measure(
        implementation,
        "forward_loss",
        () -> new_loss(predicted_demand, mu_in_value, mu_ref_value; loss_kwargs...),
    )
    measure(
        implementation,
        "gradient_demand",
        () -> Flux.gradient(
            d -> new_loss(d, mu_in_value, mu_ref_value; loss_kwargs...),
            predicted_demand,
        )[1],
    )
    measure(
        implementation,
        "surrogate_solve",
        () -> surrogate_solve(predicted_demand, mu_in_value; solve_kwargs...),
    )
    measure(
        implementation,
        "recourse_cost",
        () -> recourse_cost(z_for_cost, mu_ref_value; solve_kwargs...),
    )
    measure(
        implementation,
        "recourse_gradient_z",
        () -> recourse_gradient_z(z_for_cost, mu_ref_value; solve_kwargs...),
    )
    measure(
        implementation,
        "relative_display_loss",
        () -> new_relative_display(
            predicted_demand,
            mu_in_value,
            mu_ref_value;
            loss_kwargs...,
        ),
    )
end

measure_suite("new_default")
measure_suite("new_tol_1e-9"; loss_kwargs=(; tol=1e-9), solve_kwargs=(; tol=1e-9))
measure_suite(
    "new_rho_only_0.1";
    mu_in_value=0.0,
    mu_ref_value=0.0,
    loss_kwargs=(; rho_in=0.1, rho_ref=0.1),
    solve_kwargs=(; rho=0.1),
)
measure_suite(
    "new_mu_rho_0.1";
    loss_kwargs=(; rho_in=0.1, rho_ref=0.1),
    solve_kwargs=(; rho=0.1),
)

profile_once(
    "new_default",
    "gradient_demand",
    () -> Flux.gradient(d -> new_loss(d, mu_in, mu_ref), predicted_demand)[1],
)
"""

old_script = write_temp_script(OLD_SCRIPT)
new_script = write_temp_script(NEW_SCRIPT)
old_project = mktempdir()

try
    old_output = run_benchmark(
        "old ProblemBasedScenarioGeneration",
        benchmark_command(old_script, old_project, REPO_ROOT),
    )
    new_output = run_benchmark(
        "new ContextualDFL",
        benchmark_command(new_script, NEW_PROJECT, REPO_ROOT),
    )

    rows = vcat(parse_results(old_output), parse_results(new_output))
    print_table(rows)
finally
    rm(old_script; force=true)
    rm(new_script; force=true)
end

# END FILE: src/ContextualDFL/ContextualDFLExperiments/experiments/resource_allocation_annealing/benchmark_old_vs_new.jl

# BEGIN FILE: src/ContextualDFL/ContextualDFLExperiments/experiments/resource_allocation_annealing/profile_old_vs_new_annealing.jl
import Dates
import Printf

const REPO_ROOT = normpath(joinpath(@__DIR__, "..", "..", "..", "..", ".."))
const NEW_PROJECT = normpath(joinpath(@__DIR__, "..", ".."))
const PROFILE_EPOCHS = parse(Int, get(ENV, "CDFL_PROFILE_EPOCHS", "3"))
const PROFILE_MODE = get(ENV, "CDFL_PROFILE_MODE", "both")
const PROFILE_REPEATS = parse(Int, get(ENV, "CDFL_PROFILE_REPEATS", "3"))
const PROFILE_WARMUPS = parse(Int, get(ENV, "CDFL_PROFILE_WARMUPS", "2"))
const PROFILE_DELAY = parse(Float64, get(ENV, "CDFL_PROFILE_DELAY", "0.001"))
const PROFILE_MINCOUNT = parse(Int, get(ENV, "CDFL_PROFILE_MINCOUNT", "5"))
const PROFILE_FULL_SAMPLES = parse(Int, get(ENV, "CDFL_PROFILE_FULL_SAMPLES", "100"))
const PROFILE_SMOKE_SAMPLES = parse(Int, get(ENV, "CDFL_PROFILE_SMOKE_SAMPLES", "10"))
const RESULT_ROOT = normpath(
    joinpath(
        @__DIR__,
        "results",
        "profiling_" * replace(string(Dates.now()), ':' => '-'),
    ),
)

function write_temp_script(source)
    path = tempname() * ".jl"
    write(path, source)
    return path
end

function worker_command(script, project, mode, stage_limit, impl, training_samples)
    out_dir = joinpath(RESULT_ROOT, mode, impl)
    mkpath(out_dir)
    return `$(Base.julia_cmd()) --project=$(project) $(script) $(REPO_ROOT) $(out_dir) $(PROFILE_EPOCHS) $(stage_limit) $(mode) $(PROFILE_REPEATS) $(PROFILE_WARMUPS) $(PROFILE_DELAY) $(PROFILE_MINCOUNT) $(training_samples)`
end

function run_worker(label, command)
    println()
    println("Running $(label)...")
    output = IOBuffer()
    io = open(command, "r")
    try
        for line in eachline(io)
            println(line)
            println(output, line)
        end
    finally
        close(io)
    end
    return String(take!(output))
end

function parse_rows(output, prefix, fields)
    rows = NamedTuple[]
    for line in split(output, '\n')
        startswith(line, prefix * "\t") || continue
        parts = split(line, '\t')
        length(parts) == length(fields) + 1 || continue
        values = parts[2:end]
        push!(rows, NamedTuple{Tuple(Symbol.(fields))}(Tuple(values)))
    end
    return rows
end

function print_bucket_table(rows)
    isempty(rows) && return
    println()
    println("Bucket timing summary")
    Printf.@printf("%-6s %-7s %-28s %11s %9s %11s %8s\n", "impl", "mode", "bucket", "seconds", "count", "alloc_MiB", "% total")
    println("-"^90)
    totals = Dict{Tuple{String,String},Float64}()
    for row in rows
        key = (row.impl, row.mode)
        totals[key] = get(totals, key, 0.0) + parse(Float64, row.seconds)
    end
    for row in rows
        seconds = parse(Float64, row.seconds)
        percent = 100 * seconds / max(totals[(row.impl, row.mode)], eps())
        Printf.@printf(
            "%-6s %-7s %-28s %11.3f %9s %11.2f %7.1f%%\n",
            row.impl,
            row.mode,
            row.bucket,
            seconds,
            row.count,
            parse(Float64, row.alloc_mib),
            percent,
        )
    end
end

function print_micro_table(rows)
    isempty(rows) && return
    println()
    println("Micro timing summary")
    Printf.@printf("%-6s %-7s %-28s %10s %10s %10s %11s %s\n", "impl", "mode", "measurement", "min_s", "mean_s", "median_s", "alloc_MiB", "value")
    println("-"^105)
    for row in rows
        Printf.@printf(
            "%-6s %-7s %-28s %10.4f %10.4f %10.4f %11.2f %s\n",
            row.impl,
            row.mode,
            row.measurement,
            parse(Float64, row.min_seconds),
            parse(Float64, row.mean_seconds),
            parse(Float64, row.median_seconds),
            parse(Float64, row.alloc_mib),
            row.value,
        )
    end
end

function print_run_table(rows)
    isempty(rows) && return
    println()
    println("Run summary")
    Printf.@printf("%-6s %-7s %7s %8s %11s %12s %12s\n", "impl", "mode", "stages", "epochs", "iterations", "total_s", "iter_ms")
    println("-"^78)
    for row in rows
        total = parse(Float64, row.total_seconds)
        iterations = parse(Int, row.iterations)
        Printf.@printf(
            "%-6s %-7s %7s %8s %11s %12.3f %12.3f\n",
            row.impl,
            row.mode,
            row.stages,
            row.epochs_per_stage,
            row.iterations,
            total,
            1000 * total / max(iterations, 1),
        )
    end
end

const SHARED_PROFILING_UTILS = raw"""
using LinearAlgebra
using Printf
using Profile
using Serialization
using Statistics

mutable struct BucketStats
    seconds::Float64
    bytes::Int
    count::Int
end

BucketStats() = BucketStats(0.0, 0, 0)

function add_bucket!(buckets, name, sample)
    bucket = get!(buckets, name, BucketStats())
    bucket.seconds += sample.time
    bucket.bytes += sample.bytes
    bucket.count += 1
    return sample.value
end

function time_bucket!(f, buckets, name)
    return add_bucket!(buckets, name, @timed f())
end

function emit_buckets(impl, mode, buckets)
    for name in sort(collect(keys(buckets)))
        bucket = buckets[name]
        Printf.@printf(
            "BUCKET\t%s\t%s\t%s\t%.9f\t%d\t%.6f\n",
            impl,
            mode,
            name,
            bucket.seconds,
            bucket.count,
            bucket.bytes / 1024^2,
        )
    end
end

function summary_value(value)
    if value isa Number
        return Printf.@sprintf("%.6g", Float64(value))
    elseif value isa AbstractArray
        return Printf.@sprintf("array(len=%d,norm=%.6g)", length(value), LinearAlgebra.norm(value))
    else
        return string(typeof(value))
    end
end

function measure(impl, mode, measurement, f, repeats, warmups)
    for _ in 1:warmups
        f()
    end
    samples = NamedTuple[]
    for _ in 1:repeats
        GC.gc()
        push!(samples, @timed f())
    end
    times = [sample.time for sample in samples]
    bytes = [sample.bytes for sample in samples]
    Printf.@printf(
        "MICRO\t%s\t%s\t%s\t%.9f\t%.9f\t%.9f\t%.6f\t%s\n",
        impl,
        mode,
        measurement,
        minimum(times),
        Statistics.mean(times),
        Statistics.median(times),
        Statistics.mean(bytes) / 1024^2,
        summary_value(samples[end].value),
    )
    return samples[end].value
end

function profile_to_file(f, path, delay, mincount)
    mkpath(dirname(path))
    Profile.init(delay=delay)
    Profile.clear()
    result = @profile f()
    open(path, "w") do io
        Profile.print(io; format=:flat, sortedby=:count, mincount=mincount, maxdepth=40)
    end
    return result
end

function emit_profile_file(impl, mode, name, path)
    println("PROFILE_FILE\t$(impl)\t$(mode)\t$(name)\t$(path)")
end
"""

const OLD_WORKER = SHARED_PROFILING_UTILS * raw"""
repo_root = ARGS[1]
out_dir = ARGS[2]
epochs_per_stage = parse(Int, ARGS[3])
stage_limit = parse(Int, ARGS[4])
mode = ARGS[5]
repeats = parse(Int, ARGS[6])
warmups = parse(Int, ARGS[7])
profile_delay = parse(Float64, ARGS[8])
profile_mincount = parse(Int, ARGS[9])
training_samples = parse(Int, ARGS[10])
impl = "old"

import Pkg
env_seconds = @elapsed redirect_stdout(devnull) do
    redirect_stderr(devnull) do
        Pkg.develop(path=joinpath(repo_root, "src", "ProblemBasedScenarioGeneration"))
        Pkg.add(["ChainRulesCore", "Flux", "Plots"])
        Pkg.instantiate()
    end
end
Printf.@printf("ENV_SETUP\t%s\t%s\t%.9f\n", impl, mode, env_seconds)

using ChainRulesCore
using Flux
using Plots
using ProblemBasedScenarioGeneration
using Random

using ProblemBasedScenarioGeneration: ResourceAllocationProblemData,
    ResourceAllocationProblem,
    dataGeneration
import ProblemBasedScenarioGeneration: loss, relative_loss, surrogate_solution

include(joinpath(repo_root, "scripts", "resource_allocation_prototype", "custom_code", "neural_net.jl"))
include(joinpath(repo_root, "scripts", "resource_allocation_prototype", "parameters.jl"))

Random.seed!(1234)

cz = vec(getfield(Main, Symbol("cz")))
qw = vec(getfield(Main, Symbol("qw")))
rho_i = vec(getfield(Main, Symbol("ρᵢ")))
service_rate_parameters = getfield(Main, Symbol("μᵢⱼ"))

problem_data = ResourceAllocationProblemData(service_rate_parameters, cz, qw, rho_i)
problem = ResourceAllocationProblem(problem_data)

Ntraining_samples = training_samples
Ntesting_samples = 30
sigma = 5
p = 2
L = 3
N_xi_per_x = 100
batchsize = 1
step_size = 1e-3
param_list = [1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.08, 0.06, 0.04, 0.02, 0.01]
stage_specs = [(reg, reg) for reg in param_list]
push!(stage_specs, (param_list[end], 0.0))
stage_specs = stage_specs[1:min(stage_limit, length(stage_specs))]

setup_buckets = Dict{String,BucketStats}()
data_set_training, data_set_testing, _, _ = time_bucket!(setup_buckets, "data_generation") do
    dataGeneration(problem, Ntraining_samples, Ntesting_samples, N_xi_per_x, sigma, p, L)
end
model = time_bucket!(setup_buckets, "model_construction") do
    construct_neural_network(problem; nr_of_scenarios=1)
end

xs = collect(keys(data_set_training))
xis = collect(values(data_set_training))
N = length(xs)

function old_loss_mb(loss_fn, model, Xb, Xib)
    return Statistics.mean(loss_fn(model(Xb[:, i:i]), Xib[:, i:i]) for i in 1:size(Xb, 2))
end

function old_relative_loss_mb(relative_fn, model, Xb, Xib)
    return Statistics.mean(relative_fn(model(Xb[:, i:i]), Xib[:, i:i]) for i in 1:size(Xb, 2))
end

function run_epoch!(model, opt, loss_fn, relative_fn, buckets)
    state = time_bucket!(buckets, "optimizer_setup") do
        Flux.setup(opt, model)
    end
    epoch_losses = Float64[]
    for idxs in Iterators.partition(1:N, batchsize)
        Xb = time_bucket!(buckets, "batch_materialization") do
            hcat(xs[idxs]...)
        end
        Xib = time_bucket!(buckets, "batch_materialization") do
            hcat(xis[idxs]...)
        end
        gs = time_bucket!(buckets, "training_gradient") do
            Flux.gradient(model) do m
                old_loss_mb(loss_fn, m, Xb, Xib)
            end
        end
        gmodel = gs isa Tuple ? gs[1] : gs
        time_bucket!(buckets, "optimizer_update") do
            Flux.update!(state, model, gmodel)
        end
        display_value = time_bucket!(buckets, "relative_display_loss") do
            old_relative_loss_mb(relative_fn, model, Xb, Xib)
        end
        push!(epoch_losses, Float64(display_value))
    end
    time_bucket!(buckets, "gc") do
        GC.gc()
    end
    return Statistics.mean(epoch_losses)
end

function run_annealing!(model, buckets)
    stage_seconds = Float64[]
    total_iterations = 0
    for (stage_index, (reg_param_surr, reg_param_prim)) in enumerate(stage_specs)
        println("Starting old stage $(stage_index) with reg_param_surr=$(reg_param_surr), reg_param_prim=$(reg_param_prim), epochs=$(epochs_per_stage)")
        stage_started = time()
        stage_losses = Float64[]
        loss_fn(output, actual) = loss(problem, reg_param_surr, reg_param_prim, output, actual)
        relative_fn(output, actual) = relative_loss(problem, reg_param_surr, reg_param_prim, output, actual)
        for epoch in 1:epochs_per_stage
            average_display = run_epoch!(model, Flux.Adam(step_size), loss_fn, relative_fn, buckets)
            push!(stage_losses, average_display)
            total_iterations += N
            println("Epoch $(epoch) with avg loss $(average_display) ($(N) iterations)")
        end
        time_bucket!(buckets, "save_model") do
            Serialization.serialize(joinpath(out_dir, "old_model_stage_$(stage_index).jls"), model)
        end
        time_bucket!(buckets, "save_state") do
            Serialization.serialize(
                joinpath(out_dir, "old_state_stage_$(stage_index).jls"),
                (; model=model, data_set_training=data_set_training, data_set_testing=data_set_testing, stage=stage_index),
            )
        end
        time_bucket!(buckets, "plot_creation") do
            Plots.plot(1:length(stage_losses), stage_losses; xlabel="Epoch", ylabel="Loss", title="Training Loss")
        end
        stage_elapsed = time() - stage_started
        push!(stage_seconds, stage_elapsed)
        Printf.@printf("STAGE\t%s\t%s\t%d\t%.9f\t%d\n", impl, mode, stage_index, stage_elapsed, epochs_per_stage * N)
    end
    return total_iterations, stage_seconds
end

function run_micro_measurements()
    Xb = hcat(xs[1])
    Xib = hcat(xis[1])
    reg_param_surr = first(stage_specs)[1]
    reg_param_prim = first(stage_specs)[2]
    loss_fn(output, actual) = loss(problem, reg_param_surr, reg_param_prim, output, actual)
    relative_fn(output, actual) = relative_loss(problem, reg_param_surr, reg_param_prim, output, actual)
    predicted_demand = reshape(50.0 .+ 0.1 .* collect(1:size(problem.problem_data.service_rate_parameters, 2)), :, 1)
    actual_demand = reshape(55.0 .+ 0.2 .* collect(1:size(problem.problem_data.service_rate_parameters, 2)), :, 1)
    z_for_cost = surrogate_solution(problem, reg_param_surr, predicted_demand)

    measure(impl, mode, "model_forward", () -> model(Xb), repeats, warmups)
    measure(impl, mode, "loss_forward", () -> loss_fn(model(Xb), Xib), repeats, warmups)
    measure(impl, mode, "training_gradient", () -> Flux.gradient(m -> old_loss_mb(loss_fn, m, Xb, Xib), model)[1], repeats, warmups)
    measure(impl, mode, "relative_display_loss", () -> old_relative_loss_mb(relative_fn, model, Xb, Xib), repeats, warmups)
    measure(impl, mode, "forward_loss_fixed_demand", () -> loss(problem, reg_param_surr, reg_param_prim, predicted_demand, actual_demand), repeats, warmups)
    measure(impl, mode, "gradient_demand", () -> Flux.gradient(d -> loss(problem, reg_param_surr, reg_param_prim, d, actual_demand), predicted_demand)[1], repeats, warmups)
    measure(impl, mode, "surrogate_solve", () -> surrogate_solution(problem, reg_param_surr, predicted_demand), repeats, warmups)
    measure(impl, mode, "recourse_cost", () -> primal_problem_cost(problem, reg_param_prim, actual_demand, z_for_cost), repeats, warmups)
    measure(impl, mode, "recourse_gradient_z", () -> derivative_primal_problem_cost(problem, reg_param_prim, actual_demand, z_for_cost), repeats, warmups)

    gradient_profile_path = joinpath(out_dir, "profile_old_training_gradient.txt")
    profile_to_file(gradient_profile_path, profile_delay, profile_mincount) do
        Flux.gradient(m -> old_loss_mb(loss_fn, m, Xb, Xib), model)[1]
    end
    emit_profile_file(impl, mode, "training_gradient", gradient_profile_path)

    relative_profile_path = joinpath(out_dir, "profile_old_relative_loss.txt")
    profile_to_file(relative_profile_path, profile_delay, profile_mincount) do
        old_relative_loss_mb(relative_fn, model, Xb, Xib)
    end
    emit_profile_file(impl, mode, "relative_loss", relative_profile_path)

    if mode == "smoke"
        iteration_profile_path = joinpath(out_dir, "profile_old_training_iteration.txt")
        profile_to_file(iteration_profile_path, profile_delay, profile_mincount) do
            state = Flux.setup(Flux.Adam(step_size), model)
            gs = Flux.gradient(model) do m
                old_loss_mb(loss_fn, m, Xb, Xib)
            end
            gmodel = gs isa Tuple ? gs[1] : gs
            Flux.update!(state, model, gmodel)
            old_relative_loss_mb(relative_fn, model, Xb, Xib)
        end
        emit_profile_file(impl, mode, "training_iteration", iteration_profile_path)
    end
end

run_micro_measurements()

full_buckets = merge(Dict{String,BucketStats}(), setup_buckets)
run_sample = @timed begin
    run_annealing!(model, full_buckets)
end
run_result = run_sample.value

total_iterations, stage_seconds = run_result
total_seconds = sum(stage_seconds)
Printf.@printf(
    "SUMMARY\t%s\t%s\t%d\t%d\t%d\t%.9f\n",
    impl,
    mode,
    length(stage_specs),
    epochs_per_stage,
    total_iterations,
    total_seconds,
)
emit_buckets(impl, mode, full_buckets)
"""

const NEW_WORKER = SHARED_PROFILING_UTILS * raw"""
repo_root = ARGS[1]
out_dir = ARGS[2]
epochs_per_stage = parse(Int, ARGS[3])
stage_limit = parse(Int, ARGS[4])
mode = ARGS[5]
repeats = parse(Int, ARGS[6])
warmups = parse(Int, ARGS[7])
profile_delay = parse(Float64, ARGS[8])
profile_mincount = parse(Int, ARGS[9])
training_samples = parse(Int, ARGS[10])
impl = "new"

using ContextualDFL
using ContextualDFLExperiments
using Random

const Flux = ContextualDFL.Flux

Random.seed!(1234)

Ntraining_samples = training_samples
Ntesting_samples = 30
sigma = 5
p = 2
L = 3
N_xi_per_x = 100
batchsize = 1
step_size = 1e-3
nr_scenarios = 1
param_list = [1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.08, 0.06, 0.04, 0.02, 0.01]
stage_specs = [(reg, reg) for reg in param_list]
push!(stage_specs, (param_list[end], 0.0))
stage_specs = stage_specs[1:min(stage_limit, length(stage_specs))]

setup_buckets = Dict{String,BucketStats}()
problem = time_bucket!(setup_buckets, "problem_construction") do
    ResourceAllocationProblem(default_resource_allocation_problem_data())
end
program = stochastic_program(problem)
solver = ContextualDFL.Solver(ContextualDFL.IpoptSolver(), ContextualDFL.HiGHSSolver())
vector_decoder = ResourceAllocationDemandVectorDecoder(problem)
parametric_decoder = ResourceAllocationDemandParametricDecoder(problem)
loss_object = ContextualDFL.DflScenLoss(
    vector_decoder,
    parametric_decoder,
    solver,
    program;
    nr_scenarios=nr_scenarios,
)

rng = Random.MersenneTwister(1234)
context_generator = ResourceAllocationContextDataGenerator(rng=rng)
scenario_generator = ResourceAllocationScenarioDataGenerator(
    problem;
    sigma=sigma,
    p=p,
    L=L,
    rng=rng,
)
data_set_training = time_bucket!(setup_buckets, "data_generation") do
    contexts = [Vector{Float64}(context_generator()) for _ in 1:Ntraining_samples]
    scenarios = [scenario_generator(context) for context in contexts]
    generate_contextual_data_set(contexts, scenarios)
end
demand_count = size(problem.problem_data.service_rate_parameters, 2)
model = time_bucket!(setup_buckets, "model_construction") do
    Flux.Chain(
        Flux.Dense(3 => 128, Flux.relu),
        Flux.Dense(128 => 128, Flux.relu),
        Flux.Dense(128 => 128, Flux.relu),
        Flux.Dense(128 => demand_count * nr_scenarios, Flux.relu),
    ) |> Flux.f64
end

N = length(data_set_training)
loss_kwargs = (; nr_scenarios=nr_scenarios)

context_at(index) = data_set_training[index].context
scenario_at(index) = data_set_training[index].scenario_parameters
display_reference_input(index) =
    reduce(vcat, (scenario.h_eq_xi for scenario in scenario_at(index)))

relative_display_loss(loss_value, reference_value) =
    (Float64(loss_value) - Float64(reference_value)) / abs(Float64(reference_value))

function compute_display_reference_values(mu_ref)
    return [
        Float64(
            loss_object(
                display_reference_input(index),
                scenario_at(index),
                mu_ref,
                mu_ref;
                loss_kwargs...,
            ),
        ) for index in 1:N
    ]
end

function display_reference_values!(cache, mu_ref, buckets)
    haskey(cache, mu_ref) && return cache[mu_ref]
    values = time_bucket!(buckets, "display_reference_precompute") do
        compute_display_reference_values(mu_ref)
    end
    cache[mu_ref] = values
    return values
end

function new_loss_batch(loss_fn, model, idxs)
    return Statistics.mean(
        loss_fn(model(context_at(index)), scenario_at(index)) for index in idxs
    )
end

function new_display_batch(loss_value, reference_values, idxs)
    reference_mean = Statistics.mean(reference_values[index] for index in idxs)
    return relative_display_loss(loss_value, reference_mean)
end

function run_epoch!(model, opt, loss_fn, reference_values, buckets)
    state = time_bucket!(buckets, "optimizer_setup") do
        Flux.setup(opt, model)
    end
    epoch_losses = Float64[]
    for idxs_iter in Iterators.partition(1:N, batchsize)
        idxs = time_bucket!(buckets, "batch_materialization") do
            collect(idxs_iter)
        end
        loss_value, gradients = time_bucket!(buckets, "training_gradient") do
            Flux.withgradient(model) do trainable_model
                new_loss_batch(loss_fn, trainable_model, idxs)
            end
        end
        time_bucket!(buckets, "optimizer_update") do
            Flux.update!(state, model, gradients[1])
        end
        display_value = time_bucket!(buckets, "relative_display_loss") do
            new_display_batch(loss_value, reference_values, idxs)
        end
        push!(epoch_losses, Float64(display_value))
    end
    return Statistics.mean(epoch_losses)
end

function run_annealing!(model, buckets)
    stage_seconds = Float64[]
    total_iterations = 0
    display_reference_cache = Dict{Any,Vector{Float64}}()
    for (stage_index, (mu_in, mu_ref)) in enumerate(stage_specs)
        println("Starting new stage $(stage_index) with mu_in=$(mu_in), mu_ref=$(mu_ref), epochs=$(epochs_per_stage)")
        stage_started = time()
        loss_fn(output, scenario_parameters) =
            loss_object(output, scenario_parameters, mu_in, mu_ref; loss_kwargs...)
        reference_values = display_reference_values!(display_reference_cache, mu_ref, buckets)
        for epoch in 1:epochs_per_stage
            average_display = run_epoch!(model, Flux.Adam(step_size), loss_fn, reference_values, buckets)
            total_iterations += N
            println("Epoch $(epoch) with avg loss $(average_display) ($(N) iterations)")
        end
        time_bucket!(buckets, "save_model") do
            Serialization.serialize(joinpath(out_dir, "new_model_stage_$(stage_index).jls"), model)
        end
        time_bucket!(buckets, "save_state") do
            Serialization.serialize(
                joinpath(out_dir, "new_state_stage_$(stage_index).jls"),
                (; model=model, data_set_training=data_set_training, problem=problem, stage=stage_index),
            )
        end
        stage_elapsed = time() - stage_started
        push!(stage_seconds, stage_elapsed)
        Printf.@printf("STAGE\t%s\t%s\t%d\t%.9f\t%d\n", impl, mode, stage_index, stage_elapsed, epochs_per_stage * N)
    end
    return total_iterations, stage_seconds
end

function run_micro_measurements()
    idxs = [1]
    mu_in, mu_ref = first(stage_specs)
    loss_fn(output, scenario_parameters) =
        loss_object(output, scenario_parameters, mu_in, mu_ref; loss_kwargs...)
    reference_values = compute_display_reference_values(mu_ref)
    loss_value_for_display = loss_fn(model(context_at(1)), scenario_at(1))

    predicted_demand = 50.0 .+ 0.1 .* collect(1:demand_count)
    actual_demand = 55.0 .+ 0.2 .* collect(1:demand_count)
    actual_collection = [ContextualDFL.ParametricScenario(; h_eq_xi=actual_demand)]
    decoded_predicted = ContextualDFL.decode_scenario_collection(
        vector_decoder,
        predicted_demand;
        nr_scenarios=1,
    )
    decoded_actual = ContextualDFL.decode_scenario_collection(parametric_decoder, actual_collection)
    z_for_cost = ContextualDFL.solve(solver, program, decoded_predicted...; μ=mu_in)[1]

    measure(impl, mode, "model_forward", () -> model(context_at(1)), repeats, warmups)
    measure(impl, mode, "loss_forward", () -> loss_fn(model(context_at(1)), scenario_at(1)), repeats, warmups)
    measure(impl, mode, "training_gradient", () -> Flux.withgradient(m -> new_loss_batch(loss_fn, m, idxs), model)[2][1], repeats, warmups)
    measure(impl, mode, "relative_display_loss", () -> new_display_batch(loss_value_for_display, reference_values, idxs), repeats, warmups)
    measure(impl, mode, "forward_loss_fixed_demand", () -> loss_object(predicted_demand, actual_collection, mu_in, mu_ref; nr_scenarios=1), repeats, warmups)
    measure(impl, mode, "gradient_demand", () -> Flux.gradient(d -> loss_object(d, actual_collection, mu_in, mu_ref; nr_scenarios=1), predicted_demand)[1], repeats, warmups)
    measure(impl, mode, "surrogate_solve", () -> ContextualDFL.solve(solver, program, decoded_predicted...; μ=mu_in)[1], repeats, warmups)
    measure(impl, mode, "recourse_cost", () -> ContextualDFL.cost_function(program, solver, z_for_cost, decoded_actual...; μ=mu_ref), repeats, warmups)
    measure(impl, mode, "recourse_gradient_z", () -> Flux.gradient(z -> ContextualDFL.cost_function(program, solver, z, decoded_actual...; μ=mu_ref), z_for_cost)[1], repeats, warmups)

    gradient_profile_path = joinpath(out_dir, "profile_new_training_gradient.txt")
    profile_to_file(gradient_profile_path, profile_delay, profile_mincount) do
        Flux.withgradient(m -> new_loss_batch(loss_fn, m, idxs), model)[2][1]
    end
    emit_profile_file(impl, mode, "training_gradient", gradient_profile_path)

    relative_profile_path = joinpath(out_dir, "profile_new_relative_loss.txt")
    profile_to_file(relative_profile_path, profile_delay, profile_mincount) do
        new_display_batch(loss_value_for_display, reference_values, idxs)
    end
    emit_profile_file(impl, mode, "relative_loss", relative_profile_path)

    if mode == "smoke"
        iteration_profile_path = joinpath(out_dir, "profile_new_training_iteration.txt")
        profile_to_file(iteration_profile_path, profile_delay, profile_mincount) do
            state = Flux.setup(Flux.Adam(step_size), model)
            loss_value, gradients = Flux.withgradient(model) do trainable_model
                new_loss_batch(loss_fn, trainable_model, idxs)
            end
            Flux.update!(state, model, gradients[1])
            new_display_batch(loss_value, reference_values, idxs)
        end
        emit_profile_file(impl, mode, "training_iteration", iteration_profile_path)
    end
end

run_micro_measurements()

full_buckets = merge(Dict{String,BucketStats}(), setup_buckets)
run_sample = @timed begin
    run_annealing!(model, full_buckets)
end
run_result = run_sample.value

total_iterations, stage_seconds = run_result
total_seconds = sum(stage_seconds)
Printf.@printf(
    "SUMMARY\t%s\t%s\t%d\t%d\t%d\t%.9f\n",
    impl,
    mode,
    length(stage_specs),
    epochs_per_stage,
    total_iterations,
    total_seconds,
)
emit_buckets(impl, mode, full_buckets)
"""

function modes_to_run()
    if PROFILE_MODE == "both"
        return [("smoke", 1), ("full", 12)]
    elseif PROFILE_MODE == "smoke"
        return [("smoke", 1)]
    elseif PROFILE_MODE == "full"
        return [("full", 12)]
    else
        error("CDFL_PROFILE_MODE must be one of: both, smoke, full")
    end
end

old_script = write_temp_script(OLD_WORKER)
new_script = write_temp_script(NEW_WORKER)
old_project = mktempdir()
mkpath(RESULT_ROOT)

all_outputs = String[]
try
    println("Profile output directory: $(RESULT_ROOT)")
    for (mode, stage_limit) in modes_to_run()
        training_samples = mode == "smoke" ? PROFILE_SMOKE_SAMPLES : PROFILE_FULL_SAMPLES
        push!(
            all_outputs,
            run_worker(
                "old $(mode) profile",
                worker_command(old_script, old_project, mode, stage_limit, "old", training_samples),
            ),
        )
        push!(
            all_outputs,
            run_worker(
                "new $(mode) profile",
                worker_command(new_script, NEW_PROJECT, mode, stage_limit, "new", training_samples),
            ),
        )
    end
finally
    rm(old_script; force=true)
    rm(new_script; force=true)
end

combined_output = join(all_outputs, "\n")
summary_rows = parse_rows(
    combined_output,
    "SUMMARY",
    ["impl", "mode", "stages", "epochs_per_stage", "iterations", "total_seconds"],
)
bucket_rows = parse_rows(
    combined_output,
    "BUCKET",
    ["impl", "mode", "bucket", "seconds", "count", "alloc_mib"],
)
micro_rows = parse_rows(
    combined_output,
    "MICRO",
    [
        "impl",
        "mode",
        "measurement",
        "min_seconds",
        "mean_seconds",
        "median_seconds",
        "alloc_mib",
        "value",
    ],
)

print_run_table(summary_rows)
print_bucket_table(bucket_rows)
print_micro_table(micro_rows)

println()
println("Profile files are under: $(RESULT_ROOT)")

# END FILE: src/ContextualDFL/ContextualDFLExperiments/experiments/resource_allocation_annealing/profile_old_vs_new_annealing.jl

# BEGIN FILE: src/ContextualDFL/ContextualDFLExperiments/src/ContextualDFLExperiments.jl
module ContextualDFLExperiments

import ChainRulesCore
import ContextualDFL

export ContextDataGenerator,
    KNearestNeighborsPolicy,
    LeastSquaresPolicy,
    Policy,
    ProgramInstance,
    ResidualSampleAverageApproximationPolicy,
    ResourceAllocationContextDataGenerator,
    ResourceAllocationDemandParametricDecoder,
    ResourceAllocationDemandVectorDecoder,
    ResourceAllocationProblem,
    ResourceAllocationProblemData,
    ResourceAllocationScenarioDataGenerator,
    RandomYieldParametricDecoder,
    RandomYieldProblem,
    ScenarioDataGenerator,
    ScenarioGenerationPolicy,
    SampleAverageApproximationPolicy,
    ShipmentPlanningParametricDecoder,
    ShipmentPlanningProblem,
    base_scenario,
    default_resource_allocation_problem_data,
    default_knn_k,
    evaluate_policy,
    evaluate_policy_against_optimum,
    generate_benchmark_contexts,
    generate_benchmark_dataset,
    generate_benchmark_scenarios,
    generate_contextual_data_set,
    generate_decision_set,
    infer,
    random_yield_probabilities,
    random_yield_support_scenarios,
    sample_random_yield_scenario,
    solve_dataset_to_optimality,
    summarize_regret,
    summarize_values,
    stochastic_program,
    transshipment_decoder,
    TransShipmentComponentVectorDecoder,
    TransShipmentExperimentProblem,
    UnreliableNewsvendorParametricDecoder,
    UnreliableNewsvendorProblem,
    UnreliableNewsvendorProblemData,
    unreliable_newsvendor_scenario

include("data_generation/generate_contextual_data_set.jl")
include("data_generation/benchmark_dataset.jl")
include("data_generation/contextual_generators/ContextDataGenerator.jl")
include("data_generation/scenario_generators/ScenarioDataGenerator.jl")
include("program_instance/ProgramInstance.jl")
include("testing/policies/Policy.jl")
include("testing/policies/ScenarioGenerationPolicy.jl")
include("testing/policies/BaselinePolicies.jl")
include("testing/evaluation/evaluation.jl")

include("implementations/resource_allocation_problem/problem_data/parameters.jl")
include("implementations/resource_allocation_problem/program_instance/ResourceAllocationProblem.jl")
include("implementations/resource_allocation_problem/scenario_decoders/ResourceAllocationDemandDecoders.jl")
include("implementations/resource_allocation_problem/data_generators/ResourceAllocationContextDataGenerator.jl")
include("implementations/resource_allocation_problem/data_generators/ResourceAllocationScenarioDataGenerator.jl")

include("implementations/shipment_planning/ShipmentPlanningProblem.jl")
include("implementations/shipment_planning/ShipmentPlanningDataGenerators.jl")
include("implementations/shipment_planning/ShipmentPlanningDecoders.jl")

include("implementations/transshipment_problem/TransShipmentExperimentProblem.jl")
include("implementations/transshipment_problem/TransShipmentDataGenerators.jl")
include("implementations/transshipment_problem/TransShipmentDecoders.jl")

include("implementations/random_yield_problem/RandomYieldProblem.jl")
include("implementations/random_yield_problem/RandomYieldDataGenerators.jl")
include("implementations/random_yield_problem/RandomYieldDecoders.jl")

include("implementations/unreliable_newsvendor/UnreliableNewsvendorProblem.jl")
include("implementations/unreliable_newsvendor/UnreliableNewsvendorDataGenerators.jl")
include("implementations/unreliable_newsvendor/UnreliableNewsvendorDecoders.jl")

end

# END FILE: src/ContextualDFL/ContextualDFLExperiments/src/ContextualDFLExperiments.jl

# BEGIN FILE: src/ContextualDFL/ContextualDFLExperiments/src/data_generation/benchmark_dataset.jl
import Random

function generate_benchmark_contexts(problem; n_contexts, rng=Random.default_rng())
    error("Benchmark context generation is not defined for $(typeof(problem)).")
end

function generate_benchmark_scenarios(
    problem,
    context;
    n_scenarios,
    rng=Random.default_rng(),
)
    error("Benchmark scenario generation is not defined for $(typeof(problem)).")
end

function generate_benchmark_dataset(
    problem;
    n_contexts,
    scenarios_per_context,
    seed=1,
    rng=Random.MersenneTwister(seed),
)
    context_count = _checked_positive_integer(n_contexts, :n_contexts)
    scenario_count = _checked_positive_integer(scenarios_per_context, :scenarios_per_context)

    contexts = generate_benchmark_contexts(problem; n_contexts=context_count, rng=rng)
    scenario_collections = [
        generate_benchmark_scenarios(problem, context; n_scenarios=scenario_count, rng=rng)
        for context in contexts
    ]

    return generate_contextual_data_set(contexts, scenario_collections)
end

function _checked_positive_integer(value, name::Symbol)
    value isa Integer ||
        throw(ArgumentError("$(name) must be a positive integer, got $(typeof(value))."))

    checked_value = Int(value)
    checked_value > 0 ||
        throw(ArgumentError("$(name) must be positive, got $checked_value."))
    return checked_value
end

function _checked_context_vector(context, expected_length)
    context isa AbstractVector ||
        throw(ArgumentError("context must be an AbstractVector."))
    length(context) == expected_length ||
        throw(DimensionMismatch("context must have length $expected_length."))
    return Vector{Float64}(context)
end

function _checked_vector_or_default(value, default, expected_length, name::Symbol)
    vector = isnothing(value) ? default : value
    checked_vector = Vector{Float64}(vector)
    length(checked_vector) == expected_length ||
        throw(DimensionMismatch("$(name) must have length $expected_length."))
    return checked_vector
end

function _checked_matrix_or_default(value, default, expected_rows, expected_cols, name::Symbol)
    matrix = isnothing(value) ? default : value
    checked_matrix = Matrix{Float64}(matrix)
    size(checked_matrix) == (expected_rows, expected_cols) ||
        throw(DimensionMismatch("$(name) must have size ($(expected_rows), $(expected_cols))."))
    return checked_matrix
end

# END FILE: src/ContextualDFL/ContextualDFLExperiments/src/data_generation/benchmark_dataset.jl

# BEGIN FILE: src/ContextualDFL/ContextualDFLExperiments/src/data_generation/contextual_generators/ContextDataGenerator.jl
abstract type ContextDataGenerator end

(generator::ContextDataGenerator)() =
    error("Context data generation is not defined for $(typeof(generator)).")

# END FILE: src/ContextualDFL/ContextualDFLExperiments/src/data_generation/contextual_generators/ContextDataGenerator.jl

# BEGIN FILE: src/ContextualDFL/ContextualDFLExperiments/src/data_generation/generate_contextual_data_set.jl
function generate_contextual_data_set(contextual_data, scenario_data)
    length(contextual_data) == length(scenario_data) ||
        throw(DimensionMismatch("contextual_data and scenario_data must have the same length."))

    data_set = ContextualDFL.ContextualDataPoint[]
    for (context, scenarios) in zip(contextual_data, scenario_data)
        context isa AbstractVector ||
            throw(ArgumentError("each context must be an AbstractVector."))

        # Store a single scenario as a one-element collection, matching ContextualDFL's dataset type.
        scenario_collection = if scenarios isa ContextualDFL.ParametricScenario
            [scenarios]
        else
            scenarios isa AbstractVector{<:ContextualDFL.ParametricScenario} ||
                throw(ArgumentError("each scenario entry must be a ParametricScenario or a vector of them."))
            isempty(scenarios) && throw(ArgumentError("scenario collections must not be empty."))
            collect(scenarios)
        end

        push!(
            data_set,
            ContextualDFL.ContextualDataPoint(collect(context), scenario_collection),
        )
    end
    return data_set
end

# END FILE: src/ContextualDFL/ContextualDFLExperiments/src/data_generation/generate_contextual_data_set.jl

# BEGIN FILE: src/ContextualDFL/ContextualDFLExperiments/src/data_generation/scenario_generators/ScenarioDataGenerator.jl
abstract type ScenarioDataGenerator end

(generator::ScenarioDataGenerator)(context) =
    error("Scenario data generation is not defined for $(typeof(generator)).")

# END FILE: src/ContextualDFL/ContextualDFLExperiments/src/data_generation/scenario_generators/ScenarioDataGenerator.jl

# BEGIN FILE: src/ContextualDFL/ContextualDFLExperiments/src/implementations/random_yield_problem/RandomYieldDataGenerators.jl
import Random

function generate_benchmark_contexts(
    problem::RandomYieldProblem;
    n_contexts,
    rng=Random.default_rng(),
)
    context_count = _checked_positive_integer(n_contexts, :n_contexts)
    return [randn(rng, problem.context_dim) for _ in 1:context_count]
end

function generate_benchmark_scenarios(
    problem::RandomYieldProblem,
    context;
    n_scenarios,
    rng=Random.default_rng(),
)
    scenario_count = _checked_positive_integer(n_scenarios, :n_scenarios)
    context_vector = _checked_context_vector(context, problem.context_dim)
    return [
        sample_random_yield_scenario(problem, context_vector; rng=rng)
        for _ in 1:scenario_count
    ]
end

function random_yield_probabilities(problem::RandomYieldProblem, context)
    context_vector = _checked_context_vector(context, problem.context_dim)
    scores = problem.alpha .+ problem.beta * context_vector
    shifted_scores = scores .- maximum(scores)
    weights = exp.(shifted_scores)
    return weights ./ sum(weights)
end

function random_yield_support_scenarios(problem::RandomYieldProblem, context)
    _checked_context_vector(context, problem.context_dim)
    return [_random_yield_scenario(problem, W_eq) for W_eq in problem.W_support]
end

function sample_random_yield_scenario(
    problem::RandomYieldProblem,
    context;
    rng=Random.default_rng(),
)
    probabilities = random_yield_probabilities(problem, context)
    support_index = _sample_probability_index(probabilities, rng)
    return _random_yield_scenario(problem, problem.W_support[support_index])
end

function _random_yield_scenario(problem::RandomYieldProblem, W_eq::AbstractMatrix)
    base = base_scenario(problem)
    return ContextualDFL.ParametricScenario(;
        W_eq_xi=copy(W_eq),
        W_ineq_xi=copy(base.W_ineq),
        T_eq_xi=copy(base.T_eq),
        T_ineq_xi=copy(base.T_ineq),
        h_eq_xi=copy(base.h_eq),
        h_ineq_xi=copy(base.h_ineq),
        q_xi=copy(base.q),
    )
end

function _sample_probability_index(probabilities::AbstractVector, rng::Random.AbstractRNG)
    u = rand(rng)
    cumulative = 0.0
    for index in eachindex(probabilities)
        cumulative += probabilities[index]
        u <= cumulative && return index
    end
    return lastindex(probabilities)
end

# END FILE: src/ContextualDFL/ContextualDFLExperiments/src/implementations/random_yield_problem/RandomYieldDataGenerators.jl

# BEGIN FILE: src/ContextualDFL/ContextualDFLExperiments/src/implementations/random_yield_problem/RandomYieldDecoders.jl
RandomYieldParametricDecoder(problem::RandomYieldProblem) =
    ContextualDFL.ParametricDecoder()

# END FILE: src/ContextualDFL/ContextualDFLExperiments/src/implementations/random_yield_problem/RandomYieldDecoders.jl

# BEGIN FILE: src/ContextualDFL/ContextualDFLExperiments/src/implementations/random_yield_problem/RandomYieldProblem.jl
import LinearAlgebra
import Random

struct RandomYieldProblem <: ProgramInstance
    product_count::Int
    activity_count::Int
    context_dim::Int
    support_count::Int
    sigma_W::Float64
    demand_mean::Vector{Float64}
    B::Float64
    alpha::Vector{Float64}
    beta::Matrix{Float64}
    W_support::Vector{Matrix{Float64}}
    stochastic_program::ContextualDFL.StochasticProgram
    base_scenario::NamedTuple
end

function RandomYieldProblem(;
    r=20,
    a=40,
    context_dim=3,
    K_support=20,
    sigma_W=0.25,
    parameter_seed=1,
    demand_mean=nothing,
    B=nothing,
    alpha=nothing,
    beta=nothing,
)
    product_count = _checked_positive_integer(r, :r)
    activity_count = _checked_positive_integer(a, :a)
    checked_context_dim = _checked_positive_integer(context_dim, :context_dim)
    support_count = _checked_positive_integer(K_support, :K_support)
    checked_sigma_W = Float64(sigma_W)
    checked_sigma_W >= 0.0 || throw(ArgumentError("sigma_W must be nonnegative."))

    rng = Random.MersenneTwister(parameter_seed)
    checked_demand_mean = _checked_vector_or_default(
        demand_mean,
        fill(2.0, product_count),
        product_count,
        :demand_mean,
    )
    checked_B = isnothing(B) ? 0.5 * sum(checked_demand_mean) : Float64(B)
    checked_B > 0.0 || throw(ArgumentError("B must be positive."))

    checked_alpha = _checked_vector_or_default(
        alpha,
        0.2 .* randn(rng, support_count),
        support_count,
        :alpha,
    )
    checked_beta = _checked_matrix_or_default(
        beta,
        0.6 .* randn(rng, support_count, checked_context_dim),
        support_count,
        checked_context_dim,
        :beta,
    )

    W_support = _sample_random_yield_support(
        rng,
        product_count,
        activity_count,
        support_count,
        checked_sigma_W,
    )
    program, scenario = _random_yield_program_and_scenario(
        product_count,
        activity_count,
        checked_B,
        checked_demand_mean,
        first(W_support),
    )

    return RandomYieldProblem(
        product_count,
        activity_count,
        checked_context_dim,
        support_count,
        checked_sigma_W,
        checked_demand_mean,
        checked_B,
        checked_alpha,
        checked_beta,
        W_support,
        program,
        scenario,
    )
end

stochastic_program(problem::RandomYieldProblem) = problem.stochastic_program

base_scenario(problem::RandomYieldProblem) = problem.base_scenario

function _random_yield_program_and_scenario(
    product_count,
    activity_count,
    budget,
    demand_mean,
    base_W_eq,
)
    recourse_count = activity_count + 2 * product_count

    program = ContextualDFL.StochasticProgram(
        A_eq=ones(Float64, 1, product_count),
        A_ineq=-Matrix{Float64}(LinearAlgebra.I, product_count, product_count),
        b_eq=[Float64(budget)],
        b_ineq=zeros(Float64, product_count),
        c=fill(1.0, product_count),
    )

    scenario = (;
        W_eq=base_W_eq,
        W_ineq=-Matrix{Float64}(LinearAlgebra.I, recourse_count, recourse_count),
        T_eq=Matrix{Float64}(LinearAlgebra.I, product_count, product_count),
        T_ineq=zeros(Float64, recourse_count, product_count),
        h_eq=copy(demand_mean),
        h_ineq=zeros(Float64, recourse_count),
        q=vcat(fill(2.0, activity_count), fill(50.0, product_count), fill(0.1, product_count)),
    )

    return program, scenario
end

function _sample_random_yield_support(
    rng::Random.AbstractRNG,
    product_count,
    activity_count,
    support_count,
    sigma_W,
)
    mask = rand(rng, product_count, activity_count) .< 0.25
    for product in 1:product_count
        mask[product, rand(rng, 1:activity_count)] = true
    end
    for activity in 1:activity_count
        mask[rand(rng, 1:product_count), activity] = true
    end

    Y_bar = zeros(Float64, product_count, activity_count)
    for index in eachindex(Y_bar)
        if mask[index]
            Y_bar[index] = 0.5 + rand(rng)
        end
    end

    support = Matrix{Float64}[]
    push!(support, _random_yield_W_eq(Y_bar))
    for _ in 2:support_count
        Y = copy(Y_bar)
        for index in eachindex(Y)
            if mask[index]
                Y[index] *= exp(sigma_W * randn(rng))
            end
        end
        push!(support, _random_yield_W_eq(Y))
    end
    return support
end

function _random_yield_W_eq(Y::AbstractMatrix)
    product_count, _ = size(Y)
    return hcat(
        Matrix{Float64}(Y),
        Matrix{Float64}(LinearAlgebra.I, product_count, product_count),
        -Matrix{Float64}(LinearAlgebra.I, product_count, product_count),
    )
end

# END FILE: src/ContextualDFL/ContextualDFLExperiments/src/implementations/random_yield_problem/RandomYieldProblem.jl

# BEGIN FILE: src/ContextualDFL/ContextualDFLExperiments/src/implementations/resource_allocation_problem/data_generators/ResourceAllocationContextDataGenerator.jl
import Distributions
import LinearAlgebra
import Random

struct ResourceAllocationContextDataGenerator{TCorrelation,TRng} <: ContextDataGenerator
    correlation_matrix::TCorrelation
    rng::TRng
end

function sample_resource_allocation_correlation_matrix(
    rng::Random.AbstractRNG,
    dimension::Integer=3,
)
    beta_parameter = 2.0
    partial_correlation = zeros(Float64, dimension, dimension)
    correlation = Matrix{Float64}(LinearAlgebra.I, dimension, dimension)

    for k in 1:(dimension - 1)
        for i in (k + 1):dimension
            partial_correlation[k, i] =
                (rand(rng, Distributions.Beta(beta_parameter, beta_parameter)) - 0.5) * 2.0
            rho = partial_correlation[k, i]
            for j in (k - 1):-1:1
                rho =
                    rho *
                    sqrt((1 - partial_correlation[j, i]^2) * (1 - partial_correlation[j, k]^2)) +
                    partial_correlation[j, i] * partial_correlation[j, k]
            end
            correlation[k, i] = rho
            correlation[i, k] = rho
        end
    end

    permutation = Random.randperm(rng, dimension)
    return correlation[permutation, permutation]
end

function ResourceAllocationContextDataGenerator(;
    rng::Random.AbstractRNG=Random.default_rng(),
    correlation_matrix=sample_resource_allocation_correlation_matrix(rng, 3),
)
    return ResourceAllocationContextDataGenerator(Matrix{Float64}(correlation_matrix), rng)
end

function (generator::ResourceAllocationContextDataGenerator)()
    distribution = Distributions.MvNormal(
        zeros(size(generator.correlation_matrix, 1)),
        LinearAlgebra.Symmetric(generator.correlation_matrix + 1e-8LinearAlgebra.I),
    )
    return abs.(rand(generator.rng, distribution))
end

# END FILE: src/ContextualDFL/ContextualDFLExperiments/src/implementations/resource_allocation_problem/data_generators/ResourceAllocationContextDataGenerator.jl

# BEGIN FILE: src/ContextualDFL/ContextualDFLExperiments/src/implementations/resource_allocation_problem/data_generators/ResourceAllocationScenarioDataGenerator.jl
import Distributions
import Random

struct ResourceAllocationScenarioDataGenerator{TIntercepts,TSlopes,TSigma,TPower,TRng} <:
       ScenarioDataGenerator
    intercepts::TIntercepts
    slopes::TSlopes
    sigma::TSigma
    p::TPower
    L::Int
    rng::TRng
end

function sample_resource_allocation_demand_parameters(
    rng::Random.AbstractRNG,
    demand_count::Integer,
)
    intercepts = 50 .+ 5 .* rand(rng, Distributions.Normal(0, 1), demand_count)
    B1 = 10 .+ rand(rng, Distributions.Uniform(-4, 4), demand_count)
    B2 = 5 .+ rand(rng, Distributions.Uniform(-4, 4), demand_count)
    B3 = 2 .+ rand(rng, Distributions.Uniform(-4, 4), demand_count)
    return intercepts, hcat(B1, B2, B3)
end

function ResourceAllocationScenarioDataGenerator(
    problem::ResourceAllocationProblem;
    sigma,
    p,
    L,
    rng::Random.AbstractRNG=Random.default_rng(),
    intercepts=nothing,
    slopes=nothing,
)
    L <= 3 || throw(ArgumentError("resource-allocation data generation has three context terms."))
    demand_count = size(problem.problem_data.service_rate_parameters, 2)
    sampled_intercepts, sampled_slopes = if isnothing(intercepts) || isnothing(slopes)
        sample_resource_allocation_demand_parameters(rng, demand_count)
    else
        intercepts, slopes
    end
    length(sampled_intercepts) == demand_count ||
        throw(DimensionMismatch("intercepts must have one entry per demand."))
    size(sampled_slopes, 1) == demand_count && size(sampled_slopes, 2) >= L ||
        throw(DimensionMismatch("slopes must have demand_count rows and at least L columns."))

    return ResourceAllocationScenarioDataGenerator(
        Vector{Float64}(sampled_intercepts),
        Matrix{Float64}(sampled_slopes),
        sigma,
        p,
        Int(L),
        rng,
    )
end

function (generator::ResourceAllocationScenarioDataGenerator)(context)
    length(context) >= generator.L ||
        throw(DimensionMismatch("context must have at least $(generator.L) entries."))

    demand = zeros(Float64, length(generator.intercepts))
    for demand_index in eachindex(demand)
        signal = generator.intercepts[demand_index]
        for term in 1:generator.L
            signal += generator.slopes[demand_index, term] * context[term]^generator.p
        end
        demand[demand_index] = signal + rand(generator.rng, Distributions.Normal(0, generator.sigma))
    end

    return ContextualDFL.ParametricScenario(;
        W_eq_xi=Float64[],
        W_ineq_xi=Float64[],
        T_eq_xi=Float64[],
        T_ineq_xi=Float64[],
        h_eq_xi=demand,
        h_ineq_xi=Float64[],
        q_xi=Float64[],
    )
end

# END FILE: src/ContextualDFL/ContextualDFLExperiments/src/implementations/resource_allocation_problem/data_generators/ResourceAllocationScenarioDataGenerator.jl

# BEGIN FILE: src/ContextualDFL/ContextualDFLExperiments/src/implementations/resource_allocation_problem/problem_data/parameters.jl
# Copied from ProblemBasedScenarioGeneration/src/problem_instances/resource_allocation/parameters.jl
# ASCII names are used here so the experiments package API stays easy to type.


#first-stage objective coefficients

const RESOURCE_ALLOCATION_FIRST_STAGE_COSTS = [0.7432683327183562 1.1091782740319707 1.153303321756463 0.7751143229777455 0.8098221660821516 1.1324011945408334 0.9859746207969543 0.9502566702752542 1.1909969906693632 0.9630816483432164 1.200255951169507 1.1412400075515754 1.0668906064783976 0.6939486330511746 1.3054592887126564 1.121268771178427 1.0238773621726285 0.9083077835585992 1.2091835133144502 1.171512305637218 ] 

# recourse objective coefficients
const RESOURCE_ALLOCATION_SECOND_STAGE_COSTS = [2.3130311834976878 1.970110451642846 2.079834948369165 2.180997535033276 2.1887583154880983 2.157134900936415 2.148981366909944 2.063591377530072 2.114553485889044 2.2016770965819905 1.9966732632984514 2.2330830064352423 2.073101936010451 2.078736481418292 2.1023788717803735 2.20322688021812 2.0598950590963474 2.186612946072563 2.2213945559591637 2.4194867761920977 2.177263511555545 2.2031710809006455 1.9909192466456946 2.2396747891757802 2.1661606250152716 2.049535136705102 2.1171635559467483 2.2147607076134714 2.203825916684725 2.3108454507501515 ] 

# yield parameters
const RESOURCE_ALLOCATION_YIELD_PARAMETERS = [0.9555664467976429 0.9739615542635913 0.9122667177690378 0.9340610389388502 0.9246641057347171 0.9945151890629177 0.9615143263620417 0.9659480369131807 0.9358045688085963 0.9147377425339717 0.9756453545272545 0.9350544644953063 0.9839239868461557 0.9953649754312432 0.9288115703035881 0.9946050201412094 0.9575793792880826 0.9224016204902284 0.905702317419421 0.9507354965994901 ] 

# service rate parameters
const RESOURCE_ALLOCATION_SERVICE_RATE_PARAMETERS = begin
    service_rates = zeros(Float64,20,30) 
    service_rates[1,:] = [0.0 1.5425864608494138 1.8677174853336527 1.7036521533592883 0.0 0.0 1.7940537688365086 1.8901902999735647 0.0 1.743223975958845 0.0 0.0 1.9010265875424377 1.8193631099999497 0.0 0.0 0.0 0.0 1.7774304885999672 0.0 1.9956705204853944 1.7673613846413179 0.0 1.6465508373626938 1.7811853533294304 0.0 1.7386645478259126 0.0 0.0 1.5126998082534597 ] 
    service_rates[2,:] = [2.2607279588202296 1.9084964021630282 0.0 2.0695620946729028 2.4463795504578956 0.0 2.159963710150123 2.256100241287179 0.0 2.1091339172724597 2.2022354542170515 0.0 0.0 2.185273051313564 1.990070719163648 0.0 0.0 2.27455757738075 2.1433404299135814 0.0 2.361580461799009 2.133271325954932 2.2274657922865684 0.0 2.147095294643045 2.3868225163693566 0.0 2.0854662478744874 2.197861409557054 0.0 ] 
    service_rates[3,:] = [0.0 1.9526214498875207 0.0 0.0 2.490504598182388 2.12721997941551 2.204088757874615 2.3002252890116717 2.362568160228464 2.153258964996952 2.2463605019415436 0.0 2.3110615765805447 0.0 2.03419576688814 0.0 2.073572838292769 0.0 2.187465477638074 1.8667986470938884 2.405705509523501 2.1773963736794246 2.2715908400110605 0.0 2.1912203423675374 2.430947564093849 0.0 2.1295912955989795 0.0 0.0 ] 
    service_rates[4,:] = [1.9266640077660047 1.574432451108803 1.899563475593042 1.7354981436186776 2.1123155994036704 0.0 1.8258997590958979 1.922036290232954 0.0 1.7750699662182343 1.8681715031628263 0.0 0.0 1.851209100259339 1.6560067681094228 0.0 0.0 0.0 0.0 1.488609648315171 2.0275165107447837 1.7992073749007071 0.0 0.0 0.0 2.0527585653151315 0.0 1.751402296820262 1.863797458502829 1.544545798512849 ] 
    service_rates[5,:] = [0.0 0.0 0.0 0.0 2.1470234425080768 1.7837388237411989 1.860607602200304 1.95674413333736 0.0 1.8097778093226404 0.0 0.0 1.967580420906233 1.885916943363745 0.0 1.8935808138490131 0.0 0.0 1.8439843219637626 1.523317491419577 2.0622243538491896 0.0 1.9281096843367493 0.0 0.0 2.087466408419538 0.0 1.7861101399246682 1.898505301607235 0.0 ] 
    service_rates[6,:] = [2.2839508793290926 0.0 2.2568503471561296 2.0927850151817653 2.4696024709667586 0.0 2.1831866306589855 0.0 2.3416660330128343 2.1323568377813222 0.0 0.0 2.290159449364915 2.208495971822427 2.0132936396725105 0.0 0.0 2.2977804978896126 2.1665633504224444 0.0 2.3848033823078714 0.0 2.250688712795431 0.0 2.170318215151908 2.4100454368782196 0.0 2.10868916838335 0.0 1.9018326700759371 ] 
    service_rates[7,:] = [2.1375243055852136 0.0 2.1104237734122506 0.0 2.3231758972228795 0.0 2.0367600569151065 2.1328965880521626 0.0 1.9859302640374432 2.079031800982035 1.806567963946108 2.1437328756210356 0.0 1.8668670659286315 0.0 1.9062441373332604 0.0 2.0201367766785654 1.6994699461343798 0.0 2.010067672719916 0.0 0.0 0.0 2.2636188631343406 0.0 1.962262594639471 2.0746577563220376 0.0 ] 
    service_rates[8,:] = [0.0 1.7495747984063117 2.0747058228905506 1.9106404909161863 0.0 0.0 2.0010421063934065 2.0971786375304626 0.0 0.0 0.0 0.0 0.0 2.0263514475568476 1.8311491154069315 2.0340153180421154 0.0 0.0 0.0 0.0 2.2026588580422923 1.9743497221982158 0.0 1.8535391749195917 1.9881736908863283 2.2279009126126406 0.0 0.0 2.0389398058003376 1.7196881458103577 ] 
    service_rates[9,:] = [2.3425466754576223 1.990315118800421 2.3154461432846594 2.151380811310295 0.0 2.1649136483284104 2.2417824267875153 2.337918957924572 2.400261829141364 2.190952633909852 0.0 2.011590333818517 2.348755245493445 2.267091767950957 2.0718894358010402 2.2747556384362246 2.111266507205669 2.3563762940181423 0.0 0.0 2.443399178436401 2.215090042592325 2.3092845089239606 0.0 2.2289140112804375 0.0 2.1863932057769193 0.0 0.0 1.9604284662044669 ] 
    service_rates[10,:] = [0.0 1.762399776474274 2.087530800958513 1.9234654689841484 2.3002829247691414 1.9369983060022635 2.013867084461369 0.0 2.1723464868152176 1.963037291583705 2.0561388285282973 1.78367499149237 2.120839903167298 2.03917642562481 0.0 0.0 0.0 0.0 1.9972438042248273 1.6765769736806417 2.2154838361102547 0.0 2.0813691665978142 0.0 2.0009986689542907 2.2407258906806025 0.0 0.0 2.0517647838683 0.0 ] 
    service_rates[11,:] = [2.351805635957766 0.0 0.0 2.160639771810439 0.0 0.0 2.2510413872876596 2.3471779184247152 2.4095207896415083 0.0 2.293313131354588 0.0 0.0 0.0 2.0811483963011845 0.0 0.0 0.0 0.0 0.0 2.4526581389365454 2.2243490030924686 2.318543469424105 2.103538455813845 2.238172971780581 2.477900193506893 2.195652166277063 2.1765439250120235 2.2889390866945902 0.0 ] 
    service_rates[12,:] = [2.2927896923398343 0.0 0.0 2.1016238281925075 2.4784412839775003 0.0 2.1920254436697277 0.0 0.0 2.1411956507920644 2.234297187736656 0.0 2.298998262375657 0.0 2.0221324526832527 2.224998655318437 2.0615095240878816 2.3066193109003548 0.0 1.8547353328890008 2.3936421953186136 0.0 2.259527525806173 0.0 2.1791570281626496 2.4188842498889613 0.0 2.117527981394092 2.229923143076659 1.910671483086679 ] 
    service_rates[13,:] = [2.2184402912666568 0.0 0.0 2.0272744271193295 0.0 0.0 2.1176760425965497 2.2138125737336063 0.0 0.0 0.0 1.8874839496275513 2.2246488613024793 2.1429853837599913 1.947783051610075 2.150649254245259 1.9871601230147036 2.2322699098271768 2.1010527623600086 1.780385931815823 0.0 2.0909836584013592 2.185178124732995 0.0 0.0 2.344534848815784 0.0 2.043178580320914 0.0 1.8363220820135013 ] 
    service_rates[14,:] = [0.0 1.4932667611822321 1.818397785666471 1.6543324536921067 0.0 0.0 1.744734069169327 0.0 1.9032134715231757 1.6939042762916634 0.0 1.5145419762003283 1.851706887875256 1.770043410332768 1.574841078182852 0.0 0.0 1.8593279363999538 0.0 1.4074439583886 1.9463508208182128 1.7180416849741362 0.0 0.0 1.7318656536622488 1.9715928753885608 1.689344848158731 1.6702366068936911 1.782631768576258 1.463380108586278 ] 
    service_rates[15,:] = [2.4570089735009155 0.0 0.0 2.2658431093535882 0.0 2.2793759463717036 2.356244724830809 2.4523812559678646 2.5147241271846577 0.0 0.0 2.12605263186181 0.0 0.0 0.0 2.389217936479518 2.2257288052489623 2.4708385920614355 0.0 2.018954614050082 0.0 2.329552340635618 2.4237468069672543 2.208741793356994 2.3433763093237303 0.0 0.0 2.281747262555173 2.3941424242377396 2.07489076424776 ] 
    service_rates[16,:] = [0.0 1.9205868993094848 2.2457179237937233 2.081652591819359 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.9418621143275807 2.2790270260025087 2.1973635484600207 0.0 0.0 0.0 0.0 2.155430927060038 1.8347640965158525 2.373670958945465 2.1453618231013887 2.2395562894330245 0.0 2.1591857917895014 2.398913013515813 0.0 0.0 0.0 0.0 ] 
    service_rates[17,:] = [2.1754270469608876 1.8231954903036862 2.148326514787925 1.9842611828135603 0.0 1.9977940198316757 2.074662798290781 2.1707993294278367 2.2331422006446298 2.0238330054131173 0.0 1.8444707053217821 2.1816356169967097 2.0999721394542217 1.9047698073043058 2.10763600993949 0.0 2.1892566655214076 2.0580395180542395 0.0 0.0 0.0 0.0 1.9271598668169663 2.0617943827837024 2.3015216045100146 0.0 0.0 2.1125604976977117 1.7933088377077322 ] 
    service_rates[18,:] = [2.0598574683468582 1.7076259116896568 0.0 0.0 2.245509059984524 0.0 1.9590932196767517 2.0552297508138078 2.1175726220306004 1.908263426799088 2.00136496374368 1.7289011267077528 2.0660660383826808 1.9844025608401927 1.7892002286902766 1.9920664313254606 1.828577300094905 0.0 1.94246993944021 1.6218031088960245 2.1607099713256375 1.9324008354815607 2.026595301813197 0.0 0.0 0.0 0.0 1.8845957574011156 1.9969909190836828 0.0 ] 
    service_rates[19,:] = [2.360733198102709 2.0085016414455077 0.0 2.1695673339553823 0.0 2.183100170973497 0.0 0.0 0.0 0.0 0.0 2.029776856463604 2.3669417681385316 2.2852782905960436 2.0900759584461275 2.292942161081312 2.1294530298507564 2.3745628166632295 0.0 1.9226788386518756 2.4615857010814883 2.2332765652374116 2.327471031569048 2.1124660179587877 2.2471005339255243 0.0 2.2045797284220066 2.185471487156967 2.2978666488395336 0.0 ] 
    service_rates[20,:] = [0.0 1.9708304337682758 2.2959614582525143 2.13189612627815 2.508713582063143 0.0 2.22229774175537 2.3184342728924268 2.380777144109219 0.0 2.2645694858222987 1.9921056487863718 0.0 0.0 2.052404750768895 2.2552709534040796 2.091781822173524 2.3368916089859972 2.205674461518829 0.0 2.423914493404256 0.0 2.2897998238918156 0.0 2.2094293262482925 0.0 2.1669085207447742 2.1478002794797346 2.2601954411623018 0.0 ] 
    service_rates
end



# END FILE: src/ContextualDFL/ContextualDFLExperiments/src/implementations/resource_allocation_problem/problem_data/parameters.jl

# BEGIN FILE: src/ContextualDFL/ContextualDFLExperiments/src/implementations/resource_allocation_problem/program_instance/ResourceAllocationProblem.jl
import LinearAlgebra

struct ResourceAllocationProblemData
    service_rate_parameters::Matrix{Float64}
    first_stage_costs::Vector{Float64}
    second_stage_costs::Vector{Float64}
    yield_parameters::Vector{Float64}

    function ResourceAllocationProblemData(
        service_rate_parameters::AbstractMatrix,
        first_stage_costs::AbstractVector,
        second_stage_costs::AbstractVector,
        yield_parameters::AbstractVector,
    )
        service_rates = Matrix{Float64}(service_rate_parameters)
        first_costs = Vector{Float64}(first_stage_costs)
        second_costs = Vector{Float64}(second_stage_costs)
        yields = Vector{Float64}(yield_parameters)

        resource_count, demand_count = size(service_rates)
        length(first_costs) == resource_count ||
            throw(DimensionMismatch("first-stage costs must match resource count."))
        length(second_costs) == demand_count ||
            throw(DimensionMismatch("second-stage costs must match demand count."))
        length(yields) == resource_count ||
            throw(DimensionMismatch("yield parameters must match resource count."))

        return new(service_rates, first_costs, second_costs, yields)
    end
end

function default_resource_allocation_problem_data()
    return ResourceAllocationProblemData(
        RESOURCE_ALLOCATION_SERVICE_RATE_PARAMETERS,
        vec(Float64.(RESOURCE_ALLOCATION_FIRST_STAGE_COSTS)),
        vec(Float64.(RESOURCE_ALLOCATION_SECOND_STAGE_COSTS)),
        vec(Float64.(RESOURCE_ALLOCATION_YIELD_PARAMETERS)),
    )
end

struct ResourceAllocationProblem <: ProgramInstance
    problem_data::ResourceAllocationProblemData
    stochastic_program::ContextualDFL.StochasticProgram
    base_scenario::NamedTuple
end

function ResourceAllocationProblem(
    problem_data::ResourceAllocationProblemData=default_resource_allocation_problem_data(),
)
    service_rates = problem_data.service_rate_parameters
    first_costs = problem_data.first_stage_costs
    second_costs = problem_data.second_stage_costs
    yields = problem_data.yield_parameters

    resource_count, demand_count = size(service_rates)
    recourse_variables =
        demand_count + resource_count * demand_count + resource_count + demand_count
    recourse_rows = resource_count + demand_count

    W_eq = zeros(Float64, recourse_rows, recourse_variables)
    for resource_index in 1:resource_count
        for demand_index in 1:demand_count
            allocation_index = demand_count + demand_count * (resource_index - 1) + demand_index
            W_eq[resource_index, allocation_index] = 1.0
        end
        W_eq[resource_index, demand_count + resource_count * demand_count + resource_index] = 1.0
    end

    for demand_index in 1:demand_count
        row = resource_count + demand_index
        W_eq[row, demand_index] = 1.0
        for resource_index in 1:resource_count
            allocation_index = demand_count + demand_count * (resource_index - 1) + demand_index
            W_eq[row, allocation_index] = service_rates[resource_index, demand_index]
        end
        slack_index = demand_count + resource_count * demand_count + resource_count + demand_index
        W_eq[row, slack_index] = -1.0
    end

    T_eq = zeros(Float64, recourse_rows, resource_count)
    for resource_index in 1:resource_count
        T_eq[resource_index, resource_index] = -yields[resource_index]
    end

    q = zeros(Float64, recourse_variables)
    q[1:demand_count] .= second_costs

    program = ContextualDFL.StochasticProgram(
        A_eq=zeros(Float64, 0, resource_count),
        A_ineq=-Matrix{Float64}(LinearAlgebra.I, resource_count, resource_count),
        b_eq=Float64[],
        b_ineq=zeros(Float64, resource_count),
        c=first_costs,
    )

    scenario = (;
        W_eq=W_eq,
        W_ineq=-Matrix{Float64}(LinearAlgebra.I, recourse_variables, recourse_variables),
        T_eq=T_eq,
        T_ineq=zeros(Float64, recourse_variables, resource_count),
        h_eq=zeros(Float64, recourse_rows),
        h_ineq=zeros(Float64, recourse_variables),
        q=q,
    )

    return ResourceAllocationProblem(problem_data, program, scenario)
end

stochastic_program(problem::ResourceAllocationProblem) = problem.stochastic_program

base_scenario(problem::ResourceAllocationProblem) = problem.base_scenario

# END FILE: src/ContextualDFL/ContextualDFLExperiments/src/implementations/resource_allocation_problem/program_instance/ResourceAllocationProblem.jl

# BEGIN FILE: src/ContextualDFL/ContextualDFLExperiments/src/implementations/resource_allocation_problem/scenario_decoders/ResourceAllocationDemandDecoders.jl
import ChainRulesCore

struct ResourceAllocationDemandVectorDecoder{TBaseScenario} <: ContextualDFL.VectorDecoder
    base_scenario::TBaseScenario
end

ResourceAllocationDemandVectorDecoder(problem::ResourceAllocationProblem) =
    ResourceAllocationDemandVectorDecoder(base_scenario(problem))

struct ResourceAllocationDemandParametricDecoder{TBaseScenario} <: ContextualDFL.ScenarioDecoder
    base_scenario::TBaseScenario
end

ResourceAllocationDemandParametricDecoder(problem::ResourceAllocationProblem) =
    ResourceAllocationDemandParametricDecoder(base_scenario(problem))

function _resource_allocation_h_eq(scenario, demand::AbstractVector)
    # Generated demand fills the demand rows; the resource-balance rows stay fixed at zero.
    resource_count = size(scenario.T_eq, 2)
    demand_count = length(scenario.h_eq) - resource_count
    length(demand) == demand_count ||
        throw(DimensionMismatch("demand vector must have length $demand_count."))

    return vcat(zeros(eltype(demand), resource_count), demand)
end

function (decoder::ResourceAllocationDemandVectorDecoder)(demand::AbstractVector)
    scenario = decoder.base_scenario

    return (
        scenario.W_eq,
        scenario.W_ineq,
        scenario.T_eq,
        scenario.T_ineq,
        _resource_allocation_h_eq(scenario, demand),
        scenario.h_ineq,
        scenario.q,
    )
end

function (decoder::ResourceAllocationDemandParametricDecoder)(
    scenario_parameters::ContextualDFL.ParametricScenario,
)
    scenario = decoder.base_scenario

    return (
        scenario.W_eq,
        scenario.W_ineq,
        scenario.T_eq,
        scenario.T_ineq,
        _resource_allocation_h_eq(scenario, scenario_parameters.h_eq_xi),
        scenario.h_ineq,
        scenario.q,
    )
end

function ChainRulesCore.rrule(
    ::typeof(ContextualDFL.decode_scenario_collection),
    decoder::ResourceAllocationDemandParametricDecoder,
    scenario_parameter_collection::AbstractVector{<:ContextualDFL.ParametricScenario},
)
    output = ContextualDFL.decode_scenario_collection(decoder, scenario_parameter_collection)
    demand_rows = _resource_allocation_demand_rows(decoder)

    function resource_allocation_parametric_decode_pullback(output_tangent)
        dh_eq_array = ContextualDFL._array_cotangent(
            output_tangent,
            5,
            output[5];
            name=:h_eq_array,
        )

        scenario_tangents = map(enumerate(scenario_parameter_collection)) do (k, scenario_parameters)
            ChainRulesCore.Tangent{typeof(scenario_parameters)}(
                W_eq_xi=ChainRulesCore.NoTangent(),
                W_ineq_xi=ChainRulesCore.NoTangent(),
                T_eq_xi=ChainRulesCore.NoTangent(),
                T_ineq_xi=ChainRulesCore.NoTangent(),
                h_eq_xi=ChainRulesCore.ProjectTo(scenario_parameters.h_eq_xi)(
                    view(dh_eq_array, demand_rows, k),
                ),
                h_ineq_xi=ChainRulesCore.NoTangent(),
                q_xi=ChainRulesCore.NoTangent(),
            )
        end

        return (
            ChainRulesCore.NoTangent(),
            ChainRulesCore.NoTangent(),
            scenario_tangents,
        )
    end

    return output, resource_allocation_parametric_decode_pullback
end

function _resource_allocation_demand_rows(decoder::ResourceAllocationDemandParametricDecoder)
    scenario = decoder.base_scenario
    resource_count = size(scenario.T_eq, 2)
    demand_count = length(scenario.h_eq) - resource_count
    return (resource_count + 1):(resource_count + demand_count)
end

function ChainRulesCore.rrule(
    ::typeof(ContextualDFL.decode_scenario_collection),
    decoder::ResourceAllocationDemandVectorDecoder,
    demand_vector::AbstractVector{<:Number};
    nr_scenarios=nothing,
)
    isnothing(nr_scenarios) &&
        throw(ArgumentError(
            "ResourceAllocationDemandVectorDecoder rrule requires explicit nr_scenarios.",
        ))
    nr_scenarios isa Integer && nr_scenarios > 0 ||
        throw(ArgumentError("nr_scenarios must be a positive integer."))

    scenario = decoder.base_scenario
    resource_count = size(scenario.T_eq, 2)
    demand_count = length(scenario.h_eq) - resource_count
    expected_length = demand_count * nr_scenarios
    length(demand_vector) == expected_length ||
        throw(DimensionMismatch(
            "demand_vector has length $(length(demand_vector)); expected " *
            "$(expected_length) for demand_count=$demand_count, " *
            "nr_scenarios=$nr_scenarios.",
        ))

    output = ContextualDFL.decode_scenario_collection(
        decoder,
        demand_vector;
        nr_scenarios=nr_scenarios,
    )
    demand_rows = (resource_count + 1):(resource_count + demand_count)
    project_demand = ChainRulesCore.ProjectTo(demand_vector)

    function resource_allocation_vector_decode_pullback(output_tangent)
        dh_eq_array = ContextualDFL._array_cotangent(
            output_tangent,
            5,
            output[5];
            name=:h_eq_array,
        )
        ddemand_vector = vec(copy(view(dh_eq_array, demand_rows, :)))

        return (
            ChainRulesCore.NoTangent(),
            ChainRulesCore.NoTangent(),
            project_demand(ddemand_vector),
        )
    end

    return output, resource_allocation_vector_decode_pullback
end

function _resource_allocation_demand_rows(decoder::ResourceAllocationDemandVectorDecoder)
    scenario = decoder.base_scenario
    resource_count = size(scenario.T_eq, 2)
    demand_count = length(scenario.h_eq) - resource_count
    return (resource_count + 1):(resource_count + demand_count)
end

# END FILE: src/ContextualDFL/ContextualDFLExperiments/src/implementations/resource_allocation_problem/scenario_decoders/ResourceAllocationDemandDecoders.jl

# BEGIN FILE: src/ContextualDFL/ContextualDFLExperiments/src/implementations/shipment_planning/ShipmentPlanningDataGenerators.jl
import Random

function generate_benchmark_contexts(
    problem::ShipmentPlanningProblem;
    n_contexts,
    rng=Random.default_rng(),
)
    context_count = _checked_positive_integer(n_contexts, :n_contexts)
    return [randn(rng, problem.context_dim) for _ in 1:context_count]
end

function generate_benchmark_scenarios(
    problem::ShipmentPlanningProblem,
    context;
    n_scenarios,
    rng=Random.default_rng(),
)
    scenario_count = _checked_positive_integer(n_scenarios, :n_scenarios)
    context_vector = _checked_context_vector(context, problem.context_dim)
    return [
        _shipment_planning_scenario(
            problem,
            _shipment_planning_demand(problem, context_vector, rng),
        ) for _ in 1:scenario_count
    ]
end

function _shipment_planning_demand(
    problem::ShipmentPlanningProblem,
    context::AbstractVector,
    rng::Random.AbstractRNG,
)
    features = Float64.(context) .^ problem.p
    demand = zeros(Float64, problem.demand_count)
    for j in 1:problem.demand_count
        signal = problem.demand_intercepts[j] +
                 sum(problem.demand_slopes[j, term] * features[term] for term in 1:problem.context_dim)
        demand[j] = max(1e-6, signal + problem.sigma * randn(rng))
    end
    return demand
end

function _shipment_planning_scenario(problem::ShipmentPlanningProblem, demand::AbstractVector)
    length(demand) == problem.demand_count ||
        throw(DimensionMismatch("demand must have length $(problem.demand_count)."))

    base = base_scenario(problem)
    h_eq = copy(base.h_eq)
    h_eq[1:problem.demand_count] = Float64.(demand)

    return ContextualDFL.ParametricScenario(;
        W_eq_xi=copy(base.W_eq),
        W_ineq_xi=copy(base.W_ineq),
        T_eq_xi=copy(base.T_eq),
        T_ineq_xi=copy(base.T_ineq),
        h_eq_xi=h_eq,
        h_ineq_xi=copy(base.h_ineq),
        q_xi=copy(base.q),
    )
end

# END FILE: src/ContextualDFL/ContextualDFLExperiments/src/implementations/shipment_planning/ShipmentPlanningDataGenerators.jl

# BEGIN FILE: src/ContextualDFL/ContextualDFLExperiments/src/implementations/shipment_planning/ShipmentPlanningDecoders.jl
ShipmentPlanningParametricDecoder(problem::ShipmentPlanningProblem) =
    ContextualDFL.ParametricDecoder()

# END FILE: src/ContextualDFL/ContextualDFLExperiments/src/implementations/shipment_planning/ShipmentPlanningDecoders.jl

# BEGIN FILE: src/ContextualDFL/ContextualDFLExperiments/src/implementations/shipment_planning/ShipmentPlanningProblem.jl
import LinearAlgebra
import Random

struct ShipmentPlanningProblem <: ProgramInstance
    warehouse_count::Int
    demand_count::Int
    context_dim::Int
    p::Float64
    sigma::Float64
    demand_intercepts::Vector{Float64}
    demand_slopes::Matrix{Float64}
    stochastic_program::ContextualDFL.StochasticProgram
    base_scenario::NamedTuple
end

function ShipmentPlanningProblem(;
    I=5,
    J=12,
    context_dim=3,
    p=2.0,
    sigma=5.0,
    parameter_seed=1,
    production_costs=nothing,
    emergency_costs=nothing,
    shipment_costs=nothing,
    shortage_costs=nothing,
    unused_costs=nothing,
    demand_intercepts=nothing,
    demand_slopes=nothing,
)
    warehouse_count = _checked_positive_integer(I, :I)
    demand_count = _checked_positive_integer(J, :J)
    checked_context_dim = _checked_positive_integer(context_dim, :context_dim)
    checked_sigma = Float64(sigma)
    checked_sigma >= 0.0 || throw(ArgumentError("sigma must be nonnegative."))

    rng = Random.MersenneTwister(parameter_seed)
    c_z = _checked_vector_or_default(
        production_costs,
        fill(1.0, warehouse_count),
        warehouse_count,
        :production_costs,
    )
    q_emergency = _checked_vector_or_default(
        emergency_costs,
        fill(5.0, warehouse_count),
        warehouse_count,
        :emergency_costs,
    )
    q_ship = _checked_matrix_or_default(
        shipment_costs,
        _sample_shipment_costs(rng, warehouse_count, demand_count),
        warehouse_count,
        demand_count,
        :shipment_costs,
    )
    q_shortage = _checked_vector_or_default(
        shortage_costs,
        fill(20.0, demand_count),
        demand_count,
        :shortage_costs,
    )
    q_unused = _checked_vector_or_default(
        unused_costs,
        fill(0.01, warehouse_count),
        warehouse_count,
        :unused_costs,
    )
    intercepts = _checked_vector_or_default(
        demand_intercepts,
        50.0 .+ 50.0 .* rand(rng, demand_count),
        demand_count,
        :demand_intercepts,
    )
    slopes = _checked_matrix_or_default(
        demand_slopes,
        5.0 .+ 10.0 .* rand(rng, demand_count, checked_context_dim),
        demand_count,
        checked_context_dim,
        :demand_slopes,
    )

    program, scenario = _shipment_planning_program_and_scenario(
        warehouse_count,
        demand_count,
        c_z,
        q_emergency,
        q_ship,
        q_shortage,
        q_unused,
    )

    return ShipmentPlanningProblem(
        warehouse_count,
        demand_count,
        checked_context_dim,
        Float64(p),
        checked_sigma,
        intercepts,
        slopes,
        program,
        scenario,
    )
end

stochastic_program(problem::ShipmentPlanningProblem) = problem.stochastic_program

base_scenario(problem::ShipmentPlanningProblem) = problem.base_scenario

function _shipment_planning_program_and_scenario(
    warehouse_count,
    demand_count,
    c_z,
    q_emergency,
    q_ship,
    q_shortage,
    q_unused,
)
    recourse_count =
        warehouse_count + warehouse_count * demand_count + demand_count + warehouse_count
    equality_count = demand_count + warehouse_count

    emergency_index(i) = i
    shipment_index(i, j) = warehouse_count + (j - 1) * warehouse_count + i
    shortage_index(j) = warehouse_count + warehouse_count * demand_count + j
    unused_index(i) = warehouse_count + warehouse_count * demand_count + demand_count + i

    W_eq = zeros(Float64, equality_count, recourse_count)
    T_eq = zeros(Float64, equality_count, warehouse_count)
    h_eq = zeros(Float64, equality_count)

    for j in 1:demand_count
        for i in 1:warehouse_count
            W_eq[j, shipment_index(i, j)] = 1.0
        end
        W_eq[j, shortage_index(j)] = 1.0
    end

    for i in 1:warehouse_count
        row = demand_count + i
        W_eq[row, emergency_index(i)] = -1.0
        for j in 1:demand_count
            W_eq[row, shipment_index(i, j)] = 1.0
        end
        W_eq[row, unused_index(i)] = 1.0
        T_eq[row, i] = -1.0
    end

    q = vcat(q_emergency, vec(q_ship), q_shortage, q_unused)

    program = ContextualDFL.StochasticProgram(
        A_eq=zeros(Float64, 0, warehouse_count),
        A_ineq=-Matrix{Float64}(LinearAlgebra.I, warehouse_count, warehouse_count),
        b_eq=Float64[],
        b_ineq=zeros(Float64, warehouse_count),
        c=c_z,
    )

    scenario = (;
        W_eq=W_eq,
        W_ineq=-Matrix{Float64}(LinearAlgebra.I, recourse_count, recourse_count),
        T_eq=T_eq,
        T_ineq=zeros(Float64, recourse_count, warehouse_count),
        h_eq=h_eq,
        h_ineq=zeros(Float64, recourse_count),
        q=q,
    )

    return program, scenario
end

function _sample_shipment_costs(rng::Random.AbstractRNG, warehouse_count, demand_count)
    warehouse_locations = rand(rng, warehouse_count, 2)
    demand_locations = rand(rng, demand_count, 2)
    costs = zeros(Float64, warehouse_count, demand_count)
    for i in 1:warehouse_count
        for j in 1:demand_count
            dx = warehouse_locations[i, 1] - demand_locations[j, 1]
            dy = warehouse_locations[i, 2] - demand_locations[j, 2]
            costs[i, j] = 1.0 + 3.0 * sqrt(dx^2 + dy^2) + 0.1 * rand(rng)
        end
    end
    return costs
end

# END FILE: src/ContextualDFL/ContextualDFLExperiments/src/implementations/shipment_planning/ShipmentPlanningProblem.jl

# BEGIN FILE: src/ContextualDFL/ContextualDFLExperiments/src/implementations/transshipment_problem/TransShipmentDataGenerators.jl
import Random

function generate_benchmark_contexts(
    problem::TransShipmentExperimentProblem;
    n_contexts,
    rng=Random.default_rng(),
)
    context_count = _checked_positive_integer(n_contexts, :n_contexts)
    return [randn(rng, problem.context_dim) for _ in 1:context_count]
end

function generate_benchmark_scenarios(
    problem::TransShipmentExperimentProblem,
    context;
    n_scenarios,
    rng=Random.default_rng(),
)
    scenario_count = _checked_positive_integer(n_scenarios, :n_scenarios)
    context_vector = _checked_context_vector(context, problem.context_dim)
    return [
        sample_transshipment_experiment_scenario(problem, context_vector; rng=rng)
        for _ in 1:scenario_count
    ]
end

function sample_transshipment_experiment_scenario(
    problem::TransShipmentExperimentProblem,
    context;
    rng=Random.default_rng(),
)
    context_vector = _checked_context_vector(context, problem.context_dim)
    mean_parameters = ContextualDFL.transshipment_mean_parameters(problem.core_problem)

    rhs = if problem.variant in (:h_only, :h_and_q)
        _contextual_positive_values(
            mean_parameters.rhs,
            problem.B_h,
            context_vector,
            problem.sigma_h,
            rng,
        )
    else
        copy(mean_parameters.rhs)
    end

    q = if problem.variant in (:q_only, :h_and_q)
        _contextual_positive_values(
            mean_parameters.q,
            problem.B_q,
            context_vector,
            problem.sigma_q,
            rng,
        )
    else
        copy(mean_parameters.q)
    end

    return ContextualDFL.ParametricScenario(; h_eq_xi=rhs, q_xi=q)
end

function _contextual_positive_values(
    mean_values::AbstractVector,
    slopes::AbstractMatrix,
    context::AbstractVector,
    sigma::Real,
    rng::Random.AbstractRNG,
)
    noise = sigma .* randn(rng, length(mean_values))
    log_values = log.(Float64.(mean_values)) .+ slopes * Float64.(context) .+ noise
    return max.(1e-4, exp.(log_values))
end

# END FILE: src/ContextualDFL/ContextualDFLExperiments/src/implementations/transshipment_problem/TransShipmentDataGenerators.jl

# BEGIN FILE: src/ContextualDFL/ContextualDFLExperiments/src/implementations/transshipment_problem/TransShipmentDecoders.jl
transshipment_decoder(problem::ContextualDFL.TransShipmentProblem) =
    ContextualDFL.TransShipmentScenarioDecoder(problem)

struct TransShipmentComponentVectorDecoder{TDecoder} <: ContextualDFL.VectorDecoder
    decoder::TDecoder
    learned_components::Tuple{Vararg{Symbol}}
end

function TransShipmentComponentVectorDecoder(
    decoder::ContextualDFL.TransShipmentScenarioDecoder;
    learned_components=(:q,),
)
    return TransShipmentComponentVectorDecoder(
        decoder,
        _checked_transshipment_learned_components(learned_components),
    )
end

TransShipmentComponentVectorDecoder(
    problem::ContextualDFL.TransShipmentProblem;
    learned_components=(:q,),
) = TransShipmentComponentVectorDecoder(
    transshipment_decoder(problem);
    learned_components=learned_components,
)

TransShipmentComponentVectorDecoder(
    problem::TransShipmentExperimentProblem;
    learned_components=(:q,),
) = TransShipmentComponentVectorDecoder(
    transshipment_decoder(problem);
    learned_components=learned_components,
)

function (decoder::TransShipmentComponentVectorDecoder)(vector::AbstractVector)
    rhs_mean = decoder.decoder.mean_rhs_values
    q_mean = decoder.decoder.mean_objective_values
    components = decoder.learned_components

    scenario = if components == (:h_eq,)
        length(vector) == length(rhs_mean) ||
            throw(DimensionMismatch("expected $(length(rhs_mean)) transshipment h_eq values."))
        (; rhs=vector, q=q_mean)
    elseif components == (:q,)
        length(vector) == length(q_mean) ||
            throw(DimensionMismatch("expected $(length(q_mean)) transshipment q values."))
        (; rhs=rhs_mean, q=vector)
    else
        expected_length = length(rhs_mean) + length(q_mean)
        length(vector) == expected_length ||
            throw(DimensionMismatch("expected $expected_length transshipment h_eq and q values."))
        rhs_range = 1:length(rhs_mean)
        q_range = (length(rhs_mean) + 1):expected_length
        (; rhs=view(vector, rhs_range), q=view(vector, q_range))
    end

    return decoder.decoder(scenario)
end

function _checked_transshipment_learned_components(learned_components)
    raw_components = learned_components isa Symbol ? (learned_components,) : Tuple(learned_components)
    components = map(raw_components) do component
        symbol = Symbol(component)
        symbol in (:h, :rhs) && return :h_eq
        return symbol
    end
    components in ((:h_eq,), (:q,), (:h_eq, :q)) ||
        throw(ArgumentError("learned_components must be (:h_eq,), (:q,), or (:h_eq, :q)."))
    return components
end

# END FILE: src/ContextualDFL/ContextualDFLExperiments/src/implementations/transshipment_problem/TransShipmentDecoders.jl

# BEGIN FILE: src/ContextualDFL/ContextualDFLExperiments/src/implementations/transshipment_problem/TransShipmentExperimentProblem.jl
import Random

struct TransShipmentExperimentProblem <: ProgramInstance
    core_problem::ContextualDFL.TransShipmentProblem
    variant::Symbol
    context_dim::Int
    sigma_h::Float64
    sigma_q::Float64
    B_h::Matrix{Float64}
    B_q::Matrix{Float64}
end

function TransShipmentExperimentProblem(;
    core_problem=ContextualDFL.TransShipmentProblem(),
    variant=:q_only,
    context_dim=3,
    sigma_h=0.20,
    sigma_q=0.20,
    parameter_seed=1,
    B_h=nothing,
    B_q=nothing,
)
    checked_variant = _checked_transshipment_variant(variant)
    checked_context_dim = _checked_positive_integer(context_dim, :context_dim)
    checked_sigma_h = Float64(sigma_h)
    checked_sigma_q = Float64(sigma_q)
    checked_sigma_h >= 0.0 || throw(ArgumentError("sigma_h must be nonnegative."))
    checked_sigma_q >= 0.0 || throw(ArgumentError("sigma_q must be nonnegative."))

    mean_parameters = ContextualDFL.transshipment_mean_parameters(core_problem)
    rng = Random.MersenneTwister(parameter_seed)
    rhs_count = length(mean_parameters.rhs)
    q_count = length(mean_parameters.q)

    checked_B_h = _checked_matrix_or_default(
        B_h,
        0.08 .* randn(rng, rhs_count, checked_context_dim),
        rhs_count,
        checked_context_dim,
        :B_h,
    )
    checked_B_q = _checked_matrix_or_default(
        B_q,
        0.08 .* randn(rng, q_count, checked_context_dim),
        q_count,
        checked_context_dim,
        :B_q,
    )

    return TransShipmentExperimentProblem(
        core_problem,
        checked_variant,
        checked_context_dim,
        checked_sigma_h,
        checked_sigma_q,
        checked_B_h,
        checked_B_q,
    )
end

stochastic_program(problem::TransShipmentExperimentProblem) =
    ContextualDFL.stochastic_program(problem.core_problem)

base_scenario(problem::TransShipmentExperimentProblem) =
    ContextualDFL.base_scenario(problem.core_problem)

transshipment_decoder(problem::TransShipmentExperimentProblem) =
    ContextualDFL.TransShipmentScenarioDecoder(problem.core_problem)

function _checked_transshipment_variant(variant)
    checked_variant = Symbol(variant)
    checked_variant in (:q_only, :h_only, :h_and_q) ||
        throw(ArgumentError("variant must be one of :q_only, :h_only, or :h_and_q."))
    return checked_variant
end

# END FILE: src/ContextualDFL/ContextualDFLExperiments/src/implementations/transshipment_problem/TransShipmentExperimentProblem.jl

# BEGIN FILE: src/ContextualDFL/ContextualDFLExperiments/src/implementations/unreliable_newsvendor/UnreliableNewsvendorDataGenerators.jl
import Random

function generate_benchmark_contexts(
    problem::UnreliableNewsvendorProblem;
    n_contexts,
    rng=Random.default_rng(),
)
    context_count = _checked_positive_integer(n_contexts, :n_contexts)
    return [
        1.0 .+ 1.0e-6 .* rand(rng, problem.context_dim)
        for _ in 1:context_count
    ]
end

function generate_benchmark_scenarios(
    problem::UnreliableNewsvendorProblem,
    context;
    n_scenarios,
    rng=Random.default_rng(),
)
    scenario_count = _checked_positive_integer(n_scenarios, :n_scenarios)
    _checked_context_vector(context, problem.context_dim)
    return [
        unreliable_newsvendor_scenario(
            problem,
            problem.demand_upper_bound * rand(rng),
            rand(rng),
        ) for _ in 1:scenario_count
    ]
end

function unreliable_newsvendor_scenario(
    problem::UnreliableNewsvendorProblem,
    demand::Real,
    reliability::Real,
)
    checked_demand = Float64(demand)
    checked_reliability = Float64(reliability)

    isfinite(checked_demand) ||
        throw(ArgumentError("demand must be finite."))
    0.0 <= checked_demand <= problem.demand_upper_bound ||
        throw(ArgumentError("demand must be between 0 and $(problem.demand_upper_bound)."))
    isfinite(checked_reliability) ||
        throw(ArgumentError("reliability must be finite."))
    0.0 <= checked_reliability <= 1.0 ||
        throw(ArgumentError("reliability must be between 0 and 1."))

    return ContextualDFL.ParametricScenario(;
        h_eq_xi=[checked_demand, checked_reliability],
    )
end

# END FILE: src/ContextualDFL/ContextualDFLExperiments/src/implementations/unreliable_newsvendor/UnreliableNewsvendorDataGenerators.jl

# BEGIN FILE: src/ContextualDFL/ContextualDFLExperiments/src/implementations/unreliable_newsvendor/UnreliableNewsvendorDecoders.jl
import ChainRulesCore

struct UnreliableNewsvendorParametricDecoder{TBaseScenario} <: ContextualDFL.ScenarioDecoder
    base_scenario::TBaseScenario
end

UnreliableNewsvendorParametricDecoder(problem::UnreliableNewsvendorProblem) =
    UnreliableNewsvendorParametricDecoder(base_scenario(problem))

function (decoder::UnreliableNewsvendorParametricDecoder)(
    scenario_parameters::ContextualDFL.ParametricScenario,
)
    demand, reliability = _unreliable_newsvendor_parameters(scenario_parameters)
    scenario = decoder.base_scenario

    return (
        scenario.W_eq,
        scenario.W_ineq,
        _unreliable_newsvendor_T_eq(demand, reliability),
        scenario.T_ineq,
        _unreliable_newsvendor_h_eq(demand, reliability),
        scenario.h_ineq,
        scenario.q,
    )
end

function _unreliable_newsvendor_parameters(
    scenario_parameters::ContextualDFL.ParametricScenario,
)
    parameters = scenario_parameters.h_eq_xi
    parameters isa AbstractVector ||
        throw(ArgumentError("newsvendor scenario parameters must be a vector [D, U]."))
    length(parameters) == 2 ||
        throw(DimensionMismatch("newsvendor scenario parameters must have length 2."))
    return parameters[1], parameters[2]
end

function _unreliable_newsvendor_T_eq(demand, reliability)
    zero_entry = zero(demand + reliability)
    return reshape([zero_entry, -reliability], 2, 1)
end

function _unreliable_newsvendor_h_eq(demand, reliability)
    zero_entry = zero(demand + reliability)
    return [-demand, zero_entry]
end

function ChainRulesCore.rrule(
    ::typeof(ContextualDFL.decode_scenario_collection),
    decoder::UnreliableNewsvendorParametricDecoder,
    scenario_parameter_collection::AbstractVector{<:ContextualDFL.ParametricScenario},
)
    output = ContextualDFL.decode_scenario_collection(decoder, scenario_parameter_collection)

    function unreliable_newsvendor_decode_pullback(output_tangent)
        dT_eq_array = ContextualDFL._array_cotangent(
            output_tangent,
            3,
            output[3];
            name=:T_eq_array,
        )
        dh_eq_array = ContextualDFL._array_cotangent(
            output_tangent,
            5,
            output[5];
            name=:h_eq_array,
        )

        scenario_tangents = map(enumerate(scenario_parameter_collection)) do (k, scenario_parameters)
            parameter_tangent = [
                -dh_eq_array[1, k],
                -dT_eq_array[2, 1, k],
            ]

            ChainRulesCore.Tangent{typeof(scenario_parameters)}(
                W_eq_xi=ChainRulesCore.NoTangent(),
                W_ineq_xi=ChainRulesCore.NoTangent(),
                T_eq_xi=ChainRulesCore.NoTangent(),
                T_ineq_xi=ChainRulesCore.NoTangent(),
                h_eq_xi=ChainRulesCore.ProjectTo(scenario_parameters.h_eq_xi)(
                    parameter_tangent,
                ),
                h_ineq_xi=ChainRulesCore.NoTangent(),
                q_xi=ChainRulesCore.NoTangent(),
            )
        end

        return (
            ChainRulesCore.NoTangent(),
            ChainRulesCore.NoTangent(),
            scenario_tangents,
        )
    end

    return output, unreliable_newsvendor_decode_pullback
end

# END FILE: src/ContextualDFL/ContextualDFLExperiments/src/implementations/unreliable_newsvendor/UnreliableNewsvendorDecoders.jl

# BEGIN FILE: src/ContextualDFL/ContextualDFLExperiments/src/implementations/unreliable_newsvendor/UnreliableNewsvendorProblem.jl
import LinearAlgebra

struct UnreliableNewsvendorProblemData
    p::Float64
    c::Float64
    pi::Float64
    eta::Float64

    function UnreliableNewsvendorProblemData(
        p::Real,
        c::Real,
        pi::Real,
        eta::Real,
    )
        values = Float64.((p, c, pi, eta))
        all(isfinite, values) ||
            throw(ArgumentError("newsvendor cost parameters must be finite."))
        return new(values...)
    end
end

UnreliableNewsvendorProblemData(; p=5.0, c=1.0, pi=5.0, eta=0.5) =
    UnreliableNewsvendorProblemData(p, c, pi, eta)

struct UnreliableNewsvendorProblem <: ProgramInstance
    problem_data::UnreliableNewsvendorProblemData
    context_dim::Int
    demand_upper_bound::Float64
    stochastic_program::ContextualDFL.StochasticProgram
    base_scenario::NamedTuple
end

function UnreliableNewsvendorProblem(;
    problem_data::UnreliableNewsvendorProblemData=UnreliableNewsvendorProblemData(),
    context_dim=1,
    demand_upper_bound=1.0,
)
    checked_context_dim = _checked_positive_integer(context_dim, :context_dim)
    checked_demand_upper_bound = Float64(demand_upper_bound)
    isfinite(checked_demand_upper_bound) && checked_demand_upper_bound > 0.0 ||
        throw(ArgumentError("demand_upper_bound must be positive and finite."))

    program, scenario = _unreliable_newsvendor_program_and_scenario(problem_data)

    return UnreliableNewsvendorProblem(
        problem_data,
        checked_context_dim,
        checked_demand_upper_bound,
        program,
        scenario,
    )
end

stochastic_program(problem::UnreliableNewsvendorProblem) = problem.stochastic_program

base_scenario(problem::UnreliableNewsvendorProblem) = problem.base_scenario

function _unreliable_newsvendor_program_and_scenario(
    problem_data::UnreliableNewsvendorProblemData,
)
    W_eq = [1.0 -1.0 -1.0; 0.0 0.0 1.0]
    q = [problem_data.p + problem_data.eta, problem_data.pi, problem_data.c - problem_data.p]

    program = ContextualDFL.StochasticProgram(
        A_eq=zeros(Float64, 0, 1),
        A_ineq=-Matrix{Float64}(LinearAlgebra.I, 1, 1),
        b_eq=Float64[],
        b_ineq=[0.0],
        c=[0.0],
    )

    scenario = (;
        W_eq=W_eq,
        W_ineq=-Matrix{Float64}(LinearAlgebra.I, 3, 3),
        T_eq=zeros(Float64, 2, 1),
        T_ineq=zeros(Float64, 3, 1),
        h_eq=zeros(Float64, 2),
        h_ineq=zeros(Float64, 3),
        q=q,
    )

    return program, scenario
end

# END FILE: src/ContextualDFL/ContextualDFLExperiments/src/implementations/unreliable_newsvendor/UnreliableNewsvendorProblem.jl

# BEGIN FILE: src/ContextualDFL/ContextualDFLExperiments/src/program_instance/ProgramInstance.jl
abstract type ProgramInstance end

stochastic_program(instance::ProgramInstance) =
    error("Stochastic-program construction is not defined for $(typeof(instance)).")

"""
    base_scenario(instance::ProgramInstance)

Return fixed scenario data for a program instance.

Concrete methods should return a `NamedTuple` with fields
`W_eq`, `W_ineq`, `T_eq`, `T_ineq`, `h_eq`, `h_ineq`, and `q`.
Scenario decoders can then combine this base scenario with generated
scenario parameters, usually by replacing one right-hand-side vector.
"""
base_scenario(instance::ProgramInstance) =
    error("Base-scenario construction is not defined for $(typeof(instance)).")

# END FILE: src/ContextualDFL/ContextualDFLExperiments/src/program_instance/ProgramInstance.jl

# BEGIN FILE: src/ContextualDFL/ContextualDFLExperiments/src/testing/evaluation/evaluation.jl
"""
Compute benchmark optima for a contextual data set.

`evaluation_batches` is the number of independent scenario collections stored
contiguously in each data point. For each context, every collection is solved to
optimality separately and the returned objective value is their average.
"""
function solve_dataset_to_optimality(
    contextual_data_set,
    program,
    parametric_decoder,
    solver;
    mu=0,
    rho=0,
    evaluation_batches=1,
    kwargs...,
)
    batch_count = _checked_evaluation_batches(evaluation_batches)
    batch_count > 1 && :probabilities in keys((; kwargs...)) &&
        throw(ArgumentError("evaluation_batches > 1 expects equally weighted scenario collections; omit explicit probabilities."))

    results = NamedTuple[]
    for data_point in contextual_data_set
        objective_values = Float64[]

        for scenario_range in _scenario_collection_ranges(data_point, batch_count)
            W_eq, W_ineq, T_eq, T_ineq, h_eq, h_ineq, q =
                ContextualDFL.decode_scenario_collection(
                    parametric_decoder,
                    view(data_point.scenario_parameters, scenario_range),
                )

            solution = ContextualDFL.solve(
                solver,
                program,
                W_eq,
                W_ineq,
                T_eq,
                T_ineq,
                h_eq,
                h_ineq,
                q;
                μ=mu,
                ρ=rho,
                kwargs...,
            )
            z = solution[1]

            push!(
                objective_values,
                ContextualDFL.cost_function(
                    program,
                    solver,
                    z,
                    W_eq,
                    W_ineq,
                    T_eq,
                    T_ineq,
                    h_eq,
                    h_ineq,
                    q;
                    μ=mu,
                    ρ=rho,
                    kwargs...,
                ),
            )
        end

        push!(
            results,
            (;
                evaluation_batches=batch_count,
                objective_values=objective_values,
                objective_value=summary_mean(objective_values),
            ),
        )
    end
    return results
end

"""
Evaluate a fixed policy or decision matrix on scenario collections.

The policy supplies one first-stage decision per context. That fixed decision is
scored on every scenario collection for the same context, and the returned value
is the mean over collections.
"""
function evaluate_policy(
    decision_set::AbstractMatrix,
    contextual_data_set,
    program,
    parametric_decoder,
    solver;
    mu=0,
    rho=0,
    evaluation_batches=1,
    kwargs...,
)
    values, _ = _evaluate_decision_set(
        decision_set,
        contextual_data_set,
        program,
        parametric_decoder,
        solver;
        mu=mu,
        rho=rho,
        evaluation_batches=evaluation_batches,
        kwargs...,
    )
    return values
end

function _evaluate_decision_set(
    decision_set::AbstractMatrix,
    contextual_data_set,
    program,
    parametric_decoder,
    solver;
    mu=0,
    rho=0,
    evaluation_batches=1,
    kwargs...,
)
    @assert size(decision_set, 2) == length(contextual_data_set) "Each context must map to exactly one decision z"
    size(decision_set, 2) == length(contextual_data_set) ||
        throw(DimensionMismatch("decision_set must have one column per data point."))

    batch_count = _checked_evaluation_batches(evaluation_batches)
    batch_count > 1 && :probabilities in keys((; kwargs...)) &&
        throw(ArgumentError("evaluation_batches > 1 expects equally weighted scenario collections; omit explicit probabilities."))

    values = Float64[]
    values_by_collection = Vector{Vector{Float64}}()

    for data_index in eachindex(contextual_data_set)
        collection_values = _evaluate_decision_on_collections(
            view(decision_set, :, data_index),
            contextual_data_set[data_index],
            program,
            parametric_decoder,
            solver;
            mu=mu,
            rho=rho,
            evaluation_batches=batch_count,
            kwargs...,
        )

        push!(values, summary_mean(collection_values))
        push!(values_by_collection, collection_values)
    end
    return values, values_by_collection
end

function _evaluate_decision_on_collections(
    z,
    data_point,
    program,
    parametric_decoder,
    solver;
    mu=0,
    rho=0,
    evaluation_batches=1,
    kwargs...,
)
    values = Float64[]
    for scenario_range in _scenario_collection_ranges(data_point, evaluation_batches)
        W_eq, W_ineq, T_eq, T_ineq, h_eq, h_ineq, q =
            ContextualDFL.decode_scenario_collection(
                parametric_decoder,
                view(data_point.scenario_parameters, scenario_range),
            )

        push!(
            values,
            ContextualDFL.cost_function(
                program,
                solver,
                z,
                W_eq,
                W_ineq,
                T_eq,
                T_ineq,
                h_eq,
                h_ineq,
                q;
                μ=mu,
                ρ=rho,
                kwargs...,
            ),
        )
    end
    return values
end

function _checked_evaluation_batches(evaluation_batches)
    isnothing(evaluation_batches) && return 1
    evaluation_batches isa Integer ||
        throw(ArgumentError(
            "evaluation_batches must be a positive integer, got $(typeof(evaluation_batches)).",
        ))

    batch_count = Int(evaluation_batches)
    batch_count > 0 ||
        throw(ArgumentError("evaluation_batches must be positive, got $batch_count."))
    return batch_count
end

function _scenario_collection_ranges(data_point, batch_count::Integer)
    scenario_count = length(data_point.scenario_parameters)
    scenario_count > 0 || throw(ArgumentError("scenario collections must not be empty."))
    scenario_count % batch_count == 0 ||
        throw(ArgumentError(
            "scenario count $scenario_count is not divisible by evaluation_batches=$batch_count.",
        ))

    batch_size = scenario_count ÷ batch_count
    return [
        ((batch_index - 1) * batch_size + 1):(batch_index * batch_size)
        for batch_index in 1:batch_count
    ]
end

function summarize_values(values; prefix)
    numeric_values = Float64.(collect(values))
    count = length(numeric_values)
    prefix = Symbol(prefix)

    summary = if count == 0
        (;
            count=0,
            mean=NaN,
            median=NaN,
            std=NaN,
            min=NaN,
            max=NaN,
            p95=NaN,
        )
    else
        (;
            count=count,
            mean=summary_mean(numeric_values),
            median=summary_median(numeric_values),
            std=summary_std(numeric_values),
            min=minimum(numeric_values),
            max=maximum(numeric_values),
            p95=percentile_95(numeric_values),
        )
    end

    return prefix_named_tuple(prefix, summary)
end

function summarize_regret(policy_values, optimal_values; prefix)
    length(policy_values) == length(optimal_values) ||
        throw(DimensionMismatch("policy_values and optimal_values must have the same length."))

    regrets = Float64.(policy_values) .- Float64.(optimal_values)
    relative_regrets = [
        regret / max(abs(Float64(optimal_value)), eps(Float64)) for
        (regret, optimal_value) in zip(regrets, optimal_values)
    ]

    return merge(
        summarize_values(regrets; prefix=Symbol(prefix, :_regret)),
        summarize_values(relative_regrets; prefix=Symbol(prefix, :_relative_regret)),
    )
end

"""
Compare a fixed policy against precomputed benchmark optima.

`optimal_results` must come from `solve_dataset_to_optimality` on the same
scenario realizations. Policy and optimal values are compared collection by
collection, then averaged per context.
"""
function evaluate_policy_against_optimum(
    policy_or_decision_set,
    contextual_data_set,
    program,
    parametric_decoder,
    solver;
    optimal_results,
    split_name=:test,
    mu=0,
    rho=0,
    kwargs...,
)
    @assert length(optimal_results) == length(contextual_data_set) "optimal_results must have one entry per data point"
    length(optimal_results) == length(contextual_data_set) ||
        throw(DimensionMismatch("optimal_results must have one entry per data point."))

    optimal_values_by_collection = [
        _optimal_objective_values(result) for result in optimal_results
    ]
    collection_counts = length.(optimal_values_by_collection)
    batch_count = if isempty(collection_counts)
        1
    else
        all(>(0), collection_counts) ||
            throw(ArgumentError("optimal_results must contain at least one objective value per sample."))
        all(==(first(collection_counts)), collection_counts) ||
            throw(ArgumentError("optimal_results contain mixed evaluation batch counts."))
        first(collection_counts)
    end

    policy_values = Float64[]
    policy_values_by_collection = Vector{Vector{Float64}}()
    policy_eval_seconds = @elapsed begin
        decision_set =
            _decision_set_for_evaluation(policy_or_decision_set, contextual_data_set)
        policy_values, policy_values_by_collection = _evaluate_decision_set(
            decision_set,
            contextual_data_set,
            program,
            parametric_decoder,
            solver;
            mu=mu,
            rho=rho,
            evaluation_batches=batch_count,
            kwargs...,
        )
    end

    optimal_values = [summary_mean(values) for values in optimal_values_by_collection]
    gap_values_by_collection = [
        Float64.(policy_values_by_collection[index]) .-
        Float64.(optimal_values_by_collection[index])
        for index in eachindex(policy_values)
    ]
    regrets = [summary_mean(values) for values in gap_values_by_collection]
    relative_regrets = [
        regret / max(abs(Float64(optimal_value)), eps(Float64)) for
        (regret, optimal_value) in zip(regrets, optimal_values)
    ]
    gap_uncertainty = [_uncertainty(values) for values in gap_values_by_collection]

    split_name = Symbol(split_name)
    metrics = merge(
        summarize_values(policy_values; prefix=Symbol(split_name, :_policy_value)),
        summarize_values(optimal_values; prefix=Symbol(split_name, :_optimal_value)),
        summarize_regret(policy_values, optimal_values; prefix=split_name),
        prefix_named_tuple(
            split_name,
            (;
                sample_count=length(contextual_data_set),
                evaluation_batches=batch_count,
                policy_eval_seconds=policy_eval_seconds,
                gap_std_mean=summary_mean(Float64[item.std for item in gap_uncertainty]),
                gap_stderr_mean=summary_mean(Float64[item.stderr for item in gap_uncertainty]),
            ),
        ),
    )

    per_sample = [
        (;
            sample_index=index,
            policy_value=Float64(policy_values[index]),
            optimal_value=Float64(optimal_values[index]),
            regret=regrets[index],
            relative_regret=relative_regrets[index],
            policy_collection_values=policy_values_by_collection[index],
            optimal_collection_values=optimal_values_by_collection[index],
            gap_values=gap_values_by_collection[index],
            gap_std=gap_uncertainty[index].std,
            gap_stderr=gap_uncertainty[index].stderr,
        ) for index in eachindex(policy_values)
    ]

    return (;
        metrics=metrics,
        per_sample=per_sample,
        optimal_results=optimal_results,
    )
end

_decision_set_for_evaluation(decision_set::AbstractMatrix, contextual_data_set) = decision_set

function _decision_set_for_evaluation(policy::Policy, contextual_data_set)
    return generate_decision_set(policy, contextual_data_set)
end

function _optimal_objective_values(result)
    if hasproperty(result, :objective_values)
        values = Float64.(collect(result.objective_values))
        isempty(values) &&
            throw(ArgumentError("optimal_results must contain at least one objective value per sample."))
        return values
    elseif hasproperty(result, :batch_objective_values)
        throw(ArgumentError(
            "optimal_results contain batch_objective_values from the old evaluation protocol; regenerate them with solve_dataset_to_optimality.",
        ))
    elseif hasproperty(result, :objective_value)
        return [Float64(result.objective_value)]
    end

    throw(ArgumentError("optimal_results entries must contain objective_values."))
end

function _uncertainty(values)
    count = length(values)
    std = summary_std(Float64.(collect(values)))
    stderr = count == 0 ? NaN : std / sqrt(count)
    return (; std=std, stderr=stderr)
end

function percentile_95(values::AbstractVector{<:Real})
    isempty(values) && return NaN
    sorted = sort!(collect(Float64.(values)))
    index = clamp(ceil(Int, 0.95 * length(sorted)), 1, length(sorted))
    return sorted[index]
end

function summary_mean(values::AbstractVector{<:Real})
    isempty(values) && return NaN
    return sum(values) / length(values)
end

function summary_median(values::AbstractVector{<:Real})
    isempty(values) && return NaN

    sorted = sort!(collect(Float64.(values)))
    count = length(sorted)
    midpoint = count ÷ 2

    if isodd(count)
        return sorted[midpoint + 1]
    end

    return (sorted[midpoint] + sorted[midpoint + 1]) / 2
end

function summary_std(values::AbstractVector{<:Real})
    count = length(values)
    count == 0 && return NaN
    count == 1 && return 0.0

    mean_value = summary_mean(values)
    return sqrt(sum((value - mean_value)^2 for value in values) / (count - 1))
end

function prefix_named_tuple(prefix::Symbol, values::NamedTuple)
    pairs = Pair{Symbol,Any}[]
    for key in keys(values)
        push!(pairs, Symbol(prefix, :_, key) => getproperty(values, key))
    end
    return NamedTuple{Tuple(first.(pairs))}(Tuple(last.(pairs)))
end

# END FILE: src/ContextualDFL/ContextualDFLExperiments/src/testing/evaluation/evaluation.jl

# BEGIN FILE: src/ContextualDFL/ContextualDFLExperiments/src/testing/policies/BaselinePolicies.jl
import LinearAlgebra

const PARAMETRIC_SCENARIO_COMPONENTS = (
    :W_eq_xi,
    :W_ineq_xi,
    :T_eq_xi,
    :T_ineq_xi,
    :h_eq_xi,
    :h_ineq_xi,
    :q_xi,
)

"""
    SampleAverageApproximationPolicy(data_set, solver, program, parametric_decoder; kwargs...)

Context-free SAA baseline. The policy solves one stochastic program using all
scenario parameters in `data_set` and returns that same first-stage decision for
every context.
"""
struct SampleAverageApproximationPolicy{
    TDecision,
    TScenarios,
    TSolver,
    TProgram,
    TDecoder,
    TMu,
    TRho,
    TKwargs,
} <: Policy
    decision::TDecision
    scenario_parameters::TScenarios
    solver::TSolver
    program::TProgram
    parametric_decoder::TDecoder
    mu::TMu
    rho::TRho
    solve_kwargs::TKwargs
end

function SampleAverageApproximationPolicy(
    scenario_parameters::AbstractVector{<:ContextualDFL.ParametricScenario},
    solver,
    program,
    parametric_decoder;
    mu=0,
    rho=0,
    kwargs...,
)
    checked_scenarios = _checked_scenario_collection(scenario_parameters)
    solve_kwargs = (; kwargs...)
    decision = _solve_scenario_collection(
        solver,
        program,
        parametric_decoder,
        checked_scenarios;
        mu=mu,
        rho=rho,
        solve_kwargs...,
    )

    return SampleAverageApproximationPolicy(
        decision,
        checked_scenarios,
        solver,
        program,
        parametric_decoder,
        mu,
        rho,
        solve_kwargs,
    )
end

function SampleAverageApproximationPolicy(
    contextual_data_set::AbstractVector,
    solver,
    program,
    parametric_decoder;
    mu=0,
    rho=0,
    kwargs...,
)
    scenario_parameters = _flatten_scenario_collections(
        _scenario_collections(contextual_data_set),
    )
    return SampleAverageApproximationPolicy(
        scenario_parameters,
        solver,
        program,
        parametric_decoder;
        mu=mu,
        rho=rho,
        kwargs...,
    )
end

infer(policy::SampleAverageApproximationPolicy, context) = copy(policy.decision)

"""
    LeastSquaresPolicy(data_set, solver, program, parametric_decoder; kwargs...)

Ordinary least-squares certainty-equivalent baseline. The policy fits a linear
map from contexts to one vector-valued `ParametricScenario` component, predicts
that component for a new context, solves the one-scenario stochastic program,
and returns the first-stage decision.

The default `target_component` is `:h_eq_xi`, matching the resource-allocation
demand scenarios. Other simple fixed-structure problems can pass another
component, such as `:h_ineq_xi` or `:q_xi`.
"""
struct LeastSquaresPolicy{
    TCoefficients,
    TTemplate,
    TSolver,
    TProgram,
    TDecoder,
    TPostprocess,
    TMu,
    TRho,
    TKwargs,
} <: Policy
    coefficients::TCoefficients
    scenario_template::TTemplate
    target_component::Symbol
    target_length::Int
    solver::TSolver
    program::TProgram
    parametric_decoder::TDecoder
    postprocess_prediction::TPostprocess
    mu::TMu
    rho::TRho
    solve_kwargs::TKwargs
end

function LeastSquaresPolicy(
    contextual_data_set::AbstractVector,
    solver,
    program,
    parametric_decoder;
    target_component=:h_eq_xi,
    postprocess_prediction=identity,
    validate_fixed_components=true,
    mu=0,
    rho=0,
    kwargs...,
)
    regression = _fit_scenario_target_regression(
        contextual_data_set,
        target_component;
        validate_fixed_components=validate_fixed_components,
    )

    return LeastSquaresPolicy(
        regression.coefficients,
        first(regression.scenario_templates),
        regression.target_component,
        regression.target_length,
        solver,
        program,
        parametric_decoder,
        postprocess_prediction,
        mu,
        rho,
        (; kwargs...),
    )
end

function infer(policy::LeastSquaresPolicy, context)
    target_vector = _processed_prediction(
        policy.postprocess_prediction,
        _predict_target(policy.coefficients, context),
        policy.target_length,
    )
    scenario = _scenario_from_target_vector(
        policy.scenario_template,
        policy.target_component,
        target_vector,
    )

    return _solve_scenario_collection(
        policy.solver,
        policy.program,
        policy.parametric_decoder,
        [scenario];
        mu=policy.mu,
        rho=policy.rho,
        policy.solve_kwargs...,
    )
end

"""
    ResidualSampleAverageApproximationPolicy(data_set, solver, program, parametric_decoder; kwargs...)

Empirical-residual SAA baseline (ER-SAA). It fits the same OLS model as
`LeastSquaresPolicy`, stores training residuals, and at inference time solves
SAA over `prediction(context) + residual` scenarios.
"""
struct ResidualSampleAverageApproximationPolicy{
    TCoefficients,
    TResiduals,
    TTemplates,
    TSolver,
    TProgram,
    TDecoder,
    TPostprocess,
    TMu,
    TRho,
    TKwargs,
} <: Policy
    coefficients::TCoefficients
    residuals::TResiduals
    scenario_templates::TTemplates
    target_component::Symbol
    target_length::Int
    solver::TSolver
    program::TProgram
    parametric_decoder::TDecoder
    postprocess_prediction::TPostprocess
    mu::TMu
    rho::TRho
    solve_kwargs::TKwargs
end

function ResidualSampleAverageApproximationPolicy(
    contextual_data_set::AbstractVector,
    solver,
    program,
    parametric_decoder;
    target_component=:h_eq_xi,
    postprocess_prediction=identity,
    validate_fixed_components=true,
    mu=0,
    rho=0,
    kwargs...,
)
    regression = _fit_scenario_target_regression(
        contextual_data_set,
        target_component;
        validate_fixed_components=validate_fixed_components,
    )

    return ResidualSampleAverageApproximationPolicy(
        regression.coefficients,
        regression.residuals,
        regression.scenario_templates,
        regression.target_component,
        regression.target_length,
        solver,
        program,
        parametric_decoder,
        postprocess_prediction,
        mu,
        rho,
        (; kwargs...),
    )
end

function infer(policy::ResidualSampleAverageApproximationPolicy, context)
    base_target = _predict_target(policy.coefficients, context)
    scenario_parameters = [
        _scenario_from_target_vector(
            policy.scenario_templates[index],
            policy.target_component,
            _processed_prediction(
                policy.postprocess_prediction,
                base_target .+ view(policy.residuals, index, :),
                policy.target_length,
            ),
        ) for index in axes(policy.residuals, 1)
    ]

    return _solve_scenario_collection(
        policy.solver,
        policy.program,
        policy.parametric_decoder,
        scenario_parameters;
        mu=policy.mu,
        rho=policy.rho,
        policy.solve_kwargs...,
    )
end

"""
    KNearestNeighborsPolicy(data_set, solver, program, parametric_decoder; k, kwargs...)

kNN-SAA baseline. At inference time, the policy finds the `k` nearest training
contexts, pools their scenario parameters, solves the resulting SAA problem, and
returns the first-stage decision.
"""
struct KNearestNeighborsPolicy{
    TContexts,
    TScenarioCollections,
    TSolver,
    TProgram,
    TDecoder,
    TMu,
    TRho,
    TKwargs,
} <: Policy
    training_contexts::TContexts
    training_scenario_collections::TScenarioCollections
    k::Int
    solver::TSolver
    program::TProgram
    parametric_decoder::TDecoder
    mu::TMu
    rho::TRho
    solve_kwargs::TKwargs
end

function KNearestNeighborsPolicy(
    contextual_data_set::AbstractVector,
    solver,
    program,
    parametric_decoder;
    k=default_knn_k(length(contextual_data_set)),
    mu=0,
    rho=0,
    kwargs...,
)
    _check_nonempty_data_set(contextual_data_set)
    training_contexts = _contexts(contextual_data_set)
    training_scenario_collections = _scenario_collections(contextual_data_set)
    checked_k = _checked_neighbor_count(k, length(training_contexts))

    return KNearestNeighborsPolicy(
        training_contexts,
        training_scenario_collections,
        checked_k,
        solver,
        program,
        parametric_decoder,
        mu,
        rho,
        (; kwargs...),
    )
end

function infer(policy::KNearestNeighborsPolicy, context)
    neighbor_indices = _nearest_neighbor_indices(
        policy.training_contexts,
        context,
        policy.k,
    )
    scenario_parameters = _flatten_scenario_collections(
        policy.training_scenario_collections[index] for index in neighbor_indices
    )

    return _solve_scenario_collection(
        policy.solver,
        policy.program,
        policy.parametric_decoder,
        scenario_parameters;
        mu=policy.mu,
        rho=policy.rho,
        policy.solve_kwargs...,
    )
end

function default_knn_k(sample_count::Integer; scale=5.0, exponent=0.4)
    n = Int(sample_count)
    n > 0 || throw(ArgumentError("sample_count must be positive, got $n."))

    return min(n, max(1, round(Int, scale * n^exponent)))
end

function _fit_scenario_target_regression(
    contextual_data_set,
    target_component;
    validate_fixed_components,
)
    checked_target_component = _checked_target_component(target_component)
    observations = _scenario_target_observations(
        contextual_data_set,
        checked_target_component,
    )
    validate_fixed_components &&
        _check_fixed_scenario_components(
            observations.scenario_templates,
            checked_target_component,
        )

    design = _design_matrix(observations.contexts)
    coefficients = design \ observations.targets
    fitted_values = design * coefficients

    return (;
        coefficients=coefficients,
        residuals=observations.targets - fitted_values,
        scenario_templates=observations.scenario_templates,
        target_component=checked_target_component,
        target_length=size(observations.targets, 2),
    )
end

function _scenario_target_observations(contextual_data_set, target_component::Symbol)
    _check_nonempty_data_set(contextual_data_set)

    context_vectors = Vector{Vector{Float64}}()
    target_vectors = Vector{Vector{Float64}}()
    scenario_templates = ContextualDFL.ParametricScenario[]
    context_dimension = 0
    target_length = 0

    for data_point in contextual_data_set
        context_vector = Float64.(collect(data_point.context))
        isempty(context_vector) &&
            throw(ArgumentError("contexts must contain at least one feature."))
        if context_dimension == 0
            context_dimension = length(context_vector)
        elseif length(context_vector) != context_dimension
            throw(DimensionMismatch("all contexts must have the same dimension."))
        end

        for scenario in _checked_scenario_collection(data_point.scenario_parameters)
            target_vector = _target_feature_vector(scenario, target_component)
            if target_length == 0
                target_length = length(target_vector)
            elseif length(target_vector) != target_length
                throw(DimensionMismatch(
                    "all target components must have the same flattened length.",
                ))
            end

            push!(context_vectors, context_vector)
            push!(target_vectors, target_vector)
            push!(scenario_templates, scenario)
        end
    end

    isempty(target_vectors) &&
        throw(ArgumentError("regression baselines require at least one scenario."))

    context_matrix = zeros(Float64, length(context_vectors), context_dimension)
    target_matrix = zeros(Float64, length(target_vectors), target_length)
    for index in eachindex(context_vectors)
        context_matrix[index, :] = context_vectors[index]
        target_matrix[index, :] = target_vectors[index]
    end

    return (;
        contexts=context_matrix,
        targets=target_matrix,
        scenario_templates=scenario_templates,
    )
end

function _design_matrix(context_matrix::AbstractMatrix)
    return hcat(context_matrix, ones(Float64, size(context_matrix, 1)))
end

function _predict_target(coefficients::AbstractMatrix, context)
    context_vector = Float64.(collect(context))
    length(context_vector) + 1 == size(coefficients, 1) ||
        throw(DimensionMismatch(
            "context has length $(length(context_vector)); expected $(size(coefficients, 1) - 1).",
        ))

    return vec(transpose(vcat(context_vector, 1.0)) * coefficients)
end

function _checked_target_component(target_component)
    component = Symbol(target_component)
    component in PARAMETRIC_SCENARIO_COMPONENTS ||
        throw(ArgumentError(
            "target_component must be one of $(PARAMETRIC_SCENARIO_COMPONENTS); got $(repr(component)).",
        ))

    return component
end

function _target_feature_vector(scenario, target_component::Symbol)
    target = getproperty(scenario, target_component)
    vector = _numeric_feature_vector(target; name=target_component)
    isempty(vector) &&
        throw(ArgumentError("target component $(target_component) must not be empty."))
    return vector
end

function _numeric_feature_vector(value; name)
    if value isa Number
        return [Float64(value)]
    elseif value isa AbstractArray
        return Float64.(vec(value))
    end

    throw(ArgumentError("$(name) must be numeric or an array of numeric values."))
end

function _processed_prediction(postprocess_prediction, target_vector, target_length::Integer)
    processed = postprocess_prediction(collect(target_vector))
    processed_vector = _numeric_feature_vector(processed; name=:postprocess_prediction)
    length(processed_vector) == target_length ||
        throw(DimensionMismatch(
            "postprocess_prediction returned length $(length(processed_vector)); expected $target_length.",
        ))

    return processed_vector
end

function _scenario_from_target_vector(
    scenario_template,
    target_component::Symbol,
    target_vector::AbstractVector,
)
    replacement = _reshape_target_vector(
        target_vector,
        getproperty(scenario_template, target_component),
        target_component,
    )

    return ContextualDFL.ParametricScenario(;
        W_eq_xi=target_component == :W_eq_xi ? replacement :
                _copy_scenario_component(scenario_template.W_eq_xi),
        W_ineq_xi=target_component == :W_ineq_xi ? replacement :
                  _copy_scenario_component(scenario_template.W_ineq_xi),
        T_eq_xi=target_component == :T_eq_xi ? replacement :
                _copy_scenario_component(scenario_template.T_eq_xi),
        T_ineq_xi=target_component == :T_ineq_xi ? replacement :
                  _copy_scenario_component(scenario_template.T_ineq_xi),
        h_eq_xi=target_component == :h_eq_xi ? replacement :
                _copy_scenario_component(scenario_template.h_eq_xi),
        h_ineq_xi=target_component == :h_ineq_xi ? replacement :
                  _copy_scenario_component(scenario_template.h_ineq_xi),
        q_xi=target_component == :q_xi ? replacement :
             _copy_scenario_component(scenario_template.q_xi),
    )
end

function _reshape_target_vector(target_vector, template_value, target_component)
    if template_value isa Number
        length(target_vector) == 1 ||
            throw(DimensionMismatch(
                "target vector for scalar $(target_component) must have length 1.",
            ))
        return only(target_vector)
    elseif template_value isa AbstractVector
        length(target_vector) == length(template_value) ||
            throw(DimensionMismatch(
                "target vector for $(target_component) has length $(length(target_vector)); expected $(length(template_value)).",
            ))
        return collect(target_vector)
    elseif template_value isa AbstractArray
        length(target_vector) == length(template_value) ||
            throw(DimensionMismatch(
                "target vector for $(target_component) has length $(length(target_vector)); expected $(length(template_value)).",
            ))
        return reshape(collect(target_vector), size(template_value))
    end

    throw(ArgumentError("template component $(target_component) must be numeric or an array."))
end

_copy_scenario_component(value::AbstractArray) = copy(value)
_copy_scenario_component(value) = value

function _check_fixed_scenario_components(scenario_templates, target_component::Symbol)
    base_scenario = first(scenario_templates)
    for scenario in Iterators.drop(scenario_templates, 1)
        for component in PARAMETRIC_SCENARIO_COMPONENTS
            component == target_component && continue
            isequal(getproperty(base_scenario, component), getproperty(scenario, component)) ||
                throw(ArgumentError(
                    "Least-squares baselines require fixed non-target scenario components; $(component) varies.",
                ))
        end
    end

    return nothing
end

function _check_nonempty_data_set(contextual_data_set)
    isempty(contextual_data_set) &&
        throw(ArgumentError("contextual_data_set must not be empty."))
    return nothing
end

function _contexts(contextual_data_set)
    _check_nonempty_data_set(contextual_data_set)
    return [collect(data_point.context) for data_point in contextual_data_set]
end

function _scenario_collections(contextual_data_set)
    _check_nonempty_data_set(contextual_data_set)
    return map(contextual_data_set) do data_point
        _checked_scenario_collection(data_point.scenario_parameters)
    end
end

function _checked_scenario_collection(
    scenario_parameters::AbstractVector{<:ContextualDFL.ParametricScenario},
)
    isempty(scenario_parameters) &&
        throw(ArgumentError("scenario collections must not be empty."))
    return collect(scenario_parameters)
end

function _flatten_scenario_collections(scenario_collections)
    scenario_parameters = ContextualDFL.ParametricScenario[]
    for collection in scenario_collections
        checked_collection = _checked_scenario_collection(collection)
        append!(scenario_parameters, checked_collection)
    end

    isempty(scenario_parameters) &&
        throw(ArgumentError("scenario collections must not be empty."))
    return scenario_parameters
end

function _checked_neighbor_count(k, sample_count::Integer)
    k isa Integer ||
        throw(ArgumentError("k must be an integer, got $(typeof(k))."))

    neighbor_count = Int(k)
    1 <= neighbor_count <= sample_count ||
        throw(ArgumentError("k must be between 1 and $sample_count, got $neighbor_count."))

    return neighbor_count
end

function _nearest_neighbor_indices(training_contexts, context, k::Integer)
    query_context = collect(context)
    distances = [
        _squared_euclidean_distance(training_context, query_context)
        for training_context in training_contexts
    ]
    return partialsortperm(distances, 1:k)
end

function _squared_euclidean_distance(a, b)
    length(a) == length(b) ||
        throw(DimensionMismatch("context dimensions must match."))

    total = 0.0
    @inbounds for index in eachindex(a, b)
        difference = Float64(a[index]) - Float64(b[index])
        total += difference * difference
    end
    return total
end

function _solve_scenario_collection(
    solver,
    program,
    parametric_decoder,
    scenario_parameters::AbstractVector{<:ContextualDFL.ParametricScenario};
    mu=0,
    rho=0,
    kwargs...,
)
    checked_scenarios = _checked_scenario_collection(scenario_parameters)
    W_eq, W_ineq, T_eq, T_ineq, h_eq, h_ineq, q =
        ContextualDFL.decode_scenario_collection(parametric_decoder, checked_scenarios)

    z, _, _, _, _, _ = ContextualDFL.solve(
        solver,
        program,
        W_eq,
        W_ineq,
        T_eq,
        T_ineq,
        h_eq,
        h_ineq,
        q;
        μ=mu,
        ρ=rho,
        kwargs...,
    )

    return collect(z)
end

# END FILE: src/ContextualDFL/ContextualDFLExperiments/src/testing/policies/BaselinePolicies.jl

# BEGIN FILE: src/ContextualDFL/ContextualDFLExperiments/src/testing/policies/Policy.jl
abstract type Policy end

infer(policy::Policy, context) =
    error("Policy inference is not defined for $(typeof(policy)).")

function generate_decision_set(policy::Policy, contextual_data_set)
    isempty(contextual_data_set) &&
        throw(ArgumentError("contextual_data_set must not be empty."))

    decisions = [infer(policy, data_point.context) for data_point in contextual_data_set]
    return reduce(hcat, decisions)
end

function evaluate_policy(policy::Policy, contextual_data_set, program, parametric_decoder, solver; kwargs...)
    decision_set = generate_decision_set(policy, contextual_data_set)
    return evaluate_policy(decision_set, contextual_data_set, program, parametric_decoder, solver; kwargs...)
end

# END FILE: src/ContextualDFL/ContextualDFLExperiments/src/testing/policies/Policy.jl

# BEGIN FILE: src/ContextualDFL/ContextualDFLExperiments/src/testing/policies/ScenarioGenerationPolicy.jl
struct ScenarioGenerationPolicy{
    TGenerator<:ContextualDFL.ScenarioGenerator,
    TSolver,
    TProgram,
    TMu,
    TRho,
} <: Policy
    scenario_generator::TGenerator
    solver::TSolver
    program::TProgram
    mu::TMu
    rho::TRho
end

function ScenarioGenerationPolicy(scenario_generator, solver, program; mu=0, rho=0)
    return ScenarioGenerationPolicy(scenario_generator, solver, program, mu, rho)
end

function infer(policy::ScenarioGenerationPolicy, context)
    scenario_parameters = policy.scenario_generator.neural_net(context)
    W_eq, W_ineq, T_eq, T_ineq, h_eq, h_ineq, q = ContextualDFL.decode_scenario_collection(
        policy.scenario_generator.scenario_decoder,
        scenario_parameters;
        nr_scenarios=1,
    )

    z, _, _, _, _, _ = ContextualDFL.solve(
        policy.solver,
        policy.program,
        W_eq,
        W_ineq,
        T_eq,
        T_ineq,
        h_eq,
        h_ineq,
        q;
        μ=policy.mu,
        ρ=policy.rho,
    )

    return z
end

# END FILE: src/ContextualDFL/ContextualDFLExperiments/src/testing/policies/ScenarioGenerationPolicy.jl

# BEGIN FILE: src/ContextualDFL/ContextualDFLExperiments/test/benchmark_instances/runtests.jl
using ContextualDFL
using ContextualDFLExperiments
using LinearAlgebra
using Random
using Test

function benchmark_solver()
    return ContextualDFL.Solver(ContextualDFL.IpoptSolver(), ContextualDFL.HiGHSSolver())
end

function assert_one_scenario_solve(problem, decoder; seed=1)
    solver = benchmark_solver()
    program = stochastic_program(problem)
    data = generate_benchmark_dataset(
        problem;
        n_contexts=1,
        scenarios_per_context=1,
        seed=seed,
    )
    arrays = ContextualDFL.decode_scenario_collection(decoder, data[1].scenario_parameters)
    z, y = ContextualDFL.solve(solver, program, arrays...)[1:2]
    @test all(isfinite, z)
    @test all(isfinite, y)
    return data
end

function smoke_saa_knn(problem, decoder; seed=1, train_contexts=3)
    solver = benchmark_solver()
    program = stochastic_program(problem)
    train = generate_benchmark_dataset(
        problem;
        n_contexts=train_contexts,
        scenarios_per_context=1,
        seed=seed,
    )
    test = generate_benchmark_dataset(
        problem;
        n_contexts=1,
        scenarios_per_context=1,
        seed=seed + 1,
    )

    saa = SampleAverageApproximationPolicy(train, solver, program, decoder)
    knn = KNearestNeighborsPolicy(train, solver, program, decoder; k=1)

    @test all(isfinite, infer(saa, test[1].context))
    @test all(isfinite, infer(knn, test[1].context))

    optimal = solve_dataset_to_optimality(test, program, decoder, solver)
    saa_eval = evaluate_policy_against_optimum(
        saa,
        test,
        program,
        decoder,
        solver;
        optimal_results=optimal,
    )
    @test isfinite(saa_eval.metrics.test_policy_value_mean)
    @test isfinite(saa_eval.metrics.test_regret_mean)
end

function smoke_regression_policies(
    problem,
    decoder;
    seed=10,
    target_component=:h_eq_xi,
    postprocess_prediction=identity,
)
    solver = benchmark_solver()
    program = stochastic_program(problem)
    train = generate_benchmark_dataset(
        problem;
        n_contexts=5,
        scenarios_per_context=1,
        seed=seed,
    )

    ls = LeastSquaresPolicy(
        train,
        solver,
        program,
        decoder;
        target_component=target_component,
        postprocess_prediction=postprocess_prediction,
    )
    er = ResidualSampleAverageApproximationPolicy(
        train,
        solver,
        program,
        decoder;
        target_component=target_component,
        postprocess_prediction=postprocess_prediction,
    )

    @test all(isfinite, infer(ls, train[1].context))
    @test all(isfinite, infer(er, train[1].context))
end

@testset "benchmark instances" begin
    @testset "shipment planning" begin
        problem = ShipmentPlanningProblem()
        program = stochastic_program(problem)
        base = base_scenario(problem)
        decoder = ShipmentPlanningParametricDecoder(problem)

        @test size(program.A_ineq) == (5, 5)
        @test size(base.W_eq) == (17, 82)
        @test size(base.W_ineq) == (82, 82)
        @test size(base.T_eq) == (17, 5)
        @test length(base.h_eq) == 17
        @test length(base.q) == 82

        context = generate_benchmark_contexts(
            problem;
            n_contexts=1,
            rng=Random.MersenneTwister(1),
        )[1]
        scenario = generate_benchmark_scenarios(
            problem,
            context;
            n_scenarios=1,
            rng=Random.MersenneTwister(2),
        )[1]
        arrays = ContextualDFL.decode_scenario_collection(decoder, [scenario])
        @test size(arrays[1]) == (17, 82, 1)
        @test size(arrays[3]) == (17, 5, 1)
        @test size(arrays[5]) == (17, 1)
        @test all(>=(1e-6), scenario.h_eq_xi[1:problem.demand_count])

        assert_one_scenario_solve(problem, decoder; seed=3)
        smoke_saa_knn(problem, decoder; seed=4)

        shipment_postprocess = target -> begin
            values = Float64.(target)
            values[1:problem.demand_count] = max.(values[1:problem.demand_count], 1e-6)
            values[(problem.demand_count + 1):end] .= 0.0
            values
        end
        smoke_regression_policies(
            problem,
            decoder;
            seed=5,
            target_component=:h_eq_xi,
            postprocess_prediction=shipment_postprocess,
        )
    end

    @testset "transshipment variants" begin
        for variant in (:q_only, :h_only, :h_and_q)
            problem = TransShipmentExperimentProblem(; variant=variant)
            decoder = transshipment_decoder(problem)
            program = stochastic_program(problem)
            context = generate_benchmark_contexts(
                problem;
                n_contexts=1,
                rng=Random.MersenneTwister(11),
            )[1]
            scenario = generate_benchmark_scenarios(
                problem,
                context;
                n_scenarios=1,
                rng=Random.MersenneTwister(12),
            )[1]
            mean_parameters = ContextualDFL.transshipment_mean_parameters(problem.core_problem)
            arrays = ContextualDFL.decode_scenario_collection(decoder, [scenario])

            @test length(context) == 3
            @test length(scenario.h_eq_xi) == 7
            @test length(scenario.q_xi) == 7
            @test all(>(0.0), scenario.h_eq_xi)
            @test all(>(0.0), scenario.q_xi)
            @test size(arrays[1]) == (35, 77, 1)
            @test size(arrays[3]) == (35, 7, 1)
            @test size(arrays[5]) == (35, 1)
            @test size(arrays[7]) == (77, 1)
            @test size(program.A_ineq) == (7, 7)

            mean_arrays = ContextualDFL.decode_scenario_collection(
                decoder,
                [(; rhs=mean_parameters.rhs, q=mean_parameters.q)],
            )
            q_decoder = TransShipmentComponentVectorDecoder(problem; learned_components=(:q,))
            q_arrays = ContextualDFL.decode_scenario_collection(
                q_decoder,
                mean_parameters.q .+ 1.0;
                nr_scenarios=1,
            )
            @test q_arrays[5] == mean_arrays[5]
            @test count(!iszero, vec(q_arrays[7] - mean_arrays[7])) == 7

            h_decoder = TransShipmentComponentVectorDecoder(problem; learned_components=:h)
            h_arrays = ContextualDFL.decode_scenario_collection(
                h_decoder,
                mean_parameters.rhs .+ 1.0;
                nr_scenarios=1,
            )
            @test count(!iszero, vec(h_arrays[5] - mean_arrays[5])) == 7
            @test h_arrays[7] == mean_arrays[7]

            both_decoder = TransShipmentComponentVectorDecoder(
                problem;
                learned_components=(:h_eq, :q),
            )
            both_arrays = ContextualDFL.decode_scenario_collection(
                both_decoder,
                vcat(mean_parameters.rhs .+ 1.0, mean_parameters.q .+ 1.0);
                nr_scenarios=1,
            )
            @test count(!iszero, vec(both_arrays[5] - mean_arrays[5])) == 7
            @test count(!iszero, vec(both_arrays[7] - mean_arrays[7])) == 7
            @test_throws DimensionMismatch ContextualDFL.decode_scenario_collection(
                q_decoder,
                vcat(mean_parameters.q, 1.0);
                nr_scenarios=1,
            )
            @test_throws ArgumentError TransShipmentComponentVectorDecoder(
                problem;
                learned_components=(:q, :h_eq),
            )

            if variant == :q_only
                scenarios = generate_benchmark_scenarios(
                    problem,
                    context;
                    n_scenarios=3,
                    rng=Random.MersenneTwister(13),
                )
                @test all(s -> s.h_eq_xi == mean_parameters.rhs, scenarios)
                @test length(unique([s.q_xi[1] for s in scenarios])) > 1
            end

            assert_one_scenario_solve(problem, decoder; seed=14)
            smoke_saa_knn(problem, decoder; seed=15, train_contexts=2)

            if variant == :q_only
                smoke_regression_policies(
                    problem,
                    decoder;
                    seed=16,
                    target_component=:q_xi,
                    postprocess_prediction=target -> max.(target, 1e-4),
                )
            elseif variant == :h_only
                smoke_regression_policies(
                    problem,
                    decoder;
                    seed=17,
                    target_component=:h_eq_xi,
                    postprocess_prediction=target -> max.(target, 1e-4),
                )
            end
        end
    end

    @testset "random yield" begin
        problem = RandomYieldProblem(; r=5, a=10, K_support=5)
        decoder = RandomYieldParametricDecoder(problem)
        context = generate_benchmark_contexts(
            problem;
            n_contexts=1,
            rng=Random.MersenneTwister(21),
        )[1]

        probabilities = random_yield_probabilities(problem, context)
        @test length(probabilities) == 5
        @test all(>=(0.0), probabilities)
        @test sum(probabilities) ≈ 1.0

        support = random_yield_support_scenarios(problem, context)
        @test length(support) == 5
        @test support[1].W_eq_xi == base_scenario(problem).W_eq

        scenario = sample_random_yield_scenario(
            problem,
            context;
            rng=Random.MersenneTwister(22),
        )
        @test size(scenario.W_eq_xi) == (5, 20)
        arrays = ContextualDFL.decode_scenario_collection(decoder, [scenario])
        @test size(arrays[1]) == (5, 20, 1)
        @test size(arrays[3]) == (5, 5, 1)
        @test size(arrays[5]) == (5, 1)

        assert_one_scenario_solve(problem, decoder; seed=23)
        smoke_saa_knn(problem, decoder; seed=24)
    end

    @testset "unreliable newsvendor" begin
        problem = UnreliableNewsvendorProblem()
        program = stochastic_program(problem)
        base = base_scenario(problem)
        decoder = UnreliableNewsvendorParametricDecoder(problem)

        @test size(program.A_ineq) == (1, 1)
        @test size(base.W_eq) == (2, 3)
        @test size(base.W_ineq) == (3, 3)
        @test size(base.T_eq) == (2, 1)
        @test length(base.h_eq) == 2
        @test length(base.q) == 3

        context = generate_benchmark_contexts(
            problem;
            n_contexts=1,
            rng=Random.MersenneTwister(31),
        )[1]
        scenario = generate_benchmark_scenarios(
            problem,
            context;
            n_scenarios=1,
            rng=Random.MersenneTwister(32),
        )[1]
        demand, reliability = scenario.h_eq_xi
        arrays = ContextualDFL.decode_scenario_collection(decoder, [scenario])

        @test length(context) == 1
        @test 0.0 <= demand <= problem.demand_upper_bound
        @test 0.0 <= reliability <= 1.0
        @test size(arrays[1]) == (2, 3, 1)
        @test size(arrays[2]) == (3, 3, 1)
        @test size(arrays[3]) == (2, 1, 1)
        @test size(arrays[5]) == (2, 1)
        @test size(arrays[7]) == (3, 1)
        @test arrays[3][:, :, 1] == reshape([0.0, -reliability], 2, 1)
        @test arrays[5][:, 1] == [-demand, 0.0]

        @test_throws ArgumentError unreliable_newsvendor_scenario(problem, -0.1, 0.5)
        @test_throws ArgumentError unreliable_newsvendor_scenario(problem, 0.5, 1.1)

        assert_one_scenario_solve(problem, decoder; seed=33)
        smoke_saa_knn(problem, decoder; seed=34)

        newsvendor_postprocess = target -> begin
            values = Float64.(target)
            values[1] = clamp(values[1], 0.0, problem.demand_upper_bound)
            values[2] = clamp(values[2], 0.0, 1.0)
            values
        end
        smoke_regression_policies(
            problem,
            decoder;
            seed=35,
            target_component=:h_eq_xi,
            postprocess_prediction=newsvendor_postprocess,
        )
    end
end

# END FILE: src/ContextualDFL/ContextualDFLExperiments/test/benchmark_instances/runtests.jl

# BEGIN FILE: src/ContextualDFL/ContextualDFLExperiments/test/runtests.jl
using ContextualDFL
using ContextualDFLExperiments
using Flux
using LinearAlgebra
using Random
using Test

import ChainRulesCore
import ContextualDFLExperiments: infer

struct ConstantPolicy <: Policy
    z::Vector{Float64}
end

infer(policy::ConstantPolicy, context) = policy.z

struct TinyVectorDecoder <: ContextualDFL.VectorDecoder end

function (::TinyVectorDecoder)(vector::AbstractVector)
    return (
        reshape([1.0], 1, 1),
        zeros(0, 1),
        reshape([1.0], 1, 1),
        zeros(0, 1),
        [only(vector)],
        Float64[],
        [3.0],
    )
end

function tiny_program()
    return ContextualDFL.StochasticProgram(
        A_eq=reshape([1.0], 1, 1),
        A_ineq=zeros(0, 1),
        b_eq=[1.0],
        b_ineq=Float64[],
        c=[2.0],
    )
end

function tiny_scenario(h)
    return ContextualDFL.ParametricScenario(;
        W_eq_xi=reshape([1.0], 1, 1),
        W_ineq_xi=zeros(0, 1),
        T_eq_xi=reshape([1.0], 1, 1),
        T_ineq_xi=zeros(0, 1),
        h_eq_xi=[h],
        h_ineq_xi=Float64[],
        q_xi=[3.0],
    )
end

function shortage_program()
    return ContextualDFL.StochasticProgram(
        A_eq=zeros(0, 1),
        A_ineq=reshape([-1.0], 1, 1),
        b_eq=Float64[],
        b_ineq=[0.0],
        c=[1.0],
    )
end

function shortage_scenario(demand)
    return ContextualDFL.ParametricScenario(;
        W_eq_xi=zeros(0, 1),
        W_ineq_xi=reshape([-1.0, -1.0], 2, 1),
        T_eq_xi=zeros(0, 1),
        T_ineq_xi=reshape([-1.0, 0.0], 2, 1),
        h_eq_xi=Float64[],
        h_ineq_xi=[-Float64(demand), 0.0],
        q_xi=[10.0],
    )
end

function shortage_scenario_with_q(demand, q)
    return ContextualDFL.ParametricScenario(;
        W_eq_xi=zeros(0, 1),
        W_ineq_xi=reshape([-1.0, -1.0], 2, 1),
        T_eq_xi=zeros(0, 1),
        T_ineq_xi=reshape([-1.0, 0.0], 2, 1),
        h_eq_xi=Float64[],
        h_ineq_xi=[-Float64(demand), 0.0],
        q_xi=[Float64(q)],
    )
end

function small_resource_allocation_ad_problem()
    data = ResourceAllocationProblemData(
        [1.0 0.8 1.2; 0.7 1.1 0.9],
        [1.0, 1.2],
        [3.0, 4.0, 5.0],
        [1.0, 1.0],
    )
    return data, ResourceAllocationProblem(data)
end

@testset "ContextualDFLExperiments" begin
    contexts = [[1.0], [2.0]]
    scenarios = [[tiny_scenario(5.0)], [tiny_scenario(6.0)]]
    data_set = generate_contextual_data_set(contexts, scenarios)

    @test length(data_set) == 2
    @test data_set[1] isa ContextualDFL.ContextualDataPoint
    @test data_set[1].context == [1.0]
    @test data_set[2].scenario_parameters[1].h_eq_xi == [6.0]

    decision_set = generate_decision_set(ConstantPolicy([1.0]), data_set)
    @test size(decision_set) == (1, 2)
    @test decision_set == reshape([1.0, 1.0], 1, 2)

    solver = ContextualDFL.Solver(ContextualDFL.IpoptSolver(), ContextualDFL.HiGHSSolver())
    program = tiny_program()
    decoder = ContextualDFL.ParametricDecoder()

    optimal_results = solve_dataset_to_optimality(data_set, program, decoder, solver)
    @test [result.objective_values for result in optimal_results] ≈ [[14.0], [17.0]]
    @test [result.evaluation_batches for result in optimal_results] == [1, 1]
    @test [result.objective_value for result in optimal_results] ≈ [14.0, 17.0]

    policy_values = evaluate_policy(decision_set, data_set, program, decoder, solver)
    @test policy_values ≈ [14.0, 17.0]

    rho_optimal_results = solve_dataset_to_optimality(
        data_set,
        program,
        decoder,
        solver;
        rho=0.5,
    )
    rho_policy_values = evaluate_policy(
        decision_set,
        data_set,
        program,
        decoder,
        solver;
        rho=0.5,
    )
    @test [result.objective_value for result in rho_optimal_results] ≈ [18.25, 23.5]
    @test rho_policy_values ≈ [18.25, 23.5]

    value_summary = summarize_values([1.0, 2.0, 3.0]; prefix=:toy)
    @test value_summary.toy_count == 3
    @test value_summary.toy_mean ≈ 2.0
    @test value_summary.toy_median ≈ 2.0
    @test value_summary.toy_std ≈ 1.0
    @test value_summary.toy_min ≈ 1.0
    @test value_summary.toy_max ≈ 3.0
    @test value_summary.toy_p95 ≈ 3.0

    regret_summary = summarize_regret([15.0, 19.0], [14.0, 17.0]; prefix=:test)
    @test regret_summary.test_regret_mean ≈ 1.5
    @test regret_summary.test_relative_regret_mean ≈ ((1.0 / 14.0) + (2.0 / 17.0)) / 2

    comparison = evaluate_policy_against_optimum(
        decision_set,
        data_set,
        program,
        decoder,
        solver;
        optimal_results=optimal_results,
        split_name=:test,
    )
    @test comparison.optimal_results === optimal_results
    @test length(comparison.per_sample) == 2
    @test comparison.metrics.test_sample_count == 2
    @test comparison.metrics.test_evaluation_batches == 1
    @test comparison.metrics.test_policy_value_mean ≈ 15.5
    @test comparison.metrics.test_optimal_value_mean ≈ 15.5
    @test comparison.metrics.test_regret_mean ≈ 0.0
    @test comparison.metrics.test_relative_regret_mean ≈ 0.0
    @test comparison.metrics.test_gap_std_mean ≈ 0.0
    @test comparison.metrics.test_policy_eval_seconds >= 0.0

    split_data_set = generate_contextual_data_set(
        [[1.0]],
        [[tiny_scenario(5.0), tiny_scenario(6.0)]],
    )
    split_decision_set = reshape([1.0], 1, 1)
    split_optimal_results = solve_dataset_to_optimality(
        split_data_set,
        program,
        decoder,
        solver;
        evaluation_batches=2,
    )
    split_result = only(split_optimal_results)
    @test split_result.objective_values ≈ [14.0, 17.0]
    @test split_result.objective_value ≈ 15.5
    @test split_result.evaluation_batches == 2

    split_policy_values = evaluate_policy(
        split_decision_set,
        split_data_set,
        program,
        decoder,
        solver;
        evaluation_batches=2,
    )
    @test split_policy_values ≈ [15.5]

    split_comparison = evaluate_policy_against_optimum(
        split_decision_set,
        split_data_set,
        program,
        decoder,
        solver;
        optimal_results=split_optimal_results,
        split_name=:test,
    )
    @test split_comparison.metrics.test_evaluation_batches == 2
    @test only(split_comparison.per_sample).policy_collection_values ≈ [14.0, 17.0]
    @test only(split_comparison.per_sample).optimal_collection_values ≈ [14.0, 17.0]
    @test only(split_comparison.per_sample).gap_values ≈ [0.0, 0.0]
    @test only(split_comparison.per_sample).gap_std ≈ 0.0
    @test split_comparison.metrics.test_regret_mean ≈ 0.0

    replication_data_set = generate_contextual_data_set(
        [[0.0]],
        [[shortage_scenario(2.0), shortage_scenario(8.0)]],
    )
    replication_decision_set = reshape([8.0], 1, 1)
    replication_optimal_results = solve_dataset_to_optimality(
        replication_data_set,
        shortage_program(),
        ContextualDFL.ParametricDecoder(),
        solver;
        evaluation_batches=2,
    )
    @test only(replication_optimal_results).objective_values ≈ [2.0, 8.0] atol = 1e-6
    @test only(replication_optimal_results).objective_value ≈ 5.0 atol = 1e-6

    replication_comparison = evaluate_policy_against_optimum(
        replication_decision_set,
        replication_data_set,
        shortage_program(),
        ContextualDFL.ParametricDecoder(),
        solver;
        optimal_results=replication_optimal_results,
        split_name=:test,
    )
    @test only(replication_comparison.per_sample).policy_collection_values ≈
        [8.0, 8.0] atol = 1e-6
    @test only(replication_comparison.per_sample).gap_values ≈ [6.0, 0.0] atol = 1e-6
    @test only(replication_comparison.per_sample).regret ≈ 3.0 atol = 1e-6
    @test only(replication_comparison.per_sample).gap_stderr ≈ 3.0 atol = 1e-6

    @test_throws ArgumentError solve_dataset_to_optimality(
        split_data_set,
        program,
        decoder,
        solver;
        evaluation_batches=3,
    )
    @test_throws ArgumentError evaluate_policy_against_optimum(
        decision_set,
        data_set,
        program,
        decoder,
        solver;
        optimal_results=[
            (; objective_values=[1.0, 2.0], objective_value=1.5),
            (; objective_values=[1.0], objective_value=1.0),
        ],
        split_name=:test,
    )

    @test_throws UndefKeywordError evaluate_policy_against_optimum(
        decision_set,
        data_set,
        program,
        decoder,
        solver;
        split_name=:test,
    )

    generator = ContextualDFL.ScenarioGenerator(
        neural_net=context -> [context[1] + 4.0],
        scenario_decoder=TinyVectorDecoder(),
    )
    scenario_policy = ScenarioGenerationPolicy(generator, solver, program)
    rho_scenario_policy = ScenarioGenerationPolicy(generator, solver, program; mu=0.1, rho=0.2)
    @test infer(scenario_policy, [1.0]) ≈ [1.0]
    @test rho_scenario_policy.mu == 0.1
    @test rho_scenario_policy.rho == 0.2
    @test infer(rho_scenario_policy, [1.0]) ≈ [1.0]

    shortage_data_set = generate_contextual_data_set(
        [[0.0], [10.0]],
        [[shortage_scenario(2.0)], [shortage_scenario(8.0)]],
    )
    shortage_decoder = ContextualDFL.ParametricDecoder()
    shortage_policy = SampleAverageApproximationPolicy(
        shortage_data_set,
        solver,
        shortage_program(),
        shortage_decoder,
    )
    @test infer(shortage_policy, [100.0]) ≈ [8.0] atol = 1e-6
    @test generate_decision_set(shortage_policy, shortage_data_set) ≈
        reshape([8.0, 8.0], 1, 2) atol = 1e-6

    direct_shortage_policy = SampleAverageApproximationPolicy(
        [shortage_scenario(2.0)],
        solver,
        shortage_program(),
        shortage_decoder,
    )
    @test infer(direct_shortage_policy, [100.0]) ≈ [2.0] atol = 1e-6

    @test default_knn_k(100) == 32
    knn_policy = KNearestNeighborsPolicy(
        shortage_data_set,
        solver,
        shortage_program(),
        shortage_decoder;
        k=1,
    )
    @test infer(knn_policy, [0.1]) ≈ [2.0] atol = 1e-6
    @test infer(knn_policy, [9.9]) ≈ [8.0] atol = 1e-6
    @test_throws ArgumentError KNearestNeighborsPolicy(
        shortage_data_set,
        solver,
        shortage_program(),
        shortage_decoder;
        k=0,
    )
    @test_throws DimensionMismatch infer(knn_policy, [1.0, 2.0])

    residual_data_set = generate_contextual_data_set(
        [[-1.0], [1.0]],
        [
            [shortage_scenario(3.0), shortage_scenario(5.0)],
            [shortage_scenario(5.0), shortage_scenario(7.0)],
        ],
    )
    least_squares_policy = LeastSquaresPolicy(
        residual_data_set,
        solver,
        shortage_program(),
        shortage_decoder;
        target_component=:h_ineq_xi,
    )
    @test size(least_squares_policy.coefficients) == (2, 2)
    @test least_squares_policy.coefficients ≈ [-1.0 0.0; -5.0 0.0]
    @test infer(least_squares_policy, [2.0]) ≈ [7.0] atol = 1e-6
    @test generate_decision_set(least_squares_policy, residual_data_set) ≈
        reshape([4.0, 6.0], 1, 2) atol = 1e-6

    residual_policy = ResidualSampleAverageApproximationPolicy(
        residual_data_set,
        solver,
        shortage_program(),
        shortage_decoder;
        target_component=:h_ineq_xi,
    )
    @test size(residual_policy.residuals) == (4, 2)
    @test residual_policy.residuals[:, 1] ≈ [1.0, -1.0, 1.0, -1.0]
    @test residual_policy.residuals[:, 2] ≈ zeros(4)
    @test infer(residual_policy, [2.0]) ≈ [8.0] atol = 1e-6

    clipped_policy = LeastSquaresPolicy(
        residual_data_set,
        solver,
        shortage_program(),
        shortage_decoder;
        target_component=:h_ineq_xi,
        postprocess_prediction=target -> max.(target, [-6.0, 0.0]),
    )
    @test infer(clipped_policy, [2.0]) ≈ [6.0] atol = 1e-6

    @test_throws ArgumentError LeastSquaresPolicy(
        residual_data_set,
        solver,
        shortage_program(),
        shortage_decoder;
        target_component=:not_a_component,
    )
    bad_postprocess_policy = LeastSquaresPolicy(
        residual_data_set,
        solver,
        shortage_program(),
        shortage_decoder;
        target_component=:h_ineq_xi,
        postprocess_prediction=target -> target[1:1],
    )
    @test_throws DimensionMismatch infer(bad_postprocess_policy, [2.0])
    @test_throws DimensionMismatch infer(least_squares_policy, [1.0, 2.0])

    varying_structure_data_set = generate_contextual_data_set(
        [[0.0], [1.0]],
        [[shortage_scenario_with_q(2.0, 10.0)], [shortage_scenario_with_q(3.0, 12.0)]],
    )
    @test_throws ArgumentError LeastSquaresPolicy(
        varying_structure_data_set,
        solver,
        shortage_program(),
        shortage_decoder;
        target_component=:h_ineq_xi,
    )

    resource_data = default_resource_allocation_problem_data()
    @test size(resource_data.service_rate_parameters) == (20, 30)
    @test length(resource_data.first_stage_costs) == 20
    @test length(resource_data.second_stage_costs) == 30
    @test length(resource_data.yield_parameters) == 20

    small_resource_data = ResourceAllocationProblemData(
        resource_data.service_rate_parameters[1:2, 1:3],
        resource_data.first_stage_costs[1:2],
        resource_data.second_stage_costs[1:3],
        resource_data.yield_parameters[1:2],
    )
    resource_problem = ResourceAllocationProblem(small_resource_data)
    resource_base_scenario = base_scenario(resource_problem)
    @test size(resource_base_scenario.W_eq) == (5, 14)
    @test size(resource_base_scenario.W_ineq) == (14, 14)
    @test size(resource_base_scenario.T_eq) == (5, 2)
    @test resource_base_scenario.h_eq == zeros(5)
    @test resource_base_scenario.h_ineq == zeros(14)

    context_generator = ResourceAllocationContextDataGenerator(
        rng=Random.MersenneTwister(1),
    )
    resource_context = context_generator()
    @test length(resource_context) == 3
    @test all(>=(0.0), resource_context)

    scenario_generator = ResourceAllocationScenarioDataGenerator(
        resource_problem;
        sigma=0.0,
        p=1.0,
        L=3,
        rng=Random.MersenneTwister(2),
    )
    resource_scenario = scenario_generator(resource_context)
    @test resource_scenario.h_eq_xi isa Vector{Float64}
    @test length(resource_scenario.h_eq_xi) == 3
    @test isempty(resource_scenario.W_eq_xi)
    @test isempty(resource_scenario.h_ineq_xi)

    resource_vector_decoder = ResourceAllocationDemandVectorDecoder(resource_problem)
    _, _, _, _, vector_h_eq, _, _ = resource_vector_decoder(resource_scenario.h_eq_xi)
    @test vector_h_eq[1:2] == zeros(2)
    @test vector_h_eq[3:5] == resource_scenario.h_eq_xi

    predicted_demand = vcat(resource_scenario.h_eq_xi, 2 .* resource_scenario.h_eq_xi)
    @test_throws ArgumentError ChainRulesCore.rrule(
        ContextualDFL.decode_scenario_collection,
        resource_vector_decoder,
        predicted_demand,
    )
    @test_throws DimensionMismatch ChainRulesCore.rrule(
        ContextualDFL.decode_scenario_collection,
        resource_vector_decoder,
        predicted_demand[1:(end - 1)];
        nr_scenarios=2,
    )

    decoded = ContextualDFL.decode_scenario_collection(
        resource_vector_decoder,
        predicted_demand;
        nr_scenarios=2,
    )
    _, vector_pullback = ChainRulesCore.rrule(
        ContextualDFL.decode_scenario_collection,
        resource_vector_decoder,
        predicted_demand;
        nr_scenarios=2,
    )
    dh_eq_cotangent = zeros(size(decoded[5]))
    dh_eq_cotangent[3:5, :] = [1.0 4.0; 2.0 5.0; 3.0 6.0]
    output_cotangent = ntuple(
        index -> index == 5 ? dh_eq_cotangent : zeros(size(decoded[index])),
        length(decoded),
    )
    vector_tangents =
        vector_pullback(ChainRulesCore.Tangent{typeof(decoded)}(output_cotangent...))
    @test vector_tangents[3] == vec(dh_eq_cotangent[3:5, :])
    @test vector_tangents[3][1] == dh_eq_cotangent[3, 1]
    @test vector_tangents[3][4] == dh_eq_cotangent[3, 2]

    zero_vector_tangents = vector_pullback(ChainRulesCore.ZeroTangent())
    @test zero_vector_tangents[3] == zeros(length(predicted_demand))
    @test zero_vector_tangents[3] isa Vector{Float64}

    resource_parametric_decoder = ResourceAllocationDemandParametricDecoder(resource_problem)
    _, _, _, _, h_eq, h_ineq, q = resource_parametric_decoder(resource_scenario)
    @test h_eq[1:2] == zeros(2)
    @test h_eq[3:5] == resource_scenario.h_eq_xi
    @test h_ineq == zeros(14)
    @test length(q) == 14

    second_resource_scenario = ContextualDFL.ParametricScenario(;
        W_eq_xi=Float64[],
        W_ineq_xi=Float64[],
        T_eq_xi=Float64[],
        T_ineq_xi=Float64[],
        h_eq_xi=2 .* resource_scenario.h_eq_xi,
        h_ineq_xi=Float64[],
        q_xi=Float64[],
    )
    resource_scenario_collection = [resource_scenario, second_resource_scenario]
    parametric_decoded = ContextualDFL.decode_scenario_collection(
        resource_parametric_decoder,
        resource_scenario_collection,
    )
    _, parametric_pullback = ChainRulesCore.rrule(
        ContextualDFL.decode_scenario_collection,
        resource_parametric_decoder,
        resource_scenario_collection,
    )
    parametric_dh_eq_cotangent = zeros(size(parametric_decoded[5]))
    parametric_dh_eq_cotangent[3:5, :] = [10.0 40.0; 20.0 50.0; 30.0 60.0]
    parametric_output_cotangent = ntuple(
        index -> index == 5 ? parametric_dh_eq_cotangent : zeros(size(parametric_decoded[index])),
        length(parametric_decoded),
    )
    parametric_tangents = parametric_pullback(parametric_output_cotangent)
    parametric_scenario_tangents = parametric_tangents[3]
    @test parametric_scenario_tangents[1].h_eq_xi == parametric_dh_eq_cotangent[3:5, 1]
    @test parametric_scenario_tangents[2].h_eq_xi == parametric_dh_eq_cotangent[3:5, 2]
    for scenario_tangent in parametric_scenario_tangents
        @test scenario_tangent.W_eq_xi isa ChainRulesCore.NoTangent
        @test scenario_tangent.W_ineq_xi isa ChainRulesCore.NoTangent
        @test scenario_tangent.T_eq_xi isa ChainRulesCore.NoTangent
        @test scenario_tangent.T_ineq_xi isa ChainRulesCore.NoTangent
        @test scenario_tangent.h_ineq_xi isa ChainRulesCore.NoTangent
        @test scenario_tangent.q_xi isa ChainRulesCore.NoTangent
    end

    @testset "ResourceAllocationDemandVectorDecoder real AD" begin
        data, problem = small_resource_allocation_ad_problem()
        decoder = ResourceAllocationDemandVectorDecoder(problem)

        resource_count = size(data.service_rate_parameters, 1)
        demand_count = size(data.service_rate_parameters, 2)
        K = 2

        demand = collect(1.0:(demand_count * K))
        H = reshape(
            collect(1.0:((resource_count + demand_count) * K)),
            resource_count + demand_count,
            K,
        )

        f(d) = begin
            _, _, _, _, h_eq_array, _, _ =
                ContextualDFL.decode_scenario_collection(decoder, d; nr_scenarios=K)
            return sum(h_eq_array .* H)
        end

        g = only(Flux.gradient(f, demand))
        expected = vec(H[(resource_count + 1):(resource_count + demand_count), :])

        @test g ≈ expected atol = 1e-10 rtol = 1e-10
        @test !all(iszero, g)
    end

    @testset "solve rrule real AD matches finite difference wrt h_eq_array" begin
        data, problem = small_resource_allocation_ad_problem()
        solver = ContextualDFL.Solver(ContextualDFL.IpoptSolver(), ContextualDFL.HiGHSSolver())
        decoder = ResourceAllocationDemandVectorDecoder(problem)

        K = 1
        demand = [5.0, 6.0, 7.0]
        W_eq, W_ineq, T_eq, T_ineq, h_eq, h_ineq, q =
            ContextualDFL.decode_scenario_collection(decoder, demand; nr_scenarios=K)

        μ = 0.25
        z_weight = [0.3, -0.7]

        f(h_candidate) = begin
            z, _, _, _, _, _ = ContextualDFL.solve(
                solver,
                stochastic_program(problem),
                W_eq,
                W_ineq,
                T_eq,
                T_ineq,
                h_candidate,
                h_ineq,
                q;
                μ=μ,
                tol=1e-9,
            )
            return dot(z_weight, z)
        end

        g = only(Flux.gradient(f, h_eq))

        direction = zeros(size(h_eq))
        direction[(size(data.service_rate_parameters, 1) + 1):end, :] .= [0.4, -0.2, 0.3]

        ϵ = 1e-4
        fd = (f(h_eq .+ ϵ .* direction) - f(h_eq .- ϵ .* direction)) / (2ϵ)

        @test abs(fd) > 1e-8
        @test sum(g .* direction) ≈ fd atol = 3e-3 rtol = 3e-2
    end

    @testset "predicted demand gradient through decode and solve is nonzero and correct" begin
        _, problem = small_resource_allocation_ad_problem()
        solver = ContextualDFL.Solver(ContextualDFL.IpoptSolver(), ContextualDFL.HiGHSSolver())
        decoder = ResourceAllocationDemandVectorDecoder(problem)

        K = 2
        demand = [5.0, 6.0, 7.0, 4.5, 6.5, 8.0]
        μ = 0.25
        z_weight = [0.3, -0.7]

        f(d) = begin
            W_eq, W_ineq, T_eq, T_ineq, h_eq, h_ineq, q =
                ContextualDFL.decode_scenario_collection(decoder, d; nr_scenarios=K)

            z, _, _, _, _, _ = ContextualDFL.solve(
                solver,
                stochastic_program(problem),
                W_eq,
                W_ineq,
                T_eq,
                T_ineq,
                h_eq,
                h_ineq,
                q;
                μ=μ,
                tol=1e-9,
            )

            return dot(z_weight, z)
        end

        g = only(Flux.gradient(f, demand))

        direction = [0.1, -0.2, 0.3, -0.4, 0.2, 0.1]
        direction ./= norm(direction)

        ϵ = 1e-4
        fd = (f(demand .+ ϵ .* direction) - f(demand .- ϵ .* direction)) / (2ϵ)

        @test abs(fd) > 1e-8
        @test !all(iszero, g)
        @test dot(g, direction) ≈ fd atol = 3e-3 rtol = 3e-2
    end

    @testset "DflScenLoss gradient wrt predicted demand matches finite difference" begin
        _, problem = small_resource_allocation_ad_problem()
        solver = ContextualDFL.Solver(ContextualDFL.IpoptSolver(), ContextualDFL.HiGHSSolver())

        input_decoder = ResourceAllocationDemandVectorDecoder(problem)
        reference_decoder = ResourceAllocationDemandParametricDecoder(problem)

        K = 2
        loss = ContextualDFL.DflScenLoss(
            input_decoder,
            reference_decoder,
            solver,
            stochastic_program(problem);
            nr_scenarios=K,
        )

        predicted_demand = [5.0, 6.0, 7.0, 4.5, 6.5, 8.0]
        reference_scenarios = [
            ContextualDFL.ParametricScenario(; h_eq_xi=[5.5, 6.0, 7.5]),
            ContextualDFL.ParametricScenario(; h_eq_xi=[4.0, 6.8, 8.2]),
        ]
        μ = 0.25

        f(d) = loss(
            d,
            reference_scenarios,
            μ,
            μ;
            tol=1e-9,
        )

        g = only(Flux.gradient(f, predicted_demand))

        direction = [0.1, -0.2, 0.3, -0.4, 0.2, 0.1]
        direction ./= norm(direction)

        ϵ = 1e-4
        fd = (f(predicted_demand .+ ϵ .* direction) -
              f(predicted_demand .- ϵ .* direction)) / (2ϵ)

        @test abs(fd) > 1e-8
        @test !all(iszero, g)
        @test dot(g, direction) ≈ fd atol = 5e-3 rtol = 5e-2
    end

    resource_data_set = generate_contextual_data_set(
        [resource_context],
        [[resource_scenario]],
    )
    resource_results = solve_dataset_to_optimality(
        resource_data_set,
        stochastic_program(resource_problem),
        resource_parametric_decoder,
        solver,
    )
    @test length(resource_results) == 1
    @test only(resource_results).evaluation_batches == 1
    @test length(only(resource_results).objective_values) == 1
    @test isfinite(only(resource_results).objective_value)
end

include("benchmark_instances/runtests.jl")

# END FILE: src/ContextualDFL/ContextualDFLExperiments/test/runtests.jl

# BEGIN FILE: src/ContextualDFL/ContextualDFLTraining/generate_optimal_solutions.jl
#!/usr/bin/env julia

using ArgParse
using ContextualDFLExperiments
using ContextualDFLTraining

function parse_commandline(args=ARGS)
    settings = ArgParseSettings(
        description="Generate precomputed optimal solutions for one ContextualDFLTraining experiment.",
    )

    @add_arg_table! settings begin
        "--experiment"
            help = "Experiment id, module name, or config path, e.g. resource_allocation/experiment_1"
            required = true
        "--splits"
            help = "Comma-separated split names to generate; defaults to every optimality split"
            default = ""
    end

    return parse_args(args, settings)
end

function requested_splits(parsed_args)
    raw = parsed_args["splits"]
    isempty(strip(raw)) && return nothing
    return Set(Symbol(strip(value)) for value in split(raw, ",") if !isempty(strip(value)))
end

function selected_experiment(parsed_args)
    return ContextualDFLTraining.load_experiment(parsed_args["experiment"])
end

function main()
    parsed_args = parse_commandline()
    experiment = selected_experiment(parsed_args)
    config = ContextualDFLTraining.experiment_base_config(experiment)
    objects = ContextualDFLTraining.experiment_call(experiment, :training_objects, config)
    split_filter = requested_splits(parsed_args)
    splits = ContextualDFLTraining.experiment_call(experiment, :optimality_splits, objects, config)
    evaluation_batches = something(
        ContextualDFLTraining.config_value(config, :optimality_evaluation_batches, 1),
        1,
    )
    generated = Symbol[]

    for (split_name, dataset) in splits
        split_name = Symbol(split_name)
        split_filter !== nothing && !(split_name in split_filter) && continue
        isempty(dataset) && continue
        if split_name == :test && ContextualDFLTraining.uses_generated_test_data(experiment)
            println(
                "Skipping test split for experiment=$(experiment.id); generate_test_data.jl owns generated test-data optimal solutions.",
            )
            continue
        end

        println(
            "Computing optimal results for experiment=$(experiment.id), split=$(split_name), samples=$(length(dataset))",
        )
        results = nothing
        solve_seconds = @elapsed begin
            results = ContextualDFLExperiments.solve_dataset_to_optimality(
                dataset,
                objects.program,
                objects.reference_scenario_decoder,
                objects.solver;
                mu=Float64(ContextualDFLTraining.config_value(config, :optimality_mu, 0.0)),
                rho=Float64(ContextualDFLTraining.config_value(config, :optimality_rho, 0.0)),
                evaluation_batches=evaluation_batches,
            )
        end
        path = ContextualDFLTraining.save_optimal_results!(
            experiment,
            split_name,
            results;
            dataset=dataset,
            metadata=(; solve_seconds=solve_seconds, evaluation_batches=evaluation_batches),
        )
        println("Wrote optimal results to $path")

        path = ContextualDFLTraining.optimal_results_path(experiment, split_name)
        payload_results = ContextualDFLTraining.load_optimal_results(
            experiment,
            split_name;
            dataset=dataset,
        )
        length(payload_results) == length(dataset) ||
            error("saved optimal results at $path have the wrong length")
        println("Finished split=$(split_name) in $(round(solve_seconds; digits=3)) seconds")
        push!(generated, split_name)
    end

    isempty(generated) && println("No optimal-result splits were generated.")
    return generated
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

# END FILE: src/ContextualDFL/ContextualDFLTraining/generate_optimal_solutions.jl

# BEGIN FILE: src/ContextualDFL/ContextualDFLTraining/generate_test_data.jl
#!/usr/bin/env julia

using ArgParse
using ContextualDFLExperiments
using ContextualDFLTraining

function parse_commandline(args=ARGS)
    settings = ArgParseSettings(
        description="Generate standalone test data and optimal solutions for one ContextualDFLTraining experiment.",
    )

    @add_arg_table! settings begin
        "--experiment"
            help = "Experiment id, module name, or config path, e.g. resource_allocation/experiment_1"
            required = true
        "--seed"
            help = "Seed used only for generated test data."
            arg_type = Int
            default = ContextualDFLTraining.DEFAULT_TEST_DATA_SEED
        "--data-set-size"
            help = "Number of generated test data rows."
            arg_type = Int
            default = ContextualDFLTraining.DEFAULT_TEST_DATA_SET_SIZE
        "--test-scenarios-per-context"
            help = "Override the number of scenarios generated for each test context. Use 0 for the experiment default."
            arg_type = Int
            default = 0
        "--evaluation-batches"
            help = "Number of scenario collections per context used for benchmark evaluation."
            arg_type = Int
            default = 1
    end

    return parse_args(args, settings)
end

function positive_int(value, name::AbstractString)
    value = Int(value)
    value > 0 || throw(ArgumentError("$name must be positive, got $value."))
    return value
end

function nonnegative_int(value, name::AbstractString)
    value = Int(value)
    value >= 0 || throw(ArgumentError("$name must be nonnegative, got $value."))
    return value
end

function main()
    parsed_args = parse_commandline()
    experiment = ContextualDFLTraining.load_experiment(parsed_args["experiment"])
    seed = Int(parsed_args["seed"])
    data_set_size = positive_int(parsed_args["data-set-size"], "data-set-size")
    test_scenarios_per_context = nonnegative_int(
        parsed_args["test-scenarios-per-context"],
        "test-scenarios-per-context",
    )
    evaluation_batches =
        positive_int(parsed_args["evaluation-batches"], "evaluation-batches")
    overrides = test_scenarios_per_context > 0 ?
        (; test_scenarios_per_context=test_scenarios_per_context) :
        NamedTuple()

    config = ContextualDFLTraining.experiment_test_data_config(
        experiment;
        seed=seed,
        data_set_size=data_set_size,
        overrides...,
    )
    bundle = ContextualDFLTraining.experiment_test_data_bundle(
        experiment;
        seed=seed,
        data_set_size=data_set_size,
        overrides...,
    )
    dataset = bundle.dataset

    println(
        "Generated test data for experiment=$(experiment.id), seed=$seed, rows=$(length(dataset)), scenarios_per_context=$(length(first(dataset).scenario_parameters))",
    )
    test_data_path = ContextualDFLTraining.save_test_data!(
        experiment,
        seed,
        dataset;
        data_set_size=data_set_size,
    )
    println("Wrote test data to $test_data_path")

    results = nothing
    solve_seconds = @elapsed begin
        results = ContextualDFLExperiments.solve_dataset_to_optimality(
            dataset,
            bundle.program,
            bundle.reference_scenario_decoder,
            bundle.solver;
            mu=Float64(ContextualDFLTraining.config_value(config, :optimality_mu, 0.0)),
            rho=Float64(ContextualDFLTraining.config_value(config, :optimality_rho, 0.0)),
            evaluation_batches=evaluation_batches,
        )
    end

    optimal_results_path = ContextualDFLTraining.save_test_optimal_results!(
        experiment,
        seed,
        results;
        dataset=dataset,
        data_set_size=data_set_size,
        metadata=(;
            solve_seconds=solve_seconds,
            evaluation_batches=evaluation_batches,
        ),
    )
    println("Wrote optimal solutions to $optimal_results_path")

    loaded_artifact = ContextualDFLTraining.load_test_data_artifact(experiment)
    loaded_seed_index = findfirst(==(seed), loaded_artifact.metadata.seeds)
    loaded_seed_index !== nothing ||
        error("saved test data at $test_data_path were not found by the default loader")
    loaded_artifact.metadata.data_set_sizes[loaded_seed_index] == data_set_size ||
        error("saved test data at $test_data_path have the wrong length")
    loaded_results = ContextualDFLTraining.load_optimal_results(experiment, :test)
    length(loaded_results) == loaded_artifact.metadata.data_set_size ||
        error("saved optimal solutions at $optimal_results_path have the wrong length")
    println("Finished in $(round(solve_seconds; digits=3)) seconds")

    return (; test_data_path=test_data_path, optimal_results_path=optimal_results_path)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

# END FILE: src/ContextualDFL/ContextualDFLTraining/generate_test_data.jl

# BEGIN FILE: src/ContextualDFL/ContextualDFLTraining/gridsearch.jl
#!/usr/bin/env julia

using Dates
using Distributed
using ArgParse
import MLFlowClient
using Random
using SHA
using Sockets
using Statistics

include(joinpath(@__DIR__, "src", "run_defaults.jl"))
include(joinpath(@__DIR__, "src", "grid_config.jl"))
include(joinpath(@__DIR__, "src", "csv_results.jl"))
include(joinpath(@__DIR__, "src", "experiments", "ExperimentAPI.jl"))
include(joinpath(@__DIR__, "src", "grid_file_config.jl"))

const DEFAULT_REMOTE_PROJECT =
    "/home/rwl/ProblemBasedScenarioGeneration/src/ContextualDFL/ContextualDFLTraining"
const DEFAULT_REMOTE_JULIA = "/home/rwl/.juliaup/bin/julia"
const MLFLOW_RETRY_ATTEMPTS = 8
const MLFLOW_RETRY_INITIAL_DELAY_SECONDS = 1.0
const MLFLOW_RETRY_BACKOFF = 1.5
const GRID_CANDIDATE_START_STAGGER_SECONDS = 0.25
const GRID_TRAINING_DATA_SEED_MAX = typemax(Int32) - 1

function _contextualdfltraining_remote_eval(config)
    started_at = unix_milliseconds()
    try
        return Main.ContextualDFLTraining.train_and_evaluate(config)
    catch error
        return (;
            status="worker_error",
            run_id=(config isa NamedTuple && :run_id in keys(config)) ? config.run_id : "",
            config=config,
            worker=(;
                worker_id=Distributed.myid(),
                hostname=Sockets.gethostname(),
                pid=getpid(),
                julia_version=string(VERSION),
            ),
            final_metrics=NamedTuple(),
            epoch_history=Dict{Symbol,Any}[],
            error=sprint(showerror, error, catch_backtrace()),
            started_at=started_at,
            finished_at=unix_milliseconds(),
            elapsed_seconds=0.0,
        )
    end
end

function env_worker_count(name, default)
    value = lowercase(strip(get(ENV, name, string(default))))
    value == "auto" && return :auto

    parsed = tryparse(Int, value)
    parsed === nothing && error("ENV[$name] must be an integer or auto, got: $value")
    return parsed
end

function remote_worker_specs()
    return [
        ("rwl@gcp-4c-1", env_worker_count("GCP_4C_1_WORKERS", :auto)),
        ("rwl@gcp-4c-2", env_worker_count("GCP_4C_2_WORKERS", :auto)),
        ("rwl@gcp-4c-3", env_worker_count("GCP_4C_3_WORKERS", :auto)),
        ("rwl@gcp-8c-1", env_worker_count("GCP_8C_1_WORKERS", :auto)),
        ("rwl@gcp-8c-2", env_worker_count("GCP_8C_2_WORKERS", :auto)),
        ("rwl@gcp-8c-3", env_worker_count("GCP_8C_3_WORKERS", :auto)),
        ("rwl@gcp-16c-4", env_worker_count("GCP_16C_4_WORKERS", :auto)),
    ]
end

function env_flag(name, default=false)
    value = lowercase(get(ENV, name, default ? "1" : "0"))
    return value in ("1", "true", "yes", "y")
end

function deterministic_mlflow_experiment_id(experiment_id)
    digest = bytes2hex(sha1(string(experiment_id)))
    value = parse(UInt64, digest[1:15]; base=16)
    return string(1 + value % UInt64(9_000_000_000))
end

function deterministic_mlflow_experiment_name(experiment)
    return "ContextualDFLTraining/" * string(experiment.name)
end

function grid_mlflow_settings(experiment)
    deterministic_id = deterministic_mlflow_experiment_id(experiment.id)
    return (;
        enabled=true,
        experiment_id=deterministic_id,
        deterministic_experiment_id=deterministic_id,
        experiment_name=deterministic_mlflow_experiment_name(experiment),
        contextualdfl_experiment_id=experiment.id,
        contextualdfl_experiment_name=experiment.name,
        tracking_uri=get(ENV, "MLFLOW_TRACKING_URI", ""),
        upload_model_artifact=env_flag("MLFLOW_UPLOAD_MODEL_ARTIFACTS", false),
    )
end

function validate_mlflow_settings(settings)
    if settings.enabled && isempty(settings.experiment_id)
        error("MLFLOW_ENABLED=true requires MLFLOW_EXPERIMENT_ID.")
    end
    return nothing
end

function mlflow_client(settings)
    return isempty(string(settings.tracking_uri)) ?
        MLFlowClient.MLFlow(; headers=mlflow_http_headers()) :
        MLFlowClient.MLFlow(string(settings.tracking_uri); headers=mlflow_http_headers())
end

function missing_mlflow_experiment_error(error)
    message = lowercase(sprint(showerror, error))
    return occursin("resource_does_not_exist", message) ||
           occursin("does not exist", message) ||
           occursin("not found", message)
end

function ensure_mlflow_grid_experiment(settings)
    settings.enabled || return settings

    mlf = mlflow_client(settings)
    experiment_id = with_mlflow_retry("ensure MLflow experiment $(settings.experiment_name)") do
        try
            experiment = MLFlowClient.getexperimentbyname(
                mlf,
                string(settings.experiment_name),
            )
            string(experiment.experiment_id)
        catch error
            missing_mlflow_experiment_error(error) || rethrow()
            MLFlowClient.createexperiment(mlf, string(settings.experiment_name))
        end
    end

    experiment_tags = (
        "contextualdfl.experiment_id" => string(settings.contextualdfl_experiment_id),
        "contextualdfl.experiment_name" => string(settings.contextualdfl_experiment_name),
        "contextualdfl.deterministic_experiment_id" => string(settings.deterministic_experiment_id),
    )
    for (key, value) in experiment_tags
        with_mlflow_retry("set MLflow experiment tag $key") do
            MLFlowClient.setexperimenttag(mlf, string(experiment_id), key, value)
        end
    end

    return merge(settings, (; experiment_id=string(experiment_id)))
end

function ensure_clean_worker_start!()
    nprocs() == 1 ||
        error("Refusing to run with pre-existing workers. Start Julia without -p or --machine-file.")
end

function sync_code!()
    if env_flag("SKIP_SYNC", false)
        println("Skipping code sync because SKIP_SYNC is set.")
        return nothing
    end

    sync_script = joinpath(homedir(), "sync-julia-code.sh")
    isfile(sync_script) || error("sync script not found: $sync_script")
    println("Syncing code to remote machines with $sync_script")
    run(Cmd(`$sync_script`; dir=homedir()))
    return nothing
end

function add_remote_workers!()
    remote_project = get(ENV, "REMOTE_CONTEXTUAL_DFL_TRAINING_PROJECT", DEFAULT_REMOTE_PROJECT)
    remote_julia = get(ENV, "REMOTE_JULIA", DEFAULT_REMOTE_JULIA)

    for (host, count) in remote_worker_specs()
        count isa Integer && count <= 0 && continue
        println("Adding $count worker(s) on $host")
        addprocs(
            [(host, count)];
            exename=remote_julia,
            exeflags="--project=$(remote_project)",
            dir=remote_project,
            tunnel=true,
        )
    end

    remote_worker_ids = setdiff(workers(), [1])
    isempty(remote_worker_ids) && error("No remote workers were added.")
    return remote_worker_ids
end

function load_worker_stdlibs!()
    for process_id in procs()
        remotecall_fetch(process_id) do
            Core.eval(Main, :(using Dates))
            Core.eval(Main, :(using Distributed))
            Core.eval(Main, :(using Pkg))
            Core.eval(Main, :(using Sockets))
            return nothing
        end
    end
end

function assert_remote_only_workers!(remote_worker_ids)
    local_hostname = Sockets.gethostname()
    worker_hosts = Dict(
        worker => remotecall_fetch(() -> Sockets.gethostname(), worker) for
        worker in remote_worker_ids
    )
    local_workers = [
        worker for (worker, hostname) in worker_hosts if hostname == local_hostname
    ]

    isempty(local_workers) ||
        error("Refusing to run training on local worker(s): $(local_workers)")

    host_summary = join(
        ["$(worker)=>$(worker_hosts[worker])" for worker in sort(remote_worker_ids)],
        ", ",
    )
    println("Workers online: ", length(remote_worker_ids), " [", host_summary, "]")
    return worker_hosts
end

function load_training_project_on_workers!(remote_worker_ids)
    println("Instantiating and loading ContextualDFLTraining on remote workers")
    for worker in remote_worker_ids
        metadata = remotecall_fetch(worker) do
            Pkg.instantiate()
            Core.eval(Main, :(using ContextualDFLTraining))
            return (;
                worker_id=Distributed.myid(),
                hostname=Sockets.gethostname(),
                pid=getpid(),
            )
        end
        println("Loaded worker $(metadata.worker_id) on $(metadata.hostname), pid $(metadata.pid)")
    end
end

function define_remote_eval!()
    definition = quote
        function _contextualdfltraining_remote_eval(config)
            started_at = round(Int64, time() * 1000)
            try
                return Main.ContextualDFLTraining.train_and_evaluate(config)
            catch error
                return (;
                    status="worker_error",
                    run_id=(config isa NamedTuple && :run_id in keys(config)) ? config.run_id : "",
                    config=config,
                    worker=(;
                        worker_id=Distributed.myid(),
                        hostname=Sockets.gethostname(),
                        pid=getpid(),
                        julia_version=string(VERSION),
                    ),
                    final_metrics=NamedTuple(),
                    epoch_history=Dict{Symbol,Any}[],
                    error=sprint(showerror, error, catch_backtrace()),
                    started_at=started_at,
                    finished_at=round(Int64, time() * 1000),
                    elapsed_seconds=0.0,
                )
            end
        end
    end

    for worker in workers()
        remotecall_fetch(Core.eval, worker, Main, definition)
    end
end

function coordinator_error_result(config, worker, worker_hosts, status, error, backtrace, elapsed_seconds)
    return (;
        status=status,
        run_id=(config isa NamedTuple && :run_id in keys(config)) ? config.run_id : "",
        config=config,
        worker=(;
            worker_id=worker,
            hostname=get(worker_hosts, worker, ""),
            pid=missing,
            julia_version="",
        ),
        final_metrics=NamedTuple(),
        epoch_history=Dict{Symbol,Any}[],
        error=sprint(showerror, error, backtrace),
        started_at=unix_milliseconds(),
        finished_at=unix_milliseconds(),
        elapsed_seconds=elapsed_seconds,
    )
end

function transport_failure(error)
    return error isa Distributed.ProcessExitedException ||
        error isa EOFError ||
        error isa Base.IOError
end

function run_grid_on_remote_workers(remote_worker_ids, configs, worker_hosts)
    results = Vector{Any}(undef, length(configs))
    pending = Tuple{Int,Any}[(index, config) for (index, config) in enumerate(configs)]
    pending_lock = ReentrantLock()

    function next_pending!()
        lock(pending_lock)
        try
            isempty(pending) && return nothing
            return popfirst!(pending)
        finally
            unlock(pending_lock)
        end
    end

    tasks = [
        @async begin
            while true
                item = next_pending!()
                item === nothing && break

                index, config = item
                started = time()
                try
                    if length(remote_worker_ids) > 1
                        sleep(
                            GRID_CANDIDATE_START_STAGGER_SECONDS *
                            mod(index - 1, length(remote_worker_ids)),
                        )
                    end
                    results[index] = remotecall_fetch(
                        _contextualdfltraining_remote_eval,
                        worker,
                        config,
                    )
                catch error
                    elapsed_seconds = time() - started
                    if transport_failure(error)
                        println(
                            "Worker $worker exited while running $(config.run_id); recording worker_lost and continuing.",
                        )
                        mark_mlflow_run_failed(config, "worker_lost")
                        results[index] = coordinator_error_result(
                            config,
                            worker,
                            worker_hosts,
                            "worker_lost",
                            error,
                            catch_backtrace(),
                            elapsed_seconds,
                        )
                        break
                    end

                    results[index] = coordinator_error_result(
                        config,
                        worker,
                        worker_hosts,
                        "coordinator_error",
                        error,
                        catch_backtrace(),
                        elapsed_seconds,
                    )
                end
            end
        end for worker in remote_worker_ids
    ]

    foreach(wait, tasks)

    for (index, config) in enumerate(configs)
        if !isassigned(results, index)
            results[index] = (;
                status="not_started",
                run_id=config.run_id,
                config=config,
                worker=NamedTuple(),
                final_metrics=NamedTuple(),
                epoch_history=Dict{Symbol,Any}[],
                error="No remote worker remained available for this configuration.",
                started_at=unix_milliseconds(),
                finished_at=unix_milliseconds(),
                elapsed_seconds=0.0,
            )
        end
    end

    return results
end

function mark_mlflow_run_failed(config, reason)
    config isa NamedTuple || return nothing
    (:mlflow_enabled in keys(config) && config.mlflow_enabled) || return nothing

    try
        uri = string(getproperty(config, :mlflow_tracking_uri))
        experiment_id = string(getproperty(config, :mlflow_experiment_id))
        run_name = string(getproperty(config, :mlflow_run_name))
        mlf = isempty(uri) ?
            MLFlowClient.MLFlow(; headers=mlflow_http_headers()) :
            MLFlowClient.MLFlow(uri; headers=mlflow_http_headers())
        filter = "tags.candidate_name = \"$(mlflow_filter_escape(run_name))\" and attributes.status = \"RUNNING\""
        runs, _ = with_mlflow_retry("search failed candidate runs") do
            MLFlowClient.searchruns(
                mlf;
                experiment_ids=[experiment_id],
                filter=filter,
                max_results=100,
            )
        end

        for run in runs
            with_mlflow_retry("set failed candidate tag") do
                MLFlowClient.setruntag(
                    mlf,
                    run,
                    "ContextualDFLTraining.coordinator_status",
                    reason,
                )
            end
            with_mlflow_retry("update failed candidate run") do
                MLFlowClient.updaterun(
                    mlf,
                    run;
                    status=MLFlowClient.RunStatus.FAILED,
                    end_time=unix_milliseconds(),
                )
            end
        end
    catch error
        println("Could not mark MLflow run failed for $(config.run_id): ", sprint(showerror, error))
    end

    return nothing
end

function create_mlflow_grid_parent_run(
    settings,
    grid_id,
    timestamp,
    configs,
    worker_hosts,
    grid_spec::GridSearchSpec,
    ;
    repeat_training_data_seeds=nothing,
)
    settings.enabled || return nothing

    mlf = mlflow_client(settings)
    tags = Dict(
        "gridsearch_id" => grid_id,
        "gridsearch_timestamp" => timestamp,
        "gridsearch_role" => "parent",
        "training_project" => "ContextualDFLTraining",
        "mlflow.experiment.name" => string(settings.experiment_name),
        "contextualdfl.experiment_id" => string(settings.contextualdfl_experiment_id),
        "contextualdfl.experiment_name" => string(settings.contextualdfl_experiment_name),
        "contextualdfl.deterministic_experiment_id" => string(settings.deterministic_experiment_id),
        "mlflow.source.name" => "ContextualDFLTraining/gridsearch.jl",
        "mlflow.source.type" => "LOCAL",
    )
    git_commit = git_commit_or_empty()
    isempty(git_commit) || (tags["mlflow.source.git.commit"] = git_commit)

    parent_params = grid_parent_params(
        grid_id,
        timestamp,
        configs,
        worker_hosts,
        grid_spec;
        repeat_training_data_seeds=repeat_training_data_seeds,
    )

    run = with_mlflow_retry("create grid parent run") do
        MLFlowClient.createrun(
            mlf,
            string(settings.experiment_id);
            run_name=grid_id,
            start_time=unix_milliseconds(),
            tags=tags,
        )
    end

    for (key, value) in parent_params
        with_mlflow_retry("log grid parent param $key") do
            MLFlowClient.logparam(mlf, run, key, value)
        end
    end

    upload_mlflow_grid_config_artifacts!(mlf, grid_spec, configs)

    return (; client=mlf, run=run)
end

function upload_mlflow_grid_config_artifacts!(mlf, grid_spec::GridSearchSpec, configs)
    source_extension = grid_spec.format == :yaml ? ".yaml" : ".json"
    source_data = read(grid_spec.path)
    resolved_json = resolved_grid_json(configs)
    resolved_data = Vector{UInt8}(codeunits(resolved_json))
    digest_data = Vector{UInt8}(codeunits(grid_config_digest(configs) * "\n"))

    with_mlflow_retry("upload grid config source artifact") do
        MLFlowClient.uploadartifact(
            mlf,
            "grid_config/source" * source_extension,
            source_data,
        )
    end
    with_mlflow_retry("upload resolved grid config artifact") do
        MLFlowClient.uploadartifact(mlf, "grid_config/resolved.json", resolved_data)
    end
    with_mlflow_retry("upload grid config digest artifact") do
        MLFlowClient.uploadartifact(mlf, "grid_config/digest.txt", digest_data)
    end

    return nothing
end

function close_mlflow_grid_parent_run(parent, config_parent_results; child_results=Any[])
    parent === nothing && return nothing

    success = all(result -> getproperty(result, :status) == "ok", config_parent_results)
    mark_failed_mlflow_candidates!(child_results)
    log_grid_aggregate_metrics!(parent.client, parent.run, config_parent_results)
    with_mlflow_retry("update grid parent run") do
        MLFlowClient.updaterun(
            parent.client,
            parent.run;
            status=success ? MLFlowClient.RunStatus.FINISHED : MLFlowClient.RunStatus.FAILED,
            end_time=unix_milliseconds(),
        )
    end
    return nothing
end

function create_mlflow_config_parent_runs(settings, config_parent_configs)
    settings.enabled || return fill(nothing, length(config_parent_configs))
    return [
        create_mlflow_config_parent_run(settings, config) for
        config in config_parent_configs
    ]
end

function create_mlflow_config_parent_run(settings, config)
    mlf = mlflow_client(settings)
    tags = Dict(
        "gridsearch_id" => string(config.gridsearch_id),
        "gridsearch_timestamp" => string(config.gridsearch_timestamp),
        "gridsearch_role" => "config_parent",
        "training_project" => "ContextualDFLTraining",
        "run_id" => string(config.run_id),
        "base_run_id" => string(config.base_run_id),
        "candidate_index" => string(config.candidate_index),
        "candidate_name" => string(config.candidate_name),
        "repeat_count" => string(config.repeat_count),
        "gridsearch_parent_run_id" => string(config.mlflow_parent_run_id),
        "mlflow.parentRunId" => string(config.mlflow_parent_run_id),
        "mlflow.experiment.name" => string(settings.experiment_name),
        "contextualdfl.experiment_id" => string(settings.contextualdfl_experiment_id),
        "contextualdfl.experiment_name" => string(settings.contextualdfl_experiment_name),
        "contextualdfl.deterministic_experiment_id" => string(settings.deterministic_experiment_id),
        "mlflow.source.name" => "ContextualDFLTraining/gridsearch.jl",
        "mlflow.source.type" => "LOCAL",
    )
    git_commit = git_commit_or_empty()
    isempty(git_commit) || (tags["mlflow.source.git.commit"] = git_commit)

    run = with_mlflow_retry("create config parent run") do
        MLFlowClient.createrun(
            mlf,
            string(settings.experiment_id);
            run_name=string(config.candidate_name),
            start_time=unix_milliseconds(),
            tags=tags,
        )
    end

    log_config_parent_params!(mlf, run, config)
    return (; client=mlf, run=run, config=config)
end

function log_config_parent_params!(mlf, run, config)
    params = Dict{String,String}(
        "gridsearch_id" => string(config.gridsearch_id),
        "candidate_index" => string(config.candidate_index),
        "base_run_id" => string(config.base_run_id),
        "repeat_count" => string(config.repeat_count),
    )

    for key in keys(config)
        value = getproperty(config, key)
        mlflow_scalar_value(value) || continue
        params["config_" * string(key)] = string(value)
    end

    for key in sort!(collect(keys(params)))
        with_mlflow_retry("log config parent param $key") do
            MLFlowClient.logparam(mlf, run, key, params[key])
        end
    end

    return nothing
end

function close_mlflow_config_parent_runs(config_parent_runs, config_parent_results)
    for (parent, result) in zip(config_parent_runs, config_parent_results)
        close_mlflow_config_parent_run(parent, result)
    end
    return nothing
end

function close_mlflow_config_parent_run(parent, result)
    parent === nothing && return nothing

    success = getproperty(result, :status) == "ok"
    log_config_parent_aggregate_metrics!(parent.client, parent.run, result)
    with_mlflow_retry("update config parent run") do
        MLFlowClient.updaterun(
            parent.client,
            parent.run;
            status=success ? MLFlowClient.RunStatus.FINISHED : MLFlowClient.RunStatus.FAILED,
            end_time=unix_milliseconds(),
        )
    end
    return nothing
end

function mark_failed_mlflow_candidates!(results)
    for result in results
        getproperty(result, :status) == "ok" && continue
        config = getproperty(result, :config)
        mark_mlflow_run_failed(config, string(getproperty(result, :status)))
    end
    return nothing
end

function fail_mlflow_grid_parent_run(parent)
    parent === nothing && return nothing
    try
        with_mlflow_retry("fail grid parent run") do
            MLFlowClient.updaterun(
                parent.client,
                parent.run;
                status=MLFlowClient.RunStatus.FAILED,
                end_time=unix_milliseconds(),
            )
        end
    catch error
        println("Could not mark parent MLflow run failed: ", sprint(showerror, error))
    end
    return nothing
end

function fail_mlflow_config_parent_runs(parents)
    for parent in parents
        parent === nothing && continue
        try
            with_mlflow_retry("fail config parent run") do
                MLFlowClient.updaterun(
                    parent.client,
                    parent.run;
                    status=MLFlowClient.RunStatus.FAILED,
                    end_time=unix_milliseconds(),
                )
            end
        catch error
            println("Could not mark config parent MLflow run failed: ", sprint(showerror, error))
        end
    end
    return nothing
end

function grid_parent_params(
    grid_id,
    timestamp,
    configs,
    worker_hosts,
    grid_spec::GridSearchSpec;
    repeat_training_data_seeds=nothing,
)
    params = Dict{String,String}(
        "gridsearch_id" => grid_id,
        "gridsearch_timestamp" => timestamp,
        "grid_candidate_count" => string(length(configs)),
        "grid_repeat_run_count" => string(sum(grid_repeat_count(config) for config in configs; init=0)),
        "grid_config_name" => grid_spec.name,
        "grid_config_path" => grid_spec.path,
        "grid_config_digest" => grid_config_digest(configs),
        "grid_config_version" => string(grid_spec.version),
        "grid_config_format" => string(grid_spec.format),
        "grid_worker_count" => string(length(worker_hosts)),
        "grid_worker_hosts" => join(sort!(unique(collect(values(worker_hosts)))), ","),
    )

    if repeat_training_data_seeds !== nothing
        repeat_seeds = normalize_repeat_training_data_seeds(
            repeat_training_data_seeds,
            grid_repeat_seed_count(configs),
        )
        if !isempty(repeat_seeds)
            params["grid_repeat_training_data_seed_count"] = string(length(repeat_seeds))
            params["grid_repeat_training_data_seed_sequence"] =
                repeat_training_data_seed_sequence(repeat_seeds)
            for (index, seed) in enumerate(repeat_seeds)
                params["grid_" * repeat_tag(index) * "_training_data_seed"] = string(seed)
            end
        end
    end

    if !isempty(configs)
        config = first(configs)
        if config isa NamedTuple && hasproperty(config, :experiment_id)
            params["experiment_id"] = string(config.experiment_id)
        end
        if config isa NamedTuple && hasproperty(config, :experiment_name)
            params["experiment_name"] = string(config.experiment_name)
        end
        if config isa NamedTuple && hasproperty(config, :mlflow_deterministic_experiment_id)
            params["mlflow_deterministic_experiment_id"] =
                string(config.mlflow_deterministic_experiment_id)
        end
    end

    for (key, value) in grid_constant_config_values(configs)
        params["grid_constant_" * string(key)] = string(value)
    end

    variable_keys = grid_variable_config_keys(configs)
    params["grid_variable_keys"] = join(string.(variable_keys), ",")
    for key in variable_keys
        values = sort!(unique([string(getproperty(config, key)) for config in configs]))
        params["grid_variable_" * string(key) * "_count"] = string(length(values))
        length(values) <= 20 && (params["grid_variable_" * string(key) * "_values"] = join(values, ","))
    end

    return params
end

function grid_constant_config_values(configs)
    isempty(configs) && return Pair{Symbol,Any}[]
    constants = Pair{Symbol,Any}[]
    first_config = first(configs)

    for key in keys(first_config)
        value = getproperty(first_config, key)
        mlflow_scalar_value(value) || continue
        all(config -> hasproperty(config, key) && getproperty(config, key) == value, configs) ||
            continue
        push!(constants, key => value)
    end

    return constants
end

function grid_variable_config_keys(configs)
    isempty(configs) && return Symbol[]
    variable_keys = Symbol[]
    first_config = first(configs)

    for key in keys(first_config)
        value = getproperty(first_config, key)
        mlflow_scalar_value(value) || continue
        all(config -> hasproperty(config, key) && getproperty(config, key) == value, configs) &&
            continue
        push!(variable_keys, key)
    end

    return sort!(variable_keys; by=String)
end

function log_grid_aggregate_metrics!(mlf, run, results)
    log_aggregate_metrics!(mlf, run, "grid", aggregate_metric_summaries(results))
    return nothing
end

function log_config_parent_aggregate_metrics!(mlf, run, result)
    summaries = getproperty(result, :aggregate_metrics)
    summaries isa AbstractDict || return nothing
    log_aggregate_metrics!(mlf, run, "config", summaries)
    return nothing
end

function log_aggregate_metrics!(mlf, run, prefix_root::AbstractString, summaries)
    for key in sort!(collect(keys(summaries)); by=String)
        summary = summaries[key]
        prefix = prefix_root * "_" * string(key)
        timestamp = unix_milliseconds()
        for field in (:count, :mean, :median, :min, :max, :std, :stderr)
            value = getproperty(summary, field)
            with_mlflow_retry("log aggregate metric $(prefix)_$(field)") do
                MLFlowClient.logmetric(
                    mlf,
                    run,
                    prefix * "_" * string(field),
                    Float64(value);
                    timestamp=timestamp,
                    step=0,
                )
            end
        end
    end

    return nothing
end

function aggregate_metric_summaries(results)
    metric_keys = Set{Symbol}()
    for result in results
        getproperty(result, :status) == "ok" || continue
        metrics = getproperty(result, :final_metrics)
        metrics isa NamedTuple || continue
        for key in keys(metrics)
            value = getproperty(metrics, key)
            mlflow_numeric_metric(value) && push!(metric_keys, key)
        end
    end

    summaries = Dict{Symbol,NamedTuple}()
    for key in sort!(collect(metric_keys); by=String)
        values = Float64[
            Float64(getproperty(getproperty(result, :final_metrics), key)) for result in results if
            getproperty(result, :status) == "ok" &&
            getproperty(result, :final_metrics) isa NamedTuple &&
            hasproperty(getproperty(result, :final_metrics), key) &&
            mlflow_numeric_metric(getproperty(getproperty(result, :final_metrics), key))
        ]
        isempty(values) && continue
        std_value = length(values) > 1 ? std(values) : 0.0
        summaries[key] = (;
            count=Float64(length(values)),
            mean=mean(values),
            median=median(values),
            min=minimum(values),
            max=maximum(values),
            std=std_value,
            stderr=std_value / sqrt(length(values)),
        )
    end

    return summaries
end

function aggregate_mean_metrics(summaries::AbstractDict)
    keys_sorted = sort!(collect(keys(summaries)); by=String)
    return NamedTuple{Tuple(keys_sorted)}(
        Tuple(getproperty(summaries[key], :mean) for key in keys_sorted),
    )
end

function with_mlflow_retry(callback, operation)
    delay = MLFLOW_RETRY_INITIAL_DELAY_SECONDS
    for attempt in 1:MLFLOW_RETRY_ATTEMPTS
        try
            return callback()
        catch error
            attempt == MLFLOW_RETRY_ATTEMPTS && rethrow()
            @warn "MLflow $operation failed; retrying" attempt error=sprint(showerror, error)
            sleep(delay)
            delay *= MLFLOW_RETRY_BACKOFF
        end
    end
end

function mlflow_parent_run_id(parent)
    parent === nothing && return ""
    try
        return string(parent.run.info.run_id)
    catch
        return ""
    end
end

mlflow_scalar_value(value) =
    value isa Number ||
    value isa Bool ||
    value isa Symbol ||
    value isa AbstractString

function mlflow_numeric_metric(value)
    value isa Bool && return false
    value isa Number || return false
    float_value = try
        Float64(value)
    catch
        return false
    end
    return isfinite(float_value)
end

function git_commit_or_empty()
    try
        return strip(read(pipeline(`git rev-parse HEAD`; stderr=devnull), String))
    catch
        return ""
    end
end

function mlflow_http_headers()
    return Dict("Connection" => "close")
end

function mlflow_filter_escape(value)
    return replace(string(value), "\\" => "\\\\", "\"" => "\\\"")
end

function experiment_problem_identity_for_grid(experiment)
    experiment_has_function(experiment, :problem_identity_config) || return NamedTuple()
    identity = experiment_call(experiment, :problem_identity_config)
    identity isa NamedTuple ||
        throw(ArgumentError("experiment problem_identity_config() must return a NamedTuple."))
    return identity
end

function validate_grid_does_not_override_problem_identity(experiment, grid_spec::GridSearchSpec)
    identity = experiment_problem_identity_for_grid(experiment)
    isempty(keys(identity)) && return nothing

    identity_keys = Set(Symbol.(keys(identity)))
    provided_keys = Set{Symbol}()
    union!(provided_keys, keys(grid_spec.base))
    union!(provided_keys, keys(grid_spec.fixed))
    union!(provided_keys, keys(grid_spec.grid))
    union!(provided_keys, keys(grid_spec.schedules))
    for schedule_candidate in grid_spec.schedule_grid
        union!(provided_keys, keys(schedule_candidate))
    end

    overridden = sort!(collect(intersect(identity_keys, provided_keys)); by=string)
    isempty(overridden) && return nothing

    throw(
        ArgumentError(
            "grid config $(grid_spec.path) may not set problem-identity key(s) owned by experiment $(experiment.id): $(join(string.(overridden), ", "))",
        ),
    )
end

function selected_grid(experiment, grid_spec::GridSearchSpec)
    validate_grid_does_not_override_problem_identity(experiment, grid_spec)
    return resolve_grid_configs(grid_spec; base_config=experiment_base_config(experiment))
end

function parse_commandline(args=ARGS)
    settings = ArgParseSettings(
        description="Run a ContextualDFLTraining grid search for one experiment.",
    )

    @add_arg_table! settings begin
        "--experiment"
            help = "Experiment id, module name, or config path to run, e.g. resource_allocation/experiment_1"
            required = true
        "--grid-config"
            help = "Path to a YAML or JSON grid-search config file."
            required = true
    end

    return parse_args(args, settings)
end

function gridsearch_id(timestamp::AbstractString)
    return "gridsearch_" * timestamp
end

function candidate_tag(index::Integer)
    return "candidate_" * lpad(string(index), 4, "0")
end

function repeat_tag(index::Integer)
    return "repeat_" * lpad(string(index), 3, "0")
end

function base_run_id(config, index::Integer)
    if config isa NamedTuple && :run_id in keys(config)
        return string(config.run_id)
    end
    return candidate_tag(index)
end

function grid_config_value(config, key::Symbol, default)
    config isa NamedTuple || return default
    return key in keys(config) ? getproperty(config, key) : default
end

function grid_repeat_count(config)
    count = Int(grid_config_value(config, :repeat_count, 1))
    count > 0 || throw(ArgumentError("repeat_count must be positive."))
    return count
end

function grid_repeat_seed_count(configs)
    return maximum((grid_repeat_count(config) for config in configs); init=0)
end

function random_training_data_seeds(count::Integer; rng=Random.default_rng())
    count >= 0 || throw(ArgumentError("repeat seed count must be non-negative."))

    seeds = Int[]
    seen = Set{Int}()
    while length(seeds) < count
        seed = rand(rng, 1:GRID_TRAINING_DATA_SEED_MAX)
        seed in seen && continue
        push!(seeds, seed)
        push!(seen, seed)
    end
    return seeds
end

function generate_repeat_training_data_seeds(configs; rng=Random.default_rng())
    return random_training_data_seeds(grid_repeat_seed_count(configs); rng=rng)
end

function normalize_repeat_training_data_seeds(seeds, required_count::Integer)
    required_count >= 0 ||
        throw(ArgumentError("required repeat seed count must be non-negative."))
    seeds === nothing && return random_training_data_seeds(required_count)

    normalized = Int.(collect(seeds))
    length(normalized) >= required_count || throw(
        ArgumentError(
            "repeat_training_data_seeds must contain at least $required_count seed(s), got $(length(normalized)).",
        ),
    )

    selected = normalized[1:required_count]
    all(seed -> 1 <= seed <= GRID_TRAINING_DATA_SEED_MAX, selected) || throw(
        ArgumentError(
            "repeat_training_data_seeds must be in 1:$(GRID_TRAINING_DATA_SEED_MAX).",
        ),
    )
    length(unique(selected)) == length(selected) || throw(
        ArgumentError("repeat_training_data_seeds must contain distinct seeds."),
    )
    return selected
end

function repeat_training_data_seed_sequence(seeds)
    return join(string.(seeds), ",")
end

function parse_repeat_training_data_seed_sequence(value)
    text = strip(string(value))
    isempty(text) && return Int[]
    return [parse(Int, strip(part)) for part in split(text, ",")]
end

function repeat_training_data_seeds_from_config_parents(config_parent_configs, required_count)
    best_seeds = nothing
    for config_parent in config_parent_configs
        config_parent isa NamedTuple || continue
        hasproperty(config_parent, :repeat_training_data_seed_sequence) || continue
        seeds = parse_repeat_training_data_seed_sequence(
            getproperty(config_parent, :repeat_training_data_seed_sequence),
        )
        length(seeds) >= required_count &&
            return normalize_repeat_training_data_seeds(seeds, required_count)
        if best_seeds === nothing || length(seeds) > length(best_seeds)
            best_seeds = seeds
        end
    end
    return best_seeds === nothing ? nothing : best_seeds
end

function annotate_grid_config_parent(
    config,
    index::Integer,
    timestamp::AbstractString,
    mlflow_settings,
    grid_parent_run_id::AbstractString="",
    coordinator_hostname::AbstractString=Sockets.gethostname(),
    ;
    repeat_training_data_seeds=nothing,
)
    grid_id = gridsearch_id(timestamp)
    candidate = candidate_tag(index)
    previous_run_id = base_run_id(config, index)
    candidate_name = grid_id * "__" * candidate * "__" * previous_run_id
    repeats = grid_repeat_count(config)
    repeat_seeds =
        normalize_repeat_training_data_seeds(repeat_training_data_seeds, repeats)
    seed_sequence = repeat_training_data_seed_sequence(repeat_seeds)

    return merge(
        config,
        (;
            run_id=candidate_name,
            base_run_id=previous_run_id,
            gridsearch_id=grid_id,
            gridsearch_timestamp=timestamp,
            candidate_index=Int(index),
            candidate_name=candidate_name,
            config_parent_name=candidate_name,
            repeat_count=repeats,
            repeat_training_data_seed_sequence=seed_sequence,
            mlflow_enabled=mlflow_settings.enabled,
            mlflow_experiment_id=mlflow_settings.experiment_id,
            mlflow_experiment_name=mlflow_settings.experiment_name,
            mlflow_deterministic_experiment_id=mlflow_settings.deterministic_experiment_id,
            mlflow_tracking_uri=mlflow_settings.tracking_uri,
            mlflow_upload_model_artifact=mlflow_settings.upload_model_artifact,
            mlflow_parent_run_id=grid_parent_run_id,
            mlflow_run_name=candidate_name,
            coordinator_hostname=coordinator_hostname,
            mlflow_tags=(;
                gridsearch_id=grid_id,
                gridsearch_timestamp=timestamp,
                candidate_index=Int(index),
                base_run_id=previous_run_id,
                candidate_name=candidate_name,
                config_parent_name=candidate_name,
                repeat_count=repeats,
                repeat_training_data_seed_sequence=seed_sequence,
                gridsearch_parent_run_id=grid_parent_run_id,
                mlflow_parentRunId=grid_parent_run_id,
                mlflow_deterministic_experiment_id=mlflow_settings.deterministic_experiment_id,
                mlflow_experiment_name=mlflow_settings.experiment_name,
                gridsearch_role="config_parent",
            ),
        ),
    )
end

function annotate_grid_config_parents(
    configs,
    timestamp::AbstractString,
    mlflow_settings,
    grid_parent_run_id::AbstractString="",
    coordinator_hostname::AbstractString=Sockets.gethostname(),
    ;
    repeat_training_data_seeds=nothing,
)
    shared_repeat_seeds = normalize_repeat_training_data_seeds(
        repeat_training_data_seeds,
        grid_repeat_seed_count(configs),
    )
    return [
        annotate_grid_config_parent(
            config,
            index,
            timestamp,
            mlflow_settings,
            grid_parent_run_id,
            coordinator_hostname;
            repeat_training_data_seeds=shared_repeat_seeds,
        ) for
        (index, config) in enumerate(configs)
    ]
end

function annotate_repeat_config(
    config_parent,
    repeat_index::Integer,
    mlflow_settings,
    config_parent_run_id::AbstractString="",
    ;
    repeat_training_data_seeds=nothing,
)
    child_name = string(config_parent.candidate_name) * "__" * repeat_tag(repeat_index)
    seed_source = if repeat_training_data_seeds === nothing &&
                     config_parent isa NamedTuple &&
                     hasproperty(config_parent, :repeat_training_data_seed_sequence)
        parse_repeat_training_data_seed_sequence(config_parent.repeat_training_data_seed_sequence)
    else
        repeat_training_data_seeds
    end
    repeat_seeds = normalize_repeat_training_data_seeds(
        seed_source,
        Int(config_parent.repeat_count),
    )
    training_seed = repeat_seeds[Int(repeat_index)]
    tags = merge(
        config_parent.mlflow_tags,
        (;
            candidate_name=child_name,
            config_parent_name=config_parent.candidate_name,
            config_parent_run_id=config_parent_run_id,
            repeat_index=Int(repeat_index),
            repeat_count=Int(config_parent.repeat_count),
            training_data_seed=training_seed,
            repeat_training_data_seed=training_seed,
            gridsearch_parent_run_id=config_parent_run_id,
            mlflow_parentRunId=config_parent_run_id,
            gridsearch_role="repeat",
        ),
    )

    return merge(
        config_parent,
        (;
            run_id=child_name,
            candidate_name=child_name,
            config_parent_name=config_parent.candidate_name,
            config_parent_run_id=config_parent_run_id,
            repeat_index=Int(repeat_index),
            repeat_count=Int(config_parent.repeat_count),
            training_data_seed=training_seed,
            repeat_training_data_seed=training_seed,
            training_data_cache=false,
            write_training_data_artifact=false,
            mlflow_enabled=mlflow_settings.enabled,
            mlflow_upload_model_artifact=mlflow_settings.upload_model_artifact,
            mlflow_parent_run_id=config_parent_run_id,
            mlflow_run_name=child_name,
            mlflow_tags=tags,
        ),
    )
end

function annotate_repeat_configs(
    config_parent_configs,
    config_parent_runs,
    mlflow_settings;
    repeat_training_data_seeds=nothing,
)
    required_count = grid_repeat_seed_count(config_parent_configs)
    seed_source = repeat_training_data_seeds === nothing ?
        repeat_training_data_seeds_from_config_parents(config_parent_configs, required_count) :
        repeat_training_data_seeds
    shared_repeat_seeds = normalize_repeat_training_data_seeds(
        seed_source,
        required_count,
    )
    child_configs = NamedTuple[]
    for (config_parent, config_parent_run) in zip(config_parent_configs, config_parent_runs)
        parent_run_id = mlflow_parent_run_id(config_parent_run)
        for repeat_index in 1:Int(config_parent.repeat_count)
            push!(
                child_configs,
                annotate_repeat_config(
                    config_parent,
                    repeat_index,
                    mlflow_settings,
                    parent_run_id;
                    repeat_training_data_seeds=shared_repeat_seeds,
                ),
            )
        end
    end
    return child_configs
end

function config_parent_results(config_parent_configs, child_results)
    return [
        config_parent_result(config_parent, child_results_for_config(config_parent, child_results)) for
        config_parent in config_parent_configs
    ]
end

function child_results_for_config(config_parent, child_results)
    parent_name = string(config_parent.candidate_name)
    return [
        result for result in child_results if
        result_config_parent_name(getproperty(result, :config)) == parent_name
    ]
end

function result_config_parent_name(config)
    config isa NamedTuple || return ""
    if :config_parent_name in keys(config)
        return string(config.config_parent_name)
    end
    return ""
end

function config_parent_result(config_parent, child_results)
    expected_repeats = Int(config_parent.repeat_count)
    successful_repeats = count(result -> getproperty(result, :status) == "ok", child_results)
    failed_repeats = length(child_results) - successful_repeats
    success = length(child_results) == expected_repeats && failed_repeats == 0
    summaries = aggregate_metric_summaries(child_results)
    metrics = merge(
        aggregate_mean_metrics(summaries),
        (;
            repeat_count=Float64(expected_repeats),
            repeat_successful_count=Float64(successful_repeats),
            repeat_failed_count=Float64(failed_repeats),
        ),
    )
    started_at = isempty(child_results) ?
        unix_milliseconds() :
        minimum(getproperty(result, :started_at) for result in child_results)
    finished_at = isempty(child_results) ?
        unix_milliseconds() :
        maximum(getproperty(result, :finished_at) for result in child_results)
    elapsed_seconds = sum(
        Float64(getproperty(result, :elapsed_seconds)) for result in child_results;
        init=0.0,
    )
    error = success ? "" : config_parent_error(child_results, expected_repeats)

    return (;
        status=success ? "ok" : "failed",
        run_id=config_parent.run_id,
        config=config_parent,
        worker=NamedTuple(),
        final_metrics=metrics,
        aggregate_metrics=summaries,
        epoch_history=Dict{Symbol,Any}[],
        error=error,
        started_at=started_at,
        finished_at=finished_at,
        elapsed_seconds=elapsed_seconds,
    )
end

function config_parent_error(child_results, expected_repeats)
    parts = String[]
    length(child_results) == expected_repeats || push!(
        parts,
        "Expected $(expected_repeats) repeat(s), recorded $(length(child_results)).",
    )
    for result in child_results
        getproperty(result, :status) == "ok" && continue
        push!(parts, "$(getproperty(result, :run_id)): $(getproperty(result, :status))")
    end
    return join(parts, " ")
end

function main()
    parsed_args = parse_commandline()
    experiment = load_experiment(parsed_args["experiment"])
    grid_spec = load_grid_config(parsed_args["grid-config"])

    ensure_clean_worker_start!()
    sync_code!()
    remote_worker_ids = add_remote_workers!()
    load_worker_stdlibs!()
    worker_hosts = assert_remote_only_workers!(remote_worker_ids)
    load_training_project_on_workers!(remote_worker_ids)
    define_remote_eval!()

    timestamp = result_timestamp()
    grid_id = gridsearch_id(timestamp)
    mlflow_settings = grid_mlflow_settings(experiment)
    validate_mlflow_settings(mlflow_settings)
    mlflow_settings = ensure_mlflow_grid_experiment(mlflow_settings)
    base_configs = selected_grid(experiment, grid_spec)
    repeat_training_data_seeds = generate_repeat_training_data_seeds(base_configs)
    parent_run = create_mlflow_grid_parent_run(
        mlflow_settings,
        grid_id,
        timestamp,
        base_configs,
        worker_hosts,
        grid_spec;
        repeat_training_data_seeds=repeat_training_data_seeds,
    )
    parent_run_id = mlflow_parent_run_id(parent_run)
    config_parent_configs = annotate_grid_config_parents(
        base_configs,
        timestamp,
        mlflow_settings,
        parent_run_id,
        Sockets.gethostname();
        repeat_training_data_seeds=repeat_training_data_seeds,
    )
    config_parent_runs = Any[]
    configs = NamedTuple[]
    try
        config_parent_runs = create_mlflow_config_parent_runs(mlflow_settings, config_parent_configs)
        configs = annotate_repeat_configs(
            config_parent_configs,
            config_parent_runs,
            mlflow_settings;
            repeat_training_data_seeds=repeat_training_data_seeds,
        )
    catch error
        fail_mlflow_config_parent_runs(config_parent_runs)
        fail_mlflow_grid_parent_run(parent_run)
        rethrow()
    end
    println("Grid search id: $grid_id")
    println(
        "Grid config: $(grid_spec.name) ($(grid_spec.path), $(length(base_configs)) candidate(s), $(length(configs)) repeat run(s))",
    )
    println(
        "Repeat training data seeds: ",
        repeat_training_data_seed_sequence(repeat_training_data_seeds),
    )
    if mlflow_settings.enabled
        println(
            "MLflow experiment id: $(mlflow_settings.experiment_id) ($(mlflow_settings.experiment_name))",
        )
    end
    println(
        "Running $(length(configs)) configuration(s) on $(length(remote_worker_ids)) remote worker(s)",
    )

    results = try
        run_grid_on_remote_workers(remote_worker_ids, configs, worker_hosts)
    catch error
        fail_mlflow_config_parent_runs(config_parent_runs)
        fail_mlflow_grid_parent_run(parent_run)
        rethrow()
    end
    config_results = config_parent_results(config_parent_configs, results)
    close_mlflow_config_parent_runs(config_parent_runs, config_results)
    close_mlflow_grid_parent_run(parent_run, config_results; child_results=results)

    output_dir = write_grid_results(
        results;
        configs=configs,
        config_results=config_results,
        output_root=joinpath(@__DIR__, "results"),
        timestamp=timestamp,
    )
    println("Wrote grid-search CSV results to $output_dir")

    failed_count = count(result -> result.status != "ok", results)
    failed_count > 0 && println("Recorded $failed_count failed configuration(s).")
    return output_dir
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

# END FILE: src/ContextualDFL/ContextualDFLTraining/gridsearch.jl

# BEGIN FILE: src/ContextualDFL/ContextualDFLTraining/profile_training.jl
#!/usr/bin/env julia

using ArgParse
using Distributed
using Sockets

include(joinpath(@__DIR__, "src", "run_defaults.jl"))
include(joinpath(@__DIR__, "src", "csv_results.jl"))
include(joinpath(@__DIR__, "src", "experiments", "ExperimentAPI.jl"))

const DEFAULT_REMOTE_PROJECT =
    "/home/rwl/ProblemBasedScenarioGeneration/src/ContextualDFL/ContextualDFLTraining"
const DEFAULT_REMOTE_JULIA = "/home/rwl/.juliaup/bin/julia"
const PROFILE_MLFLOW_EXPERIMENT_ID = "3"
const PROFILE_MLFLOW_EXPERIMENT_NAME = "ContextualDFLProfiling"

function env_int(name, default)
    value = get(ENV, name, string(default))
    parsed = tryparse(Int, value)
    parsed === nothing && error("ENV[$name] must be an integer, got: $value")
    return parsed
end

function env_float(name, default)
    value = get(ENV, name, string(default))
    parsed = tryparse(Float64, value)
    parsed === nothing && error("ENV[$name] must be a number, got: $value")
    return parsed
end

function env_symbol(name, default)
    return Symbol(get(ENV, name, string(default)))
end

function env_flag(name, default=false)
    value = lowercase(get(ENV, name, default ? "1" : "0"))
    return value in ("1", "true", "yes", "y")
end

function parse_commandline(args=ARGS)
    settings = ArgParseSettings(
        description="Run a ContextualDFLTraining profiling job for one experiment.",
    )

    @add_arg_table! settings begin
        "--experiment"
            help = "Experiment id, module name, or config path to profile, e.g. resource_allocation/experiment_1"
            required = true
    end

    return parse_args(args, settings)
end

function ensure_clean_worker_start!()
    nprocs() == 1 ||
        error("Refusing to run with pre-existing workers. Start Julia without -p or --machine-file.")
end

function sync_code!()
    if env_flag("SKIP_SYNC", false)
        println("Skipping code sync because SKIP_SYNC is set.")
        return nothing
    end

    sync_script = joinpath(homedir(), "sync-julia-code.sh")
    isfile(sync_script) || error("sync script not found: $sync_script")
    println("Syncing code to remote machines with $sync_script")
    run(Cmd(`$sync_script`; dir=homedir()))
    return nothing
end

function profile_config_from_env(experiment)
    run_id = get(ENV, "PROFILE_RUN_ID", "profile_standard_seed3")
    mlflow_enabled = env_flag("PROFILE_MLFLOW_ENABLED", true)
    profile_mlflow_progress = env_flag("PROFILE_MLFLOW_PROGRESS", mlflow_enabled)
    profile_rho = env_float("PROFILE_RHO", DEFAULT_RUN_SETTINGS.rho)
    profile_policy_inference_rho = haskey(ENV, "PROFILE_POLICY_INFERENCE_RHO") ?
        env_float("PROFILE_POLICY_INFERENCE_RHO", profile_rho) :
        nothing
    base = if experiment_has_function(experiment, :profile_config)
        experiment_call(experiment, :profile_config)
    else
        experiment_call(experiment, :base_config)
    end

    cfg = merge(
        base,
        (;
            epochs=env_int("PROFILE_EPOCHS", 10),
            warmup_epochs=env_int("PROFILE_WARMUP_EPOCHS", 2),
            mu=env_float("PROFILE_MU", 1.0),
            mu_start=env_float("PROFILE_MU_START", 1.0),
            mu_end=env_float("PROFILE_MU_END", 1.0),
            mu_schedule=env_symbol("PROFILE_MU_SCHEDULE", :constant),
            rho=profile_rho,
            rho_start=env_float("PROFILE_RHO_START", DEFAULT_RUN_SETTINGS.rho_start),
            rho_end=env_float("PROFILE_RHO_END", DEFAULT_RUN_SETTINGS.rho_end),
            rho_schedule=env_symbol("PROFILE_RHO_SCHEDULE", DEFAULT_RUN_SETTINGS.rho_schedule),
            rho_ref=env_float("PROFILE_RHO_REF", DEFAULT_RUN_SETTINGS.rho_ref),
            rho_ref_start=env_float("PROFILE_RHO_REF_START", DEFAULT_RUN_SETTINGS.rho_ref_start),
            rho_ref_end=env_float("PROFILE_RHO_REF_END", DEFAULT_RUN_SETTINGS.rho_ref_end),
            rho_ref_schedule=env_symbol("PROFILE_RHO_REF_SCHEDULE", DEFAULT_RUN_SETTINGS.rho_ref_schedule),
            tolerance_relative=env_float("PROFILE_TOLERANCE_RELATIVE", DEFAULT_RUN_SETTINGS.tolerance_relative),
            tolerance_absolute_floor=env_float(
                "PROFILE_TOLERANCE_ABSOLUTE_FLOOR",
                DEFAULT_RUN_SETTINGS.tolerance_absolute_floor,
            ),
            optimality_evaluation=env_flag(
                "PROFILE_OPTIMALITY_EVALUATION",
                DEFAULT_RUN_SETTINGS.optimality_evaluation,
            ),
            optimality_test_sample_count=env_int(
                "PROFILE_OPTIMALITY_TEST_SAMPLE_COUNT",
                DEFAULT_RUN_SETTINGS.optimality_test_sample_count,
            ),
            optimality_train_sample_count=env_int(
                "PROFILE_OPTIMALITY_TRAIN_SAMPLE_COUNT",
                DEFAULT_RUN_SETTINGS.optimality_train_sample_count,
            ),
            optimality_validation_sample_count=env_int(
                "PROFILE_OPTIMALITY_VALIDATION_SAMPLE_COUNT",
                DEFAULT_RUN_SETTINGS.optimality_validation_sample_count,
            ),
            optimality_mu=env_float("PROFILE_OPTIMALITY_MU", DEFAULT_RUN_SETTINGS.optimality_mu),
            optimality_rho=env_float("PROFILE_OPTIMALITY_RHO", DEFAULT_RUN_SETTINGS.optimality_rho),
            policy_inference_rho=profile_policy_inference_rho,
            loss=env_symbol("PROFILE_LOSS", :dfl_scen),
            learning_rate=env_float("PROFILE_LEARNING_RATE", 1e-3),
            hidden_size=env_int("PROFILE_HIDDEN_SIZE", 128),
            depth=env_int("PROFILE_DEPTH", 2),
            batch_size=env_int("PROFILE_BATCH_SIZE", 64),
            dropout=env_float("PROFILE_DROPOUT", 0.0),
            seed=env_int("PROFILE_SEED", 3),
            run_id=run_id,
            base_run_id=run_id,
            candidate_name=run_id,
            mlflow_enabled=mlflow_enabled,
            profile_mlflow_progress=profile_mlflow_progress,
            mlflow_experiment_id=PROFILE_MLFLOW_EXPERIMENT_ID,
            mlflow_experiment_name=PROFILE_MLFLOW_EXPERIMENT_NAME,
            mlflow_tracking_uri=get(
                ENV,
                "PROFILE_MLFLOW_TRACKING_URI",
                get(ENV, "MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"),
            ),
            mlflow_run_name=get(ENV, "PROFILE_MLFLOW_RUN_NAME", run_id),
            mlflow_upload_model_artifact=false,
            mlflow_source_name="ContextualDFLTraining/profile_training.jl",
            mlflow_dataset_context="profiling",
            method_variant="profiling",
            mlflow_params=(;
                profile_target="ContextualDFL.train!",
                profile_loss="ContextualDFL.DflScenLoss",
                profile_progress_logged_by="remote_worker",
            ),
            mlflow_tags=(;
                source="ContextualDFLTraining.profile_training",
                run_kind="profiling",
                profile_run=true,
                exclude_from_model_selection=true,
                exclude_from_gridsearch=true,
                mlflow_experiment_name=PROFILE_MLFLOW_EXPERIMENT_NAME,
                profile_target="ContextualDFL.train!",
                profile_loss="ContextualDFL.DflScenLoss",
                profile_progress_logged_by="remote_worker",
                profile_artifacts="local_csv_svg_jlprof",
                coordinator_hostname=Sockets.gethostname(),
            ),
        ),
    )
    return with_experiment_metadata(experiment, cfg)
end

function with_profile_output_config(config, output_dir)
    tags = merge(
        config.mlflow_tags,
        (;
            profile_local_output_dir=output_dir,
            profile_timestamp=basename(output_dir),
        ),
    )
    return merge(config, (; profile_local_output_dir=output_dir, mlflow_tags=tags))
end

function add_profile_worker!()
    remote_project = get(ENV, "REMOTE_CONTEXTUAL_DFL_TRAINING_PROJECT", DEFAULT_REMOTE_PROJECT)
    remote_julia = get(ENV, "REMOTE_JULIA", DEFAULT_REMOTE_JULIA)
    remote_threads = env_int("PROFILE_REMOTE_THREADS", 2)

    println("Adding one profiling worker on rwl@gcp-big with $remote_threads Julia thread(s)")
    addprocs(
        [("rwl@gcp-big", 1)];
        exename=remote_julia,
        exeflags=["--project=$(remote_project)", "--threads=$(remote_threads)"],
        dir=remote_project,
        tunnel=true,
    )

    remote_worker_ids = setdiff(workers(), [1])
    length(remote_worker_ids) == 1 ||
        error("Expected exactly one remote profiling worker, got $(remote_worker_ids).")
    return only(remote_worker_ids)
end

function load_worker!(worker)
    remotecall_fetch(worker) do
        Core.eval(Main, quote
            using Dates
            using Distributed
            using Pkg
            using Sockets
            Pkg.instantiate()
            using ContextualDFLTraining
        end)
        return Core.eval(Main, quote
            (;
                worker_id=Distributed.myid(),
                hostname=Sockets.gethostname(),
                pid=getpid(),
                thread_count=Threads.nthreads(),
                julia_version=string(VERSION),
            )
        end)
    end
end

function assert_remote_profile_worker!(worker, metadata)
    local_hostname = Sockets.gethostname()
    metadata.hostname == local_hostname &&
        error("Refusing to run profiling on local host $(local_hostname).")
    metadata.thread_count == env_int("PROFILE_REMOTE_THREADS", 2) ||
        error("Remote worker has $(metadata.thread_count) thread(s), expected $(env_int("PROFILE_REMOTE_THREADS", 2)).")
    println(
        "Profiling worker online: id=$(worker), host=$(metadata.hostname), pid=$(metadata.pid), threads=$(metadata.thread_count)",
    )
end

function profile_output_dir()
    stamp = string(unix_milliseconds())
    output_dir = joinpath(@__DIR__, "results", "profile_" * stamp)
    mkpath(joinpath(output_dir, "assets"))
    return output_dir
end

function result_row(result)
    row = Dict{Symbol,Any}()
    row[:status] = result.status
    row[:run_id] = result.run_id
    row[:started_at] = result.started_at
    row[:finished_at] = result.finished_at
    row[:elapsed_seconds] = result.elapsed_seconds
    row[:error] = result.error
    flatten_to_dict!(row, "config", result.config)
    flatten_to_dict!(row, "", result.worker)
    flatten_to_dict!(row, "", result.final_metrics)
    return row
end

function write_profile_outputs(result, output_dir)
    write_csv(joinpath(output_dir, "profile_metadata.csv"), [result_row(result)])
    write_csv(joinpath(output_dir, "profile_epochs.csv"), epoch_result_rows([result]))

    if result.status == "ok"
        write(joinpath(output_dir, "assets", "prof.svg"), result.profile_svg_bytes)
        write(joinpath(output_dir, "assets", "prof.jlprof"), result.profile_jlprof_bytes)
    else
        open(joinpath(output_dir, "profile_error.txt"), "w") do io
            print(io, result.error)
        end
    end

    return output_dir
end

function main()
    parsed_args = parse_commandline()
    experiment = load_experiment(parsed_args["experiment"])
    ensure_clean_worker_start!()
    worker = nothing

    try
        output_dir = profile_output_dir()
        config = with_profile_output_config(
            merge(profile_config_from_env(experiment), (; coordinator_hostname=Sockets.gethostname())),
            output_dir,
        )
        println("Running remote profile $(config.run_id) with $(config.epochs) profiled epoch(s)")
        if config.mlflow_enabled && config.profile_mlflow_progress
            println(
                "Remote MLflow profiling progress enabled: ",
                "experiment=$(config.mlflow_experiment_id) ($(config.mlflow_experiment_name)), ",
                "tracking_uri=$(config.mlflow_tracking_uri), run_name=$(config.mlflow_run_name)",
            )
        end

        sync_code!()
        worker = add_profile_worker!()
        metadata = load_worker!(worker)
        assert_remote_profile_worker!(worker, metadata)

        result = remotecall_fetch(worker) do
            Main.ContextualDFLTraining.profile_standard_training(config)
        end

        write_profile_outputs(result, output_dir)
        println("Wrote profile outputs to $output_dir")

        if result.status == "ok"
            metrics = result.final_metrics
            println(
                "Train MSE: $(metrics.initial_train_mse) -> $(metrics.final_train_mse) ",
                "(delta=$(metrics.loss_delta))",
            )
            println("Artifacts: assets/prof.svg and assets/prof.jlprof")
        else
            println("Profile failed; see profile_error.txt")
        end

        return output_dir
    finally
        if worker !== nothing && worker in workers()
            try
                rmprocs(worker; waitfor=5)
            catch error
                @warn "Failed to remove profiling worker" worker exception=(error, catch_backtrace())
            end
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

# END FILE: src/ContextualDFL/ContextualDFLTraining/profile_training.jl

# BEGIN FILE: src/ContextualDFL/ContextualDFLTraining/src/ContextualDFLTraining.jl
module ContextualDFLTraining

include("run_defaults.jl")
include("grid_config.jl")
include("training_helpers.jl")
include(joinpath("experiments", "ExperimentAPI.jl"))
include("grid_file_config.jl")
include("mlflow_support.jl")
include("train_run.jl")
include("profile_run.jl")
include("csv_results.jl")

export default_grid,
    DEFAULT_TEST_DATA_SEED,
    DEFAULT_TEST_DATA_SET_SIZE,
    experiment_artifact_dir,
    experiment_base_config,
    experiment_call,
    experiment_from_config,
    experiment_test_data_bundle,
    experiment_test_data_config,
    smoke_grid,
    GridSearchSpec,
    grid_config_digest,
    load_experiment,
    load_grid_config,
    load_optimal_results,
    load_test_data,
    load_test_data_artifact,
    optimal_results_path,
    resolve_grid_configs,
    resolved_grid_json,
    save_optimal_results!,
    save_test_data!,
    save_test_optimal_results!,
    test_data_dir,
    test_data_path,
    test_optimal_results_path,
    train_and_evaluate,
    training_objects_for_config,
    profile_standard_training,
    write_grid_results

end

# END FILE: src/ContextualDFL/ContextualDFLTraining/src/ContextualDFLTraining.jl

# BEGIN FILE: src/ContextualDFL/ContextualDFLTraining/src/csv_results.jl
import CSV

if !isdefined(@__MODULE__, :unix_milliseconds)
    # Explicit Unix epoch milliseconds, independent of the local timezone.
    unix_milliseconds() = round(Int64, time() * 1000)
end

function result_timestamp()
    return string(unix_milliseconds())
end

function write_grid_results(
    results;
    configs=nothing,
    config_results=nothing,
    output_root=joinpath(@__DIR__, "..", "results"),
    timestamp=result_timestamp(),
)
    output_dir = joinpath(output_root, timestamp)
    mkpath(output_dir)

    run_rows = [run_result_row(result) for result in results]
    write_csv(joinpath(output_dir, "runs.csv"), run_rows)

    epoch_rows = epoch_result_rows(results)
    write_csv(joinpath(output_dir, "epochs.csv"), epoch_rows)

    failure_rows = [row for row in run_rows if get(row, :status, "") != "ok"]
    write_csv(joinpath(output_dir, "failures.csv"), failure_rows)

    successful_rows = [row for row in run_rows if get(row, :status, "") == "ok"]
    best_rows = sort(successful_rows; by=row -> get(row, :validation_mse, Inf))
    write_csv(joinpath(output_dir, "best.csv"), best_rows)

    aggregate_rows = config_aggregate_rows(config_results)
    write_csv(joinpath(output_dir, "config_aggregates.csv"), aggregate_rows)

    config_summary_rows = grid_config_summary_rows(configs, results; config_results=config_results)
    write_csv(joinpath(output_dir, "config.csv"), config_summary_rows)

    if configs !== nothing
        config_rows = [flatten_to_dict(config; prefix="config") for config in configs]
        write_csv(joinpath(output_dir, "configs.csv"), config_rows)
    end

    return output_dir
end

function run_result_row(result)
    row = Dict{Symbol,Any}()
    row[:status] = getproperty(result, :status)
    row[:run_id] = getproperty(result, :run_id)
    row[:started_at] = getproperty(result, :started_at)
    row[:finished_at] = getproperty(result, :finished_at)
    row[:elapsed_seconds] = getproperty(result, :elapsed_seconds)
    row[:error] = getproperty(result, :error)

    flatten_to_dict!(row, "config", getproperty(result, :config))
    flatten_to_dict!(row, "", getproperty(result, :worker))
    flatten_to_dict!(row, "", getproperty(result, :final_metrics))
    return row
end

function epoch_result_rows(results)
    rows = Dict{Symbol,Any}[]

    for result in results
        history = getproperty(result, :epoch_history)
        for history_row in history
            row = Dict{Symbol,Any}()
            row[:run_id] = getproperty(result, :run_id)
            row[:status] = getproperty(result, :status)
            flatten_to_dict!(row, "config", getproperty(result, :config))
            flatten_to_dict!(row, "", history_row)
            push!(rows, row)
        end
    end

    return rows
end

function config_aggregate_rows(config_results)
    config_results === nothing && return Dict{Symbol,Any}[]

    return [config_aggregate_row(result) for result in config_results]
end

function config_aggregate_row(result)
    row = Dict{Symbol,Any}()
    row[:status] = getproperty(result, :status)
    row[:run_id] = getproperty(result, :run_id)
    row[:started_at] = getproperty(result, :started_at)
    row[:finished_at] = getproperty(result, :finished_at)
    row[:elapsed_seconds] = getproperty(result, :elapsed_seconds)
    row[:error] = getproperty(result, :error)

    flatten_to_dict!(row, "config", getproperty(result, :config))
    flatten_to_dict!(row, "", getproperty(result, :final_metrics))

    aggregate_metrics = getproperty(result, :aggregate_metrics)
    if aggregate_metrics isa AbstractDict
        for key in sort!(collect(keys(aggregate_metrics)); by=String)
            summary = aggregate_metrics[key]
            for field in (:count, :mean, :median, :min, :max, :std, :stderr)
                row[Symbol(string(key) * "_" * string(field))] = getproperty(summary, field)
            end
        end
    end

    return row
end

function grid_config_summary_rows(configs, results; config_results=nothing)
    rows = Dict{Symbol,Any}[
        Dict(:key => "created_at_unix_ms", :value => unix_milliseconds()),
        Dict(:key => "result_count", :value => length(results)),
        Dict(:key => "successful_count", :value => count(result -> result.status == "ok", results)),
        Dict(:key => "failed_count", :value => count(result -> result.status != "ok", results)),
    ]

    if config_results !== nothing
        push!(rows, Dict(:key => "config_parent_count", :value => length(config_results)))
        push!(
            rows,
            Dict(
                :key => "config_parent_successful_count",
                :value => count(result -> result.status == "ok", config_results),
            ),
        )
        push!(
            rows,
            Dict(
                :key => "config_parent_failed_count",
                :value => count(result -> result.status != "ok", config_results),
            ),
        )
    end

    if configs !== nothing
        push!(rows, Dict(:key => "config_count", :value => length(configs)))
        first_config = isempty(configs) ? nothing : first(configs)
        if first_config !== nothing
            for key in keys(first_config)
                value = getproperty(first_config, key)
                if !(value isa AbstractArray)
                    push!(rows, Dict(:key => "default_" * string(key), :value => value))
                end
            end
        end
    end

    return rows
end

function flatten_to_dict(value; prefix="")
    row = Dict{Symbol,Any}()
    flatten_to_dict!(row, prefix, value)
    return row
end

function flatten_to_dict!(row::Dict{Symbol,Any}, prefix::AbstractString, value::NamedTuple)
    for key in keys(value)
        child_prefix = join_prefix(prefix, key)
        flatten_to_dict!(row, child_prefix, getproperty(value, key))
    end
    return row
end

function flatten_to_dict!(row::Dict{Symbol,Any}, prefix::AbstractString, value::AbstractDict)
    for (key, child_value) in value
        child_prefix = join_prefix(prefix, Symbol(key))
        flatten_to_dict!(row, child_prefix, child_value)
    end
    return row
end

function flatten_to_dict!(row::Dict{Symbol,Any}, prefix::AbstractString, value)
    isempty(prefix) && return row
    row[Symbol(prefix)] = value
    return row
end

function join_prefix(prefix::AbstractString, key)
    key_text = string(key)
    return isempty(prefix) ? key_text : prefix * "_" * key_text
end

function write_csv(path, rows)
    if isempty(rows)
        write(path, "")
        return path
    end

    headers = Symbol[]
    seen = Set{Symbol}()
    for row in rows, header in keys(row)
        header in seen && continue
        push!(seen, header)
        push!(headers, header)
    end
    sort!(headers; by=String)

    columns = map(headers) do header
        header => [something(get(row, header, missing), missing) for row in rows]
    end
    CSV.write(path, (; columns...); missingstring="")
    return path
end

# END FILE: src/ContextualDFL/ContextualDFLTraining/src/csv_results.jl

# BEGIN FILE: src/ContextualDFL/ContextualDFLTraining/src/experiments/ExperimentAPI.jl
using Dates
using Serialization
using SHA

const EXPERIMENT_CONFIG_FILENAME = "Config.jl"
const EXPERIMENTS_ROOT = @__DIR__
const OPTIMAL_RESULTS_FORMAT_VERSION = 1
const TEST_DATA_FORMAT_VERSION = 1
const TRAINING_DATA_FORMAT_VERSION = 1
const DEFAULT_TEST_DATA_SEED = 1
const DEFAULT_TEST_DATA_SET_SIZE = 30

struct ExperimentSpec
    id::String
    name::String
    module_name::Symbol
    module_ref::Module
    root_dir::String
    config_path::String
end

struct LatestExperimentFunction <: Function
    module_ref::Module
    name::Symbol
end

experiment_binding_isdefined(module_ref::Module, name::Symbol) =
    Base.invokelatest(isdefined, module_ref, name)

function experiment_binding(module_ref::Module, name::Symbol)
    experiment_binding_isdefined(module_ref, name) ||
        throw(ArgumentError("experiment module $(module_ref) does not define $name"))
    return Base.invokelatest(getfield, module_ref, name)
end

function (fn::LatestExperimentFunction)(args...; kwargs...)
    target = experiment_binding(fn.module_ref, fn.name)
    target isa Function ||
        throw(ArgumentError("experiment binding $(fn.name) must be a function, got $(typeof(target))"))
    return Base.invokelatest(target, args...; kwargs...)
end

const REQUIRED_EXPERIMENT_FUNCTIONS = (
    :experiment_id,
    :experiment_name,
    :artifact_dir,
    :base_config,
    :training_objects,
    :optimality_splits,
    :optimal_results_path,
)

function load_experiment()
    throw(
        ArgumentError(
            "load_experiment requires an explicit selector. Scripts should pass --experiment <experiment>.",
        ),
    )
end

function load_experiment(selector)
    selector isa ExperimentSpec && return selector
    selector_string = string(selector)
    path = resolve_experiment_config_path(selector_string)
    spec = load_experiment_config(path)

    if !experiment_selector_matches(spec, selector_string)
        throw(
            ArgumentError(
                "experiment selector '$selector_string' resolved to $(spec.config_path), but that config declares experiment_id=$(spec.id), experiment_name=$(spec.name), module_name=$(spec.module_name)",
            ),
        )
    end

    return spec
end

function resolve_experiment_config_path(selector::AbstractString)
    if isfile(selector)
        return abspath(selector)
    end

    direct = direct_experiment_config_path(selector)
    direct === nothing || return direct

    matches = ExperimentSpec[]
    for path in experiment_config_paths()
        spec = load_experiment_config(path)
        experiment_selector_matches(spec, selector) && push!(matches, spec)
    end

    isempty(matches) && throw(
        ArgumentError(
            "no experiment config found for selector '$selector'. Expected a file path, an experiment id such as resource_allocation/experiment_1, or a declared module/name under $EXPERIMENTS_ROOT.",
        ),
    )
    length(matches) == 1 || throw(
        ArgumentError(
            "experiment selector '$selector' matched multiple configs: $(join((spec.config_path for spec in matches), ", "))",
        ),
    )

    return only(matches).config_path
end

function direct_experiment_config_path(selector::AbstractString)
    normalized = replace(strip(selector), "\\" => "/")
    isempty(normalized) && return nothing

    candidates = String[]
    push!(candidates, joinpath(EXPERIMENTS_ROOT, split(normalized, "/")..., EXPERIMENT_CONFIG_FILENAME))
    push!(candidates, joinpath(EXPERIMENTS_ROOT, normalized))

    for candidate in candidates
        isfile(candidate) && return abspath(candidate)
    end

    return nothing
end

function experiment_config_paths()
    paths = String[]
    for (root, _, files) in walkdir(EXPERIMENTS_ROOT)
        EXPERIMENT_CONFIG_FILENAME in files &&
            push!(paths, abspath(joinpath(root, EXPERIMENT_CONFIG_FILENAME)))
    end
    sort!(paths)
    return paths
end

function load_experiment_config(path::AbstractString)
    absolute_path = abspath(path)
    isfile(absolute_path) ||
        throw(ArgumentError("experiment config file does not exist: $absolute_path"))

    module_name = dynamic_experiment_module_name(absolute_path)
    module_ref = Module(module_name)
    Core.eval(module_ref, :(import ContextualDFLTraining))
    Core.eval(module_ref, :(import ContextualDFLExperiments))
    Core.eval(module_ref, :(import ContextualDFL))
    Base.include(module_ref, absolute_path)

    validate_experiment_module!(module_ref, absolute_path)

    id = String(experiment_call(module_ref, :experiment_id))
    name = String(experiment_call(module_ref, :experiment_name))
    declared_module_name = if experiment_binding_isdefined(module_ref, :experiment_module_name)
        Symbol(experiment_call(module_ref, :experiment_module_name))
    else
        module_name
    end

    spec = ExperimentSpec(
        id,
        name,
        declared_module_name,
        module_ref,
        dirname(absolute_path),
        absolute_path,
    )
    validate_experiment!(spec)
    return spec
end

function dynamic_experiment_module_name(path::AbstractString)
    relative = try
        relpath(path, EXPERIMENTS_ROOT)
    catch
        basename(path)
    end
    safe = replace(relative, r"[^A-Za-z0-9_]" => "_")
    digest = bytes2hex(sha1(path))[1:8]
    return Symbol(:ContextualDFLTrainingExperiment_, safe, :_, digest)
end

function validate_experiment_module!(module_ref::Module, path::AbstractString)
    missing = Symbol[]
    for name in REQUIRED_EXPERIMENT_FUNCTIONS
        experiment_binding_isdefined(module_ref, name) || push!(missing, name)
    end
    isempty(missing) || throw(
        ArgumentError(
            "experiment config $path is missing required function(s): $(join(string.(missing), ", "))",
        ),
    )
    return nothing
end

function validate_experiment!(spec::ExperimentSpec)
    isempty(spec.id) && throw(ArgumentError("experiment_id() must not be empty."))
    isempty(spec.name) && throw(ArgumentError("experiment_name() must not be empty."))
    isabspath(experiment_artifact_dir(spec)) ||
        throw(ArgumentError("artifact_dir() must return an absolute path."))
    return spec
end

function experiment_selector_matches(spec::ExperimentSpec, selector::AbstractString)
    normalized = normalize_experiment_selector(selector)
    return normalized in (
        normalize_experiment_selector(spec.id),
        normalize_experiment_selector(spec.name),
        normalize_experiment_selector(string(spec.module_name)),
        normalize_experiment_selector(spec.config_path),
        normalize_experiment_selector(relpath(spec.root_dir, EXPERIMENTS_ROOT)),
    )
end

function normalize_experiment_selector(selector::AbstractString)
    value = lowercase(strip(selector))
    value = replace(value, "\\" => "/")
    value = replace(value, r"[^a-z0-9/]+" => "_")
    value = replace(value, r"_+" => "_")
    return strip(value, ['_', '/'])
end

function experiment_call(spec::ExperimentSpec, name::Symbol, args...; kwargs...)
    return experiment_call(spec.module_ref, name, args...; kwargs...)
end

function experiment_call(module_ref::Module, name::Symbol, args...; kwargs...)
    fn = experiment_binding(module_ref, name)
    fn isa Function ||
        throw(ArgumentError("experiment binding $name must be a function, got $(typeof(fn))"))
    return Base.invokelatest(fn, args...; kwargs...)
end

function experiment_has_function(spec::ExperimentSpec, name::Symbol)
    return experiment_binding_isdefined(spec.module_ref, name) &&
           experiment_binding(spec.module_ref, name) isa Function
end

experiment_artifact_dir(spec::ExperimentSpec) = abspath(String(experiment_call(spec, :artifact_dir)))

function experiment_base_config(spec::ExperimentSpec)
    return with_experiment_metadata(spec, experiment_call(spec, :base_config))
end

function with_experiment_metadata(spec::ExperimentSpec, config::NamedTuple)
    return merge(
        config,
        (;
            experiment_id=spec.id,
            experiment_name=spec.name,
            experiment_module_name=spec.module_name,
            experiment_config_path=spec.config_path,
            experiment_artifact_dir=experiment_artifact_dir(spec),
        ),
    )
end

function experiment_from_config(config)
    if config isa ExperimentSpec
        return config
    end
    config isa NamedTuple || return nothing

    if hasproperty(config, :experiment_config_path)
        return load_experiment(getproperty(config, :experiment_config_path))
    end

    for key in (:experiment_id, :experiment_module_name, :experiment_name)
        if hasproperty(config, key)
            return load_experiment(getproperty(config, key))
        end
    end

    return nothing
end

function training_objects_for_config(config::NamedTuple)
    spec = experiment_from_config(config)
    spec === nothing && throw(
        ArgumentError(
            "config must declare experiment_id, experiment_module_name, experiment_name, or experiment_config_path to construct training objects.",
        ),
    )
    config = with_experiment_metadata(spec, config)

    if experiment_has_function(spec, :training_data)
        artifact = load_or_generate_training_data(spec, config)
        objects = experiment_call(spec, :training_objects, config, artifact.dataset)
        return with_training_data_metadata(objects, artifact)
    end

    return experiment_call(spec, :training_objects, config)
end

function experiment_config_value(config, key::Symbol, default)
    config isa NamedTuple || return default
    return key in keys(config) ? getproperty(config, key) : default
end

function training_data_cache_enabled(config)
    return Bool(experiment_config_value(config, :training_data_cache, true)) &&
           Bool(experiment_config_value(config, :write_training_data_artifact, true))
end

function training_data_dir(spec::ExperimentSpec, config::NamedTuple)
    return abspath(
        string(
            experiment_config_value(
                config,
                :training_data_dir,
                joinpath(experiment_artifact_dir(spec), "training_data"),
            ),
        ),
    )
end

function training_dataset_name(spec::ExperimentSpec, config::NamedTuple)
    if experiment_has_function(spec, :training_dataset_name)
        return string(experiment_call(spec, :training_dataset_name, config))
    end

    training_seed = experiment_config_value(config, :training_data_seed, nothing)
    return isnothing(training_seed) ? spec.name : string(spec.name, "-", Int(training_seed))
end

function training_data_identity(spec::ExperimentSpec, config::NamedTuple)
    if experiment_has_function(spec, :training_data_identity)
        return experiment_call(spec, :training_data_identity, config)
    end

    return (;
        experiment_id=spec.id,
        experiment_name=spec.name,
        training_data_seed=experiment_config_value(config, :training_data_seed, missing),
    )
end

function training_data_identity_digest(spec::ExperimentSpec, config::NamedTuple)
    return experiment_dataset_digest(training_data_identity(spec, config))
end

function training_data_digest_slug(digest)
    text = replace(string(digest), r"[^A-Za-z0-9]+" => "-")
    return text[1:min(lastindex(text), 24)]
end

function safe_training_dataset_name(name)
    return replace(string(name), r"[^A-Za-z0-9_.=-]+" => "_")
end

function training_data_path(spec::ExperimentSpec, config::NamedTuple, name, identity_digest)
    filename = string(
        safe_training_dataset_name(name),
        "_",
        training_data_digest_slug(identity_digest),
        ".jls",
    )
    return joinpath(training_data_dir(spec, config), filename)
end

function load_or_generate_training_data(spec::ExperimentSpec, config::NamedTuple)
    name = training_dataset_name(spec, config)
    identity = training_data_identity(spec, config)
    identity_digest = experiment_dataset_digest(identity)
    path = training_data_path(spec, config, name, identity_digest)

    if !training_data_cache_enabled(config)
        dataset = experiment_call(spec, :training_data, config)
        return training_data_artifact(
            dataset;
            path="",
            name=name,
            identity_digest=identity_digest,
            cache_hit=false,
        )
    end

    isfile(path) && return load_training_data_artifact(spec, path, identity_digest; cache_hit=true)

    mkpath(dirname(path))
    return with_training_data_lock(path, config) do
        if isfile(path)
            load_training_data_artifact(spec, path, identity_digest; cache_hit=true)
        else
            dataset = experiment_call(spec, :training_data, config)
            save_training_data_artifact!(
                spec,
                path,
                dataset;
                name=name,
                identity=identity,
                identity_digest=identity_digest,
            )
            load_training_data_artifact(spec, path, identity_digest; cache_hit=false)
        end
    end
end

function with_training_data_lock(callback, path::AbstractString, config::NamedTuple)
    lock_dir = path * ".lock"
    timeout_seconds = Float64(
        experiment_config_value(config, :training_data_lock_timeout_seconds, 600.0),
    )
    started = time()
    acquired = false

    while !acquired
        try
            mkdir(lock_dir)
            acquired = true
        catch error
            if isdir(lock_dir)
                time() - started <= timeout_seconds ||
                    throw(ErrorException("timed out waiting for training data cache lock: $lock_dir"))
                sleep(0.25)
            else
                rethrow()
            end
        end
    end

    try
        return callback()
    finally
        rm(lock_dir; force=true, recursive=true)
    end
end

function save_training_data_artifact!(
    spec::ExperimentSpec,
    path::AbstractString,
    dataset;
    name,
    identity,
    identity_digest,
)
    mkpath(dirname(path))
    payload = (;
        format_version=TRAINING_DATA_FORMAT_VERSION,
        experiment_id=spec.id,
        experiment_name=spec.name,
        dataset_name=string(name),
        generated_at=Dates.format(now(UTC), dateformat"yyyy-mm-ddTHH:MM:SS.sZ"),
        training_data_identity=identity,
        training_data_identity_digest=string(identity_digest),
        dataset_digest=experiment_dataset_digest(dataset),
        dataset=dataset,
    )
    atomic_serialize(path, payload)
    return path
end

function load_training_data_artifact(
    spec::ExperimentSpec,
    path::AbstractString,
    identity_digest;
    cache_hit::Bool,
)
    payload = open(Serialization.deserialize, path)
    validate_training_data_payload(spec, payload, path, identity_digest)
    return training_data_artifact(
        training_data_from_payload(payload);
        path=path,
        name=payload.dataset_name,
        identity_digest=payload.training_data_identity_digest,
        dataset_digest=payload.dataset_digest,
        cache_hit=cache_hit,
    )
end

function training_data_from_payload(payload)
    payload isa NamedTuple && hasproperty(payload, :dataset) && return payload.dataset
    throw(ArgumentError("training data artifact payload must contain a dataset field."))
end

function validate_training_data_payload(
    spec::ExperimentSpec,
    payload,
    path::AbstractString,
    identity_digest,
)
    payload isa NamedTuple ||
        throw(ArgumentError("training data artifact $path must contain a metadata payload."))
    hasproperty(payload, :format_version) && payload.format_version == TRAINING_DATA_FORMAT_VERSION ||
        throw(ArgumentError("unsupported training-data format in $path"))
    String(payload.experiment_id) == spec.id ||
        throw(ArgumentError("training data artifact $path belongs to experiment $(payload.experiment_id), expected $(spec.id)."))
    String(payload.training_data_identity_digest) == string(identity_digest) ||
        throw(ArgumentError("training data artifact $path does not match the current training data identity."))

    dataset = training_data_from_payload(payload)
    expected = experiment_dataset_digest(dataset)
    String(payload.dataset_digest) == expected ||
        throw(ArgumentError("training data artifact $path has an invalid dataset digest."))
    return nothing
end

function training_data_artifact(
    dataset;
    path,
    name,
    identity_digest,
    dataset_digest=experiment_dataset_digest(dataset),
    cache_hit,
)
    metadata = (;
        path=string(path),
        dataset_name=string(name),
        dataset_digest=string(dataset_digest),
        training_data_identity_digest=string(identity_digest),
        cache_hit=Bool(cache_hit),
    )
    return (; dataset=dataset, metadata=metadata)
end

function with_training_data_metadata(objects, artifact)
    objects isa NamedTuple ||
        throw(ArgumentError("training objects must be a NamedTuple when using central training_data memoization."))

    existing_metadata = if hasproperty(objects, :data_metadata) &&
                           getproperty(objects, :data_metadata) isa NamedTuple
        getproperty(objects, :data_metadata)
    else
        NamedTuple()
    end

    metadata = artifact.metadata
    path = metadata.path
    central_metadata = (;
        dataset_name=metadata.dataset_name,
        dataset_digest=metadata.dataset_digest,
        dataset_path=path,
        dataset_source=isempty(path) ? metadata.training_data_identity_digest : path,
        dataset_identity_digest=metadata.training_data_identity_digest,
        training_data_artifact=path,
        training_data_cache_hit=metadata.cache_hit,
    )

    return merge(
        objects,
        (;
            data_metadata=merge(existing_metadata, central_metadata),
            training_data_artifact=metadata,
        ),
    )
end

function optimality_splits_for_config(objects, config::NamedTuple)
    spec = experiment_from_config(config)
    spec === nothing && throw(
        ArgumentError(
            "config must declare experiment_id, experiment_module_name, experiment_name, or experiment_config_path to select optimality splits.",
        ),
    )
    return experiment_call(spec, :optimality_splits, objects, with_experiment_metadata(spec, config))
end

function experiment_test_data_config(
    spec::ExperimentSpec;
    seed::Integer=DEFAULT_TEST_DATA_SEED,
    data_set_size::Integer=DEFAULT_TEST_DATA_SET_SIZE,
    overrides...,
)
    if experiment_has_function(spec, :test_data_config)
        return with_experiment_metadata(
            spec,
            experiment_call(
                spec,
                :test_data_config;
                seed=Int(seed),
                data_set_size=Int(data_set_size),
                overrides...,
            ),
        )
    end

    return merge(
        experiment_base_config(spec),
        (;
            seed=Int(seed),
            test_data_seed=Int(seed),
            data_set_size=Int(data_set_size),
        ),
        NamedTuple(overrides),
    )
end

function experiment_test_data_bundle(
    spec::ExperimentSpec;
    seed::Integer=DEFAULT_TEST_DATA_SEED,
    data_set_size::Integer=DEFAULT_TEST_DATA_SET_SIZE,
    overrides...,
)
    config = experiment_test_data_config(
        spec;
        seed=seed,
        data_set_size=data_set_size,
        overrides...,
    )
    if experiment_has_function(spec, :test_data_bundle)
        return experiment_call(
            spec,
            :test_data_bundle,
            config;
            seed=Int(seed),
            data_set_size=Int(data_set_size),
        )
    end

    throw(
        ArgumentError(
            "experiment $(spec.id) does not define test_data_bundle; generated test data must be owned by the experiment config.",
        ),
    )
end

function test_data_dir(spec::ExperimentSpec)
    if experiment_has_function(spec, :test_data_dir)
        return abspath(String(experiment_call(spec, :test_data_dir)))
    end
    return joinpath(experiment_artifact_dir(spec), "test_data")
end

function test_data_path(spec::ExperimentSpec, seed::Integer=DEFAULT_TEST_DATA_SEED)
    if experiment_has_function(spec, :test_data_path)
        return abspath(String(experiment_call(spec, :test_data_path, Int(seed))))
    end
    return joinpath(test_data_dir(spec), "test_data_seed$(Int(seed)).jls")
end

function test_optimal_results_path(
    spec::ExperimentSpec,
    seed::Integer=DEFAULT_TEST_DATA_SEED,
)
    if experiment_has_function(spec, :test_optimal_results_path)
        return abspath(String(experiment_call(spec, :test_optimal_results_path, Int(seed))))
    end
    return joinpath(test_data_dir(spec), "optimal_solutions_seed$(Int(seed)).jls")
end

function optimal_results_path(spec::ExperimentSpec, split_name::Symbol)
    return abspath(String(experiment_call(spec, :optimal_results_path, split_name)))
end

function optimal_results_path(config::NamedTuple, split_name::Symbol)
    spec = experiment_from_config(config)
    spec === nothing && throw(
        ArgumentError(
            "config must declare experiment_id, experiment_module_name, or experiment_name to locate optimal results.",
        ),
    )
    return optimal_results_path(spec, split_name)
end

function load_optimal_results(
    spec::ExperimentSpec,
    split_name::Symbol;
    dataset=nothing,
)
    if split_name == :test && uses_generated_test_data(spec)
        test_artifact = load_test_data_artifact(spec)
        test_seeds = test_artifact.metadata.seeds
        dataset_digests_by_seed = Dict(
            seed => digest for
            (seed, digest) in zip(test_seeds, test_artifact.metadata.dataset_digests)
        )
        paths = test_artifact_paths(spec, "optimal_solutions")
        isempty(paths) && throw(
            ArgumentError(
                "missing generated optimal-results artifacts for experiment $(spec.id) in $(test_data_dir(spec)). Run ContextualDFLTraining/generate_test_data.jl --experiment $(spec.id) first.",
            ),
        )

        results_by_seed = Dict{Int,Any}()
        for path in paths
            payload = open(Serialization.deserialize, path)
            hasproperty(payload, :test_data_seed) ||
                throw(ArgumentError("optimal-results artifact $path is missing test_data_seed."))
            hasproperty(payload, :data_set_size) ||
                throw(ArgumentError("optimal-results artifact $path is missing data_set_size."))
            hasproperty(payload, :dataset_digest) ||
                throw(ArgumentError("optimal-results artifact $path is missing dataset_digest."))

            seed = Int(payload.test_data_seed)
            haskey(results_by_seed, seed) &&
                throw(ArgumentError("duplicate optimal-results artifact for seed $seed."))
            haskey(dataset_digests_by_seed, seed) || throw(
                ArgumentError(
                    "optimal-results artifact $path has seed $seed, but no matching test-data artifact was loaded.",
                ),
            )

            results = optimal_results_from_payload(payload)
            length(results) == Int(payload.data_set_size) ||
                throw(ArgumentError("optimal-results artifact $path has $(length(results)) rows, expected $(payload.data_set_size)."))
            String(payload.dataset_digest) == dataset_digests_by_seed[seed] || throw(
                ArgumentError(
                    "optimal-results artifact $path does not match test-data artifact seed $seed. Regenerate it with generate_test_data.jl.",
                ),
            )
            validate_optimal_results_payload(spec, split_name, nothing, payload, path)
            results_by_seed[seed] = results
        end

        missing_seeds = setdiff(Set(test_seeds), Set(keys(results_by_seed)))
        isempty(missing_seeds) || throw(
            ArgumentError(
                "missing optimal-results artifact(s) for test data seed(s): $(join(sort!(collect(missing_seeds)), ", ")).",
            ),
        )

        results = vcat([results_by_seed[seed] for seed in test_seeds]...)
        if dataset !== nothing
            full_dataset = test_artifact.dataset
            if length(dataset) == length(full_dataset)
                experiment_dataset_digest(dataset) == experiment_dataset_digest(full_dataset) ||
                    throw(ArgumentError("optimal-results artifacts do not match the current dataset for split test. Regenerate them with generate_test_data.jl."))
            elseif length(dataset) < length(full_dataset)
                prefix_dataset = full_dataset[1:length(dataset)]
                experiment_dataset_digest(dataset) == experiment_dataset_digest(prefix_dataset) ||
                    throw(ArgumentError("limited test dataset must be a prefix of the generated test-data artifacts."))
                return results[1:length(dataset)]
            else
                throw(DimensionMismatch("test dataset has $(length(dataset)) rows, but generated optimal-results artifacts cover $(length(full_dataset)) rows."))
            end
        end
        return results
    end

    path = optimal_results_path(spec, split_name)
    isfile(path) || throw(
        ArgumentError(
            "missing precomputed optimal results for experiment $(spec.id), split $(split_name). Expected $path. Run ContextualDFLTraining/generate_test_data.jl for this experiment first.",
        ),
    )

    payload = open(Serialization.deserialize, path)
    results = optimal_results_from_payload(payload)
    validate_optimal_results_payload(spec, split_name, dataset, payload, path)
    return results
end

function load_optimal_results(config::NamedTuple, split_name::Symbol; dataset=nothing)
    spec = experiment_from_config(config)
    spec === nothing && throw(
        ArgumentError(
            "optimality_evaluation=true requires config.experiment_id so precomputed optimal results can be loaded.",
        ),
    )
    return load_optimal_results(spec, split_name; dataset=dataset)
end

function uses_generated_test_data(spec::ExperimentSpec)
    return experiment_has_function(spec, :test_data_config) ||
           experiment_has_function(spec, :test_data_bundle) ||
           experiment_has_function(spec, :test_data_dir) ||
           experiment_has_function(spec, :test_data_path) ||
           experiment_has_function(spec, :test_optimal_results_path)
end

function test_artifact_paths(spec::ExperimentSpec, prefix::AbstractString)
    dir = test_data_dir(spec)
    isdir(dir) || return String[]
    pattern = Regex("^" * prefix * "_seed([0-9]+)\\.jls\$")
    paths = [
        joinpath(dir, name) for name in readdir(dir) if occursin(pattern, name)
    ]
    return sort!(
        paths;
        by=path -> parse(Int, only(match(pattern, basename(path)).captures)),
    )
end

function save_test_data!(
    spec::ExperimentSpec,
    seed::Integer,
    dataset;
    data_set_size::Integer=length(dataset),
    metadata=NamedTuple(),
)
    seed = Int(seed)
    data_set_size = Int(data_set_size)
    data_set_size > 0 || throw(ArgumentError("data_set_size must be positive."))
    length(dataset) == data_set_size ||
        throw(ArgumentError("test dataset length $(length(dataset)) != data_set_size $data_set_size."))

    path = test_data_path(spec, seed)
    mkpath(dirname(path))
    payload = merge(
        metadata,
        (;
            format_version=TEST_DATA_FORMAT_VERSION,
            experiment_id=spec.id,
            experiment_name=spec.name,
            split_name=:test,
            generated_at=Dates.format(now(UTC), dateformat"yyyy-mm-ddTHH:MM:SS.sZ"),
            test_data_seed=seed,
            data_set_size=data_set_size,
            dataset_digest=experiment_dataset_digest(dataset),
            dataset=dataset,
        ),
    )
    atomic_serialize(path, payload)
    return path
end

function load_test_data_artifact(spec::ExperimentSpec)
    paths = test_artifact_paths(spec, "test_data")
    isempty(paths) && throw(
        ArgumentError(
            "missing generated test data artifacts for experiment $(spec.id) in $(test_data_dir(spec)). Run ContextualDFLTraining/generate_test_data.jl --experiment $(spec.id) first.",
        ),
    )

    datasets = Any[]
    seeds = Int[]
    data_set_sizes = Int[]
    dataset_digests = String[]
    expected_data_set_size = nothing
    expected_context_dimension = nothing
    expected_scenarios_per_context = nothing

    for path in paths
        payload = open(Serialization.deserialize, path)
        dataset = test_data_from_payload(payload)
        validate_test_data_payload(spec, dataset, payload, path)
        isempty(dataset) &&
            throw(ArgumentError("test data artifact $path must not contain an empty dataset."))

        seed = Int(payload.test_data_seed)
        seed in seeds &&
            throw(ArgumentError("duplicate test data artifact for seed $seed."))
        data_set_size = Int(payload.data_set_size)
        context_dimension = length(first(dataset).context)
        scenarios_per_context = length(first(dataset).scenario_parameters)

        if expected_data_set_size === nothing
            expected_data_set_size = data_set_size
            expected_context_dimension = context_dimension
            expected_scenarios_per_context = scenarios_per_context
        elseif data_set_size != expected_data_set_size ||
               context_dimension != expected_context_dimension ||
               scenarios_per_context != expected_scenarios_per_context
            throw(
                ArgumentError(
                    "test data artifact $path has shape (rows=$data_set_size, context_dimension=$context_dimension, scenarios_per_context=$scenarios_per_context), expected (rows=$expected_data_set_size, context_dimension=$expected_context_dimension, scenarios_per_context=$expected_scenarios_per_context).",
                ),
            )
        end

        push!(datasets, dataset)
        push!(seeds, seed)
        push!(data_set_sizes, data_set_size)
        push!(dataset_digests, String(payload.dataset_digest))
    end

    dataset = vcat(datasets...)
    return (;
        dataset=dataset,
        metadata=(;
            path=join(paths, ","),
            paths=paths,
            seed=first(seeds),
            seeds=seeds,
            data_set_size=length(dataset),
            data_set_sizes=data_set_sizes,
            dataset_digest=experiment_dataset_digest(dataset),
            dataset_digests=dataset_digests,
        ),
    )
end

function load_test_data_artifact(config::NamedTuple)
    spec = experiment_from_config(config)
    spec === nothing && throw(
        ArgumentError(
            "loading generated test data requires config.experiment_id, experiment_module_name, or experiment_name.",
        ),
    )
    return load_test_data_artifact(spec)
end

load_test_data(spec::ExperimentSpec) = load_test_data_artifact(spec).dataset
load_test_data(config::NamedTuple) = load_test_data_artifact(config).dataset

function save_test_optimal_results!(
    spec::ExperimentSpec,
    seed::Integer,
    results;
    dataset,
    data_set_size::Integer=length(dataset),
    metadata=NamedTuple(),
)
    seed = Int(seed)
    data_set_size = Int(data_set_size)
    data_set_size > 0 || throw(ArgumentError("data_set_size must be positive."))
    length(results) == data_set_size ||
        throw(ArgumentError("optimal results length $(length(results)) != data_set_size $data_set_size."))
    path = test_optimal_results_path(spec, seed)
    mkpath(dirname(path))
    payload = merge(
        metadata,
        (;
            format_version=OPTIMAL_RESULTS_FORMAT_VERSION,
            experiment_id=spec.id,
            experiment_name=spec.name,
            split_name=:test,
            generated_at=Dates.format(now(UTC), dateformat"yyyy-mm-ddTHH:MM:SS.sZ"),
            test_data_seed=seed,
            data_set_size=data_set_size,
            dataset_digest=experiment_dataset_digest(dataset),
            optimal_results=results,
        ),
    )
    atomic_serialize(path, payload)
    return path
end

function save_optimal_results!(
    spec::ExperimentSpec,
    split_name::Symbol,
    results;
    dataset=nothing,
    metadata=NamedTuple(),
)
    path = optimal_results_path(spec, split_name)
    mkpath(dirname(path))
    payload = merge(
        metadata,
        (;
            format_version=OPTIMAL_RESULTS_FORMAT_VERSION,
            experiment_id=spec.id,
            experiment_name=spec.name,
            split_name=Symbol(split_name),
            generated_at=Dates.format(now(UTC), dateformat"yyyy-mm-ddTHH:MM:SS.sZ"),
            dataset_digest=dataset === nothing ? missing : experiment_dataset_digest(dataset),
            optimal_results=results,
        ),
    )

    atomic_serialize(path, payload)
    return path
end

function atomic_serialize(path::AbstractString, payload)
    temp_path = tempname(dirname(path))
    open(temp_path, "w") do io
        Serialization.serialize(io, payload)
    end
    mv(temp_path, path; force=true)
    return path
end

function test_data_from_payload(payload)
    payload isa NamedTuple && hasproperty(payload, :dataset) && return payload.dataset
    throw(ArgumentError("test data artifact payload must contain a dataset field."))
end

function validate_test_data_payload(
    spec::ExperimentSpec,
    dataset,
    payload,
    path::AbstractString,
)
    payload isa NamedTuple ||
        throw(ArgumentError("test data artifact $path must contain a metadata payload."))

    if hasproperty(payload, :format_version)
        payload.format_version == TEST_DATA_FORMAT_VERSION ||
            throw(ArgumentError("unsupported test-data format in $path"))
    end
    if hasproperty(payload, :experiment_id)
        String(payload.experiment_id) == spec.id ||
            throw(ArgumentError("test data artifact $path belongs to experiment $(payload.experiment_id), expected $(spec.id)."))
    end
    if hasproperty(payload, :split_name)
        Symbol(payload.split_name) == :test ||
            throw(ArgumentError("test data artifact $path belongs to split $(payload.split_name), expected test."))
    end
    hasproperty(payload, :test_data_seed) ||
        throw(ArgumentError("test data artifact $path is missing test_data_seed."))
    hasproperty(payload, :data_set_size) ||
        throw(ArgumentError("test data artifact $path is missing data_set_size."))
    hasproperty(payload, :dataset_digest) ||
        throw(ArgumentError("test data artifact $path is missing dataset_digest."))
    length(dataset) == Int(payload.data_set_size) ||
        throw(ArgumentError("test data artifact $path has $(length(dataset)) rows, expected $(payload.data_set_size)."))
    expected = experiment_dataset_digest(dataset)
    String(payload.dataset_digest) == expected ||
        throw(ArgumentError("test data artifact $path has an invalid dataset digest."))

    return nothing
end

function optimal_results_from_payload(payload)
    if payload isa NamedTuple && hasproperty(payload, :optimal_results)
        return payload.optimal_results
    end
    return payload
end

function validate_optimal_results_payload(
    spec::ExperimentSpec,
    split_name::Symbol,
    dataset,
    payload,
    path::AbstractString,
)
    payload isa NamedTuple || return nothing

    if hasproperty(payload, :format_version)
        payload.format_version == OPTIMAL_RESULTS_FORMAT_VERSION ||
            throw(ArgumentError("unsupported optimal-results format in $path"))
    end
    if hasproperty(payload, :experiment_id)
        String(payload.experiment_id) == spec.id ||
            throw(ArgumentError("optimal-results artifact $path belongs to experiment $(payload.experiment_id), expected $(spec.id)."))
    end
    if hasproperty(payload, :split_name)
        Symbol(payload.split_name) == split_name ||
            throw(ArgumentError("optimal-results artifact $path belongs to split $(payload.split_name), expected $(split_name)."))
    end
    if dataset !== nothing && hasproperty(payload, :dataset_digest) &&
       payload.dataset_digest !== missing
        expected = experiment_dataset_digest(dataset)
        String(payload.dataset_digest) == expected ||
            throw(ArgumentError("optimal-results artifact $path does not match the current dataset for split $(split_name). Regenerate it with generate_optimal_solutions.jl."))
    end

    return nothing
end

function experiment_dataset_digest(dataset)
    io = IOBuffer()
    Serialization.serialize(io, dataset)
    return "sha1:" * bytes2hex(sha1(take!(io)))
end

# END FILE: src/ContextualDFL/ContextualDFLTraining/src/experiments/ExperimentAPI.jl

# BEGIN FILE: src/ContextualDFL/ContextualDFLTraining/src/experiments/resource_allocation/experiment_1/Config.jl
import ContextualDFL
import ContextualDFLExperiments
import ContextualDFLTraining
import Random
import Serialization
import SHA

const EXPERIMENT_ID = "resource_allocation/experiment_1"
const EXPERIMENT_NAME = "resource_allocation_experiment_1"
const DEFAULT_TEST_DATA_SEED = 1
const DEFAULT_TEST_DATA_SET_SIZE = 30

const DEFAULT_TRAINING_CONTEXT_COUNT = 150
const DEFAULT_TRAINING_SCENARIOS_PER_CONTEXT = 1
const DEFAULT_COLLECTION_DUPLICATES_PER_CONTEXT = 1
const DEFAULT_VALIDATION_FRACTION = 0.13333333333333333
const DEFAULT_GENERATED_SPLIT_TEST_FRACTION = 0.20
const DEFAULT_TEST_SCENARIOS_PER_CONTEXT = 100
const DEFAULT_TRAINING_DATA_SEED = 1

const DEMAND_SIGMA = 5.0
const DEMAND_POWER = 2.0
const CONTEXT_TERMS = 3

experiment_id() = EXPERIMENT_ID
experiment_name() = EXPERIMENT_NAME
experiment_module_name() = :ResourceAllocationExperiment1
artifact_dir() = joinpath(@__DIR__, "artifacts")
test_data_dir() = joinpath(artifact_dir(), "test_data")
test_data_path(seed::Integer=DEFAULT_TEST_DATA_SEED) =
    joinpath(test_data_dir(), "test_data_seed$(Int(seed)).jls")
test_optimal_results_path(seed::Integer=DEFAULT_TEST_DATA_SEED) =
    joinpath(test_data_dir(), "optimal_solutions_seed$(Int(seed)).jls")

function experiment_overrides(; overrides...)
    return merge(
        (;
            experiment_id=EXPERIMENT_ID,
            experiment_name=EXPERIMENT_NAME,
            problem=:resource_allocation,
            mlflow_dataset_name=EXPERIMENT_NAME,
        ),
        NamedTuple(overrides),
    )
end

function problem_config()
    return (;
        problem=:resource_allocation,
        solver=:highs,
    )
end

function problem_identity_config()
    return (;
        problem=:resource_allocation,
        demand_sigma=DEMAND_SIGMA,
        sigma=DEMAND_SIGMA,
        demand_power=DEMAND_POWER,
        context_terms=CONTEXT_TERMS,
    )
end

function training_data_defaults()
    return (;
        training_context_count=DEFAULT_TRAINING_CONTEXT_COUNT,
        training_scenarios_per_context=DEFAULT_TRAINING_SCENARIOS_PER_CONTEXT,
        collection_duplicates_per_context=DEFAULT_COLLECTION_DUPLICATES_PER_CONTEXT,
        validation_fraction=DEFAULT_VALIDATION_FRACTION,
        generated_split_test_fraction=DEFAULT_GENERATED_SPLIT_TEST_FRACTION,
        training_data_seed=DEFAULT_TRAINING_DATA_SEED,
    )
end

function base_config(; overrides...)
    return merge(
        ContextualDFLTraining.DEFAULT_RUN_SETTINGS,
        problem_config(),
        training_data_defaults(),
        experiment_overrides(; overrides...),
    )
end

function test_data_config(;
    seed=DEFAULT_TEST_DATA_SEED,
    data_set_size=DEFAULT_TEST_DATA_SET_SIZE,
    overrides...,
)
    return base_config(;
        seed=Int(seed),
        test_data_seed=Int(seed),
        data_set_size=Int(data_set_size),
        overrides...,
    )
end

function profile_config(; overrides...)
    return merge(
        base_config(;
            epochs=100,
            warmup_epochs=2,
            mu=1.0,
            mu_start=1.0,
            mu_end=1.0,
            mu_schedule=:constant,
            learning_rate=1e-3,
            hidden_size=128,
            depth=2,
            batch_size=64,
            dropout=0.0,
            seed=3,
            run_id="profile_standard_seed3",
            base_run_id="profile_standard_seed3",
            candidate_name="profile_standard_seed3",
            method_variant="profiling",
            overrides...,
        ),
        (;
            mlflow_dataset_context="profiling",
            mlflow_source_name="ContextualDFLTraining/profile_training.jl",
            mlflow_params=(;
                profile_target="ContextualDFL.train!",
                profile_loss="ContextualDFL.DflScenLoss",
                profile_progress_logged_by="remote_worker",
            ),
        ),
    )
end

problem_data() = ContextualDFLExperiments.default_resource_allocation_problem_data()

ResourceAllocationProblem(
    data::ContextualDFLExperiments.ResourceAllocationProblemData=problem_data(),
) = ContextualDFLExperiments.ResourceAllocationProblem(data)

problem() = ResourceAllocationProblem(problem_data())

program(problem_instance=problem()) = ContextualDFLExperiments.stochastic_program(problem_instance)

solver(config=base_config()) = ContextualDFLTraining.build_solver(config)

scenario_decoder(problem_instance=problem()) =
    ContextualDFLExperiments.ResourceAllocationDemandVectorDecoder(problem_instance)

reference_scenario_decoder(problem_instance=problem()) =
    ContextualDFLExperiments.ResourceAllocationDemandParametricDecoder(problem_instance)

ContextDataGenerator(; rng::Random.AbstractRNG=Random.default_rng()) =
    ContextualDFLExperiments.ResourceAllocationContextDataGenerator(rng=rng)

function ScenarioDataGenerator(
    problem_instance=problem();
    rng::Random.AbstractRNG=Random.default_rng(),
)
    return ContextualDFLExperiments.ResourceAllocationScenarioDataGenerator(
        problem_instance;
        sigma=DEMAND_SIGMA,
        p=DEMAND_POWER,
        L=CONTEXT_TERMS,
        rng=rng,
    )
end

function problem_objects(config=base_config())
    problem_instance = problem()
    return (;
        problem=problem_instance,
        program=program(problem_instance),
        solver=solver(config),
        scenario_decoder=scenario_decoder(problem_instance),
        reference_scenario_decoder=reference_scenario_decoder(problem_instance),
    )
end

function demand_count(problem_instance)
    return size(problem_instance.problem_data.service_rate_parameters, 2)
end

function resource_count(problem_instance)
    return size(problem_instance.problem_data.service_rate_parameters, 1)
end

function training_context_count(config)
    return Int(
        ContextualDFLTraining.config_value(
            config,
            :training_context_count,
            ContextualDFLTraining.config_value(
                config,
                :Nr_contexts,
                DEFAULT_TRAINING_CONTEXT_COUNT,
            ),
        ),
    )
end

function training_scenarios_per_context(config)
    return Int(
        ContextualDFLTraining.config_value(
            config,
            :training_scenarios_per_context,
            ContextualDFLTraining.config_value(
                config,
                :scenarios_per_context,
                ContextualDFLTraining.config_value(
                    config,
                    :nr_scenarios,
                    DEFAULT_TRAINING_SCENARIOS_PER_CONTEXT,
                ),
            ),
        ),
    )
end

function collection_duplicates_count(config)
    return Int(
        ContextualDFLTraining.config_value(
            config,
            :collection_duplicates_per_context,
            DEFAULT_COLLECTION_DUPLICATES_PER_CONTEXT,
        ),
    )
end

function validation_split_fraction(config)
    return Float64(
        ContextualDFLTraining.config_value(
            config,
            :validation_fraction,
            DEFAULT_VALIDATION_FRACTION,
        ),
    )
end

function generated_split_test_fraction(config)
    return Float64(
        ContextualDFLTraining.config_value(
            config,
            :generated_split_test_fraction,
            ContextualDFLTraining.config_value(
                config,
                :test_fraction,
                DEFAULT_GENERATED_SPLIT_TEST_FRACTION,
            ),
        ),
    )
end

function test_scenarios_per_context(config)
    return Int(
        ContextualDFLTraining.config_value(
            config,
            :test_scenarios_per_context,
            ContextualDFLTraining.config_value(
                config,
                :scenarios_per_context,
                ContextualDFLTraining.config_value(
                    config,
                    :nr_scenarios,
                    DEFAULT_TEST_SCENARIOS_PER_CONTEXT,
                ),
            ),
        ),
    )
end

function dataset_scenarios_per_context(dataset, config)
    isempty(dataset) && return training_scenarios_per_context(config)
    return length(first(dataset).scenario_parameters)
end

function generate_dataset(
    problem_instance,
    rng::Random.AbstractRNG;
    context_count::Integer=DEFAULT_TRAINING_CONTEXT_COUNT,
    scenario_count::Integer=DEFAULT_TRAINING_SCENARIOS_PER_CONTEXT,
    duplicates_per_context::Integer=DEFAULT_COLLECTION_DUPLICATES_PER_CONTEXT,
)
    context_count > 0 || throw(ArgumentError("context_count must be positive."))
    scenario_count > 0 || throw(ArgumentError("scenario_count must be positive."))
    duplicates_per_context > 0 ||
        throw(ArgumentError("duplicates_per_context must be positive."))

    context_generator = ContextDataGenerator(rng=rng)
    scenario_generator = ScenarioDataGenerator(problem_instance; rng=rng)

    contexts = Vector{Vector{Float64}}()
    scenario_collections = Vector{Vector{ContextualDFL.ParametricScenario}}()

    for _ in 1:Int(context_count)
        context = Vector{Float64}(context_generator())
        for _ in 1:Int(duplicates_per_context)
            push!(contexts, copy(context))
            push!(
                scenario_collections,
                [scenario_generator(context) for _ in 1:Int(scenario_count)],
            )
        end
    end

    return ContextualDFLExperiments.generate_contextual_data_set(
        contexts,
        scenario_collections,
    )
end

function generated_training_splits(
    problem_instance,
    config,
    rng::Random.AbstractRNG;
    test_fraction::Real=0.0,
)
    dataset = generate_dataset(
        problem_instance,
        rng;
        context_count=training_context_count(config),
        scenario_count=training_scenarios_per_context(config),
        duplicates_per_context=collection_duplicates_count(config),
    )
    return ContextualDFLTraining.split_contextual_dataset(
        dataset;
        validation_fraction=validation_split_fraction(config),
        test_fraction=Float64(test_fraction),
        rng=rng,
    )
end

function generated_test_artifact(config)
    Bool(ContextualDFLTraining.config_value(config, :use_generated_test_data_artifact, true)) ||
        return nothing
    spec = ContextualDFLTraining.experiment_from_config(config)
    spec === nothing && return nothing
    isempty(ContextualDFLTraining.test_artifact_paths(spec, "test_data")) && return nothing
    return ContextualDFLTraining.load_test_data_artifact(spec)
end

function test_data_artifact_metadata(config)
    artifact = generated_test_artifact(config)
    return artifact === nothing ? (; source=:generated_split) : artifact.metadata
end

function data_splits(problem_instance, config, rng::Random.AbstractRNG)
    test_artifact = generated_test_artifact(config)
    split_test_fraction =
        test_artifact === nothing ? generated_split_test_fraction(config) : 0.0
    generated_splits = generated_training_splits(
        problem_instance,
        config,
        rng;
        test_fraction=split_test_fraction,
    )
    return (;
        train=generated_splits.train,
        validation=generated_splits.validation,
        test=test_artifact === nothing ? generated_splits.test : test_artifact.dataset,
    )
end

function training_data_seed(config)
    return Int(
        ContextualDFLTraining.config_value(
            config,
            :training_data_seed,
            DEFAULT_TRAINING_DATA_SEED,
        ),
    )
end

function training_data(config)
    rng = Random.MersenneTwister(training_data_seed(config))
    return data_splits(problem(), config, rng)
end

function training_dataset_name(config)
    explicit_name =
        ContextualDFLTraining.config_value(config, :training_dataset_name, nothing)
    isnothing(explicit_name) || return string(explicit_name)
    return join(
        (
            EXPERIMENT_NAME,
            "ctx$(training_context_count(config))",
            "scen$(training_scenarios_per_context(config))",
            "dup$(collection_duplicates_count(config))",
            "training_seed$(training_data_seed(config))",
        ),
        "-",
    )
end

function demand_from_scenario(scenario::ContextualDFL.ParametricScenario)
    isempty(scenario.h_eq_xi) &&
        throw(ArgumentError("expected a demand vector in h_eq_xi"))
    return scenario.h_eq_xi
end

function target_from_contextual_point(point)
    isempty(point.scenario_parameters) &&
        throw(ArgumentError("contextual data point has no scenario parameters."))
    return reduce(vcat, (demand_from_scenario(scenario) for scenario in point.scenario_parameters))
end

function problem_instance_id(problem_instance)
    values = (
        "service_rate=$(vec(problem_instance.problem_data.service_rate_parameters))",
        "first_stage=$(problem_instance.problem_data.first_stage_costs)",
        "second_stage=$(problem_instance.problem_data.second_stage_costs)",
        "yield=$(problem_instance.problem_data.yield_parameters)",
        "demand_sigma=$(DEMAND_SIGMA)",
        "demand_power=$(DEMAND_POWER)",
        "context_terms=$(CONTEXT_TERMS)",
    )
    return "sha256:" * SHA.bytes2hex(SHA.sha256(join(values, "\n")))
end

function serialized_digest(value)
    io = IOBuffer()
    Serialization.serialize(io, value)
    return "sha1:" * SHA.bytes2hex(SHA.sha1(take!(io)))
end

function test_artifact_identity(config)
    artifact = generated_test_artifact(config)
    artifact === nothing && return (; source=:generated_split)
    return (;
        source=:artifact,
        path=artifact.metadata.path,
        seed=artifact.metadata.seed,
        data_set_size=artifact.metadata.data_set_size,
        dataset_digest=artifact.metadata.dataset_digest,
    )
end

function training_data_identity(config)
    test_artifact = test_artifact_identity(config)
    identity = (;
        experiment_id=EXPERIMENT_ID,
        problem_instance_id=problem_instance_id(problem()),
        training_data_seed=training_data_seed(config),
        training_context_count=training_context_count(config),
        training_scenarios_per_context=training_scenarios_per_context(config),
        collection_duplicates_per_context=collection_duplicates_count(config),
        validation_fraction=validation_split_fraction(config),
        demand_sigma=DEMAND_SIGMA,
        demand_power=DEMAND_POWER,
        context_terms=CONTEXT_TERMS,
        test_artifact=test_artifact,
    )
    return test_artifact.source == :generated_split ?
        merge(
            identity,
            (; generated_split_test_fraction=generated_split_test_fraction(config)),
        ) :
        identity
end

function problem_metadata(problem_instance)
    return (;
        problem="resource_allocation",
        instance_id=problem_instance_id(problem_instance),
        resource_count=resource_count(problem_instance),
        demand_count=demand_count(problem_instance),
        service_rate_shape=size(problem_instance.problem_data.service_rate_parameters),
        demand_sigma=DEMAND_SIGMA,
        demand_power=DEMAND_POWER,
        context_terms=CONTEXT_TERMS,
    )
end

function data_metadata(splits, config)
    test_artifact =
        ContextualDFLTraining.config_value(config, :test_data_artifact, NamedTuple())
    test_source = hasproperty(test_artifact, :path) ? :artifact : :generated_split
    dataset_recipe = join(
        (
            "ContextualDFLTraining.experiment",
            "experiment_id=$(EXPERIMENT_ID)",
            "training_data_seed=$(training_data_seed(config))",
            "training_context_count=$(training_context_count(config))",
            "training_scenarios_per_context=$(training_scenarios_per_context(config))",
            "collection_duplicates_per_context=$(collection_duplicates_count(config))",
            "validation_fraction=$(validation_split_fraction(config))",
            "test_source=$(test_source)",
        ),
        ";",
    )

    return (;
        generator="ContextualDFLExperiments.resource_allocation",
        dataset_recipe=dataset_recipe,
        train_size=length(splits.train),
        validation_size=length(splits.validation),
        test_size=length(splits.test),
        context_dimension=isempty(splits.train) ? 0 : length(first(splits.train).context),
        target_dimension=isempty(splits.train) ? 0 : length(target_from_contextual_point(first(splits.train))),
        training_context_count=training_context_count(config),
        training_scenarios_per_context=training_scenarios_per_context(config),
        collection_duplicates_per_context=collection_duplicates_count(config),
        validation_fraction=validation_split_fraction(config),
        generated_split_test_fraction=
            test_source == :generated_split ? generated_split_test_fraction(config) : 0.0,
        test_source=test_source,
        train_context_seed=training_data_seed(config),
        train_scenario_seed=training_data_seed(config),
        split_seed=training_data_seed(config),
        test_data_artifact=get(Dict(pairs(test_artifact)), :path, ""),
    )
end

function model_metadata(model, problem_instance, splits, config)
    scenario_count = dataset_scenarios_per_context(splits.train, config)
    return (;
        architecture="Flux.Chain",
        depth=Int(config.depth),
        width=Int(config.hidden_size),
        activation=string(ContextualDFLTraining.config_value(config, :activation, "relu")),
        output_activation="softplus",
        dropout=Float64(config.dropout),
        initialization_seed=Int(config.seed),
        input_dimension=isempty(splits.train) ? 0 : length(first(splits.train).context),
        output_dimension=demand_count(problem_instance) * scenario_count,
    )
end

function test_data_bundle(
    config;
    seed=DEFAULT_TEST_DATA_SEED,
    data_set_size=DEFAULT_TEST_DATA_SET_SIZE,
)
    data_set_size > 0 || throw(ArgumentError("data_set_size must be positive."))
    objects = problem_objects(config)
    rng = Random.MersenneTwister(Int(seed))
    dataset = generate_dataset(
        objects.problem,
        rng;
        context_count=Int(data_set_size),
        scenario_count=test_scenarios_per_context(config),
        duplicates_per_context=1,
    )
    return merge(
        objects,
        (;
            dataset=dataset,
            problem_metadata=problem_metadata(objects.problem),
            data_metadata=(;
                generator="ContextualDFLExperiments.resource_allocation",
                dataset_name=EXPERIMENT_NAME,
                dataset_digest=serialized_digest(dataset),
                dataset_source=join(
                    (
                        "ContextualDFLTraining.experiment_test_data",
                        "experiment_id=$(EXPERIMENT_ID)",
                        "seed=$(Int(seed))",
                        "data_set_size=$(Int(data_set_size))",
                    ),
                    ";",
                ),
                data_set_size=Int(data_set_size),
                context_dimension=isempty(dataset) ? 0 : length(first(dataset).context),
                target_dimension=isempty(dataset) ? 0 : length(target_from_contextual_point(first(dataset))),
                scenarios_per_context=test_scenarios_per_context(config),
            ),
        ),
    )
end

training_objects(config) = training_objects(config, training_data(config))

function training_objects(config, data)
    objects = problem_objects(config)
    test_data_artifact = test_data_artifact_metadata(config)
    scenario_count = dataset_scenarios_per_context(data.train, config)
    model_initialization_seed = Int(config.seed)
    effective_config = merge(
        config,
        (; nr_scenarios=scenario_count, model_initialization_seed=model_initialization_seed),
    )

    neural_net = ContextualDFLTraining.build_neural_net(
        length(first(data.train).context),
        demand_count(objects.problem) * scenario_count;
        hidden_size=Int(config.hidden_size),
        depth=Int(config.depth),
        dropout=Float64(config.dropout),
        activation=ContextualDFLTraining.config_value(config, :activation, :relu),
        seed=model_initialization_seed,
    )
    generator = ContextualDFL.ScenarioGenerator(
        neural_net=neural_net,
        scenario_decoder=objects.scenario_decoder,
    )
    loss = ContextualDFLTraining.build_loss(
        effective_config,
        objects.scenario_decoder,
        objects.reference_scenario_decoder,
        objects.solver,
        objects.program,
    )

    return merge(
        objects,
        (;
            loss=loss,
            scenario_generator=generator,
            data=data,
            target_extractor=ContextualDFLTraining.LatestExperimentFunction(
                @__MODULE__,
                :target_from_contextual_point,
            ),
            test_data_artifact=test_data_artifact,
            problem_metadata=problem_metadata(objects.problem),
            data_metadata=data_metadata(
                data,
                merge(config, (; test_data_artifact=test_data_artifact)),
            ),
            model_metadata=model_metadata(neural_net, objects.problem, data, effective_config),
            schedules=(;
                mu=ContextualDFLTraining.ConstantSchedule(config.mu),
                rho=ContextualDFLTraining.ConstantSchedule(config.rho),
                batch_size=ContextualDFLTraining.ConstantSchedule(config.batch_size),
                step_size=ContextualDFLTraining.ConstantSchedule(config.learning_rate),
            ),
        ),
    )
end

function optimality_splits(objects, config)
    return ContextualDFLTraining.optimality_evaluation_datasets(objects, config)
end

function optimal_results_path(split_name::Symbol)
    return joinpath(artifact_dir(), "optimal_solutions", string(split_name) * ".jls")
end

# END FILE: src/ContextualDFL/ContextualDFLTraining/src/experiments/resource_allocation/experiment_1/Config.jl

# BEGIN FILE: src/ContextualDFL/ContextualDFLTraining/src/grid_config.jl
const DEFAULT_GRID_VALUES = (;
    learning_rate=[1e-3, 5e-4, 3e-4],
    hidden_size=[64, 128],
    depth=[2, 3],
    batch_size=[8, 16],
    dropout=[0.0],
    seed=[143],
)

function _merge_settings(overrides::NamedTuple)
    return merge(DEFAULT_RUN_SETTINGS, overrides)
end

function _run_id(index, cfg)
    return "run_" * lpad(string(index), 4, "0") * "_seed" * string(cfg.seed)
end

function default_grid(; overrides...)
    settings = _merge_settings(NamedTuple(overrides))
    configs = NamedTuple[]
    index = 0

    for learning_rate in DEFAULT_GRID_VALUES.learning_rate,
        hidden_size in DEFAULT_GRID_VALUES.hidden_size,
        depth in DEFAULT_GRID_VALUES.depth,
        batch_size in DEFAULT_GRID_VALUES.batch_size,
        dropout in DEFAULT_GRID_VALUES.dropout,
        seed in DEFAULT_GRID_VALUES.seed

        index += 1
        cfg = merge(
            settings,
            (;
                learning_rate=Float64(learning_rate),
                hidden_size=Int(hidden_size),
                depth=Int(depth),
                batch_size=Int(batch_size),
                dropout=Float64(dropout),
                seed=Int(seed),
            ),
        )
        push!(configs, merge(cfg, (; run_id=_run_id(index, cfg))))
    end

    return configs
end

function smoke_grid(; overrides...)
    settings = _merge_settings((; epochs=2, overrides...))
    cfg = merge(
        settings,
        (;
            learning_rate=1e-3,
            hidden_size=16,
            depth=1,
            batch_size=4,
            dropout=0.0,
            seed=1,
        ),
    )
    return [merge(cfg, (; run_id="smoke_0001_seed1"))]
end

# END FILE: src/ContextualDFL/ContextualDFLTraining/src/grid_config.jl

# BEGIN FILE: src/ContextualDFL/ContextualDFLTraining/src/grid_file_config.jl
using Configurations: @option, from_dict
using JSON3
using SHA
using YAML

@option struct GridScheduleSegmentConfig
    epochs::Int
    value::Float64
end

@option struct GridScheduleConfig
    kind::String = ""
    type::String = ""
    start::Union{Nothing,Float64} = nothing
    stop::Union{Nothing,Float64} = nothing
    value::Union{Nothing,Float64} = nothing
    values::Union{Nothing,Vector{Float64}} = nothing
    segments::Vector{GridScheduleSegmentConfig} = GridScheduleSegmentConfig[]
end

@option struct GridFileConfig
    version::Int
    name::Union{Nothing,String} = nothing
    description::Union{Nothing,String} = nothing
    base::Dict{String,Any} = Dict{String,Any}()
    fixed::Dict{String,Any} = Dict{String,Any}()
    grid::Dict{String,Any} = Dict{String,Any}()
    schedules::Dict{String,GridScheduleConfig} = Dict{String,GridScheduleConfig}()
    run_id_template::Union{Nothing,String} = nothing
end

struct GridSearchSpec
    path::String
    format::Symbol
    version::Int
    name::String
    base::Dict{Symbol,Any}
    fixed::Dict{Symbol,Any}
    grid::Dict{Symbol,Vector{Any}}
    schedules::Dict{Symbol,Any}
    schedule_grid::Vector{Dict{Symbol,Any}}
    run_id_template::Union{Nothing,String}
    raw::Dict{String,Any}
end

const GRID_SYMBOL_KEYS = Set(
    [
        :activation,
        :checkpoint_format,
        :loss,
        :method,
        :mu_ref_schedule,
        :mu_schedule,
        :rho_ref_schedule,
        :rho_schedule,
    ],
)

const GRID_INT_KEYS = Set(
    [
        :batch_size,
        :candidate_index,
        :collection_duplicates_per_context,
        :depth,
        :display_real,
        :epochs,
        :hidden_size,
        :Nr_contexts,
        :optimality_evaluation_batches,
        :optimality_test_sample_count,
        :optimality_train_sample_count,
        :optimality_validation_sample_count,
        :replicate_index,
        :repeat_count,
        :scenarios_per_context,
        :seed,
        :test_scenarios_per_context,
        :training_context_count,
        :training_scenarios_per_context,
        :warmup_epochs,
    ],
)

const GRID_FLOAT_KEYS = Set(
    [
        :dropout,
        :generated_split_test_fraction,
        :learning_rate,
        :mu,
        :mu_end,
        :mu_ref,
        :mu_ref_end,
        :mu_ref_start,
        :mu_start,
        :optimality_mu,
        :optimality_rho,
        :policy_inference_mu,
        :policy_inference_rho,
        :rho,
        :rho_end,
        :rho_ref,
        :rho_ref_end,
        :rho_ref_start,
        :rho_start,
        :test_fraction,
        :tolerance_absolute_floor,
        :tolerance_relative,
        :validation_fraction,
    ],
)

const GRID_BOOL_KEYS = Set(
    [
        :annealing,
        :display_smooth,
        :fine_tuning,
        :knn_homogenization,
        :checkpoint_enabled,
        :checkpoint_required,
        :checkpoint_upload_mlflow,
        :log_barrier_inference,
        :mlflow_enabled,
        :mlflow_upload_model_artifact,
        :optimality_evaluation,
        :reset_optimizer_each_epoch,
        :shuffle,
    ],
)

function load_grid_config(path::AbstractString)
    absolute_path = abspath(path)
    isfile(absolute_path) ||
        throw(ArgumentError("grid config file does not exist: $absolute_path"))

    format = grid_config_format(absolute_path)
    raw = grid_config_data(read_grid_config_file(absolute_path, format))
    parsed = parse_grid_file_config(raw, absolute_path)

    version = parsed.version
    version == 1 ||
        throw(ArgumentError("unsupported grid config version $version in $absolute_path"))

    name = parsed.name === nothing ? splitext(basename(absolute_path))[1] : string(parsed.name)
    isempty(strip(name)) &&
        throw(ArgumentError("grid config name must not be empty in $absolute_path"))

    base = settings_section(parsed.base)
    fixed = settings_section(parsed.fixed)
    raw_grid = copy(parsed.grid)
    schedule_grid = schedule_grid_section(pop_grid_schedules!(raw_grid))
    grid = grid_section(raw_grid)
    schedules = schedule_settings(parsed.schedules)
    run_id_template = parsed.run_id_template === nothing ?
        nothing :
        string(parsed.run_id_template)

    return GridSearchSpec(
        absolute_path,
        format,
        version,
        name,
        base,
        fixed,
        grid,
        schedules,
        schedule_grid,
        run_id_template,
        raw,
    )
end

function grid_config_format(path::AbstractString)
    extension = lowercase(splitext(path)[2])
    extension in (".yaml", ".yml") && return :yaml
    extension == ".json" && return :json
    throw(ArgumentError("grid config must be .yaml, .yml, or .json, got: $path"))
end

function read_grid_config_file(path::AbstractString, format::Symbol)
    format == :yaml && return YAML.load_file(path)
    format == :json && return JSON3.read(read(path, String))
    throw(ArgumentError("unsupported grid config format $format"))
end

function grid_config_data(value)
    if value isa AbstractDict || value isa JSON3.Object
        output = Dict{String,Any}()
        for (key, item) in pairs(value)
            output[string(key)] = grid_config_data(item)
        end
        return output
    elseif value isa AbstractVector || value isa JSON3.Array
        return Any[grid_config_data(item) for item in value]
    elseif value isa AbstractString || value isa Number || value isa Bool ||
           value === nothing || value === missing
        return value
    end
    return string(value)
end

function parse_grid_file_config(raw, path::AbstractString)
    raw isa AbstractDict ||
        throw(ArgumentError("grid config $path must contain a mapping/object at the top level."))

    normalize_grid_schedule_aliases!(raw)
    try
        return from_dict(GridFileConfig, raw)
    catch error
        throw(ArgumentError("invalid grid config $path: $(sprint(showerror, error))"))
    end
end

function normalize_grid_schedule_aliases!(raw)
    normalize_schedule_aliases_in_section!(get(raw, "schedules", nothing))

    grid = get(raw, "grid", nothing)
    if grid isa AbstractDict
        grid_schedules = get(grid, "schedules", nothing)
        if grid_schedules isa AbstractDict
            for (_, schedules) in grid_schedules
                if schedules isa AbstractVector
                    for schedule in schedules
                        normalize_single_schedule_aliases!(schedule)
                    end
                else
                    normalize_single_schedule_aliases!(schedules)
                end
            end
        end
    end

    return raw
end

function normalize_schedule_aliases_in_section!(schedules)
    schedules isa AbstractDict || return schedules
    for (_, schedule) in schedules
        normalize_single_schedule_aliases!(schedule)
    end
    return schedules
end

function normalize_single_schedule_aliases!(schedule)
    schedule isa AbstractDict || return schedule
    if haskey(schedule, "end")
        haskey(schedule, "stop") || (schedule["stop"] = schedule["end"])
        delete!(schedule, "end")
    end
    return schedule
end

function pop_grid_schedules!(grid_values::Dict{String,Any})
    haskey(grid_values, "schedules") || return nothing
    schedules = grid_values["schedules"]
    delete!(grid_values, "schedules")
    return schedules
end

function settings_section(section)
    output = Dict{Symbol,Any}()
    for (setting_key, value) in section
        symbol_key = Symbol(setting_key)
        output[symbol_key] = normalize_grid_setting_value(symbol_key, value)
    end
    return output
end

function grid_section(grid_values)
    output = Dict{Symbol,Vector{Any}}()
    for (setting_key, values) in grid_values
        values isa AbstractVector ||
            throw(ArgumentError("grid entry '$setting_key' must be a non-empty array."))
        isempty(values) &&
            throw(ArgumentError("grid entry '$setting_key' must not be empty."))
        symbol_key = Symbol(setting_key)
        output[symbol_key] = [
            normalize_grid_setting_value(symbol_key, value) for value in values
        ]
    end
    return output
end

function normalize_grid_setting_value(key::Symbol, value)
    value === nothing && return nothing
    value === missing && return missing

    if key in GRID_SYMBOL_KEYS
        return Symbol(value)
    elseif key in GRID_INT_KEYS
        return Int(value)
    elseif key in GRID_FLOAT_KEYS
        return Float64(value)
    elseif key in GRID_BOOL_KEYS
        return Bool(value)
    elseif value isa AbstractDict
        return Dict(Symbol(k) => normalize_grid_setting_value(Symbol(k), v) for (k, v) in value)
    elseif value isa AbstractVector
        return Any[normalize_grid_setting_value(key, item) for item in value]
    end

    return value
end

function schedule_settings(schedules)
    output = Dict{Symbol,Any}()

    for (name, spec) in schedules
        schedule_name = Symbol(name)
        merge!(output, normalize_schedule(schedule_name, spec))
    end

    return output
end

function schedule_grid_section(schedules)
    schedules === nothing && return [Dict{Symbol,Any}()]
    schedules isa AbstractDict ||
        throw(ArgumentError("grid.schedules must be a mapping/object."))

    options = Dict{Symbol,Vector{Dict{Symbol,Any}}}()
    for (name, values) in schedules
        values isa AbstractVector ||
            throw(ArgumentError("grid.schedules entry '$name' must be a non-empty array."))
        isempty(values) &&
            throw(ArgumentError("grid.schedules entry '$name' must not be empty."))

        schedule_name = Symbol(name)
        options[schedule_name] = [
            normalize_schedule(schedule_name, grid_schedule_config(schedule_name, value)) for
            value in values
        ]
    end

    return schedule_grid_candidates(options)
end

function grid_schedule_config(schedule_name::Symbol, value)
    value isa GridScheduleConfig && return value
    value isa AbstractDict || throw(
        ArgumentError(
            "grid.schedules entry '$schedule_name' candidates must be schedule mappings/objects.",
        ),
    )

    string_keyed = Dict{String,Any}(string(key) => item for (key, item) in value)
    normalize_single_schedule_aliases!(string_keyed)
    try
        return from_dict(GridScheduleConfig, string_keyed)
    catch error
        throw(
            ArgumentError(
                "invalid grid.schedules entry '$schedule_name': $(sprint(showerror, error))",
            ),
        )
    end
end

function schedule_grid_candidates(options::Dict{Symbol,Vector{Dict{Symbol,Any}}})
    keys_sorted = sort!(collect(keys(options)); by=string)
    isempty(keys_sorted) && return [Dict{Symbol,Any}()]

    reversed_keys = reverse(keys_sorted)
    return vec([
        merge_schedule_candidate_values(keys_sorted, reverse(values)) for
        values in Iterators.product((options[key] for key in reversed_keys)...)
    ])
end

function merge_schedule_candidate_values(keys_sorted, values)
    output = Dict{Symbol,Any}()
    for (_, settings) in zip(keys_sorted, values)
        merge!(output, settings)
    end
    return output
end

function normalize_schedule(name::Symbol, spec::GridScheduleConfig)
    raw_kind = isempty(strip(spec.kind)) ? spec.type : spec.kind
    kind = Symbol(strip(raw_kind))
    kind === Symbol("") &&
        throw(ArgumentError("schedule '$name' must define a kind."))

    if name == :mu
        return normalize_mu_schedule(kind, spec)
    elseif name == :mu_ref
        return normalize_mu_ref_schedule(kind, spec)
    elseif name == :rho
        return normalize_rho_schedule(kind, spec)
    elseif name == :rho_ref
        return normalize_rho_ref_schedule(kind, spec)
    end

    throw(ArgumentError("unsupported schedule '$name'; supported schedules are mu, mu_ref, rho, and rho_ref."))
end

function normalize_mu_schedule(kind::Symbol, spec)
    manual_values = manual_schedule_values(kind, spec, "mu")
    manual_values === nothing || return Dict{Symbol,Any}(:mu_schedule => manual_values)

    kind in (:constant, :linear, :geometric, :exponential) ||
        throw(ArgumentError("unsupported mu schedule kind '$kind'."))

    output = Dict{Symbol,Any}(:mu_schedule => kind)
    if kind == :constant
        spec.value !== nothing && (output[:mu] = Float64(spec.value))
        return output
    end

    output[:mu_start] = Float64(required_schedule_value(spec, :start, "mu"))
    output[:mu_end] = Float64(schedule_stop_value(spec, "mu"))
    return output
end

function normalize_mu_ref_schedule(kind::Symbol, spec)
    manual_values = manual_schedule_values(kind, spec, "mu_ref")
    manual_values === nothing || return Dict{Symbol,Any}(:mu_ref_schedule => manual_values)

    if kind in (:match_input, :same, :input, :zero, :zeros, :none)
        return Dict{Symbol,Any}(:mu_ref_schedule => kind)
    end
    kind in (:constant, :linear, :geometric, :exponential) ||
        throw(ArgumentError("unsupported mu_ref schedule kind '$kind'."))

    output = Dict{Symbol,Any}(:mu_ref_schedule => kind)
    if kind == :constant
        spec.value !== nothing && (output[:mu_ref] = Float64(spec.value))
        return output
    end

    output[:mu_ref_start] = Float64(required_schedule_value(spec, :start, "mu_ref"))
    output[:mu_ref_end] = Float64(schedule_stop_value(spec, "mu_ref"))
    return output
end

function normalize_rho_schedule(kind::Symbol, spec)
    manual_values = manual_schedule_values(kind, spec, "rho")
    manual_values === nothing || return Dict{Symbol,Any}(:rho_schedule => manual_values)

    kind in (:constant, :linear, :geometric, :exponential) ||
        throw(ArgumentError("unsupported rho schedule kind '$kind'."))

    output = Dict{Symbol,Any}(:rho_schedule => kind)
    if kind == :constant
        spec.value !== nothing && (output[:rho] = Float64(spec.value))
        return output
    end

    output[:rho_start] = Float64(required_schedule_value(spec, :start, "rho"))
    output[:rho_end] = Float64(schedule_stop_value(spec, "rho"))
    return output
end

function normalize_rho_ref_schedule(kind::Symbol, spec)
    manual_values = manual_schedule_values(kind, spec, "rho_ref")
    manual_values === nothing || return Dict{Symbol,Any}(:rho_ref_schedule => manual_values)

    if kind in (:match_input, :same, :input, :zero, :zeros, :none)
        return Dict{Symbol,Any}(:rho_ref_schedule => kind)
    end
    kind in (:constant, :linear, :geometric, :exponential) ||
        throw(ArgumentError("unsupported rho_ref schedule kind '$kind'."))

    output = Dict{Symbol,Any}(:rho_ref_schedule => kind)
    if kind == :constant
        spec.value !== nothing && (output[:rho_ref] = Float64(spec.value))
        return output
    end

    output[:rho_ref_start] = Float64(required_schedule_value(spec, :start, "rho_ref"))
    output[:rho_ref_end] = Float64(schedule_stop_value(spec, "rho_ref"))
    return output
end

function manual_schedule_values(
    kind::Symbol,
    spec::GridScheduleConfig,
    schedule_name::AbstractString,
)
    if kind in (:values, :manual)
        spec.values !== nothing ||
            throw(ArgumentError("schedule '$schedule_name' must define non-empty values."))
        isempty(spec.values) &&
            throw(ArgumentError("schedule '$schedule_name' values must not be empty."))
        return Float64.(spec.values)
    elseif kind in (:piecewise, :step)
        isempty(spec.segments) &&
            throw(ArgumentError("schedule '$schedule_name' segments must not be empty."))

        values = Float64[]
        for (index, segment) in enumerate(spec.segments)
            segment.epochs > 0 || throw(
                ArgumentError(
                    "schedule '$schedule_name' segment $index epochs must be positive.",
                ),
            )
            append!(values, fill(Float64(segment.value), segment.epochs))
        end
        return values
    end

    return nothing
end

function required_schedule_value(
    spec::GridScheduleConfig,
    key::Symbol,
    schedule_name::AbstractString,
)
    value = getproperty(spec, key)
    value === nothing &&
        throw(ArgumentError("schedule '$schedule_name' must define '$key'."))
    return value
end

function schedule_stop_value(spec::GridScheduleConfig, schedule_name::AbstractString)
    spec.stop !== nothing && return spec.stop
    throw(ArgumentError("schedule '$schedule_name' must define 'stop' or 'end'."))
end

function resolve_grid_configs(
    spec::GridSearchSpec;
    base_config::NamedTuple=NamedTuple(),
    defaults::NamedTuple=DEFAULT_RUN_SETTINGS,
)
    static_config = Dict{Symbol,Any}()
    merge!(static_config, Dict{Symbol,Any}(pairs(defaults)))
    merge!(static_config, Dict{Symbol,Any}(pairs(base_config)))
    merge!(static_config, spec.base)
    merge!(static_config, spec.fixed)

    configs_without_digest = NamedTuple[]
    index = 0
    for candidate_values in grid_candidates(spec.grid),
        schedule_candidate_values in spec.schedule_grid

        index += 1
        config = copy(static_config)
        merge!(config, spec.schedules)
        merge!(config, candidate_values)
        merge!(config, schedule_candidate_values)

        config[:grid_config_name] = spec.name
        config[:grid_config_path] = spec.path
        config[:grid_config_version] = spec.version
        config[:grid_candidate_index] = index

        validate_explicit_schedule_lengths!(config)

        haskey(config, :run_id) || (config[:run_id] = grid_run_id(spec, config, index))
        push!(configs_without_digest, namedtuple_from_dict(config))
    end

    digest = grid_config_digest(configs_without_digest)
    return [
        merge(config, (; grid_config_digest=digest)) for
        config in configs_without_digest
    ]
end

function validate_explicit_schedule_lengths!(config::Dict{Symbol,Any})
    epochs = Int(get(config, :epochs, 0))
    for key in (:mu_schedule, :mu_ref_schedule, :rho_schedule, :rho_ref_schedule)
        haskey(config, key) || continue
        values = config[key]
        values isa AbstractVector || continue
        length(values) == epochs || throw(
            ArgumentError(
                "$(key) vector length $(length(values)) must equal epochs $(epochs).",
            ),
        )
    end
    return nothing
end

function grid_candidates(grid::Dict{Symbol,Vector{Any}})
    keys_sorted = sort!(collect(keys(grid)); by=string)
    isempty(keys_sorted) && return [Dict{Symbol,Any}()]

    reversed_keys = reverse(keys_sorted)
    return [
        Dict{Symbol,Any}(key => value for (key, value) in zip(keys_sorted, reverse(values))) for
        values in Iterators.product((grid[key] for key in reversed_keys)...)
    ]
end

function grid_run_id(spec::GridSearchSpec, config::Dict{Symbol,Any}, index::Integer)
    hash = bytes2hex(sha1(JSON3.write(json_ready(config))))[1:8]
    index_text = lpad(string(index), 4, "0")
    template = spec.run_id_template
    template === nothing &&
        return "grid_" * safe_identifier(spec.name) * "_" * index_text * "_" * hash

    output = replace(template, "{index}" => index_text)
    output = replace(output, "{name}" => safe_identifier(spec.name))
    output = replace(output, "{hash}" => hash)
    return output
end

function safe_identifier(value)
    text = lowercase(string(value))
    text = replace(text, r"[^A-Za-z0-9_.=-]+" => "_")
    return strip(text, ['_'])
end

function namedtuple_from_dict(values::Dict{Symbol,Any})
    keys_sorted = sort!(collect(keys(values)); by=string)
    return NamedTuple{Tuple(keys_sorted)}(Tuple(values[key] for key in keys_sorted))
end

function grid_config_digest(configs)
    data = Vector{UInt8}(codeunits(resolved_grid_json(configs)))
    return "sha256:" * bytes2hex(sha256(data))
end

const RESOLVED_GRID_JSON_OMITTED_KEYS = Set(["grid_config_digest"])

function resolved_grid_json(configs)
    return sprint(io -> JSON3.pretty(io, json_ready(configs, RESOLVED_GRID_JSON_OMITTED_KEYS)))
end

function json_ready(value, omitted_keys::Set{String}=Set{String}())
    value === missing && return nothing
    value === nothing && return nothing
    value isa Symbol && return string(value)
    value isa Number && return value
    value isa Bool && return value
    value isa AbstractString && return value

    if value isa NamedTuple
        ready_keys = Symbol[
            key for key in sort!(collect(keys(value)); by=string) if
            !(string(key) in omitted_keys)
        ]
        return NamedTuple{Tuple(ready_keys)}(
            Tuple(json_ready(getproperty(value, key), omitted_keys) for key in ready_keys),
        )
    elseif value isa AbstractDict
        source_keys = [
            key for key in sort!(collect(keys(value)); by=string) if
            !(string(key) in omitted_keys)
        ]
        ready_keys = Symbol.(string.(source_keys))
        return NamedTuple{Tuple(ready_keys)}(
            Tuple(json_ready(value[key], omitted_keys) for key in source_keys),
        )
    elseif value isa AbstractVector || value isa Tuple
        return Any[json_ready(item, omitted_keys) for item in value]
    end

    return string(value)
end

# END FILE: src/ContextualDFL/ContextualDFLTraining/src/grid_file_config.jl

# BEGIN FILE: src/ContextualDFL/ContextualDFLTraining/src/mlflow_support.jl
import MLFlowClient

const RunStatus = MLFlowClient.RunStatus
const MLFLOW_RETRY_ATTEMPTS = 8
const MLFLOW_RETRY_INITIAL_DELAY_SECONDS = 1.0
const MLFLOW_RETRY_BACKOFF = 1.5

function with_mlflow_retry(callback, operation)
    delay = MLFLOW_RETRY_INITIAL_DELAY_SECONDS
    for attempt in 1:MLFLOW_RETRY_ATTEMPTS
        try
            return callback()
        catch error
            attempt == MLFLOW_RETRY_ATTEMPTS && rethrow()
            @warn "MLflow $operation failed; retrying" attempt error=sprint(showerror, error)
            sleep(delay)
            delay *= MLFLOW_RETRY_BACKOFF
        end
    end
end

mutable struct NamedMLFlowClient
    client::MLFlowClient.MLFlow
    run_name::String
    tags::Dict{String,String}
    params::Dict{String,String}
    run::Any
end

function NamedMLFlowClient(; tracking_uri="", run_name, tags, params)
    client = isempty(string(tracking_uri)) ?
        MLFlowClient.MLFlow(; headers=mlflow_http_headers()) :
        MLFlowClient.MLFlow(string(tracking_uri); headers=mlflow_http_headers())
    return NamedMLFlowClient(
        client,
        string(run_name),
        string_dict(tags),
        string_dict(params),
        nothing,
    )
end

function createrun(mlf::NamedMLFlowClient, experiment_id; start_time=missing)
    run = with_mlflow_retry("create run") do
        MLFlowClient.createrun(
            mlf.client,
            string(experiment_id);
            run_name=mlf.run_name,
            start_time=start_time,
            tags=mlf.tags,
        )
    end
    mlf.run = run

    for key in sort!(collect(keys(mlf.params)))
        with_mlflow_retry("log param $key") do
            MLFlowClient.logparam(mlf.client, run, key, mlf.params[key])
        end
    end

    return run
end

function logparam(mlf::NamedMLFlowClient, run, key, value)
    return with_mlflow_retry("log param $key") do
        MLFlowClient.logparam(mlf.client, run, string(key), string(value))
    end
end

function logmetric(
    mlf::NamedMLFlowClient,
    run,
    key,
    value;
    step,
    timestamp=round(Int64, time() * 1000),
)
    return with_mlflow_retry("log metric $key") do
        MLFlowClient.logmetric(
            mlf.client,
            run,
            string(key),
            Float64(value);
            timestamp=Int64(timestamp),
            step=Int(step),
        )
    end
end

function logbatch(
    mlf::NamedMLFlowClient,
    run;
    metrics=MLFlowClient.Metric[],
    params=MLFlowClient.Param[],
    tags=MLFlowClient.Tag[],
)
    return with_mlflow_retry("log batch") do
        MLFlowClient.logbatch(mlf.client, run; metrics=metrics, params=params, tags=tags)
    end
end

function setruntag(mlf::NamedMLFlowClient, run, key, value)
    return with_mlflow_retry("set tag $key") do
        MLFlowClient.setruntag(mlf.client, run, string(key), string(value))
    end
end

function loginputs(mlf::NamedMLFlowClient, run; datasets)
    return with_mlflow_retry("log inputs") do
        MLFlowClient.loginputs(mlf.client, run, datasets)
    end
end

function loginputs(mlf::NamedMLFlowClient, run, datasets)
    return with_mlflow_retry("log inputs") do
        MLFlowClient.loginputs(mlf.client, run, datasets)
    end
end

const Dataset = MLFlowClient.Dataset
const DatasetInput = MLFlowClient.DatasetInput
const Metric = MLFlowClient.Metric
const Tag = MLFlowClient.Tag
const MLFLOW_STACKTRACE_ARTIFACT_PATH = "errors/stacktrace.txt"

function uploadartifact(mlf::NamedMLFlowClient, run, path)
    return uploadartifact(mlf, run, path, basename(path))
end

function uploadartifact(mlf::NamedMLFlowClient, run, path, artifact_path)
    return uploadartifact(mlf, run, string(artifact_path), read(path))
end

function uploadartifact(
    mlf::NamedMLFlowClient,
    run,
    artifact_path::AbstractString,
    data::Vector{UInt8},
)
    run_id = string(getproperty(getproperty(run, :info), :run_id))
    return with_mlflow_retry("upload artifact $artifact_path") do
        upload_run_artifact_data(mlf.client, run_id, artifact_path, data)
    end
end

function uploadartifact(mlf::NamedMLFlowClient, artifact_path::AbstractString, data::Vector{UInt8})
    return with_mlflow_retry("upload artifact $artifact_path") do
        MLFlowClient.uploadartifact(mlf.client, string(artifact_path), data)
    end
end

function log_mlflow_stacktrace_artifact!(mlf, run, error_text)
    uploadartifact(mlf, run, MLFLOW_STACKTRACE_ARTIFACT_PATH, Vector{UInt8}(codeunits(string(error_text))))
    return nothing
end

function upload_run_artifact_data(client::MLFlowClient.MLFlow, run_id, artifact_path, data::Vector{UInt8})
    uri = mlflow_ajax_uri(
        client,
        "upload-artifact";
        parameters=Dict{Symbol,Any}(
            :run_uuid => string(run_id),
            :path => clean_mlflow_artifact_path(artifact_path),
        ),
    )
    headers = MLFlowClient.headers(
        client,
        Dict("Content-Type" => "application/octet-stream"),
    )
    MLFlowClient.HTTP.post(uri, headers, data)
    return true
end

function mlflow_ajax_uri(client::MLFlowClient.MLFlow, endpoint; parameters=Dict{Symbol,Any}())
    root = replace(string(client.apiroot), r"/api/?$" => "")
    return MLFlowClient.URIs.URI(
        "$(root)/ajax-api/2.0/mlflow/$(endpoint)";
        query=parameters,
    )
end

function mlflow_run_artifact_path(run, artifact_path)
    relative_path = clean_mlflow_artifact_path(artifact_path)
    artifact_uri = try
        string(getproperty(getproperty(run, :info), :artifact_uri))
    catch
        ""
    end

    startswith(artifact_uri, "mlflow-artifacts:/") || return relative_path
    root_path = replace(artifact_uri, r"^mlflow-artifacts:/*" => "")
    return join_mlflow_artifact_path(root_path, relative_path)
end

function join_mlflow_artifact_path(prefix, artifact_path)
    clean_prefix = clean_mlflow_artifact_path(prefix)
    clean_path = clean_mlflow_artifact_path(artifact_path)
    isempty(clean_prefix) && return clean_path
    isempty(clean_path) && return clean_prefix
    return clean_prefix * "/" * clean_path
end

function clean_mlflow_artifact_path(path)
    return strip(replace(string(path), "\\" => "/"), '/')
end

function updaterun(mlf::NamedMLFlowClient, run; status, end_time=missing)
    return with_mlflow_retry("update run") do
        MLFlowClient.updaterun(mlf.client, run; status=status, end_time=end_time)
    end
end

function log_mlflow_metric!(mlf, run, key, value; timestamp, step)
    return logmetric(
        mlf,
        run,
        key,
        Float64(value);
        timestamp=Int64(timestamp),
        step=Int(step),
    )
end

function log_mlflow_epoch!(mlf, run, epoch, loss_value, display_loss, metadata)
    timestamp = mlflow_unix_milliseconds()
    step = Int64(epoch)
    metrics = Metric[Metric("loss", Float64(loss_value), timestamp, step)]
    mlflow_metric_value(display_loss) &&
        push!(metrics, Metric("display_loss", Float64(display_loss), timestamp, step))
    append!(metrics, mlflow_epoch_metadata_metrics(metadata; timestamp=timestamp, step=step))
    return logbatch(mlf, run; metrics=metrics)
end

function mlflow_epoch_metadata_metrics(metadata; timestamp, step)
    metadata isa NamedTuple || return Metric[]
    metrics = Metric[]

    for (metric_name, field_name) in (
        ("epoch_seconds", :epoch_seconds),
        ("epoch_mu_in", :mu_in),
        ("epoch_mu_ref", :mu_ref),
        ("epoch_rho_in", :rho_in),
        ("epoch_rho_ref", :rho_ref),
        ("epoch_iterations", :iterations),
        ("real_display_loss", :real_display_loss),
    )
        field_name in keys(metadata) || continue
        value = getproperty(metadata, field_name)
        mlflow_metric_value(value) || continue
        push!(metrics, Metric(metric_name, Float64(value), Int64(timestamp), Int64(step)))
    end

    return metrics
end

function log_mlflow_params!(mlf, run, prefix::AbstractString, values)
    params = Dict{String,String}()
    flatten_mlflow_params!(params, prefix, values)

    for key in sort!(collect(keys(params)))
        logparam(mlf, run, key, params[key])
    end

    return nothing
end

function flatten_mlflow_params!(
    params::Dict{String,String},
    prefix::AbstractString,
    values::NamedTuple,
)
    for key in keys(values)
        flatten_mlflow_params!(params, join_mlflow_key(prefix, key), getproperty(values, key))
    end
    return params
end

function flatten_mlflow_params!(
    params::Dict{String,String},
    prefix::AbstractString,
    values::AbstractDict,
)
    for (key, value) in values
        flatten_mlflow_params!(params, join_mlflow_key(prefix, key), value)
    end
    return params
end

function flatten_mlflow_params!(params::Dict{String,String}, prefix::AbstractString, value)
    isempty(prefix) && return params
    mlflow_param_value(value) || return params
    params[prefix] = string(value)
    return params
end

function log_mlflow_evaluation_result!(mlf, run, name::AbstractString, result)
    if result isa NamedTuple || result isa AbstractDict
        metrics = mlflow_evaluation_field(result, :metrics, result)
        artifacts = mlflow_evaluation_field(result, :artifacts, nothing)
        log_mlflow_metrics!(mlf, run, name, metrics; step=0)
        log_mlflow_artifacts!(mlf, run, artifacts)
    elseif mlflow_metric_value(result)
        log_mlflow_metrics!(mlf, run, "", Dict(name => result); step=0)
    end

    return nothing
end

function mlflow_evaluation_field(values::NamedTuple, key::Symbol, default)
    return key in keys(values) ? getproperty(values, key) : default
end

function mlflow_evaluation_field(values::AbstractDict, key::Symbol, default)
    return haskey(values, key) ? values[key] : get(values, string(key), default)
end

function log_mlflow_metrics!(mlf, run, prefix::AbstractString, values; step::Integer)
    metrics = Dict{String,Float64}()
    flatten_mlflow_metrics!(metrics, prefix, values)

    timestamp = mlflow_unix_milliseconds()
    for key in sort!(collect(keys(metrics)))
        log_mlflow_metric!(
            mlf,
            run,
            key,
            metrics[key];
            timestamp=timestamp,
            step=step,
        )
    end

    return nothing
end

function flatten_mlflow_metrics!(
    metrics::Dict{String,Float64},
    prefix::AbstractString,
    values::NamedTuple,
)
    for key in keys(values)
        flatten_mlflow_metrics!(metrics, join_mlflow_key(prefix, key), getproperty(values, key))
    end
    return metrics
end

function flatten_mlflow_metrics!(
    metrics::Dict{String,Float64},
    prefix::AbstractString,
    values::AbstractDict,
)
    for (key, value) in values
        flatten_mlflow_metrics!(metrics, join_mlflow_key(prefix, key), value)
    end
    return metrics
end

function flatten_mlflow_metrics!(metrics::Dict{String,Float64}, prefix::AbstractString, value)
    isempty(prefix) && return metrics
    mlflow_metric_value(value) || return metrics
    metrics[prefix] = Float64(value)
    return metrics
end

function mlflow_metric_value(value)
    value isa Bool && return false
    value isa Number || return false
    float_value = try
        Float64(value)
    catch
        return false
    end
    return isfinite(float_value)
end

function log_mlflow_artifacts!(mlf, run, artifacts)
    artifacts === nothing && return nothing

    if artifacts isa AbstractString
        isfile(artifacts) && upload_mlflow_artifact!(mlf, run, artifacts; artifact_path=basename(artifacts))
        return nothing
    elseif artifacts isa Pair
        path = last(artifacts)
        path isa AbstractString && isfile(path) &&
            upload_mlflow_artifact!(mlf, run, path; artifact_path=string(first(artifacts)))
        return nothing
    elseif artifacts isa NamedTuple || artifacts isa AbstractDict
        for (name, path) in pairs(artifacts)
            path isa AbstractString && isfile(path) &&
                upload_mlflow_artifact!(mlf, run, path; artifact_path=string(name))
        end
        return nothing
    elseif artifacts isa AbstractVector || artifacts isa Tuple
        for artifact in artifacts
            log_mlflow_artifacts!(mlf, run, artifact)
        end
    end

    return nothing
end

function upload_mlflow_artifact!(mlf, run, path; artifact_path)
    if applicable(uploadartifact, mlf, run, path, artifact_path)
        uploadartifact(mlf, run, path, artifact_path)
    elseif applicable(uploadartifact, mlf, run, path)
        uploadartifact(mlf, run, path)
    else
        uploadartifact(mlf, string(artifact_path), read(path))
    end

    return nothing
end

function log_mlflow_source_tags!(
    mlf,
    run;
    source_name,
    source_type,
    source_git_commit,
)
    setruntag(mlf, run, "mlflow.source.name", string(source_name))
    setruntag(mlf, run, "mlflow.source.type", string(source_type))
    isnothing(source_git_commit) ||
        setruntag(mlf, run, "mlflow.source.git.commit", string(source_git_commit))

    return nothing
end

function log_mlflow_dataset!(
    mlf,
    run;
    dataset_inputs=nothing,
    dataset_name=nothing,
    dataset_digest=nothing,
    dataset_source_type="local",
    dataset_source=nothing,
    dataset_context="training",
)
    inputs = if !isnothing(dataset_inputs)
        dataset_inputs
    elseif !any(isnothing, (dataset_name, dataset_digest, dataset_source))
        dataset = Dataset(
            string(dataset_name),
            mlflow_dataset_digest_value(dataset_digest),
            string(dataset_source_type),
            string(dataset_source),
            nothing,
            nothing,
        )
        [DatasetInput([Tag("context", string(dataset_context))], dataset)]
    else
        return nothing
    end

    try
        loginputs(mlf, run; datasets=inputs)
    catch error
        error isa MethodError || rethrow()
        loginputs(mlf, run, inputs)
    end

    return nothing
end

function mlflow_dataset_digest_value(digest)
    value = string(digest)
    length(value) <= 36 && return value
    return bytes2hex(sha256(value))[1:32]
end

function tag_optional_mlflow_evaluation_error!(mlf, run, name, error)
    setruntag(
        mlf,
        run,
        "mlflow.optional_evaluation.$(name).error",
        sprint(showerror, error),
    )
    return nothing
end

function run_mlflow_evaluation_callbacks!(
    mlf,
    run,
    callbacks,
    train_result;
    optional::Bool,
)
    isempty(mlflow_callback_pairs(callbacks)) && return nothing

    for (name, callback) in mlflow_callback_pairs(callbacks)
        try
            result = applicable(callback, train_result) ? callback(train_result) : callback()
            log_mlflow_evaluation_result!(mlf, run, string(name), result)
        catch error
            optional || rethrow()
            tag_optional_mlflow_evaluation_error!(mlf, run, name, error)
        end
    end

    return nothing
end

mlflow_callback_pairs(callbacks::NamedTuple) = collect(pairs(callbacks))
mlflow_callback_pairs(callbacks::AbstractDict) = collect(pairs(callbacks))
mlflow_callback_pairs(callbacks::Tuple) = collect(pairs(callbacks))
mlflow_callback_pairs(callbacks::AbstractVector) = collect(pairs(callbacks))
mlflow_callback_pairs(::Nothing) = Pair{Symbol,Any}[]

function join_mlflow_key(prefix::AbstractString, key)
    key_text = string(key)
    return isempty(prefix) ? key_text : prefix * "_" * key_text
end

mlflow_unix_milliseconds() = round(Int64, time() * 1000)

function git_commit_or_nothing()
    try
        return strip(read(pipeline(`git rev-parse HEAD`; stderr=devnull), String))
    catch
        return nothing
    end
end

function mlflow_enabled(config)
    return Bool(config_value(config, :mlflow_enabled, false))
end

function mlflow_client_for_config(config)
    experiment_id = string(config_value(config, :mlflow_experiment_id, ""))
    isempty(experiment_id) &&
        throw(ArgumentError("mlflow_enabled=true requires config.mlflow_experiment_id"))

    run_name = string(
        config_value(
            config,
            :mlflow_run_name,
            config_value(config, :candidate_name, config_value(config, :run_id, "training-run")),
        ),
    )
    tracking_uri = string(config_value(config, :mlflow_tracking_uri, ""))

    mlf = NamedMLFlowClient(
        tracking_uri=tracking_uri,
        run_name=run_name,
        tags=mlflow_tags_for_config(config),
        params=mlflow_params_for_config(config),
    )
    return mlf, experiment_id
end

function mlflow_tags_for_config(config)
    tags = Dict{String,String}(
        "source" => "ContextualDFLTraining.gridsearch",
        "run_id" => string(config_value(config, :run_id, "")),
        "base_run_id" => string(config_value(config, :base_run_id, "")),
        "candidate_name" => string(config_value(config, :candidate_name, "")),
        "gridsearch_id" => string(config_value(config, :gridsearch_id, "")),
        "gridsearch_timestamp" => string(config_value(config, :gridsearch_timestamp, "")),
        "candidate_index" => string(config_value(config, :candidate_index, "")),
        "gridsearch_parent_run_id" => string(config_value(config, :mlflow_parent_run_id, "")),
        "mlflow.parentRunId" => string(config_value(config, :mlflow_parent_run_id, "")),
        "training_project" => "ContextualDFLTraining",
    )

    extra_tags = config_value(config, :mlflow_tags, nothing)
    add_string_pairs!(tags, extra_tags)
    return drop_empty_values(tags)
end

function mlflow_params_for_config(config)
    params = Dict{String,String}()
    config isa NamedTuple || return params

    for key in keys(config)
        key in (:mlflow_tags, :mlflow_tracking_uri) && continue
        value = getproperty(config, key)
        mlflow_param_value(value) || continue
        params["config_" * string(key)] = string(value)
    end

    return params
end

function config_value(config, key::Symbol, default)
    config isa NamedTuple || return default
    return key in keys(config) ? getproperty(config, key) : default
end

function string_dict(values)
    output = Dict{String,String}()
    add_string_pairs!(output, values)
    return output
end

function add_string_pairs!(output::Dict{String,String}, values::NamedTuple)
    for key in keys(values)
        output[string(key)] = string(getproperty(values, key))
    end
    return output
end

function add_string_pairs!(output::Dict{String,String}, values::AbstractDict)
    for (key, value) in values
        output[string(key)] = string(value)
    end
    return output
end

add_string_pairs!(output::Dict{String,String}, ::Nothing) = output

function add_string_pairs!(output::Dict{String,String}, values)
    throw(ArgumentError("MLflow tags/params must be a NamedTuple or Dict, got $(typeof(values))"))
end

function drop_empty_values(values::Dict{String,String})
    return Dict(key => value for (key, value) in values if !isempty(value))
end

mlflow_param_value(value) =
    value isa Number ||
    value isa Bool ||
    value isa Symbol ||
    value isa AbstractString

function mlflow_http_headers()
    return Dict("Connection" => "close")
end

# END FILE: src/ContextualDFL/ContextualDFLTraining/src/mlflow_support.jl

# BEGIN FILE: src/ContextualDFL/ContextualDFLTraining/src/profile_run.jl
using FileIO
using FlameGraphs
using Profile
using ProfileSVG

function strict_contextualdfl_training(objects, config; profile_mlflow_state=nothing)
    assert_remote_training_worker!(config)
    mu_schedule = mu_schedule_for_config(config)

    if profile_mlflow_active(config)
        profile_mlflow_state === nothing &&
            error("profile MLflow progress requires an active remote MLflow run")
        return profile_train_with_epoch_progress!(objects, config, mu_schedule, profile_mlflow_state)
    end

    return ContextualDFL.train!(
        objects.scenario_generator.neural_net,
        objects.loss,
        mu_schedule,
        objects.data.train;
        learning_rate=config.learning_rate,
        optimizer_type=Flux.Adam,
        epochs=config.epochs,
        batchsize=config.batch_size,
        shuffle=true,
        rng=MersenneTwister(config.seed + 10_000),
        display_plot=false,
        verbose=false,
        nr_scenarios=effective_nr_scenarios(objects, config),
        display_smooth=Bool(config_value(config, :display_smooth, false)),
        display_real=config_value(config, :display_real, nothing),
        display_reference_input=display_reference_input(objects, config),
    )
end

function profile_train_with_epoch_progress!(objects, config, mu_schedule, mlflow_state)
    run_started = time()

    return ContextualDFL.train!(
        objects.scenario_generator.neural_net,
        objects.loss,
        mu_schedule,
        objects.data.train;
        learning_rate=config.learning_rate,
        optimizer_type=Flux.Adam,
        epochs=config.epochs,
        batchsize=config.batch_size,
        shuffle=true,
        rng=MersenneTwister(config.seed + 10_000),
        display_plot=false,
        verbose=false,
        nr_scenarios=effective_nr_scenarios(objects, config),
        display_smooth=Bool(config_value(config, :display_smooth, false)),
        display_real=config_value(config, :display_real, nothing),
        display_reference_input=display_reference_input(objects, config),
        on_epoch_end=(epoch, loss_value, display_loss, metadata) -> begin
            elapsed_seconds = time() - run_started
            profile_mlflow_log_epoch!(
                mlflow_state,
                epoch,
                loss_value,
                display_loss,
                metadata,
                elapsed_seconds,
            )
            println(
                "MLflow profiling progress: epoch=$(epoch)/$(config.epochs) ",
                "loss=$(Float64(loss_value)) elapsed_seconds=$(round(elapsed_seconds; digits=2))",
            )
        end,
    )
end

function standard_profile_config(; overrides...)
    settings = merge(
        DEFAULT_RUN_SETTINGS,
        (;
            epochs=100,
            learning_rate=1e-3,
            hidden_size=128,
            depth=2,
            batch_size=64,
            dropout=0.0,
            seed=3,
            run_id="profile_standard_seed3",
        ),
        NamedTuple(overrides),
    )
    return settings
end

function profile_mlflow_active(config)
    return mlflow_enabled(config) && Bool(config_value(config, :profile_mlflow_progress, false))
end

function profile_mlflow_start!(config)
    profile_mlflow_active(config) || return nothing

    experiment_id = string(config_value(config, :mlflow_experiment_id, ""))
    isempty(experiment_id) &&
        throw(ArgumentError("profile MLflow logging requires config.mlflow_experiment_id"))

    mlf = NamedMLFlowClient(
        tracking_uri=string(config_value(config, :mlflow_tracking_uri, "http://127.0.0.1:5000")),
        run_name=string(
            config_value(
                config,
                :mlflow_run_name,
                config_value(config, :run_id, "profile-run"),
            ),
        ),
        tags=profile_mlflow_tags(config),
        params=profile_mlflow_params(config),
    )
    run = createrun(mlf, experiment_id; start_time=unix_milliseconds())
    return (; mlf=mlf, run=run, run_id=string(config_value(config, :run_id, "")))
end

function profile_mlflow_tags(config)
    tags = Dict{String,String}()
    add_string_pairs!(tags, config_value(config, :mlflow_tags, nothing))
    tags["run_kind"] = "profiling"
    tags["profile_run"] = "true"
    tags["exclude_from_model_selection"] = "true"
    tags["exclude_from_gridsearch"] = "true"
    tags["profile_target"] = "ContextualDFL.train!"
    tags["profile_loss"] = "ContextualDFL.DflScenLoss"
    tags["profile_progress_logged_by"] = "remote_worker"
    tags["run_id"] = string(config_value(config, :run_id, ""))
    tags["training_project"] = "ContextualDFLTraining"
    tags["worker_id"] = string(Distributed.myid())
    tags["worker_hostname"] = Sockets.gethostname()
    tags["worker_pid"] = string(getpid())
    return drop_empty_values(tags)
end

function profile_mlflow_params(config)
    params = Dict{String,String}()
    add_string_pairs!(params, config_value(config, :mlflow_params, nothing))
    params["profile_target"] = "ContextualDFL.train!"
    params["profile_loss"] = "ContextualDFL.DflScenLoss"
    params["profile_progress_logged_by"] = "remote_worker"

    config isa NamedTuple || return params
    for key in keys(config)
        key in (:mlflow_tags, :mlflow_params, :mlflow_tracking_uri) && continue
        value = getproperty(config, key)
        mlflow_param_value(value) || continue
        params["config_" * string(key)] = string(value)
    end

    return drop_empty_values(params)
end

function profile_mlflow_log_epoch!(
    state,
    epoch,
    loss_value,
    display_loss,
    metadata,
    elapsed_seconds,
)
    state === nothing && return nothing

    try
        step = Int(epoch)
        profile_mlflow_logmetric(state, "loss", Float64(loss_value); step=step)
        profile_mlflow_logmetric(state, "display_loss", Float64(display_loss); step=step)
        profile_mlflow_logmetric(
            state,
            "profile_elapsed_seconds",
            Float64(elapsed_seconds);
            step=step,
        )

        if metadata isa NamedTuple
            for (metric_name, field_name) in (
                "epoch_seconds" => :epoch_seconds,
                "epoch_mu" => :mu,
                "epoch_iterations" => :iterations,
                "real_display_loss" => :real_display_loss,
            )
                haskey(metadata, field_name) || continue
                value = getproperty(metadata, field_name)
                value isa Number || continue
                profile_mlflow_logmetric(state, metric_name, Float64(value); step=step)
            end
        end
    catch error
        println("Failed to log MLflow profiling progress: ", sprint(showerror, error))
    end

    return nothing
end

function profile_mlflow_log_final!(state, metrics; status, error="")
    state === nothing && return nothing

    setruntag(state.mlf, state.run, "profile_status", string(status))
    isempty(error) ||
        setruntag(state.mlf, state.run, "profile_error", string(first(split(error, '\n'))))

    metrics isa NamedTuple || return nothing
    for key in keys(metrics)
        value = getproperty(metrics, key)
        value isa Number || continue
        profile_mlflow_logmetric(state, "final_" * string(key), Float64(value); step=0)
    end

    return nothing
end

function profile_mlflow_finish!(state, status::Symbol; error="")
    state === nothing && return nothing

    isempty(error) ||
        setruntag(state.mlf, state.run, "profile_error", string(first(split(error, '\n'))))
    run_status = status == :ok ? RunStatus.FINISHED : RunStatus.FAILED
    updaterun(state.mlf, state.run; status=run_status, end_time=unix_milliseconds())
    return nothing
end

function profile_mlflow_logmetric(state, key, value; step)
    for attempt in 1:3
        try
            return MLFlowClient.logmetric(
                state.mlf.client,
                state.run,
                string(key),
                Float64(value);
                timestamp=unix_milliseconds(),
                step=Int(step),
            )
        catch
            attempt == 3 && rethrow()
            sleep(0.25 * attempt)
        end
    end
end

function profile_standard_training(config::NamedTuple)
    cfg = normalize_config(config)
    assert_remote_training_worker!(cfg)
    started_at = unix_milliseconds()
    elapsed_seconds = 0.0
    remote_output_dir = ""
    mlflow_state = nothing
    mlflow_finished = false

    try
        mlflow_state = profile_mlflow_start!(cfg)

        remote_output_dir = mktempdir(; prefix="contextualdfl_profile_")
        remote_assets_dir = joinpath(remote_output_dir, "assets")
        mkpath(remote_assets_dir)
        svg_path = joinpath(remote_assets_dir, "prof.svg")
        jlprof_path = joinpath(remote_assets_dir, "prof.jlprof")

        profile_result = nothing
        initial_train_mse = NaN
        final_train_mse = NaN
        metrics = NamedTuple()
        history = Dict{Symbol,Any}[]
        svg_bytes = UInt8[]
        jlprof_bytes = UInt8[]

        elapsed_seconds = @elapsed begin
            warmup_epochs = max(Int(hasproperty(cfg, :warmup_epochs) ? cfg.warmup_epochs : 2), 0)
            if warmup_epochs > 0
                warmup_cfg = merge(
                    cfg,
                    (;
                        epochs=warmup_epochs,
                        run_id=string(cfg.run_id, "_warmup"),
                        mlflow_enabled=false,
                        profile_mlflow_progress=false,
                    ),
                )
                warmup_objects = training_objects_for_config(warmup_cfg)
                strict_contextualdfl_training(warmup_objects, warmup_cfg)
                GC.gc()
            end

            objects = training_objects_for_config(cfg)
            model = objects.scenario_generator.neural_net
            initial_train_mse = split_mse(model, objects.data.train, objects)

            Profile.clear()
            profile_result = Profile.@profile strict_contextualdfl_training(
                objects,
                cfg;
                profile_mlflow_state=mlflow_state,
            )

            ProfileSVG.save(svg_path)
            FileIO.save(jlprof_path, Profile.retrieve()...)
            svg_bytes = read(svg_path)
            jlprof_bytes = read(jlprof_path)

            trained_model = extract_model(profile_result, objects.scenario_generator)
            final_train_mse = split_mse(trained_model, objects.data.train, objects)
            metrics = merge(
                evaluate_model_for_reporting(trained_model, objects, cfg),
                (;
                    initial_train_mse=initial_train_mse,
                    final_train_mse=final_train_mse,
                    loss_delta=initial_train_mse - final_train_mse,
                    loss_decreased=final_train_mse < initial_train_mse,
                    training_backend=mlflow_enabled(cfg) && Bool(config_value(cfg, :profile_mlflow_progress, false)) ?
                        "ContextualDFL.train! with MLflow profiling progress" :
                        "ContextualDFL.train!",
                    remote_output_dir=remote_output_dir,
                    thread_count=Threads.nthreads(),
                ),
            )
            history = extract_epoch_history(profile_result)
        end

        require_train_mse_decrease = Bool(
            config_value(cfg, :require_train_mse_decrease, false),
        )
        (!require_train_mse_decrease || final_train_mse < initial_train_mse) ||
            error("profiled training did not reduce train MSE: initial=$(initial_train_mse), final=$(final_train_mse)")

        result = (;
            status="ok",
            run_id=cfg.run_id,
            config=cfg,
            worker=worker_metadata(),
            final_metrics=metrics,
            epoch_history=history,
            profile_svg_bytes=svg_bytes,
            profile_jlprof_bytes=jlprof_bytes,
            error="",
            started_at=started_at,
            finished_at=unix_milliseconds(),
            elapsed_seconds=elapsed_seconds,
        )

        profile_mlflow_log_final!(mlflow_state, result.final_metrics; status=result.status)
        profile_mlflow_finish!(mlflow_state, :ok)
        mlflow_finished = true

        return result
    catch error
        error_text = exception_text(error, catch_backtrace())
        if mlflow_state !== nothing && !mlflow_finished
            try
                profile_mlflow_log_final!(
                    mlflow_state,
                    (;);
                    status="failed",
                    error=error_text,
                )
                profile_mlflow_finish!(mlflow_state, :failed; error=error_text)
                mlflow_finished = true
            catch mlflow_error
                error_text *=
                    "\n\nMLflow failure while marking profile run failed:\n" *
                    exception_text(mlflow_error, catch_backtrace())
            end
        end

        return (;
            status="failed",
            run_id=hasproperty(cfg, :run_id) ? cfg.run_id : "",
            config=cfg,
            worker=worker_metadata(),
            final_metrics=NamedTuple(),
            epoch_history=Dict{Symbol,Any}[],
            profile_svg_bytes=UInt8[],
            profile_jlprof_bytes=UInt8[],
            error=error_text,
            started_at=started_at,
            finished_at=unix_milliseconds(),
            elapsed_seconds=elapsed_seconds,
        )
    finally
        if !isempty(remote_output_dir) && isdir(remote_output_dir)
            try
                rm(remote_output_dir; recursive=true, force=true)
            catch
            end
        end
    end
end

# END FILE: src/ContextualDFL/ContextualDFLTraining/src/profile_run.jl

# BEGIN FILE: src/ContextualDFL/ContextualDFLTraining/src/run_defaults.jl
const DEFAULT_RUN_SETTINGS = (;
    epochs=10,
    learning_rate=1e-3,
    hidden_size=64,
    depth=2,
    batch_size=8,
    dropout=0.0,
    activation=:relu,
    seed=143,
    repeat_count=1,
    checkpoint_enabled=true,
    checkpoint_upload_mlflow=true,
    checkpoint_required=false,
    checkpoint_dir="",
    checkpoint_format=:jls,
    mu=1e-2,
    mu_start=1.0,
    mu_end=1e-2,
    mu_schedule=:geometric,
    mu_ref_schedule=:match_input,
    rho=0.0,
    rho_start=0.0,
    rho_end=0.0,
    rho_schedule=:constant,
    rho_ref=0.0,
    rho_ref_start=0.0,
    rho_ref_end=0.0,
    rho_ref_schedule=:match_input,
    tolerance_relative=0.10,
    tolerance_absolute_floor=1.0,
    optimality_evaluation=true,
    optimality_test_sample_count=30,
    optimality_train_sample_count=0,
    optimality_validation_sample_count=0,
    optimality_mu=0.0,
    optimality_rho=0.0,
    optimality_evaluation_batches=nothing,
    policy_inference_rho=nothing,
    loss=:dfl_scen,
    display_smooth=false,
    display_real=nothing,
)

# END FILE: src/ContextualDFL/ContextualDFLTraining/src/run_defaults.jl

# BEGIN FILE: src/ContextualDFL/ContextualDFLTraining/src/train_run.jl
using Dates
using ContextualDFLExperiments
using Distributed
using Flux
using Random
using Serialization
using SHA
using Sockets
using Statistics

function normalize_config(config::NamedTuple)
    return merge(DEFAULT_RUN_SETTINGS, config)
end

# Explicit Unix epoch milliseconds, independent of the worker timezone.
unix_milliseconds() = round(Int64, time() * 1000)

function worker_metadata()
    return (;
        worker_id=Distributed.myid(),
        hostname=Sockets.gethostname(),
        pid=getpid(),
        julia_version=string(VERSION),
    )
end

function exception_text(error, backtrace)
    return sprint(showerror, error, backtrace)
end

function train_and_evaluate(config::NamedTuple)
    cfg = normalize_config(config)
    assert_remote_training_worker!(cfg)
    started_at = unix_milliseconds()
    elapsed_seconds = 0.0

    try
        train_result = nothing
        training_backend = ""
        fallback_reason = ""
        objects = nothing
        object_build_seconds = 0.0
        training_seconds = 0.0
        evaluation_seconds = 0.0

        object_build_seconds = @elapsed begin
            objects = training_objects_for_config(cfg)
        end
        training_seconds = @elapsed begin
            training = train_with_contextualdfl(objects, cfg)
            train_result = training.result
            training_backend = training.backend
            fallback_reason = training.fallback_reason
        end

        model = extract_model(train_result, objects.scenario_generator)
        split_metrics = if hasproperty(training, :final_metrics) && !isnothing(training.final_metrics)
            training.final_metrics
        else
            measured_metrics = nothing
            evaluation_seconds = @elapsed begin
                measured_metrics = evaluate_model_for_reporting(model, objects, cfg)
            end
            measured_metrics
        end
        elapsed_seconds = object_build_seconds + training_seconds + evaluation_seconds
        metrics = merge(
            split_metrics,
            (;
                training_backend=training_backend,
                fallback_reason=fallback_reason,
                object_build_seconds=object_build_seconds,
                training_seconds=training_seconds,
                evaluation_seconds=evaluation_seconds,
                total_elapsed_seconds=elapsed_seconds,
            ),
        )
        history = extract_epoch_history(train_result)

        return (;
            status="ok",
            run_id=string(config_value(cfg, :run_id, "")),
            config=cfg,
            worker=worker_metadata(),
            final_metrics=metrics,
            epoch_history=history,
            error="",
            started_at=started_at,
            finished_at=unix_milliseconds(),
            elapsed_seconds=elapsed_seconds,
        )
    catch error
        return (;
            status="failed",
            run_id=hasproperty(cfg, :run_id) ? cfg.run_id : "",
            config=cfg,
            worker=worker_metadata(),
            final_metrics=NamedTuple(),
            epoch_history=Dict{Symbol,Any}[],
            error=exception_text(error, catch_backtrace()),
            started_at=started_at,
            finished_at=unix_milliseconds(),
            elapsed_seconds=elapsed_seconds,
        )
    end
end

function train_with_contextualdfl(objects, config)
    if mlflow_enabled(config)
        mlflow_result = train_with_contextualdfl_mlflow(objects, config)
        return (;
            result=mlflow_result.result,
            backend="ContextualDFL.train! with ContextualDFLTraining MLflow",
            fallback_reason="",
            final_metrics=mlflow_result.final_metrics,
        )
    end

    result = ContextualDFL.train!(
        objects.scenario_generator.neural_net,
        objects.loss,
        mu_schedule_for_config(config),
        mu_ref_schedule_for_config(config),
        objects.data.train;
        rho_in_schedule=rho_schedule_for_config(config),
        rho_ref_schedule=rho_ref_schedule_for_config(config),
        learning_rate=config.learning_rate,
        optimizer_type=Flux.Adam,
        epochs=config.epochs,
        batchsize=config.batch_size,
        shuffle=Bool(config_value(config, :shuffle, false)),
        display_smooth=Bool(config_value(config, :display_smooth, false)),
        display_real=config_value(config, :display_real, nothing),
        display_reference_input=display_reference_input(objects, config),
    )
    trained_model = extract_model(result, objects.scenario_generator)
    save_flux_checkpoint_after_training!(trained_model, result, objects, config)

    return (;
        result=result,
        backend="ContextualDFL.train",
        fallback_reason="",
        final_metrics=nothing,
    )
end

function train_with_contextualdfl_mlflow(objects, config)
    mlflow_config = add_worker_mlflow_tags(config)
    mlf, experiment_id = mlflow_client_for_config(mlflow_config)
    loss = contextual_dfl_loss(objects, config)
    upload_model_artifact = Bool(config_value(config, :mlflow_upload_model_artifact, false))
    model_save_path = mlflow_model_save_path(config)
    final_metrics = Ref{Any}(nothing)
    model = objects.scenario_generator.neural_net
    mu_schedule = mu_schedule_for_config(config)
    mu_ref_schedule = mu_ref_schedule_for_config(config, mu_schedule)
    rho_schedule = rho_schedule_for_config(config)
    rho_ref_schedule = rho_ref_schedule_for_config(config, rho_schedule)
    run = createrun(mlf, experiment_id; start_time=unix_milliseconds())
    training_succeeded = false

    try
        log_contextualdfl_training_params!(
            mlf,
            run,
            loss,
            config,
            mu_schedule,
            mu_ref_schedule,
            rho_schedule,
            rho_ref_schedule,
        )
        log_mlflow_params!(mlf, run, "experiment", mlflow_experiment_spec(objects, config))
        log_mlflow_params!(mlf, run, "data", mlflow_data_spec(objects, config))
        log_mlflow_params!(mlf, run, "model", mlflow_model_spec(model, objects, config))
        log_mlflow_params!(mlf, run, "method", mlflow_method_spec(objects, config))
        log_mlflow_source_tags!(
            mlf,
            run;
            source_name=string(
                config_value(
                    config,
                    :mlflow_source_name,
                    "ContextualDFLTraining/gridsearch.jl",
                ),
            ),
            source_type=string(config_value(config, :mlflow_source_type, "LOCAL")),
            source_git_commit=git_commit_or_nothing(),
        )
        log_mlflow_dataset!(
            mlf,
            run;
            dataset_name=mlflow_dataset_name(objects, config),
            dataset_digest=mlflow_dataset_digest(objects, config),
            dataset_source_type=mlflow_dataset_source_type(objects, config),
            dataset_source=mlflow_dataset_source(objects, config),
            dataset_context="training",
        )

        result = ContextualDFL.train!(
            model,
            loss,
            mu_schedule,
            mu_ref_schedule,
            objects.data.train;
            rho_in_schedule=rho_schedule,
            rho_ref_schedule=rho_ref_schedule,
            learning_rate=config.learning_rate,
            optimizer_type=Flux.Adam,
            epochs=config.epochs,
            batchsize=config.batch_size,
            shuffle=Bool(config_value(config, :shuffle, false)),
            reset_optimizer_each_epoch=Bool(
                config_value(config, :reset_optimizer_each_epoch, false),
            ),
            save_model=upload_model_artifact,
            model_save_path=model_save_path,
            on_epoch_end=(epoch, loss_value, display_loss, metadata) -> log_mlflow_epoch!(
                mlf,
                run,
                epoch,
                loss_value,
                display_loss,
                metadata,
            ),
            nr_scenarios=effective_nr_scenarios(objects, config),
            display_smooth=Bool(config_value(config, :display_smooth, false)),
            display_real=config_value(config, :display_real, nothing),
            display_reference_input=display_reference_input(objects, config),
        )

        trained_model = extract_model(result, objects.scenario_generator)
        checkpoint = save_flux_checkpoint_after_training!(
            trained_model,
            result,
            objects,
            mlflow_config,
        )
        log_mlflow_checkpoint_artifact!(mlf, run, checkpoint, mlflow_config)

        if upload_model_artifact && isfile(model_save_path)
            upload_mlflow_artifact!(
                mlf,
                run,
                model_save_path;
                artifact_path="models/" * basename(model_save_path),
            )
        end

        metrics = evaluate_model_for_reporting(trained_model, objects, config)
        final_metrics[] = metrics
        log_mlflow_evaluation_result!(mlf, run, "", metrics)

        training_succeeded = true
        return (; result=result, final_metrics=final_metrics[])
    catch error
        error_text = exception_text(error, catch_backtrace())
        try
            log_mlflow_stacktrace_artifact!(mlf, run, error_text)
        catch mlflow_error
            @warn "Failed to upload MLflow stacktrace artifact" error=exception_text(
                mlflow_error,
                catch_backtrace(),
            )
        end
        rethrow()
    finally
        status = training_succeeded ? RunStatus.FINISHED : RunStatus.FAILED
        try
            updaterun(mlf, run; status=status, end_time=unix_milliseconds())
        catch
            training_succeeded && rethrow()
        end
    end
end

function log_contextualdfl_training_params!(
    mlf,
    run,
    loss,
    config,
    mu_schedule,
    mu_ref_schedule,
    rho_schedule,
    rho_ref_schedule,
)
    logparam(mlf, run, "learning_rate", string(config.learning_rate))
    logparam(mlf, run, "optimizer_type", string(Flux.Adam))
    logparam(mlf, run, "epochs", string(config.epochs))
    logparam(mlf, run, "batchsize", string(config.batch_size))
    logged_scenarios = logged_nr_scenarios(
        loss,
        config_value(config, :nr_scenarios, nothing),
    )
    isnothing(logged_scenarios) ||
        logparam(mlf, run, "nr_scenarios", string(logged_scenarios))
    logparam(mlf, run, "mu_in_schedule", string(collect(mu_schedule)))
    logparam(mlf, run, "mu_ref_schedule", string(collect(mu_ref_schedule)))
    logparam(mlf, run, "rho_in_schedule", string(collect(rho_schedule)))
    logparam(mlf, run, "rho_ref_schedule", string(collect(rho_ref_schedule)))
    logparam(
        mlf,
        run,
        "display_smooth",
        string(Bool(config_value(config, :display_smooth, false))),
    )
    logparam(
        mlf,
        run,
        "display_real",
        string(config_value(config, :display_real, nothing)),
    )
    logparam(mlf, run, "shuffle", string(Bool(config_value(config, :shuffle, false))))
    logparam(
        mlf,
        run,
        "reset_optimizer_each_epoch",
        string(Bool(config_value(config, :reset_optimizer_each_epoch, false))),
    )
    training_seed = config_value(config, :training_data_seed, nothing)
    if training_seed !== nothing && training_seed !== missing
        logparam(mlf, run, "training_data_seed", string(training_seed))
    end
    repeat_training_seed = config_value(config, :repeat_training_data_seed, training_seed)
    if repeat_training_seed !== nothing && repeat_training_seed !== missing
        logparam(mlf, run, "repeat_training_data_seed", string(repeat_training_seed))
    end
    repeat_index = config_value(config, :repeat_index, nothing)
    if repeat_index !== nothing && repeat_index !== missing
        logparam(mlf, run, "repeat_index", string(repeat_index))
    end
    return nothing
end

function display_reference_input(objects, config)
    display_smooth = Bool(config_value(config, :display_smooth, false))
    display_real = config_value(config, :display_real, nothing)
    (display_smooth || !isnothing(display_real)) || return nothing
    hasproperty(objects, :target_extractor) && return objects.target_extractor
    return nothing
end

function logged_nr_scenarios(loss, nr_scenarios)
    if hasproperty(loss, :nr_scenarios)
        return Int(getproperty(loss, :nr_scenarios))
    end
    isnothing(nr_scenarios) || return Int(nr_scenarios)
    return nothing
end

function effective_nr_scenarios(objects, config)
    if hasproperty(objects, :loss)
        value = logged_nr_scenarios(objects.loss, nothing)
        isnothing(value) || return value
    end

    value = config_value(config, :nr_scenarios, nothing)
    isnothing(value) && return nothing
    return Int(value)
end

function add_worker_mlflow_tags(config)
    tags = string_dict(config_value(config, :mlflow_tags, nothing))
    tags["worker_id"] = string(Distributed.myid())
    tags["worker_hostname"] = Sockets.gethostname()
    tags["worker_pid"] = string(getpid())
    parent_run_id = string(config_value(config, :mlflow_parent_run_id, ""))
    if !isempty(parent_run_id)
        tags["gridsearch_parent_run_id"] = parent_run_id
        tags["mlflow.parentRunId"] = parent_run_id
    end
    return merge(config, (; mlflow_tags=tags))
end

function assert_remote_training_worker!(config)
    Bool(config_value(config, :allow_local_training, false)) && return nothing

    Distributed.myid() == 1 &&
        error("Refusing to run training on Distributed worker 1. Use the remote gridsearch/profile entry points.")

    coordinator_hostname = string(config_value(config, :coordinator_hostname, ""))
    if !isempty(coordinator_hostname) && Sockets.gethostname() == coordinator_hostname
        error("Refusing to run training on coordinator host $(coordinator_hostname).")
    end

    return nothing
end

function object_metadata(objects, field::Symbol)
    hasproperty(objects, field) || return NamedTuple()
    value = getproperty(objects, field)
    return value isa NamedTuple ? value : NamedTuple()
end

function metadata_value(metadata::NamedTuple, key::Symbol, default)
    return key in keys(metadata) ? getproperty(metadata, key) : default
end

function mlflow_dataset_name(config)
    return string(
        config_value(
            config,
            :mlflow_dataset_name,
            config_value(config, :experiment_name, "generated_dataset"),
        ),
    )
end

function mlflow_dataset_name(objects, config)
    data_metadata = object_metadata(objects, :data_metadata)
    name = metadata_value(data_metadata, :dataset_name, nothing)
    isnothing(name) || return string(name)
    return mlflow_dataset_name(config)
end

function mlflow_dataset_source(objects, config)
    data_metadata = object_metadata(objects, :data_metadata)
    source = metadata_value(data_metadata, :dataset_source, nothing)
    isnothing(source) || return string(source)

    parts = String["ContextualDFLTraining.experiment"]
    hasproperty(config, :experiment_id) && push!(parts, "experiment_id=$(config.experiment_id)")
    hasproperty(config, :training_data_seed) &&
        push!(parts, "training_data_seed=$(config.training_data_seed)")
    hasproperty(config, :validation_fraction) &&
        push!(parts, "validation_fraction=$(config.validation_fraction)")
    hasproperty(config, :test_fraction) && push!(parts, "test_fraction=$(config.test_fraction)")
    return join(parts, ";")
end

function mlflow_dataset_source_type(objects, config)
    data_metadata = object_metadata(objects, :data_metadata)
    path = metadata_value(data_metadata, :dataset_path, "")
    (path === nothing || path === missing || isempty(string(path))) || return "local"
    return string(config_value(config, :mlflow_dataset_source_type, "generated"))
end

function mlflow_dataset_digest(objects, config)
    data_metadata = object_metadata(objects, :data_metadata)
    digest = metadata_value(data_metadata, :dataset_digest, nothing)
    isnothing(digest) || return string(digest)

    split_summary = (
        "dataset=$(mlflow_dataset_name(objects, config))",
        "training_data_seed=$(config_value(config, :training_data_seed, ""))",
        "train_x=$(size(dataset_context_matrix(objects.data.train)))",
        "train_y=$(size(dataset_target_matrix(objects.data.train, objects)))",
        "validation_x=$(size(dataset_context_matrix(objects.data.validation)))",
        "validation_y=$(size(dataset_target_matrix(objects.data.validation, objects)))",
        "test_x=$(size(dataset_context_matrix(objects.data.test)))",
        "test_y=$(size(dataset_target_matrix(objects.data.test, objects)))",
    )
    return short_mlflow_digest(split_summary)
end

function short_mlflow_digest(values)
    return bytes2hex(sha256(join(values, "\n")))[1:32]
end

function mlflow_model_save_path(config)
    run_id = string(config_value(config, :run_id, "training-run"))
    safe_run_id = replace(run_id, r"[^A-Za-z0-9_.=-]" => "_")
    return joinpath(tempdir(), safe_run_id * ".jls")
end

function checkpoint_enabled(config)
    return Bool(config_value(config, :checkpoint_enabled, true))
end

function checkpoint_upload_mlflow(config)
    return Bool(config_value(config, :checkpoint_upload_mlflow, true))
end

function checkpoint_required(config)
    return Bool(config_value(config, :checkpoint_required, false))
end

function checkpoint_format(config)
    format = Symbol(config_value(config, :checkpoint_format, :jls))
    format == :jls ||
        throw(ArgumentError("unsupported checkpoint_format $(format); use :jls."))
    return format
end

function default_checkpoint_root()
    return joinpath(dirname(@__DIR__), "results", "checkpoints")
end

function checkpoint_directory(config)
    configured_dir = string(config_value(config, :checkpoint_dir, ""))
    if isempty(strip(configured_dir))
        grid_id = string(config_value(config, :gridsearch_id, "standalone"))
        return joinpath(default_checkpoint_root(), safe_checkpoint_identifier(grid_id))
    end
    return abspath(configured_dir)
end

function checkpoint_save_path(config)
    format = checkpoint_format(config)
    run_id = string(config_value(config, :run_id, "training-run"))
    filename = safe_checkpoint_identifier(run_id) * "_checkpoint." * string(format)
    return joinpath(checkpoint_directory(config), filename)
end

function safe_checkpoint_identifier(value)
    text = replace(string(value), r"[^A-Za-z0-9_.=-]+" => "_")
    text = strip(text, ['_'])
    return isempty(text) ? "training-run" : text
end

function save_flux_checkpoint_after_training!(model, train_result, objects, config)
    checkpoint_enabled(config) || return nothing

    path = checkpoint_save_path(config)
    try
        save_flux_checkpoint!(path, model, train_result, objects, config)
        return path
    catch error
        checkpoint_required(config) && rethrow()
        @warn "Failed to save Flux checkpoint" path error=exception_text(error, catch_backtrace())
        return nothing
    end
end

function save_flux_checkpoint!(path::AbstractString, model, train_result, objects, config)
    mkpath(dirname(path))
    payload = flux_checkpoint_payload(model, train_result, objects, config, path)
    temp_path = tempname(dirname(path))
    open(temp_path, "w") do io
        Serialization.serialize(io, payload)
    end
    mv(temp_path, path; force=true)
    return path
end

function flux_checkpoint_payload(model, train_result, objects, config, path)
    return (;
        format_version=1,
        checkpoint_format=:jls,
        saved_at=Dates.format(now(UTC), dateformat"yyyy-mm-ddTHH:MM:SS.sZ"),
        saved_at_unix_ms=unix_milliseconds(),
        checkpoint_path=String(path),
        run_id=string(config_value(config, :run_id, "")),
        gridsearch_id=string(config_value(config, :gridsearch_id, "")),
        candidate_index=config_value(config, :candidate_index, missing),
        repeat_index=config_value(config, :repeat_index, missing),
        worker=worker_metadata(),
        config=config,
        model_state=Flux.state(model),
        optimizer_state=optimizer_flux_state(train_result),
        epoch_history=extract_epoch_history(train_result),
        model_metadata=mlflow_model_spec(model, objects, config),
        data_metadata=mlflow_data_spec(objects, config),
    )
end

function optimizer_flux_state(train_result)
    hasproperty(train_result, :opt_state) || return missing
    opt_state = getproperty(train_result, :opt_state)
    (opt_state === nothing || opt_state === missing) && return missing
    return Flux.state(opt_state)
end

function log_mlflow_checkpoint_artifact!(mlf, run, checkpoint_path, config)
    checkpoint_path === nothing && return nothing
    logparam(mlf, run, "checkpoint_path", string(checkpoint_path))
    logparam(mlf, run, "checkpoint_format", string(checkpoint_format(config)))

    checkpoint_upload_mlflow(config) || return nothing

    artifact_path = "checkpoints/" * basename(checkpoint_path)
    try
        upload_mlflow_artifact!(mlf, run, checkpoint_path; artifact_path=artifact_path)
        logparam(mlf, run, "checkpoint_artifact_path", artifact_path)
    catch error
        checkpoint_required(config) && rethrow()
        @warn "Failed to upload Flux checkpoint artifact" artifact_path error=exception_text(
            error,
            catch_backtrace(),
        )
    end

    return nothing
end

function contextual_dfl_loss(objects, config)
    return objects.loss
end

function mu_schedule_for_config(config)
    epochs = Int(config.epochs)
    epochs >= 0 || throw(ArgumentError("epochs must be non-negative."))
    epochs == 0 && return Float64[]

    raw_schedule = config_value(config, :mu_schedule, :constant)
    if raw_schedule isa AbstractVector
        length(raw_schedule) == epochs ||
            throw(ArgumentError("mu_schedule vector must have one value per epoch."))
        return Float64.(raw_schedule)
    end

    schedule = Symbol(raw_schedule)
    mu_start = Float64(config_value(config, :mu_start, config.mu))
    mu_end = Float64(config_value(config, :mu_end, config.mu))

    if schedule == :constant
        return fill(Float64(config.mu), epochs)
    elseif schedule == :linear
        epochs == 1 && return [mu_start]
        return collect(range(mu_start, mu_end; length=epochs))
    elseif schedule == :geometric || schedule == :exponential
        mu_start > 0 && mu_end > 0 ||
            throw(ArgumentError("$schedule mu annealing requires positive mu_start and mu_end."))
        epochs == 1 && return [mu_start]
        return exp.(range(log(mu_start), log(mu_end); length=epochs))
    end

    throw(ArgumentError("unsupported mu_schedule $(schedule)"))
end

function policy_inference_mu_for_config(config, mu_schedule=nothing)
    if config isa NamedTuple && :policy_inference_mu in keys(config)
        policy_mu = config.policy_inference_mu
        (policy_mu === nothing || policy_mu === missing) || return Float64(policy_mu)
    end

    resolved_mu_schedule =
        isnothing(mu_schedule) ? mu_schedule_for_config(config) : mu_schedule
    isempty(resolved_mu_schedule) && return Float64(config_value(config, :mu, 0.0))
    return Float64(last(resolved_mu_schedule))
end

function policy_inference_rho_for_config(config, rho_schedule=nothing)
    if config isa NamedTuple && :policy_inference_rho in keys(config)
        policy_rho = config.policy_inference_rho
        (policy_rho === nothing || policy_rho === missing) || return Float64(policy_rho)
    end

    resolved_rho_schedule =
        isnothing(rho_schedule) ? rho_schedule_for_config(config) : rho_schedule
    isempty(resolved_rho_schedule) && return Float64(config_value(config, :rho, 0.0))
    return Float64(last(resolved_rho_schedule))
end

function mu_ref_schedule_for_config(config, mu_schedule=mu_schedule_for_config(config))
    epochs = Int(config.epochs)
    raw_schedule = config_value(config, :mu_ref_schedule, :match_input)

    if raw_schedule isa AbstractVector
        length(raw_schedule) == epochs ||
            throw(ArgumentError("mu_ref_schedule vector must have one value per epoch."))
        return Float64.(raw_schedule)
    end

    schedule = Symbol(raw_schedule)
    if schedule in (:match_input, :same, :input)
        length(mu_schedule) == epochs ||
            throw(ArgumentError("mu_schedule must have one value per epoch."))
        return Float64.(mu_schedule)
    elseif schedule in (:zero, :zeros, :none)
        return zeros(Float64, epochs)
    elseif schedule == :constant
        return fill(Float64(config_value(config, :mu_ref, config.mu)), epochs)
    elseif schedule == :linear
        epochs == 1 && return [Float64(config_value(config, :mu_ref_start, config_value(config, :mu_start, config.mu)))]
        return collect(
            range(
                Float64(config_value(config, :mu_ref_start, config_value(config, :mu_start, config.mu))),
                Float64(config_value(config, :mu_ref_end, config_value(config, :mu_end, config.mu)));
                length=epochs,
            ),
        )
    elseif schedule == :geometric || schedule == :exponential
        mu_ref_start = Float64(config_value(config, :mu_ref_start, config_value(config, :mu_start, config.mu)))
        mu_ref_end = Float64(config_value(config, :mu_ref_end, config_value(config, :mu_end, config.mu)))
        mu_ref_start > 0 && mu_ref_end > 0 ||
            throw(ArgumentError("$schedule mu_ref annealing requires positive mu_ref_start and mu_ref_end."))
        epochs == 1 && return [mu_ref_start]
        return exp.(range(log(mu_ref_start), log(mu_ref_end); length=epochs))
    end

    throw(ArgumentError("unsupported mu_ref_schedule $(schedule)"))
end

function rho_schedule_for_config(config)
    epochs = Int(config.epochs)
    epochs >= 0 || throw(ArgumentError("epochs must be non-negative."))
    epochs == 0 && return Float64[]

    raw_schedule = config_value(config, :rho_schedule, :constant)
    if raw_schedule isa AbstractVector
        length(raw_schedule) == epochs ||
            throw(ArgumentError("rho_schedule vector must have one value per epoch."))
        return Float64.(raw_schedule)
    end

    schedule = Symbol(raw_schedule)
    rho_start = Float64(config_value(config, :rho_start, config.rho))
    rho_end = Float64(config_value(config, :rho_end, config.rho))

    if schedule == :constant
        return fill(Float64(config.rho), epochs)
    elseif schedule == :linear
        epochs == 1 && return [rho_start]
        return collect(range(rho_start, rho_end; length=epochs))
    elseif schedule == :geometric || schedule == :exponential
        rho_start > 0 && rho_end > 0 ||
            throw(ArgumentError("$schedule rho annealing requires positive rho_start and rho_end."))
        epochs == 1 && return [rho_start]
        return exp.(range(log(rho_start), log(rho_end); length=epochs))
    end

    throw(ArgumentError("unsupported rho_schedule $(schedule)"))
end

function rho_ref_schedule_for_config(config, rho_schedule=rho_schedule_for_config(config))
    epochs = Int(config.epochs)
    raw_schedule = config_value(config, :rho_ref_schedule, :match_input)

    if raw_schedule isa AbstractVector
        length(raw_schedule) == epochs ||
            throw(ArgumentError("rho_ref_schedule vector must have one value per epoch."))
        return Float64.(raw_schedule)
    end

    schedule = Symbol(raw_schedule)
    if schedule in (:match_input, :same, :input)
        length(rho_schedule) == epochs ||
            throw(ArgumentError("rho_schedule must have one value per epoch."))
        return Float64.(rho_schedule)
    elseif schedule in (:zero, :zeros, :none)
        return zeros(Float64, epochs)
    elseif schedule == :constant
        return fill(Float64(config_value(config, :rho_ref, config.rho)), epochs)
    elseif schedule == :linear
        epochs == 1 && return [Float64(config_value(config, :rho_ref_start, config_value(config, :rho_start, config.rho)))]
        return collect(
            range(
                Float64(config_value(config, :rho_ref_start, config_value(config, :rho_start, config.rho))),
                Float64(config_value(config, :rho_ref_end, config_value(config, :rho_end, config.rho)));
                length=epochs,
            ),
        )
    elseif schedule == :geometric || schedule == :exponential
        rho_ref_start = Float64(config_value(config, :rho_ref_start, config_value(config, :rho_start, config.rho)))
        rho_ref_end = Float64(config_value(config, :rho_ref_end, config_value(config, :rho_end, config.rho)))
        rho_ref_start > 0 && rho_ref_end > 0 ||
            throw(ArgumentError("$schedule rho_ref annealing requires positive rho_ref_start and rho_ref_end."))
        epochs == 1 && return [rho_ref_start]
        return exp.(range(log(rho_ref_start), log(rho_ref_end); length=epochs))
    end

    throw(ArgumentError("unsupported rho_ref_schedule $(schedule)"))
end

function mlflow_experiment_spec(objects, config)
    problem_metadata = object_metadata(objects, :problem_metadata)
    return (;
        problem=string(
            metadata_value(
                problem_metadata,
                :problem,
                config_value(config, :problem, :experiment),
            ),
        ),
        instance_id=metadata_value(problem_metadata, :instance_id, missing),
        method=string(config_value(config, :method, config.loss)),
        variant=string(config_value(config, :method_variant, "default")),
        run_group=string(config_value(config, :gridsearch_id, "")),
        candidate_index=config_value(config, :candidate_index, ""),
        replicate_index=config_value(
            config,
            :repeat_index,
            config_value(config, :replicate_index, missing),
        ),
        base_run_id=string(config_value(config, :base_run_id, "")),
    )
end

function mlflow_data_spec(objects, config)
    train_size = length(objects.data.train)
    validation_size = length(objects.data.validation)
    test_size = length(objects.data.test)
    context_dimension = isempty(objects.data.train) ? 0 : length(first(objects.data.train).context)
    scenario_count =
        isempty(objects.data.train) ? 0 : length(first(objects.data.train).scenario_parameters)
    target_dimension = isempty(objects.data.train) ?
        0 :
        length(target_from_contextual_point(first(objects.data.train), objects))

    defaults = (;
        generator="experiment_config",
        dataset_name=mlflow_dataset_name(objects, config),
        dataset_digest=mlflow_dataset_digest(objects, config),
        train_size=train_size,
        validation_size=validation_size,
        test_size=test_size,
        context_dimension=context_dimension,
        scenario_count=scenario_count,
        target_dimension=target_dimension,
        validation_fraction=config_value(config, :validation_fraction, missing),
        test_fraction=config_value(config, :test_fraction, missing),
        training_data_seed=config_value(config, :training_data_seed, missing),
        repeat_training_data_seed=config_value(
            config,
            :repeat_training_data_seed,
            config_value(config, :training_data_seed, missing),
        ),
        train_context_seed=config_value(config, :training_data_seed, missing),
        train_scenario_seed=config_value(config, :training_data_seed, missing),
        split_seed=config_value(config, :training_data_seed, missing),
        optimization_seed=config_value(config, :optimization_seed, missing),
    )
    return merge(defaults, object_metadata(objects, :data_metadata))
end

function mlflow_model_spec(model, objects, config)
    defaults = (;
        architecture="Flux.Chain",
        depth=config_value(config, :depth, missing),
        width=config_value(config, :hidden_size, missing),
        activation=string(config_value(config, :activation, "relu")),
        output_activation="softplus",
        dropout=config_value(config, :dropout, missing),
        parameter_count=model_parameter_count(model),
        initialization_seed=string(
            config_value(
                config,
                :model_initialization_seed,
                config_value(config, :seed, "global_rng"),
            ),
        ),
        input_dimension=isempty(objects.data.train) ? 0 : length(first(objects.data.train).context),
        output_dimension=model_output_dimension(objects),
    )
    return merge(defaults, object_metadata(objects, :model_metadata))
end

function mlflow_method_spec(objects, config)
    mu_schedule = mu_schedule_for_config(config)
    mu_ref_schedule = mu_ref_schedule_for_config(config, mu_schedule)
    rho_schedule = rho_schedule_for_config(config)
    rho_ref_schedule = rho_ref_schedule_for_config(config, rho_schedule)
    policy_inference_mu = policy_inference_mu_for_config(config, mu_schedule)
    policy_inference_rho = policy_inference_rho_for_config(config, rho_schedule)
    return (;
        loss=string(config.loss),
        solver=string(config.solver),
        decoder=string(typeof(objects.scenario_decoder)),
        reference_decoder=string(typeof(objects.reference_scenario_decoder)),
        learned_components="h",
        nr_scenarios=something(effective_nr_scenarios(objects, config), 1),
        mu=config.mu,
        mu_start=isempty(mu_schedule) ? missing : first(mu_schedule),
        mu_end=isempty(mu_schedule) ? missing : last(mu_schedule),
        mu_schedule=string(config_value(config, :mu_schedule, :constant)),
        mu_ref=Float64(config_value(config, :mu_ref, config.mu)),
        mu_ref_start=isempty(mu_ref_schedule) ? missing : first(mu_ref_schedule),
        mu_ref_end=isempty(mu_ref_schedule) ? missing : last(mu_ref_schedule),
        mu_ref_schedule=string(config_value(config, :mu_ref_schedule, :match_input)),
        rho=config.rho,
        rho_start=isempty(rho_schedule) ? missing : first(rho_schedule),
        rho_end=isempty(rho_schedule) ? missing : last(rho_schedule),
        rho_schedule=string(config_value(config, :rho_schedule, :constant)),
        rho_ref=Float64(config_value(config, :rho_ref, config.rho)),
        rho_ref_start=isempty(rho_ref_schedule) ? missing : first(rho_ref_schedule),
        rho_ref_end=isempty(rho_ref_schedule) ? missing : last(rho_ref_schedule),
        rho_ref_schedule=string(config_value(config, :rho_ref_schedule, :match_input)),
        homotopy_schedule=string(config_value(config, :mu_schedule, :constant)),
        log_barrier_training=any(!iszero, mu_schedule),
        reference_log_barrier_training=any(!iszero, mu_ref_schedule),
        quadratic_smoothing_training=any(!iszero, rho_schedule),
        reference_quadratic_smoothing_training=any(!iszero, rho_ref_schedule),
        log_barrier_inference=Bool(config_value(config, :log_barrier_inference, any(!iszero, mu_schedule))),
        display_smooth=Bool(config_value(config, :display_smooth, false)),
        display_real=config_value(config, :display_real, nothing),
        optimality_evaluation=Bool(config_value(config, :optimality_evaluation, false)),
        optimality_test_sample_count=Int(config_value(config, :optimality_test_sample_count, 0)),
        optimality_train_sample_count=Int(config_value(config, :optimality_train_sample_count, 0)),
        optimality_validation_sample_count=Int(config_value(config, :optimality_validation_sample_count, 0)),
        optimality_mu=Float64(config_value(config, :optimality_mu, 0.0)),
        optimality_rho=Float64(config_value(config, :optimality_rho, 0.0)),
        optimality_evaluation_batches=config_value(config, :optimality_evaluation_batches, nothing),
        policy_inference_mu=policy_inference_mu,
        policy_inference_rho=policy_inference_rho,
        fine_tuning=Bool(config_value(config, :fine_tuning, false)),
        annealing=Bool(config_value(config, :annealing, false)),
        knn_homogenization=Bool(config_value(config, :knn_homogenization, false)),
        rrule_variant=string(config_value(config, :rrule_variant, "default")),
    )
end

function model_parameter_count(model)
    try
        return sum(length, Flux.trainables(model))
    catch
        try
            return sum(length, Flux.params(model))
        catch
            return missing
        end
    end
end

function split_mse(model, dataset, objects)
    target = dataset_target_matrix(dataset, objects)
    prediction = matrix_like(model(dataset_context_matrix(dataset)), target)
    return mean(abs2, prediction .- target)
end

function dataset_context_matrix(dataset)
    isempty(dataset) && return zeros(Float64, 0, 0)
    return reduce(hcat, (point.context for point in dataset))
end

function dataset_target_matrix(dataset, objects)
    isempty(dataset) && return zeros(Float64, 0, 0)
    return reduce(hcat, (target_from_contextual_point(point, objects) for point in dataset))
end

function target_from_contextual_point(point, objects)
    extractor = target_extractor(objects)
    target = extractor(point)
    target isa AbstractVector ||
        throw(ArgumentError("training object target_extractor must return an AbstractVector."))
    return target
end

function target_extractor(objects)
    if hasproperty(objects, :target_extractor)
        extractor = getproperty(objects, :target_extractor)
        extractor isa Function ||
            throw(ArgumentError("training object target_extractor must be a function."))
        return extractor
    end

    throw(
        ArgumentError(
            "training objects must provide target_extractor for reporting and MSE evaluation.",
        ),
    )
end

function model_output_dimension(objects)
    model_metadata = object_metadata(objects, :model_metadata)
    output_dimension = metadata_value(model_metadata, :output_dimension, nothing)
    isnothing(output_dimension) || return output_dimension

    isempty(objects.data.train) && return 0
    return length(target_from_contextual_point(first(objects.data.train), objects))
end

function extract_model(train_result, fallback_generator)
    candidates = Any[train_result, fallback_generator]

    if train_result isa Tuple
        append!(candidates, collect(train_result))
    end

    for candidate in candidates
        candidate === nothing && continue

        if hasproperty(candidate, :scenario_generator)
            scenario_generator = getproperty(candidate, :scenario_generator)
            hasproperty(scenario_generator, :neural_net) &&
                return getproperty(scenario_generator, :neural_net)
        end

        hasproperty(candidate, :neural_net) && return getproperty(candidate, :neural_net)
        hasproperty(candidate, :model) && return getproperty(candidate, :model)
    end

    return fallback_generator.neural_net
end

function extract_epoch_history(train_result)
    raw_history = find_history_payload(train_result)
    return normalize_history(raw_history)
end

function find_history_payload(train_result)
    train_result === nothing && return nothing

    if hasproperty(train_result, :history)
        return getproperty(train_result, :history)
    end
    if hasproperty(train_result, :metrics)
        return getproperty(train_result, :metrics)
    end
    if hasproperty(train_result, :epoch_history)
        return getproperty(train_result, :epoch_history)
    end

    if train_result isa Tuple
        for item in train_result
            payload = find_history_payload(item)
            payload === nothing || return payload
        end
    end

    return train_result isa AbstractVector ? train_result : nothing
end

function normalize_history(raw_history)
    raw_history === nothing && return Dict{Symbol,Any}[]

    if raw_history isa NamedTuple
        return normalize_namedtuple_history(raw_history)
    end

    if raw_history isa AbstractVector
        rows = Dict{Symbol,Any}[]
        for (index, row) in enumerate(raw_history)
            push!(rows, normalize_history_row(row, index))
        end
        return rows
    end

    return [Dict{Symbol,Any}(:epoch => 1, :value => string(raw_history))]
end

function normalize_namedtuple_history(history::NamedTuple)
    vector_lengths = [
        length(value) for value in values(history) if value isa AbstractVector
    ]
    isempty(vector_lengths) && return [Dict{Symbol,Any}(pairs(history))]

    row_count = maximum(vector_lengths)
    rows = Dict{Symbol,Any}[]

    for index in 1:row_count
        row = Dict{Symbol,Any}(:epoch => index)
        for key in keys(history)
            value = getproperty(history, key)
            if value isa AbstractVector
                row[key] = index <= length(value) ? value[index] : missing
            else
                row[key] = value
            end
        end
        push!(rows, row)
    end

    return rows
end

function normalize_history_row(row::NamedTuple, index)
    output = Dict{Symbol,Any}(pairs(row))
    haskey(output, :epoch) || (output[:epoch] = index)
    return output
end

function normalize_history_row(row::AbstractDict, index)
    output = Dict{Symbol,Any}()
    for (key, value) in row
        output[Symbol(key)] = value
    end
    haskey(output, :epoch) || (output[:epoch] = index)
    return output
end

function normalize_history_row(row::Number, index)
    return Dict{Symbol,Any}(:epoch => index, :value => Float64(row))
end

function normalize_history_row(row, index)
    return Dict{Symbol,Any}(:epoch => index, :value => string(row))
end

function evaluate_model_on_splits(model, splits, objects, config)
    try
        Flux.testmode!(model)
    catch
    end

    train_metrics = evaluate_split(model, splits.train, objects, config, "train")
    validation_metrics = evaluate_split(model, splits.validation, objects, config, "validation")
    test_metrics = evaluate_split(model, splits.test, objects, config, "test")
    return merge(train_metrics, validation_metrics, test_metrics)
end

function evaluate_model_for_reporting(model, objects, config)
    metrics = evaluate_model_on_splits(model, objects.data, objects, config)
    Bool(config_value(config, :optimality_evaluation, false)) || return metrics
    return merge(metrics, evaluate_optimality_on_splits(model, objects, config))
end

function evaluate_optimality_on_splits(model, objects, config)
    spec = experiment_from_config(config)
    spec === nothing && throw(
        ArgumentError(
            "optimality_evaluation=true requires config.experiment_id so precomputed optimal results can be loaded.",
        ),
    )

    policy = optimality_policy(model, objects, config)
    metrics = NamedTuple()

    for (split_name, dataset) in optimality_splits_for_config(objects, config)
        isempty(dataset) && continue
        optimal_results = load_optimal_results(spec, split_name; dataset=dataset)
        result = ContextualDFLExperiments.evaluate_policy_against_optimum(
            policy,
            dataset,
            objects.program,
            objects.reference_scenario_decoder,
            objects.solver;
            optimal_results=optimal_results,
            split_name=split_name,
            mu=Float64(config_value(config, :optimality_mu, 0.0)),
            rho=Float64(config_value(config, :optimality_rho, 0.0)),
        )
        metrics = merge(metrics, result.metrics)
    end

    return metrics
end

function optimality_policy(model, objects, config)
    scenario_generator = ContextualDFL.ScenarioGenerator(
        neural_net=model,
        scenario_decoder=objects.scenario_decoder,
    )
    policy_mu = policy_inference_mu_for_config(config)
    policy_rho = policy_inference_rho_for_config(config)
    return ContextualDFLExperiments.ScenarioGenerationPolicy(
        scenario_generator,
        objects.solver,
        objects.program;
        mu=policy_mu,
        rho=policy_rho,
    )
end

function optimality_evaluation_datasets(objects, config)
    datasets = Pair{Symbol,Any}[]

    push!(
        datasets,
        :test => limited_dataset(
            objects.data.test,
            Int(config_value(config, :optimality_test_sample_count, 0)),
        ),
    )

    train_count = Int(config_value(config, :optimality_train_sample_count, 0))
    train_count > 0 && push!(
        datasets,
        :train_subset => limited_dataset(objects.data.train, train_count),
    )

    validation_count = Int(config_value(config, :optimality_validation_sample_count, 0))
    validation_count > 0 && push!(
        datasets,
        :validation_subset => limited_dataset(objects.data.validation, validation_count),
    )

    return datasets
end

function limited_dataset(dataset, limit::Integer)
    limit <= 0 && return dataset
    return dataset[1:min(Int(limit), length(dataset))]
end

function evaluate_split(model, dataset, objects, config, prefix)
    x_data = dataset_context_matrix(dataset)
    target = dataset_target_matrix(dataset, objects)
    predictions, inference_timings = timed_model_prediction(model, x_data, config)
    target = reporting_target_for_prediction(target, predictions)
    prediction_matrix = matrix_like(predictions, target)

    errors = prediction_matrix .- target
    absolute_errors = abs.(errors)
    denominator = max.(abs.(target), config.tolerance_absolute_floor)
    tolerance = max.(abs.(target) .* config.tolerance_relative, config.tolerance_absolute_floor)

    metrics = (;
        mse=mean(abs2, errors),
        mae=mean(absolute_errors),
        rmse=sqrt(mean(abs2, errors)),
        relative_mae=mean(absolute_errors ./ denominator),
        tolerance_accuracy=mean(absolute_errors .<= tolerance),
        sample_count=size(target, 2),
        inference_seconds_mean=mean(inference_timings),
        inference_seconds_p95=percentile_95(inference_timings),
        inference_seconds_total=sum(inference_timings),
    )

    return prefix_named_tuple(Symbol(prefix), metrics)
end

function timed_model_prediction(model, x_data, config)
    repetitions = max(Int(config_value(config, :inference_repetitions, 1)), 1)
    timings = Float64[]
    predictions = nothing

    for _ in 1:repetitions
        elapsed = @elapsed begin
            predictions = model(x_data)
        end
        push!(timings, elapsed)
    end

    return predictions, timings
end

function reporting_target_for_prediction(target, prediction)
    output_dimension = reporting_prediction_output_dimension(prediction, target)
    output_dimension === nothing && return target
    size(target, 1) == output_dimension && return target
    size(target, 1) % output_dimension == 0 || return target

    scenario_count = size(target, 1) ÷ output_dimension
    scenario_count > 1 || return target

    scenario_target = reshape(target, output_dimension, scenario_count, size(target, 2))
    return dropdims(mean(scenario_target; dims=2); dims=2)
end

function reporting_prediction_output_dimension(prediction, target)
    prediction_matrix = Array(prediction)
    if ndims(prediction_matrix) == 2 && size(prediction_matrix, 2) == size(target, 2)
        return size(prediction_matrix, 1)
    elseif ndims(prediction_matrix) == 1 && size(target, 2) == 1
        return length(prediction_matrix)
    end
    return nothing
end

function percentile_95(values::AbstractVector{<:Real})
    isempty(values) && return NaN
    sorted = sort!(collect(Float64.(values)))
    index = clamp(ceil(Int, 0.95 * length(sorted)), 1, length(sorted))
    return sorted[index]
end

function matrix_like(value, target)
    matrix = Array(value)
    size(matrix) == size(target) && return matrix
    length(matrix) == length(target) && return reshape(matrix, size(target))

    throw(
        DimensionMismatch(
            "prediction size $(size(matrix)) cannot be compared with target size $(size(target))",
        ),
    )
end

function prefix_named_tuple(prefix::Symbol, values::NamedTuple)
    prefixed_pairs = Pair{Symbol,Any}[]
    for key in keys(values)
        push!(prefixed_pairs, Symbol(prefix, "_", key) => getproperty(values, key))
    end
    return (; prefixed_pairs...)
end

# END FILE: src/ContextualDFL/ContextualDFLTraining/src/train_run.jl

# BEGIN FILE: src/ContextualDFL/ContextualDFLTraining/src/training_helpers.jl
using ContextualDFL
using Flux
using Random

struct ConstantSchedule{T}
    value::T
end

(schedule::ConstantSchedule)(args...; kwargs...) = schedule.value

function build_neural_net(
    input_dimension,
    output_dimension;
    hidden_size,
    depth,
    dropout,
    activation=:relu,
    seed=nothing,
)
    input_dimension > 0 ||
        throw(ArgumentError("input_dimension must be positive."))
    output_dimension > 0 ||
        throw(ArgumentError("output_dimension must be positive."))
    hidden_size > 0 ||
        throw(ArgumentError("hidden_size must be positive."))
    depth > 0 ||
        throw(ArgumentError("depth must be positive."))

    activation_fn = activation_function(activation)
    init = dense_initializer(seed)

    layers = Any[Dense(input_dimension => hidden_size, activation_fn; init=init)]

    for _ in 2:depth
        dropout > 0 && push!(layers, Dropout(dropout))
        push!(layers, Dense(hidden_size => hidden_size, activation_fn; init=init))
    end

    dropout > 0 && push!(layers, Dropout(dropout))
    push!(layers, Dense(hidden_size => output_dimension; init=init))
    push!(layers, x -> Flux.softplus.(x))

    return Chain(layers...) |> f64
end

function activation_function(activation)
    name = Symbol(lowercase(string(activation)))
    name == :relu && return Flux.relu
    name in (:silu, :swish) && return Flux.swish
    name in (:gelu, :geelu) && return Flux.gelu
    name == :tanh && return tanh
    name == :sigmoid && return Flux.sigmoid
    name in (:identity, :linear, :none) && return identity
    throw(
        ArgumentError(
            "unsupported activation $(activation); use relu, silu/swish, gelu/geelu, tanh, sigmoid, or identity.",
        ),
    )
end

function dense_initializer(seed)
    seed === nothing && return Flux.glorot_uniform
    seed === missing && return Flux.glorot_uniform
    return Flux.glorot_uniform(Random.MersenneTwister(Int(seed)))
end

function build_solver(config)
    solver_name = Symbol(config_value(config, :solver, :highs))
    if solver_name == :highs
        return ContextualDFL.Solver(ContextualDFL.IpoptSolver(), ContextualDFL.HiGHSSolver())
    end
    throw(ArgumentError("unsupported solver $(solver_name)"))
end

function build_loss(config, vector_decoder, reference_decoder, solver, program)
    loss_name = Symbol(config_value(config, :loss, :dfl_scen))
    if loss_name in (:dfl_scen, :spo_plus)
        nr_scenarios = config_value(config, :nr_scenarios, nothing)
        loss_type = loss_name == :dfl_scen ? ContextualDFL.DflScenLoss : ContextualDFL.SPOPlusLoss
        return loss_type(
            vector_decoder,
            reference_decoder,
            solver,
            program;
            nr_scenarios=isnothing(nr_scenarios) ? 1 : Int(nr_scenarios),
        )
    end
    throw(ArgumentError("unsupported loss $(loss_name); use :dfl_scen or :spo_plus"))
end

function split_contextual_dataset(
    dataset::AbstractVector;
    validation_fraction,
    test_fraction,
    rng::AbstractRNG,
)
    0 <= validation_fraction < 1 ||
        throw(ArgumentError("validation_fraction must be in [0, 1)."))
    0 <= test_fraction < 1 ||
        throw(ArgumentError("test_fraction must be in [0, 1)."))
    validation_fraction + test_fraction < 1 ||
        throw(ArgumentError("validation_fraction + test_fraction must be less than 1."))

    sample_count = length(dataset)
    sample_count > 0 || throw(ArgumentError("dataset must not be empty."))

    indices = randperm(rng, sample_count)
    test_count = floor(Int, test_fraction * sample_count)
    validation_count = floor(Int, validation_fraction * sample_count)
    train_count = sample_count - validation_count - test_count
    train_count > 0 || throw(ArgumentError("split leaves no training samples."))

    train_indices = indices[1:train_count]
    validation_indices = indices[(train_count + 1):(train_count + validation_count)]
    test_indices = indices[(train_count + validation_count + 1):end]

    return (;
        train=dataset[train_indices],
        validation=dataset[validation_indices],
        test=dataset[test_indices],
    )
end

# END FILE: src/ContextualDFL/ContextualDFLTraining/src/training_helpers.jl

# BEGIN FILE: src/ContextualDFL/ContextualDFLTraining/test/runtests.jl
using ContextualDFLTraining
using ContextualDFL
using ContextualDFLExperiments
using Flux
using Serialization
using Test

mutable struct FakeRun
    params::Vector{Tuple{String,String}}
    metrics::Vector{Tuple{String,Float64,Int}}
    tags::Vector{Tuple{String,String}}
    inputs::Vector{Any}
    artifacts::Vector{Tuple{String,Vector{UInt8}}}
    events::Vector{Symbol}
end

struct TrainingTestVectorDecoder <: ContextualDFL.VectorDecoder end

function flattened_trainables(model)
    return reduce(vcat, (vec(Array(parameter)) for parameter in Flux.trainables(model)))
end

@testset "ContextualDFLTraining model construction" begin
    for activation in (:relu, :silu, :swish, :gelu, :geelu, :tanh, :sigmoid, :identity)
        model = ContextualDFLTraining.build_neural_net(
            3,
            2;
            hidden_size=4,
            depth=2,
            dropout=0.0,
            activation=activation,
            seed=7,
        )
        @test size(model(ones(Float64, 3, 5))) == (2, 5)
    end

    @test_throws ArgumentError ContextualDFLTraining.build_neural_net(
        3,
        2;
        hidden_size=4,
        depth=1,
        dropout=0.0,
        activation=:unsupported,
        seed=7,
    )

    first_model = ContextualDFLTraining.build_neural_net(
        3,
        2;
        hidden_size=4,
        depth=2,
        dropout=0.0,
        seed=11,
    )
    same_seed_model = ContextualDFLTraining.build_neural_net(
        3,
        2;
        hidden_size=4,
        depth=2,
        dropout=0.0,
        seed=11,
    )
    different_seed_model = ContextualDFLTraining.build_neural_net(
        3,
        2;
        hidden_size=4,
        depth=2,
        dropout=0.0,
        seed=12,
    )

    @test flattened_trainables(first_model) == flattened_trainables(same_seed_model)
    @test flattened_trainables(first_model) != flattened_trainables(different_seed_model)
end

@testset "ContextualDFLTraining Flux checkpoints" begin
    mktempdir() do dir
        model = Chain(Dense(2 => 3, relu), Dense(3 => 1))
        opt_state = Flux.setup(Flux.Adam(0.01), model)
        train_result = (; model=model, opt_state=opt_state, history=[(; epoch=1, loss=0.25)])
        objects = (;
            data=(; train=[], validation=[], test=[]),
            model_metadata=(; output_dimension=1),
            data_metadata=(; dataset_name="checkpoint_test", dataset_digest="sha1:test"),
        )
        config = (;
            run_id="checkpoint/run 1",
            gridsearch_id="checkpoint_grid",
            checkpoint_dir=dir,
            checkpoint_format=:jls,
            validation_fraction=0.0,
        )
        path = ContextualDFLTraining.save_flux_checkpoint!(
            joinpath(dir, "checkpoint.jls"),
            model,
            train_result,
            objects,
            config,
        )
        payload = Serialization.deserialize(path)
        reloaded_model = Chain(Dense(2 => 3, relu), Dense(3 => 1))

        @test isfile(path)
        @test payload.format_version == 1
        @test payload.run_id == "checkpoint/run 1"
        @test payload.gridsearch_id == "checkpoint_grid"
        @test only(payload.epoch_history)[:epoch] == 1
        @test only(payload.epoch_history)[:loss] == 0.25
        @test payload.optimizer_state !== missing
        Flux.loadmodel!(reloaded_model, payload.model_state)
        @test Flux.state(reloaded_model) == payload.model_state
    end
end

@testset "ContextualDFLTraining experiments" begin
    spec = ContextualDFLTraining.load_experiment("ResourceAllocationExperiment1")
    @test spec.id == "resource_allocation/experiment_1"
    @test spec.name == "resource_allocation_experiment_1"
    @test ContextualDFLTraining.experiment_base_config(spec).experiment_id == spec.id
    @test isabspath(ContextualDFLTraining.optimal_results_path(spec, :test))
    @test !ContextualDFLTraining.experiment_has_function(spec, :grid_configs)
    @test !ContextualDFLTraining.experiment_has_function(spec, :smoke_configs)
    @test !isdefined(ContextualDFLTraining, :experiment_grid_configs)
    @test !isdefined(ContextualDFLTraining, :experiment_smoke_configs)
    @test !isdefined(ContextualDFLTraining, :experiment_problem_identity)
    @test !isdefined(ContextualDFLTraining, :resource_allocation_training_objects)
    @test !isdefined(ContextualDFLTraining, :resource_allocation_test_data_bundle)
    @test_throws ArgumentError ContextualDFLTraining.training_objects_for_config((; seed=1))

    training_data_dir = mktempdir()
    config = merge(
        ContextualDFLTraining.experiment_base_config(spec),
        (;
            optimality_evaluation=false,
            use_generated_test_data_artifact=false,
            training_data_dir=training_data_dir,
        ),
    )
    @test hasproperty(config, :training_context_count)
    @test hasproperty(config, :training_scenarios_per_context)
    @test hasproperty(config, :collection_duplicates_per_context)
    @test hasproperty(config, :validation_fraction)
    @test hasproperty(config, :generated_split_test_fraction)
    @test !hasproperty(config, :Nr_contexts)
    @test !hasproperty(config, :scenarios_per_context)
    @test !hasproperty(config, :test_fraction)
    @test !hasproperty(config, :n_samples)
    @test !hasproperty(config, :sigma)
    @test !hasproperty(config, :demand_power)
    @test !hasproperty(config, :context_terms)

    objects = ContextualDFLTraining.training_objects_for_config(config)
    @test objects.problem isa ResourceAllocationProblem
    @test objects.program isa ContextualDFL.StochasticProgram
    @test objects.solver isa ContextualDFL.Solver
    @test objects.scenario_decoder isa ResourceAllocationDemandVectorDecoder
    @test objects.reference_scenario_decoder isa ResourceAllocationDemandParametricDecoder
    @test objects.loss isa ContextualDFL.DflScenLoss
    @test hasproperty(objects, :target_extractor)
    target = objects.target_extractor(first(objects.data.train))
    @test target isa AbstractVector
    @test length(target) == objects.model_metadata.output_dimension
    @test objects.problem_metadata.problem == "resource_allocation"
    expected_dataset_name =
        "resource_allocation_experiment_1-ctx$(config.training_context_count)-scen$(config.training_scenarios_per_context)-dup$(config.collection_duplicates_per_context)-training_seed$(config.training_data_seed)"
    @test objects.data_metadata.dataset_name == expected_dataset_name
    @test startswith(objects.data_metadata.dataset_path, training_data_dir)
    @test isfile(objects.data_metadata.dataset_path)
    @test objects.data_metadata.training_data_cache_hit == false
    @test objects.data_metadata.training_context_count == config.training_context_count
    @test objects.data_metadata.training_scenarios_per_context ==
          config.training_scenarios_per_context
    @test objects.data_metadata.generated_split_test_fraction ==
          config.generated_split_test_fraction
    @test length(objects.data.train) == 100
    @test length(objects.data.validation) == 20
    @test length(objects.data.test) == 30
    artifact_path = objects.data_metadata.dataset_path
    @test ContextualDFLTraining.mlflow_dataset_name(objects, config) ==
          expected_dataset_name
    @test ContextualDFLTraining.mlflow_dataset_source(objects, config) == artifact_path
    @test ContextualDFLTraining.mlflow_dataset_source_type(objects, config) == "local"
    artifact_mtime = stat(artifact_path).mtime
    cached_objects = ContextualDFLTraining.training_objects_for_config(config)
    @test cached_objects.data_metadata.dataset_path == artifact_path
    @test cached_objects.data_metadata.training_data_cache_hit == true
    @test stat(artifact_path).mtime == artifact_mtime

    mktempdir() do dir
        config_dir = joinpath(dir, "toy")
        mkpath(config_dir)
        config_path = joinpath(config_dir, "Config.jl")
        write(
            config_path,
            """
            import ContextualDFLTraining

            experiment_id() = "toy/experiment"
            experiment_name() = "toy_experiment"
            experiment_module_name() = :ToyExperiment
            artifact_dir() = joinpath(@__DIR__, "artifacts")
            base_config() = (; experiment_id=experiment_id())
            training_objects(config) = nothing
            optimality_splits(objects, config) = Pair{Symbol,Any}[]
            optimal_results_path(split_name::Symbol) =
                joinpath(artifact_dir(), string(split_name) * ".jls")
            """,
        )

        toy_spec = ContextualDFLTraining.load_experiment(config_path)
        dataset = [(; context=[1.0], scenario_parameters=[2.0])]
        results = [(; objective_value=3.0)]
        path = ContextualDFLTraining.save_optimal_results!(
            toy_spec,
            :test,
            results;
            dataset=dataset,
        )

        @test isfile(path)
        @test ContextualDFLTraining.load_optimal_results(
            toy_spec,
            :test;
            dataset=dataset,
        ) == results
        @test_throws ArgumentError ContextualDFLTraining.load_optimal_results(
            toy_spec,
            :train;
            dataset=dataset,
        )
    end

    mktempdir() do dir
        config_dir = joinpath(dir, "toy_cached")
        mkpath(config_dir)
        config_path = joinpath(config_dir, "Config.jl")
        write(
            config_path,
            """
            experiment_id() = "toy/cached"
            experiment_name() = "toy_cached"
            experiment_module_name() = :ToyCachedExperiment
            artifact_dir() = joinpath(@__DIR__, "artifacts")
            base_config() = (; experiment_id=experiment_id(), seed=1, training_data_seed=1)
            training_data(config) = rand()
            training_data_identity(config) = (; training_data_seed=config.training_data_seed)
            training_dataset_name(config) = "toy_cached-\$(config.training_data_seed)"
            training_objects(config) = error("legacy path should not be used")
            training_objects(config, data) = (; data=data, data_metadata=(; generated_by="toy"))
            optimality_splits(objects, config) = Pair{Symbol,Any}[]
            optimal_results_path(split_name::Symbol) =
                joinpath(artifact_dir(), string(split_name) * ".jls")
            """,
        )

        cached_spec = ContextualDFLTraining.load_experiment(config_path)
        cached_config = merge(
            ContextualDFLTraining.experiment_base_config(cached_spec),
            (; seed=70, training_data_seed=7, training_data_dir=joinpath(dir, "cache")),
        )
        first_objects = ContextualDFLTraining.training_objects_for_config(cached_config)
        second_objects = ContextualDFLTraining.training_objects_for_config(cached_config)

        @test first_objects.data == second_objects.data
        @test first_objects.data_metadata.generated_by == "toy"
        @test first_objects.data_metadata.dataset_name == "toy_cached-7"
        @test isfile(first_objects.data_metadata.dataset_path)
        @test second_objects.data_metadata.training_data_cache_hit
    end
end

@testset "ContextualDFLTraining CSV results" begin
    mktempdir() do dir
        output_dir = ContextualDFLTraining.write_grid_results(
            [];
            output_root=dir,
            timestamp="empty",
        )

        @test output_dir == joinpath(dir, "empty")
        for filename in ("runs.csv", "epochs.csv", "failures.csv", "best.csv", "config.csv")
            @test isfile(joinpath(output_dir, filename))
        end
    end

    mktempdir() do dir
        result = (;
            status="ok",
            run_id="ok_1",
            started_at=0,
            finished_at=1,
            elapsed_seconds=1.0,
            error="",
            config=(; seed=1, learning_rate=0.001),
            worker=(; worker_id=1),
            final_metrics=(; validation_mse=0.1, train_mse=0.2),
            epoch_history=NamedTuple[],
        )

        output_dir = ContextualDFLTraining.write_grid_results(
            [result];
            output_root=dir,
            timestamp="success",
        )

        failures_path = joinpath(output_dir, "failures.csv")
        @test isfile(failures_path)
        @test filesize(failures_path) == 0
    end
end

@testset "ContextualDFLTraining profile training script" begin
    profile_module = Module(:ProfileTrainingScriptTest)
    Core.eval(profile_module, :(using Base))
    Core.eval(profile_module, :(include(path) = Base.include($profile_module, path)))
    Base.include(
        profile_module,
        joinpath(dirname(dirname(pathof(ContextualDFLTraining))), "profile_training.jl"),
    )
    profile_config_from_env = getfield(profile_module, :profile_config_from_env)
    experiment = getfield(profile_module, :load_experiment)("resource_allocation/experiment_1")

    withenv(
        "PROFILE_MLFLOW_ENABLED" => nothing,
        "PROFILE_MLFLOW_PROGRESS" => nothing,
        "PROFILE_MLFLOW_EXPERIMENT_ID" => "9999",
        "MLFLOW_EXPERIMENT_ID" => "8888",
    ) do
        config = profile_config_from_env(experiment)
        @test config.mlflow_enabled
        @test config.profile_mlflow_progress
        @test config.mlflow_experiment_id == "3"
        @test config.mlflow_experiment_name == "ContextualDFLProfiling"
        @test config.mlflow_tags.mlflow_experiment_name == "ContextualDFLProfiling"
    end

    withenv(
        "PROFILE_MLFLOW_ENABLED" => "0",
        "PROFILE_MLFLOW_PROGRESS" => nothing,
        "PROFILE_MLFLOW_EXPERIMENT_ID" => "9999",
        "MLFLOW_EXPERIMENT_ID" => "8888",
    ) do
        config = profile_config_from_env(experiment)
        @test !config.mlflow_enabled
        @test !config.profile_mlflow_progress
        @test config.mlflow_experiment_id == "3"
    end
end

@testset "ContextualDFLTraining generated test data" begin
    script_module = Module(:GenerateTestDataScriptTest)
    Base.include(
        script_module,
        joinpath(dirname(dirname(pathof(ContextualDFLTraining))), "generate_test_data.jl"),
    )
    parsed = getfield(script_module, :parse_commandline)(
        ["--experiment", "resource_allocation/experiment_1"],
    )
    @test parsed["seed"] == 1
    @test parsed["data-set-size"] == 30

    resource_spec = ContextualDFLTraining.load_experiment("resource_allocation/experiment_1")
    resource_bundle = ContextualDFLTraining.experiment_test_data_bundle(
        resource_spec;
        seed=5,
        data_set_size=2,
    )
    @test length(resource_bundle.dataset) == 2
    @test resource_bundle.problem isa ResourceAllocationProblem
    @test resource_bundle.data_metadata.data_set_size == 2

    mktempdir() do dir
        config_dir = joinpath(dir, "toy_generated")
        mkpath(config_dir)
        config_path = joinpath(config_dir, "Config.jl")
        write(
            config_path,
            """
            experiment_id() = "toy/generated"
            experiment_name() = "toy_generated"
            experiment_module_name() = :ToyGeneratedExperiment
            artifact_dir() = joinpath(@__DIR__, "artifacts")
            test_data_dir() = joinpath(artifact_dir(), "test_data")
            test_data_path(seed::Integer) =
                joinpath(test_data_dir(), "test_data_seed\$(Int(seed)).jls")
            test_optimal_results_path(seed::Integer) =
                joinpath(test_data_dir(), "optimal_solutions_seed\$(Int(seed)).jls")
            base_config() = (; experiment_id=experiment_id())
            training_objects(config) = nothing
            optimality_splits(objects, config) = Pair{Symbol,Any}[]
            optimal_results_path(split_name::Symbol) =
                joinpath(artifact_dir(), "legacy", string(split_name) * ".jls")
            """,
        )

        spec = ContextualDFLTraining.load_experiment(config_path)
        dataset = [
            (; context=[Float64(index)], scenario_parameters=[Float64(index + 1)]) for
            index in 1:3
        ]
        results = [(; objective_value=Float64(index)) for index in 1:3]
        dataset_seed10 = [
            (; context=[Float64(index)], scenario_parameters=[Float64(index + 1)]) for
            index in 11:13
        ]
        results_seed10 = [(; objective_value=Float64(index)) for index in 11:13]

        data_path = ContextualDFLTraining.save_test_data!(
            spec,
            7,
            dataset;
            data_set_size=3,
        )
        optimal_path = ContextualDFLTraining.save_test_optimal_results!(
            spec,
            7,
            results;
            dataset=dataset,
            data_set_size=3,
        )
        data_path_seed10 = ContextualDFLTraining.save_test_data!(
            spec,
            10,
            dataset_seed10;
            data_set_size=3,
        )
        optimal_path_seed10 = ContextualDFLTraining.save_test_optimal_results!(
            spec,
            10,
            results_seed10;
            dataset=dataset_seed10,
            data_set_size=3,
        )

        @test basename(data_path) == "test_data_seed7.jls"
        @test basename(optimal_path) == "optimal_solutions_seed7.jls"
        @test basename(data_path_seed10) == "test_data_seed10.jls"
        @test basename(optimal_path_seed10) == "optimal_solutions_seed10.jls"
        artifact = ContextualDFLTraining.load_test_data_artifact(spec)
        combined_dataset = vcat(dataset, dataset_seed10)
        combined_results = vcat(results, results_seed10)
        @test artifact.dataset == combined_dataset
        @test artifact.metadata.seed == 7
        @test artifact.metadata.seeds == [7, 10]
        @test artifact.metadata.data_set_size == 6
        @test artifact.metadata.data_set_sizes == [3, 3]
        @test basename.(artifact.metadata.paths) == ["test_data_seed7.jls", "test_data_seed10.jls"]
        @test length(artifact.metadata.dataset_digests) == 2
        @test ContextualDFLTraining.load_test_data(spec) == combined_dataset
        @test ContextualDFLTraining.load_optimal_results(spec, :test) ==
              combined_results
        @test ContextualDFLTraining.load_optimal_results(
            spec,
            :test;
            dataset=combined_dataset,
        ) == combined_results
        @test ContextualDFLTraining.load_optimal_results(
            spec,
            :test;
            dataset=combined_dataset[1:4],
        ) == combined_results[1:4]
        @test_throws ArgumentError ContextualDFLTraining.load_optimal_results(
            spec,
            :test;
            dataset=combined_dataset[2:4],
        )

        missing_optimal_dataset = [
            (; context=[Float64(index)], scenario_parameters=[Float64(index + 1)]) for
            index in 21:23
        ]
        ContextualDFLTraining.save_test_data!(
            spec,
            12,
            missing_optimal_dataset;
            data_set_size=3,
        )
        @test_throws ArgumentError ContextualDFLTraining.load_optimal_results(spec, :test)

        mismatched_dataset = [
            (; context=[Float64(index)], scenario_parameters=[Float64(index + 1), Float64(index + 2)]) for
            index in 31:33
        ]
        ContextualDFLTraining.save_test_data!(
            spec,
            13,
            mismatched_dataset;
            data_set_size=3,
        )
        @test_throws ArgumentError ContextualDFLTraining.load_test_data(spec)
    end
end

@testset "ContextualDFLTraining grid file config" begin
    gridsearch_module = Module(:GridSearchScriptTest)
    Core.eval(gridsearch_module, :(using Base))
    Core.eval(gridsearch_module, :(include(path) = Base.include($gridsearch_module, path)))
    Base.include(
        gridsearch_module,
        joinpath(dirname(dirname(pathof(ContextualDFLTraining))), "gridsearch.jl"),
    )
    grid_load_experiment = getfield(gridsearch_module, :load_experiment)
    grid_load_grid_config = getfield(gridsearch_module, :load_grid_config)
    selected_grid = getfield(gridsearch_module, :selected_grid)

    experiment = grid_load_experiment("resource_allocation/experiment_1")

    @testset "bundled resource allocation configs" begin
        default_spec = grid_load_grid_config(
            joinpath(experiment.root_dir, "grid_configs", "default.yaml"),
        )
        smoke_spec = grid_load_grid_config(
            joinpath(experiment.root_dir, "grid_configs", "smoke.yaml"),
        )

        default_configs = selected_grid(experiment, default_spec)
        smoke_configs = selected_grid(experiment, smoke_spec)
        @test length(default_configs) == 24
        @test length(smoke_configs) == 1
        for config in vcat(default_configs, smoke_configs)
            @test !hasproperty(config, :n_samples)
            @test !hasproperty(config, :sigma)
            @test !hasproperty(config, :demand_power)
            @test !hasproperty(config, :context_terms)
            @test !hasproperty(config, :Nr_contexts)
            @test !hasproperty(config, :scenarios_per_context)
            @test !hasproperty(config, :test_fraction)
            @test hasproperty(config, :training_context_count)
            @test hasproperty(config, :training_scenarios_per_context)
            @test hasproperty(config, :collection_duplicates_per_context)
        end
    end

    mktempdir() do dir
        yaml_path = joinpath(dir, "data_grid.yaml")
        write(
            yaml_path,
            """
            version: 1
            name: data_grid
            fixed:
              learning_rate: 0.001
              hidden_size: 16
              depth: 1
              batch_size: 4
              dropout: 0.0
              training_context_count: 12
              training_scenarios_per_context: 2
              collection_duplicates_per_context: 1
              validation_fraction: 0.25
              generated_split_test_fraction: 0.0
            grid:
              seed: [1]
            """,
        )

        configs = selected_grid(experiment, grid_load_grid_config(yaml_path))
        @test length(configs) == 1
        @test only(configs).training_context_count == 12
        @test only(configs).training_scenarios_per_context == 2
        @test only(configs).validation_fraction == 0.25
    end

    mktempdir() do dir
        yaml_path = joinpath(dir, "repeat_activation_grid.yaml")
        write(
            yaml_path,
            """
            version: 1
            name: repeat_activation_grid
            base:
              repeat_count: 2
            fixed:
              learning_rate: 0.001
              hidden_size: 16
              depth: 1
              batch_size: 4
              dropout: 0.0
            grid:
              activation: [relu, gelu]
              seed: [1]
            """,
        )

        configs = selected_grid(experiment, grid_load_grid_config(yaml_path))
        @test length(configs) == 2
        @test Set(config.activation for config in configs) == Set([:relu, :gelu])
        @test all(config -> config.repeat_count == 2, configs)
    end

    mktempdir() do dir
        yaml_path = joinpath(dir, "grid.yaml")
        write(
            yaml_path,
            """
            version: 1
            name: yaml_grid
            base:
              epochs: 3
              optimality_evaluation: false
            fixed:
              depth: 1
              batch_size: 4
              dropout: 0.0
            grid:
              learning_rate: [0.001, 0.0005]
              hidden_size: [16, 32]
              seed: [1]
            schedules:
              mu:
                kind: geometric
                start: 1.0
                stop: 0.01
              mu_ref:
                kind: match_input
              rho:
                kind: linear
                start: 0.3
                stop: 0.1
              rho_ref:
                kind: zero
            run_id_template: "{name}_{index}_{hash}"
            """,
        )

        spec = grid_load_grid_config(yaml_path)
        configs = selected_grid(experiment, spec)
        resolved_json = ContextualDFLTraining.resolved_grid_json(configs)
        digest = ContextualDFLTraining.grid_config_digest(configs)

        @test spec.format == :yaml
        @test length(configs) == 4
        @test all(config -> config.experiment_id == experiment.id, configs)
        @test all(config -> config.optimality_evaluation == false, configs)
        @test Set(config.learning_rate for config in configs) == Set([0.001, 0.0005])
        @test Set(config.hidden_size for config in configs) == Set([16, 32])
        @test all(config -> config.mu_schedule == :geometric, configs)
        @test all(config -> config.mu_start == 1.0, configs)
        @test all(config -> config.mu_end == 0.01, configs)
        @test all(config -> config.mu_ref_schedule == :match_input, configs)
        @test all(config -> config.rho_schedule == :linear, configs)
        @test all(config -> config.rho_start == 0.3, configs)
        @test all(config -> config.rho_end == 0.1, configs)
        @test all(config -> config.rho_ref_schedule == :zero, configs)
        @test all(config -> startswith(config.run_id, "yaml_grid_"), configs)
        @test startswith(digest, "sha256:")
        @test all(config -> config.grid_config_digest == digest, configs)
        @test occursin("\"grid_config_name\"", resolved_json)
        @test !occursin("grid_config_digest", resolved_json)

        write(
            yaml_path,
            """

            version: 1
            name: yaml_grid
            base:
              epochs: 3
              optimality_evaluation: false
            fixed:
              depth: 1
              batch_size: 4
              dropout: 0.0
            grid:
              learning_rate: [0.001, 0.0005]
              hidden_size: [16, 32]
              seed: [1]
            schedules:
              mu:
                kind: geometric
                start: 1.0
                stop: 0.01
              mu_ref:
                kind: match_input
              rho:
                kind: linear
                start: 0.3
                stop: 0.1
              rho_ref:
                kind: zero
            run_id_template: "{name}_{index}_{hash}"

            """,
        )
        blank_line_configs = selected_grid(
            experiment,
            grid_load_grid_config(yaml_path),
        )
        @test ContextualDFLTraining.resolved_grid_json(blank_line_configs) == resolved_json
        @test ContextualDFLTraining.grid_config_digest(blank_line_configs) == digest
    end

    mktempdir() do dir
        nested_path = joinpath(dir, "nested_schedules.yaml")
        write(
            nested_path,
            """
            version: 1
            name: nested_schedule_grid
            base:
              epochs: 3
            fixed:
              learning_rate: 0.001
              hidden_size: 16
              depth: 1
              batch_size: 4
              dropout: 0.0
            grid:
              seed: [1]
              schedules:
                mu:
                  - kind: geometric
                    start: 1.0
                    end: 0.01
                  - kind: values
                    values: [1.0, 0.5, 0.25]
                mu_ref:
                  - kind: match_input
                rho:
                  - kind: constant
                    value: 0.0
                rho_ref:
                  - kind: match_input
            schedules:
              mu:
                kind: constant
                value: 0.5
            """,
        )

        configs = selected_grid(experiment, grid_load_grid_config(nested_path))
        @test length(configs) == 2
        @test count(config -> config.mu_schedule == :geometric, configs) == 1
        @test count(config -> config.mu_schedule == [1.0, 0.5, 0.25], configs) == 1
        geometric_config = only([config for config in configs if config.mu_schedule == :geometric])
        manual_config = only([config for config in configs if config.mu_schedule isa AbstractVector])
        @test geometric_config.mu_end == 0.01
        @test manual_config.mu_ref_schedule == :match_input
        @test all(config -> config.rho_schedule == :constant, configs)
        @test all(config -> config.rho == 0.0, configs)
        @test all(config -> config.rho_ref_schedule == :match_input, configs)
    end

    mktempdir() do dir
        for problem_key in (
            "problem",
            "demand_sigma",
            "sigma",
            "demand_power",
            "context_terms",
        )
            invalid_problem_key_path = joinpath(dir, "problem_key_$(problem_key).yaml")
            write(
                invalid_problem_key_path,
                """
                version: 1
                name: invalid_problem_key
                fixed:
                  learning_rate: 0.001
                  hidden_size: 16
                  depth: 1
                  batch_size: 4
                  dropout: 0.0
                  $(problem_key): 16
                grid:
                  seed: [1]
                """,
            )

            @test_throws ArgumentError selected_grid(
                experiment,
                grid_load_grid_config(invalid_problem_key_path),
            )
        end
    end

    mktempdir() do dir
        json_path = joinpath(dir, "grid.json")
        write(
            json_path,
            """
            {
              "version": 1,
              "name": "json_grid",
              "fixed": {
                "learning_rate": 0.001,
                "hidden_size": 16,
                "depth": 1,
                "batch_size": 4,
                "dropout": 0.0
              },
              "grid": {
                "seed": [1, 2]
              },
              "schedules": {
                "mu": {"kind": "constant", "value": 0.25}
              }
            }
            """,
        )

        spec = grid_load_grid_config(json_path)
        configs = selected_grid(experiment, spec)

        @test spec.format == :json
        @test length(configs) == 2
        @test Set(config.seed for config in configs) == Set([1, 2])
        @test all(config -> config.mu_schedule == :constant, configs)
        @test all(config -> config.mu == 0.25, configs)
    end

    mktempdir() do dir
        piecewise_path = joinpath(dir, "piecewise.yaml")
        write(
            piecewise_path,
            """
            version: 1
            name: manual_schedule_grid
            base:
              epochs: 6
            fixed:
              learning_rate: 0.001
              hidden_size: 16
              depth: 1
              batch_size: 4
              dropout: 0.0
            grid:
              seed: [1]
            schedules:
              mu:
                kind: piecewise
                segments:
                  - epochs: 2
                    value: 1.0
                  - epochs: 3
                    value: 0.9
                  - epochs: 1
                    value: 0.4
              mu_ref:
                kind: values
                values: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
              rho:
                kind: piecewise
                segments:
                  - epochs: 3
                    value: 0.2
                  - epochs: 3
                    value: 0.05
              rho_ref:
                kind: values
                values: [0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
            """,
        )

        spec = grid_load_grid_config(piecewise_path)
        config = only(selected_grid(experiment, spec))

        @test config.mu_schedule == [1.0, 1.0, 0.9, 0.9, 0.9, 0.4]
        @test config.mu_ref_schedule == [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        @test config.rho_schedule == [0.2, 0.2, 0.2, 0.05, 0.05, 0.05]
        @test config.rho_ref_schedule == [0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
        @test ContextualDFLTraining.mu_schedule_for_config(config) ==
              [1.0, 1.0, 0.9, 0.9, 0.9, 0.4]
        @test ContextualDFLTraining.mu_ref_schedule_for_config(config) ==
              [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        @test ContextualDFLTraining.rho_schedule_for_config(config) ==
              [0.2, 0.2, 0.2, 0.05, 0.05, 0.05]
        @test ContextualDFLTraining.rho_ref_schedule_for_config(config) ==
              [0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
    end

    mktempdir() do dir
        values_path = joinpath(dir, "values.yaml")
        write(
            values_path,
            """
            version: 1
            name: values_schedule_grid
            base:
              epochs: 3
            fixed:
              learning_rate: 0.001
              hidden_size: 16
              depth: 1
              batch_size: 4
              dropout: 0.0
            grid:
              seed: [1]
            schedules:
              mu:
                kind: values
                values: [1.0, 0.5, 0.25]
            """,
        )

        config = only(
            selected_grid(
                experiment,
                grid_load_grid_config(values_path),
            ),
        )

        @test config.mu_schedule == [1.0, 0.5, 0.25]
        @test ContextualDFLTraining.mu_schedule_for_config(config) == [1.0, 0.5, 0.25]
        @test ContextualDFLTraining.mu_ref_schedule_for_config(config) == [1.0, 0.5, 0.25]
    end

    @testset "manual schedule validation" begin
        @test ContextualDFLTraining.mu_schedule_for_config(
            (; epochs=3, mu=0.0, mu_schedule=[1, 2, 3]),
        ) == [1.0, 2.0, 3.0]
        @test_throws ArgumentError ContextualDFLTraining.mu_schedule_for_config(
            (; epochs=3, mu=0.0, mu_schedule=[1, 2]),
        )
        @test ContextualDFLTraining.rho_schedule_for_config(
            (; epochs=3, rho=0.0, rho_schedule=[0.3, 0.2, 0.1]),
        ) == [0.3, 0.2, 0.1]
        @test ContextualDFLTraining.rho_ref_schedule_for_config(
            (; epochs=3, rho=0.0, rho_schedule=[0.3, 0.2, 0.1], rho_ref_schedule=:match_input),
        ) == [0.3, 0.2, 0.1]
        @test_throws ArgumentError ContextualDFLTraining.rho_schedule_for_config(
            (; epochs=3, rho=0.0, rho_schedule=[0.3, 0.2]),
        )

        mktempdir() do dir
            empty_values_path = joinpath(dir, "empty_values.yaml")
            write(
                empty_values_path,
                """
                version: 1
                schedules:
                  mu:
                    kind: values
                    values: []
                """,
            )
            @test_throws ArgumentError ContextualDFLTraining.load_grid_config(empty_values_path)

            empty_segments_path = joinpath(dir, "empty_segments.yaml")
            write(
                empty_segments_path,
                """
                version: 1
                schedules:
                  mu:
                    kind: piecewise
                    segments: []
                """,
            )
            @test_throws ArgumentError ContextualDFLTraining.load_grid_config(empty_segments_path)

            non_positive_path = joinpath(dir, "non_positive.yaml")
            write(
                non_positive_path,
                """
                version: 1
                schedules:
                  mu:
                    kind: piecewise
                    segments:
                      - epochs: 0
                        value: 1.0
                """,
            )
            @test_throws ArgumentError ContextualDFLTraining.load_grid_config(non_positive_path)

            wrong_length_path = joinpath(dir, "wrong_length.yaml")
            write(
                wrong_length_path,
                """
                version: 1
                base:
                  epochs: 3
                schedules:
                  mu:
                    kind: values
                    values: [1.0, 0.5]
                """,
            )
            @test_throws ArgumentError selected_grid(
                experiment,
                grid_load_grid_config(wrong_length_path),
            )
        end
    end

    @testset "repeat annotation and aggregates" begin
        annotate_parents = getfield(gridsearch_module, :annotate_grid_config_parents)
        create_parent_runs = getfield(gridsearch_module, :create_mlflow_config_parent_runs)
        annotate_repeats = getfield(gridsearch_module, :annotate_repeat_configs)
        parent_results = getfield(gridsearch_module, :config_parent_results)

        settings = (;
            enabled=false,
            experiment_id="",
            experiment_name="",
            deterministic_experiment_id="",
            tracking_uri="",
            upload_model_artifact=false,
        )
        base_config = (;
            run_id="base",
            seed=11,
            repeat_count=2,
            learning_rate=0.001,
        )
        repeat_seeds = [101, 202]
        parents = annotate_parents(
            [base_config],
            "12345",
            settings,
            "";
            repeat_training_data_seeds=repeat_seeds,
        )
        parent_runs = create_parent_runs(settings, parents)
        children = annotate_repeats(
            parents,
            parent_runs,
            settings;
            repeat_training_data_seeds=repeat_seeds,
        )

        @test length(parents) == 1
        @test only(parents).run_id ==
              "gridsearch_12345__candidate_0001__base"
        @test only(parents).repeat_training_data_seed_sequence == "101,202"
        @test length(children) == 2
        @test children[1].run_id ==
              "gridsearch_12345__candidate_0001__base__repeat_001"
        @test children[2].run_id ==
              "gridsearch_12345__candidate_0001__base__repeat_002"
        @test all(child -> child.seed == 11, children)
        @test children[1].training_data_seed == 101
        @test children[2].training_data_seed == 202
        @test children[1].repeat_training_data_seed == 101
        @test children[2].repeat_training_data_seed == 202
        @test children[1].mlflow_tags.training_data_seed == 101
        @test children[1].mlflow_tags.repeat_training_data_seed == 101
        @test all(child -> child.training_data_cache == false, children)
        @test all(child -> child.write_training_data_artifact == false, children)

        mlflow_params = ContextualDFLTraining.mlflow_params_for_config(children[1])
        @test mlflow_params["config_training_data_seed"] == "101"
        @test mlflow_params["config_repeat_training_data_seed"] == "101"

        comparison_configs = [
            base_config,
            merge(base_config, (; run_id="base_other", learning_rate=0.002)),
        ]
        comparison_parents = annotate_parents(
            comparison_configs,
            "12345",
            settings,
            "";
            repeat_training_data_seeds=repeat_seeds,
        )
        comparison_children = annotate_repeats(
            comparison_parents,
            create_parent_runs(settings, comparison_parents),
            settings;
            repeat_training_data_seeds=repeat_seeds,
        )
        @test comparison_children[1].training_data_seed == comparison_children[3].training_data_seed
        @test comparison_children[2].training_data_seed == comparison_children[4].training_data_seed
        @test comparison_children[1].training_data_seed != comparison_children[2].training_data_seed

        child_results = [
            (;
                status="ok",
                run_id=children[1].run_id,
                config=children[1],
                worker=NamedTuple(),
                final_metrics=(; validation_mse=1.0),
                epoch_history=Dict{Symbol,Any}[],
                error="",
                started_at=1,
                finished_at=2,
                elapsed_seconds=1.0,
            ),
            (;
                status="ok",
                run_id=children[2].run_id,
                config=children[2],
                worker=NamedTuple(),
                final_metrics=(; validation_mse=3.0),
                epoch_history=Dict{Symbol,Any}[],
                error="",
                started_at=2,
                finished_at=3,
                elapsed_seconds=1.0,
            ),
        ]
        config_result = only(parent_results(parents, child_results))
        summary = config_result.aggregate_metrics[:validation_mse]

        @test config_result.status == "ok"
        @test config_result.final_metrics.validation_mse == 2.0
        @test summary.count == 2.0
        @test summary.mean == 2.0
        @test summary.std ≈ sqrt(2.0)
        @test summary.stderr == 1.0
    end

    @testset "policy inference smoothing defaults" begin
        policy_objects = (;
            scenario_decoder=TrainingTestVectorDecoder(),
            reference_scenario_decoder=TrainingTestVectorDecoder(),
            solver=:solver,
            program=:program,
        )
        base_policy_config = (; loss=:dfl_scen, solver=:highs, mu=0.0, mu_schedule=:constant, rho=0.0, rho_schedule=:constant)
        annealed_config = merge(
            base_policy_config,
            (;
                epochs=3,
                mu=9.0,
                mu_schedule=:geometric,
                mu_start=1.0,
                mu_end=0.05,
            ),
        )
        manual_config = merge(
            base_policy_config,
            (; epochs=3, mu=9.0, mu_schedule=[1.0, 0.5, 0.25]),
        )
        override_config = merge(manual_config, (; policy_inference_mu=0.7))
        null_override_config = merge(manual_config, (; policy_inference_mu=nothing))
        zero_epoch_config =
            merge(base_policy_config, (; epochs=0, mu=0.4, mu_schedule=:constant))

        @test ContextualDFLTraining.policy_inference_mu_for_config(annealed_config) ≈ 0.05
        @test ContextualDFLTraining.policy_inference_mu_for_config(manual_config) == 0.25
        @test ContextualDFLTraining.policy_inference_mu_for_config(override_config) == 0.7
        @test ContextualDFLTraining.policy_inference_mu_for_config(null_override_config) ==
              0.25
        @test ContextualDFLTraining.policy_inference_mu_for_config(zero_epoch_config) == 0.4

        policy = ContextualDFLTraining.optimality_policy(
            identity,
            policy_objects,
            annealed_config,
        )
        annealed_method_spec =
            ContextualDFLTraining.mlflow_method_spec(policy_objects, annealed_config)
        method_spec = ContextualDFLTraining.mlflow_method_spec(policy_objects, manual_config)

        @test policy.mu ≈ 0.05
        @test annealed_method_spec.policy_inference_mu == policy.mu
        @test method_spec.policy_inference_mu == 0.25

        rho_annealed_config = merge(
            base_policy_config,
            (;
                epochs=3,
                rho=9.0,
                rho_schedule=:linear,
                rho_start=0.3,
                rho_end=0.1,
                policy_inference_rho=nothing,
            ),
        )
        rho_manual_config = merge(
            base_policy_config,
            (; epochs=3, rho=9.0, rho_schedule=[0.3, 0.2, 0.1], policy_inference_rho=nothing),
        )
        rho_override_config = merge(rho_manual_config, (; policy_inference_rho=0.7))
        rho_zero_epoch_config =
            merge(base_policy_config, (; epochs=0, rho=0.4, rho_schedule=:constant, policy_inference_rho=nothing))

        @test ContextualDFLTraining.policy_inference_rho_for_config(rho_annealed_config) ≈ 0.1
        @test ContextualDFLTraining.policy_inference_rho_for_config(rho_manual_config) == 0.1
        @test ContextualDFLTraining.policy_inference_rho_for_config(rho_override_config) == 0.7
        @test ContextualDFLTraining.policy_inference_rho_for_config(rho_zero_epoch_config) == 0.4

        rho_policy = ContextualDFLTraining.optimality_policy(
            identity,
            policy_objects,
            rho_annealed_config,
        )
        rho_method_spec =
            ContextualDFLTraining.mlflow_method_spec(policy_objects, rho_manual_config)

        @test rho_policy.rho ≈ 0.1
        @test rho_method_spec.policy_inference_rho == 0.1
        @test rho_method_spec.quadratic_smoothing_training == true
    end

    mktempdir() do dir
        invalid_path = joinpath(dir, "invalid.yaml")
        write(
            invalid_path,
            """
            version: 1
            surprise: true
            """,
        )

        @test_throws ArgumentError ContextualDFLTraining.load_grid_config(invalid_path)
    end
end

FakeRun() = FakeRun(
    Tuple{String,String}[],
    Tuple{String,Float64,Int}[],
    Tuple{String,String}[],
    Any[],
    Tuple{String,Vector{UInt8}}[],
    Symbol[],
)

struct FakeMLFlow end

function ContextualDFLTraining.logparam(::FakeMLFlow, run::FakeRun, key, value)
    value isa String || throw(ArgumentError("MLflow params must be strings."))
    push!(run.params, (string(key), value))
    push!(run.events, :param)
    return nothing
end

function ContextualDFLTraining.logmetric(
    ::FakeMLFlow,
    run::FakeRun,
    key,
    value;
    step,
    timestamp=missing,
)
    value isa Float64 || throw(ArgumentError("MLflow metrics must be Float64."))
    timestamp === missing || timestamp isa Int64 ||
        throw(ArgumentError("MLflow metric timestamps must be Int64."))
    push!(run.metrics, (string(key), value, Int(step)))
    push!(run.events, :metric)
    return nothing
end

function ContextualDFLTraining.logbatch(::FakeMLFlow, run::FakeRun; metrics=[], params=[], tags=[])
    for metric in metrics
        step = getproperty(metric, :step)
        push!(
            run.metrics,
            (
                string(getproperty(metric, :key)),
                Float64(getproperty(metric, :value)),
                step === nothing ? 0 : Int(step),
            ),
        )
    end

    for param in params
        push!(
            run.params,
            (string(getproperty(param, :key)), string(getproperty(param, :value))),
        )
    end

    for tag in tags
        push!(
            run.tags,
            (string(getproperty(tag, :key)), string(getproperty(tag, :value))),
        )
    end

    push!(run.events, :batch)
    return nothing
end

function ContextualDFLTraining.setruntag(::FakeMLFlow, run::FakeRun, key, value)
    push!(run.tags, (string(key), string(value)))
    push!(run.events, :tag)
    return nothing
end

function ContextualDFLTraining.loginputs(::FakeMLFlow, run::FakeRun; datasets)
    append!(run.inputs, datasets)
    push!(run.events, :input)
    return nothing
end

function ContextualDFLTraining.uploadartifact(
    ::FakeMLFlow,
    artifact_path::AbstractString,
    data::Vector{UInt8},
)
    push!(GLOBAL_ARTIFACT_RUN[], (string(artifact_path), data))
    return nothing
end

function ContextualDFLTraining.uploadartifact(
    ::FakeMLFlow,
    ::FakeRun,
    artifact_path::AbstractString,
    data::Vector{UInt8},
)
    push!(GLOBAL_ARTIFACT_RUN[], (string(artifact_path), data))
    return nothing
end

const GLOBAL_ARTIFACT_RUN = Ref{Vector{Tuple{String,Vector{UInt8}}}}(
    Tuple{String,Vector{UInt8}}[],
)

@testset "ContextualDFLTraining MLflow support" begin
    @testset "logs params and epoch metrics" begin
        mlf = FakeMLFlow()
        run = FakeRun()

        ContextualDFLTraining.log_mlflow_params!(
            mlf,
            run,
            "model",
            (; depth=2, hidden_size=64, nested=(; activation=:relu), skipped=[1, 2]),
        )
        ContextualDFLTraining.log_contextualdfl_training_params!(
            mlf,
            run,
            :dfl_scen,
            (;
                learning_rate=0.001,
                epochs=2,
                batch_size=8,
                training_data_seed=101,
                repeat_training_data_seed=101,
                repeat_index=1,
            ),
            [0.1, 0.01],
            [0.1, 0.01],
            [0.0, 0.0],
            [0.0, 0.0],
        )
        ContextualDFLTraining.log_mlflow_epoch!(
            mlf,
            run,
            2,
            1.5,
            2.5,
            (;
                mu=0.1,
                mu_in=0.1,
                mu_ref=0.0,
                rho_in=0.2,
                rho_ref=0.05,
                iterations=3,
                epoch_seconds=0.25,
                real_display_loss=0.75,
            ),
        )

        params = Dict(run.params)
        @test params["model_depth"] == "2"
        @test params["model_hidden_size"] == "64"
        @test params["model_nested_activation"] == "relu"
        @test !haskey(params, "model_skipped")
        @test params["training_data_seed"] == "101"
        @test params["repeat_training_data_seed"] == "101"
        @test params["repeat_index"] == "1"

        metrics = Dict(metric[1] => metric[2] for metric in run.metrics)
        @test metrics["loss"] == 1.5
        @test metrics["epoch_mu_in"] == 0.1
        @test metrics["epoch_mu_ref"] == 0.0
        @test metrics["epoch_rho_in"] == 0.2
        @test metrics["epoch_rho_ref"] == 0.05
        @test metrics["epoch_iterations"] == 3.0
        @test metrics["epoch_seconds"] == 0.25
        @test metrics["display_loss"] == 2.5
        @test metrics["real_display_loss"] == 0.75
        @test !haskey(metrics, "epoch_mu")
        @test all(metric -> metric[3] == 2, run.metrics)
        @test count(==(:batch), run.events) == 1
    end

    @testset "logs evaluation metrics, datasets, tags, and artifacts" begin
        mlf = FakeMLFlow()
        run = FakeRun()
        empty!(GLOBAL_ARTIFACT_RUN[])

        mktempdir() do dir
            artifact_path = joinpath(dir, "report.txt")
            write(artifact_path, "ok")
            ContextualDFLTraining.log_mlflow_evaluation_result!(
                mlf,
                run,
                "",
                (; metrics=(; validation_mse=1.25), artifacts=(; report=artifact_path)),
            )
        end
        append!(run.artifacts, GLOBAL_ARTIFACT_RUN[])

        ContextualDFLTraining.log_mlflow_source_tags!(
            mlf,
            run;
            source_name="ContextualDFLTraining/gridsearch.jl",
            source_type="LOCAL",
            source_git_commit="abc123",
        )
        ContextualDFLTraining.log_mlflow_dataset!(
            mlf,
            run;
            dataset_name="resource_allocation_generated",
            dataset_digest="sha1:" * repeat("a", 40),
            dataset_source_type="generated",
            dataset_source="generated:test",
            dataset_context="training",
        )

        @test ("validation_mse", 1.25, 0) in run.metrics
        @test only(run.artifacts)[1] == "report"
        @test !isempty(only(run.artifacts)[2])
        @test Dict(run.tags)["mlflow.source.name"] == "ContextualDFLTraining/gridsearch.jl"
        @test length(run.inputs) == 1
        @test length(only(run.inputs).dataset.digest) <= 36
    end

    @testset "logs failure stacktraces as artifacts" begin
        mlf = FakeMLFlow()
        run = FakeRun()
        empty!(GLOBAL_ARTIFACT_RUN[])

        ContextualDFLTraining.log_mlflow_stacktrace_artifact!(mlf, run, "full stacktrace")
        append!(run.artifacts, GLOBAL_ARTIFACT_RUN[])

        @test only(run.artifacts)[1] == "errors/stacktrace.txt"
        @test String(only(run.artifacts)[2]) == "full stacktrace"
        @test isempty(run.tags)
        @test ContextualDFLTraining.mlflow_run_artifact_path(
            (; info=(; artifact_uri="mlflow-artifacts:/3/run-id/artifacts")),
            "errors/stacktrace.txt",
        ) == "3/run-id/artifacts/errors/stacktrace.txt"
    end
end

# END FILE: src/ContextualDFL/ContextualDFLTraining/test/runtests.jl
