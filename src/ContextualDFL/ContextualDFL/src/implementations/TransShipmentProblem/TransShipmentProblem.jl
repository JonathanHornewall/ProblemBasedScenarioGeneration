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
