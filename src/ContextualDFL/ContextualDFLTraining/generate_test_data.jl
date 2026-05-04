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
            help = "Number of Monte Carlo evaluation batches stored with the optimal solutions."
            arg_type = Int
            default = 1
        "--evaluate-mode"
            help = "Optimal-solution evaluation mode: mean_only or batched."
            default = "batched"
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

function checked_evaluate_mode(value)
    mode = Symbol(value)
    mode in (:mean_only, :batched) ||
        throw(ArgumentError("evaluate-mode must be mean_only or batched, got $value."))
    return mode
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
    evaluate_mode = checked_evaluate_mode(parsed_args["evaluate-mode"])
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
            evaluate_mode=evaluate_mode,
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
            evaluate_mode=string(evaluate_mode),
            evaluation_batches=evaluation_batches,
        ),
    )
    println("Wrote optimal solutions to $optimal_results_path")

    loaded_dataset = ContextualDFLTraining.load_test_data(experiment)
    length(loaded_dataset) == data_set_size ||
        error("saved test data at $test_data_path have the wrong length")
    loaded_results = ContextualDFLTraining.load_optimal_results(
        experiment,
        :test;
        dataset=dataset,
    )
    length(loaded_results) == data_set_size ||
        error("saved optimal solutions at $optimal_results_path have the wrong length")
    println("Finished in $(round(solve_seconds; digits=3)) seconds")

    return (; test_data_path=test_data_path, optimal_results_path=optimal_results_path)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
