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
    end

    return parse_args(args, settings)
end

function positive_int(value, name::AbstractString)
    value = Int(value)
    value > 0 || throw(ArgumentError("$name must be positive, got $value."))
    return value
end

function main()
    parsed_args = parse_commandline()
    experiment = ContextualDFLTraining.load_experiment(parsed_args["experiment"])
    seed = Int(parsed_args["seed"])
    data_set_size = positive_int(parsed_args["data-set-size"], "data-set-size")

    config = ContextualDFLTraining.experiment_test_data_config(
        experiment;
        seed=seed,
        data_set_size=data_set_size,
    )
    bundle = ContextualDFLTraining.experiment_test_data_bundle(
        experiment;
        seed=seed,
        data_set_size=data_set_size,
    )
    dataset = bundle.dataset

    println(
        "Generated test data for experiment=$(experiment.id), seed=$seed, rows=$(length(dataset))",
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
        )
    end

    optimal_results_path = ContextualDFLTraining.save_test_optimal_results!(
        experiment,
        seed,
        results;
        dataset=dataset,
        data_set_size=data_set_size,
        metadata=(; solve_seconds=solve_seconds),
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
